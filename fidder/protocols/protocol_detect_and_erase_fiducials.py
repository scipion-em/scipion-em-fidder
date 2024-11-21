# -*- coding: utf-8 -*-
# **************************************************************************
# *
# * Authors:     Scipion Team
# *
# * National Center of Biotechnology, CSIC, Spain
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307  USA
# *
# *  All comments concerning this program package may be sent to the
# *  e-mail address 'scipion@cnb.csic.es'
# *
# **************************************************************************
import glob
import logging
from enum import Enum
from os.path import join, basename
from typing import Union, List
import mrcfile
import numpy as np
from typing_extensions import Tuple
from fidder import Plugin
from pwem.emlib import DT_FLOAT
from pwem.emlib.image import ImageHandler
from pwem.protocols import EMProtocol
from pyworkflow.constants import BETA
from pyworkflow.object import Set, Pointer
from pyworkflow.protocol import PointerParam, FloatParam, GT, LE, GPU_LIST, StringParam, BooleanParam, LEVEL_ADVANCED, \
    STEPS_PARALLEL
from pyworkflow.utils import Message, makePath, cyanStr
from tomo.objects import SetOfTiltSeries, TiltSeries, TiltImage

logger = logging.getLogger(__name__)
# Form variables
IN_TS_SET = 'inTsSet'
PROB_THRESHOLD = 'probThreshold'
# Auxiliar variables
MRC = '.mrc'
MRCS = '.mrcs'
OUT_MASKS_DIR = 'masks'
OUT_TS_DIR = 'results'
EVEN_SUFFIX = '_even'
ODD_SUFFIX = '_odd'
MASK_SUFFIX = '_mask'


class fidderOutputs(Enum):
    tiltSeries = SetOfTiltSeries


class ProtFidderDetectAndEraseFiducials(EMProtocol):
    """Fidder is a Python package for detecting and erasing gold fiducials in cryo-EM images.
    The fiducials are detected using a pre-trained residual 2D U-Net at 8 Å/px. Segmented regions are replaced
    with white noise matching the local mean and global standard deviation of the image."""

    _label = 'detect and erase fiducials'
    _devStatus = BETA
    _possibleOutputs = fidderOutputs
    stepsExecutionMode = STEPS_PARALLEL

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tsDict = None
        self.sRate = -1
        self.ih = ImageHandler()

    # --------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addSection(label=Message.LABEL_INPUT)
        form.addParam(IN_TS_SET, PointerParam,
                      pointerClass='SetOfTiltSeries',
                      important=True,
                      label='Tilt-Series')
        form.addParam(PROB_THRESHOLD, FloatParam,
                      default=0.5,
                      validators=[GT(0), LE(1)],
                      label='Fiducial probability threshold',
                      help='Threshold in range (0, 1] above which pixels are considered part of a fiducial.')
        form.addParam('doEvenOdd', BooleanParam,
                      label='Erase the fiducials in the odd/even tilt-series?',
                      default=False)
        form.addParam('saveMaskStack', BooleanParam,
                      default=False,
                      label='Save the fiducial-segmented stack?',
                      expertLevel=LEVEL_ADVANCED,
                      help='If set to Yes, the stack generated for each tilt-series with fiducial-based '
                           'segmentation will be saved (but not registered as Scipion objects. They can be '
                           'found in the protocol directory > extra.')
        form.addHidden(GPU_LIST, StringParam,
                       default='0',
                       label="Choose GPU IDs")
        form.addParallelSection(threads=1, mpi=0)

    # --------------------------- INSERT steps functions ----------------------
    def _insertAllSteps(self):
        self._initialize()
        closeSetStepDeps = []
        for ts in self._getInTsSet().iterItems():
            tsId = ts.getTsId()
            cInputId = self._insertFunctionStep(self.convertInputStep, tsId,
                                                prerequisites=[],
                                                needsGPU=False)
            predFidId = self._insertFunctionStep(self.predictAndEraseFiducialMaskStep, tsId,
                                                 prerequisites=cInputId,
                                                 needsGPU=True)
            cOutId = self._insertFunctionStep(self.createOutputStep, tsId,
                                              prerequisites=predFidId,
                                              needsGPU=False)
            closeSetStepDeps.append(cOutId)
        self._insertFunctionStep(self._closeOutputSet,
                                 prerequisites=closeSetStepDeps,
                                 needsGPU=False)

    # -------------------------- STEPS functions ------------------------------
    def _initialize(self):
        self.sRate = self._getInTsSet().getSamplingRate()

    def convertInputStep(self, tsId: str):
        logger.info(cyanStr(f'===> tsId = {tsId}: Unstacking...'))
        ts = self._getInTsSet().getItem(TiltSeries.TS_ID_FIELD, tsId)
        tsFileName = ts.getFirstItem().getFileName()
        nImgs = len(ts)
        # Create the necessary directories in tmp
        self._createTmpDirs(tsId, doEvenOdd=self.doEvenOdd.get())
        # Fidder works with individual MRC images --> the tilt-series must be un-stacked
        for i, ti in enumerate(ts.iterItems(orderBy=TiltImage.INDEX_FIELD)):
            logger.info(cyanStr(f'======> tsId = {tsId}: unstacked image {i + 1} of {nImgs}'))
            index = ti.getIndex()
            self._generateUnstakedImg(tsId, tsFileName, index)

    def predictAndEraseFiducialMaskStep(self, tsId: str):
        logger.info(cyanStr(f'===> tsId = {tsId}: Predicting the fiducial mask and erasing them...'))
        self._runFidder(tsId)
        # Odd/Even
        if self.doEvenOdd.get():
            self._runFidder(tsId, suffix=EVEN_SUFFIX)
            self._runFidder(tsId, suffix=ODD_SUFFIX)

    def createOutputStep(self, tsId: str):
        with self._lock:
            logger.info(cyanStr(f'===> tsId = {tsId}: Creating the resulting tilt-series...'))
            doEvenOdd = self.doEvenOdd.get()
            if self.saveMaskStack.get():
                # Mount the segmented stack
                self._mountSegmentedStack(tsId)

            # Mount the resulting tilt-series
            tsFName, tsFnameEven, tsFnameOdd = self._mountTiltSeries(tsId, doEvenOdd=doEvenOdd)

            inTs = self._getInTsSet().getItem(TiltSeries.TS_ID_FIELD, tsId)
            outTsSet = self._getOutputTsSet()
            newTs = TiltSeries()
            newTs.copyInfo(inTs)
            outTsSet.append(newTs)

            for inTi in inTs.iterItems(orderBy=TiltImage.INDEX_FIELD):
                newTi = TiltImage()
                newTi.copyInfo(inTi)
                newTi.setFileName(tsFName)
                if doEvenOdd:
                    newTi.setOddEven([tsFnameOdd, tsFnameEven])
                newTs.append(newTi)

            newTs.write()
            outTsSet.update(newTs)
            outTsSet.write()
            self._store(outTsSet)

    # --------------------------- UTILS functions -----------------------------
    def _getInTsSet(self, returnPointer: bool = False) -> Union[SetOfTiltSeries, Pointer]:
        inTsPointer = getattr(self, IN_TS_SET)
        return inTsPointer if returnPointer else inTsPointer.get()

    def _getCurrentTsTmpDir(self, tsId: str) -> str:
        return self._getTmpPath(tsId)

    def _getUnstackedImgsDir(self, tsId: str, suffix: str = '') -> str:
        return join(self._getCurrentTsTmpDir(tsId), 'unstackedImgs' + suffix)

    def _getUnstackedMasksDir(self, tsId: str, suffix: str = '') -> str:
        return join(self._getCurrentTsTmpDir(tsId), 'unstackedMasks' + suffix)

    def _getUnstackedErasedImgsDir(self, tsId: str, suffix: str = '') -> str:
        return join(self._getCurrentTsTmpDir(tsId), 'unstackedResults' + suffix)

    def _getOutputMaskFileName(self, tsId: str, inImageFileName: str, suffix: str = '') -> str:
        return join(self._getUnstackedMasksDir(tsId), basename(inImageFileName) + suffix)

    def _getOutputImgFileName(self, tsId: str, inImageFileName: str, suffix: str = '') -> str:
        return join(self._getUnstackedErasedImgsDir(tsId, suffix=suffix), basename(inImageFileName))

    def _getTsNewFileName(self, tsId, suffix: str = '') -> str:
        return self._getExtraPath(f'{tsId}{suffix}{MRCS}')

    @staticmethod
    def _getNewTiFileName(tsId: str, index: int, suffix: str = '') -> str:
        return f'{tsId}_{index:03}{suffix}{MRC}'

    def _getNewTiTmpFileName(self, tsId: str, index: int, suffix: str = '') -> str:
        return join(self._getUnstackedImgsDir(tsId, suffix=suffix),
                    self._getNewTiFileName(tsId, index, suffix=suffix))

    def _createTmpDirs(self, tsId: str, doEvenOdd: bool = False) -> None:
        currentTsTmpDir = self._getCurrentTsTmpDir(tsId)
        inImgsDir = self._getUnstackedImgsDir(tsId)
        masksDir = self._getUnstackedMasksDir(tsId)
        outImgsDir = self._getUnstackedErasedImgsDir(tsId)
        dirList = [currentTsTmpDir,
                   inImgsDir,
                   masksDir,
                   outImgsDir]
        if doEvenOdd:
            outImgsDirEven = self._getUnstackedErasedImgsDir(tsId, suffix=EVEN_SUFFIX)
            outImgsDirOdd = self._getUnstackedErasedImgsDir(tsId, suffix=ODD_SUFFIX)
            evenOdddirList = [outImgsDirEven,
                              outImgsDirOdd]
            dirList.extend(evenOdddirList)
        makePath(*dirList)

    def _getPredictArgs(self, inImage: str, outMask: str) -> str:
        cmd = [
            'predict',
            f'--input-image {inImage}',
            f'--output-mask {outMask}',
            f'--pixel-spacing {self.sRate:.3f}',
            f'--probability-threshold {getattr(self, PROB_THRESHOLD).get():.2f}'
        ]
        return ' '.join(cmd)

    @staticmethod
    def _getEraseFidArgs(inImage: str, maskedImage: str, outImage: str) -> str:
        cmd = [
            'erase',
            f'--input-image {inImage}',
            f'--input-mask {maskedImage}',
            f'--output-image {outImage}'
        ]
        return ' '.join(cmd)

    def _runFidder(self, tsId: str, suffix: str = ''):
        imagesList = glob.glob(join(self._getUnstackedImgsDir(tsId, suffix=suffix), '*' + MRC))
        nImgs = len(imagesList)
        for i, inImage in enumerate(sorted(imagesList)):
            logger.info(cyanStr(f'======> tsId = {tsId}{suffix}: processing image {i + 1} of {nImgs}'))
            outImgMask = self._getOutputMaskFileName(tsId, inImage)
            outResultImg = self._getOutputImgFileName(tsId, inImage, suffix=suffix)
            # Predict: only for the whole TS
            if not suffix:
                args = self._getPredictArgs(inImage, outImgMask)
                Plugin.runFidder(self, args)
            # Erase: do always this part, no matter if it's the whole TS, the even or the odd
            args = self._getEraseFidArgs(inImage, outImgMask, outResultImg)
            Plugin.runFidder(self, args)

    def _generateUnstakedImg(self, tsId: str, tsFileName: str, index: int, suffix: str = '') -> None:
        newTiFileName = self._getNewTiTmpFileName(tsId, index, suffix=suffix)
        tiFName = f'{index}@{tsFileName}:mrcs'
        self.ih.convert(tiFName, newTiFileName, DT_FLOAT)

    def _getOutputTsSet(self) -> SetOfTiltSeries:
        outSetSetAttrib = self._possibleOutputs.tiltSeries.name
        outTsSet = getattr(self, outSetSetAttrib, None)
        if outTsSet:
            outTsSet.enableAppend()
        else:
            outTsSet = SetOfTiltSeries.create(self._getPath(), template='tiltseries')
            outTsSet.copyInfo(self._getInTsSet())
            outTsSet.setStreamState(Set.STREAM_OPEN)
            self._defineOutputs(**{outSetSetAttrib: outTsSet})
            self._defineSourceRelation(self._getInTsSet(returnPointer=True), outTsSet)
        return outTsSet

    def _mountCurrentStack(self, tsId: str, imagesDir: str, suffix: str = '') -> str:
        logger.info(cyanStr(f'===> tsId = {tsId}{suffix}: mounting the stack file...'))
        outStackFile = self._getTsNewFileName(tsId, suffix=suffix)
        resultImgs = sorted(glob.glob(join(imagesDir, '*' + MRC)))

        # Read the first image to get the dimensions
        with mrcfile.mmap(resultImgs[0], mode='r+') as mrc:
            img = mrc.data
            nx, ny = img.shape

        # Create an empty array in which the stack of images will be stored
        shape = (len(resultImgs), nx, ny)
        stackArray = np.empty(shape, dtype=img.dtype)

        # Fill it with the images sorted by angle
        for i, img in enumerate(resultImgs):
            with mrcfile.mmap(img) as mrc:
                logger.info(f'Inserting image - index [{i}], {img}')
                stackArray[i] = mrc.data

        # Save the stack in a new mrc file
        with mrcfile.new_mmap(outStackFile, shape, overwrite=True) as mrc:
            mrc.set_data(stackArray)
            mrc.update_header_from_data()
            mrc.update_header_stats()
            mrc.voxel_size = self.sRate
        return outStackFile

    def _mountSegmentedStack(self, tsId: str) -> None:
        self._mountCurrentStack(tsId,
                                self._getUnstackedMasksDir(tsId),
                                suffix=MASK_SUFFIX)

    def _mountTiltSeries(self, tsId: str, doEvenOdd: bool=False) -> Tuple[str, str, str]:
        resultingTsFileNameEven = ''
        resultingTsFileNameOdd = ''
        unstackedErasedImgsDir = self._getUnstackedErasedImgsDir(tsId)
        resultingTsFileName = self._mountCurrentStack(tsId, unstackedErasedImgsDir)
        if doEvenOdd:
            # Even
            unstackedErasedImgsDirEven = self._getUnstackedErasedImgsDir(tsId,
                                                                         suffix=EVEN_SUFFIX)
            resultingTsFileNameEven = self._mountCurrentStack(tsId,
                                                              unstackedErasedImgsDirEven,
                                                              suffix=EVEN_SUFFIX)
            # Odd
            unstackedErasedImgsDirOdd = self._getUnstackedErasedImgsDir(tsId,
                                                                        suffix=ODD_SUFFIX)
            resultingTsFileNameOdd = self._mountCurrentStack(tsId,
                                                             unstackedErasedImgsDirOdd,
                                                             suffix=ODD_SUFFIX)

        return resultingTsFileName, resultingTsFileNameEven, resultingTsFileNameOdd

    # --------------------------- INFO functions ------------------------------
    def _validate(self) -> List[str]:
        errors = []
        if self.doEvenOdd.get() and not self._getInTsSet().hasOddEven():
            errors.append('The even/odd tilt-series cannot be processed as no even/odd tilt-series '
                          'are found in the metadata of the introduced tilt-series.')

        return errors
