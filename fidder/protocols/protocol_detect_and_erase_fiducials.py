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
from fidder import Plugin
from pwem.emlib import DT_FLOAT
from pwem.emlib.image import ImageHandler
from pwem.protocols import EMProtocol
from pyworkflow.constants import BETA
from pyworkflow.object import Set, Pointer
from pyworkflow.protocol import PointerParam, FloatParam, GT, LE, GPU_LIST, StringParam, BooleanParam, LEVEL_ADVANCED
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


class fidderOutputs(Enum):
    tiltSeries = SetOfTiltSeries


class ProtFidderDetectAndEraseFiducials(EMProtocol):
    """Fidder is a Python package for detecting and erasing gold fiducials in cryo-EM images.
    The fiducials are detected using a pre-trained residual 2D U-Net at 8 Å/px. Segmented regions are replaced
    with white noise matching the local mean and global standard deviation of the image."""

    _label = 'detect and erase fiducials'
    _devStatus = BETA
    _possibleOutputs = fidderOutputs

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
                      label='Input set of Tilt-Series')
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
                           'segmentation will be saved.')
        form.addHidden(GPU_LIST, StringParam,
                       default='0',
                       label="Choose GPU IDs")

    # --------------------------- INSERT steps functions ----------------------
    def _insertAllSteps(self):
        self._initialize()
        closeSetStepDeps = []
        for tsId in self.tsDict.keys():
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
        self.tsDict = {ts.getTsId(): ts.clone() for ts in self._getInTsSet()}

    def convertInputStep(self, tsId: str):
        ts = self.tsDict[tsId]
        tsFileName = ts.getFirstItem().getFileName()
        # Fidder works with individual MRC images --> the tilt-series must be un-stacked
        currentTsTmpDir = self._getCurrentTsTmpDir(tsId)
        inImgsDir = self._getUnstackedImgsDir(tsId)
        masksDir = self._getUnstackedMasksDir(tsId)
        outImgsDir = self._getUnstackedErasedImgsDir(tsId)
        makePath(*[currentTsTmpDir,
                   inImgsDir,
                   masksDir,
                   outImgsDir])
        for ti in ts.iterItems(orderBy=TiltImage.INDEX_FIELD):
            index = ti.getIndex()
            newTiFileName = self._getNewTiTmpFileName(tsId, index)
            tiFName = f'{index}@{tsFileName}'
            self.ih.convert(tiFName, newTiFileName, DT_FLOAT)

    def predictAndEraseFiducialMaskStep(self, tsId: str):
        logger.info(cyanStr(f'===> tsId = {tsId}: Predicting the fiducial mask and erasing them...'))
        imagesList = glob.glob(join(self._getUnstackedImgsDir(tsId), '*' + MRC))
        nImgs = len(imagesList)
        for i, inImage in enumerate(sorted(imagesList)):
            logger.info(cyanStr(f'======> tsId = {tsId}: processing image {i + 1} of {nImgs}'))
            outImgMask = self._getOutputMaskFileName(tsId, inImage)
            outResultImg = self._getOutputImgFileName(tsId, inImage)
            # Predict
            args = self._getPredictArgs(inImage, outImgMask)
            Plugin.runFidder(self, args)
            # Erase
            args = self._getEraseFidArgs(inImage, outImgMask, outResultImg)
            Plugin.runFidder(self, args)

    def createOutputStep(self, tsId: str):
        logger.info(cyanStr(f'===> tsId = {tsId}: Creating the resulting tilt-series...'))
        if self.saveMaskStack.get():
            # Mount the segmented stack
            self._mountCurrentStack(tsId, self._getUnstackedMasksDir(tsId), suffix='mask')

        # Mount the resulting tilt-series
        resultingTsFileName = self._mountCurrentStack(tsId, self._getUnstackedErasedImgsDir(tsId))

        inTs = self._getInTsSet().getItem(TiltSeries.TS_ID_FIELD, tsId)
        outTsSet = self._getOutputTsSet()
        newTs = TiltSeries()
        newTs.copyInfo(inTs)
        outTsSet.append(newTs)

        for inTi in inTs.iterItems(orderBy=TiltImage.INDEX_FIELD):
            newTi = inTi.clone()
            newTi.setFileName(resultingTsFileName)
            newTs.append(newTi)

        newTs.write()
        outTsSet.update(newTs)
        outTsSet.write()
        self._store(outTsSet)

    # --------------------------- UTILS functions -----------------------------
    def _getInTsSet(self, returnPointer: bool=False) -> Union[SetOfTiltSeries, Pointer]:
        inTsPointer = getattr(self, IN_TS_SET)
        return inTsPointer if returnPointer else inTsPointer.get()

    def _getCurrentTsTmpDir(self, tsId: str) -> str:
        return self._getTmpPath(tsId)

    def _getUnstackedImgsDir(self, tsId: str) -> str:
        return join(self._getCurrentTsTmpDir(tsId), 'unstackedImgs')

    def _getUnstackedMasksDir(self, tsId: str) -> str:
        return join(self._getCurrentTsTmpDir(tsId), 'unstackedMasks')

    def _getUnstackedErasedImgsDir(self, tsId: str) -> str:
        return join(self._getCurrentTsTmpDir(tsId), 'unstackedResults')

    def _getOutputMaskFileName(self, tsId: str, inImageFileName: str) -> str:
        return join(self._getUnstackedMasksDir(tsId), basename(inImageFileName))

    def _getOutputImgFileName(self, tsId: str, inImageFileName: str) -> str:
        return join(self._getUnstackedErasedImgsDir(tsId), basename(inImageFileName))

    @staticmethod
    def _getTsNewFileName(tsId, suffix: str='') -> str:
        return f'{tsId}_{suffix}{MRCS}'

    @staticmethod
    def _getNewTiFileName(tsId: str, index: int) -> str:
        return f'{tsId}_{index:03}{MRC}'

    def _getNewTiTmpFileName(self, tsId: str, index: int) -> str:
        return join(self._getUnstackedImgsDir(tsId), self._getNewTiFileName(tsId, index))


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

    def _mountCurrentStack(self, tsId: str, imagesDir: str, suffix: str='') -> str:
        suffix = '_' + suffix if suffix else suffix
        logger.info(cyanStr(f'===> tsId = {tsId}{suffix}: mounting the stack file...'))
        outStackFile = self._getExtraPath(f'{tsId}{suffix}{MRCS}')
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

    # --------------------------- INFO functions ------------------------------
    def _validate(self) -> List[str]:
        errors = []
        if self.doEvenOdd.get() and not self._getInTsSet().hasOddEven():
            errors.append('The even/odd tilt-series cannot be processed as no even/odd tilt-series '
                          'are found in the metadata of the introduced tilt-series.')

        return errors