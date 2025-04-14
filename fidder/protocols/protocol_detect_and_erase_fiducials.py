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
import shutil
import time
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
    STEPS_PARALLEL, ProtStreamingBase
from pyworkflow.utils import Message, makePath, cyanStr, redStr
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
# Other variables
OUTPUT_TS_FAILED_NAME = "FailedTiltSeries"


class fidderOutputs(Enum):
    tiltSeries = SetOfTiltSeries


class ProtFidderDetectAndEraseFiducials(EMProtocol, ProtStreamingBase):
    """Fidder is a Python package for detecting and erasing gold fiducials in cryo-EM images.
    The fiducials are detected using a pre-trained residual 2D U-Net at 8 Å/px. Segmented regions are replaced
    with white noise matching the local mean and global standard deviation of the image."""

    _label = 'detect and erase fiducials'
    _devStatus = BETA
    _possibleOutputs = fidderOutputs
    stepsExecutionMode = STEPS_PARALLEL

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.itemTsIdReadList = []
        self.failedItems = []
        self.sRate = -1
        self.ih = ImageHandler()

    @classmethod
    def worksInStreaming(cls):
        return True

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
        form.addParallelSection(threads=2, mpi=0)

    # --------------------------- INSERT steps functions ----------------------
    def stepsGeneratorStep(self) -> None:
        closeSetStepDeps = []
        inTsSet = self._getInTsSet()
        self.sRate = self._getInTsSet().getSamplingRate()
        self.readingOutput()

        while True:
            listInTsIds = inTsSet.getTSIds()
            if not inTsSet.isStreamOpen() and self.itemTsIdReadList == listInTsIds:
                logger.info(cyanStr('Input set closed.\n'))
                self._insertFunctionStep(self._closeOutputSet,
                                         prerequisites=closeSetStepDeps,
                                         needsGPU=False)
                break
            for ts in inTsSet.iterItems():
                tsId = ts.getTsId()
                if tsId not in self.itemTsIdReadList and ts.getSize() > 0:  # Avoid processing empty TS (before the Tis are added)
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
                    logger.info(cyanStr(f"Steps created for tsId = {tsId}"))
                    self.itemTsIdReadList.append(tsId)
            time.sleep(10)
            if inTsSet.isStreamOpen():
                with self._lock:
                    inTsSet.loadAllProperties()  # refresh status for the streaming

    # -------------------------- STEPS functions ------------------------------
    def convertInputStep(self, tsId: str):
        logger.info(cyanStr(f'===> tsId = {tsId}: Unstacking...'))
        ts = self._getCurrentItem(tsId)
        tsFileName = ts.getFirstItem().getFileName()
        # Create the necessary directories in tmp
        self._createTmpDirs(tsId, doEvenOdd=self.doEvenOdd.get())
        # Fidder works with individual MRC images --> the tilt-series must be un-stacked
        for i, ti in enumerate(ts.iterItems(orderBy=TiltImage.INDEX_FIELD)):
            index = i + 1
            self._generateUnstakedImg(tsId, tsFileName, index)

    def predictAndEraseFiducialMaskStep(self, tsId: str):
        logger.info(cyanStr(f'===> tsId = {tsId}: Predicting the fiducial mask and erasing them...'))
        try:
            self._runFidder(tsId)
            # Odd/Even
            if self.doEvenOdd.get():
                self._runFidder(tsId, suffix=EVEN_SUFFIX)
                self._runFidder(tsId, suffix=ODD_SUFFIX)
        except Exception as e:
            self.failedItems.append(tsId)
            logger.error(redStr(f'Fidder execution failed for tsId {tsId} -> {e}'))

    def createOutputStep(self, tsId: str):
        with self._lock:
            logger.info(cyanStr(f'===> tsId = {tsId}: Creating the resulting tilt-series...'))
            inTs = self._getCurrentItem(tsId, doLock=False)
            if tsId in self.failedItems:
                self.createOutputFailedSet(inTs)
                failedTs = getattr(self, OUTPUT_TS_FAILED_NAME, None)
                if failedTs:
                    failedTs.close()
            else:
                doEvenOdd = self.doEvenOdd.get()
                if self.saveMaskStack.get():
                    # Mount the segmented stack
                    self._mountSegmentedStack(tsId)

                # Mount the resulting tilt-series
                tsFName, tsFnameEven, tsFnameOdd = self._mountTiltSeries(tsId, doEvenOdd=doEvenOdd)
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
                for outputName in self._possibleOutputs:
                    output = getattr(self, outputName.name, None)
                    if output:
                        output.close()

        # Clean the current ts folder/s in /tmp
        tsIdTmpDir = self._getTmpPath(tsId)
        if tsIdTmpDir:
            shutil.rmtree(tsIdTmpDir)

    # --------------------------- UTILS functions -----------------------------
    def readingOutput(self) -> None:
        outTsSet = getattr(self, self._possibleOutputs.tiltSeries.name, None)
        if outTsSet:
            for item in outTsSet:
                self.itemTsIdReadList.append(item.getTsId())
            self.info(cyanStr(f'TsIds processed: {self.itemTsIdReadList}'))
        else:
            self.info(cyanStr('No tilt-series have been processed yet'))

    def _getInTsSet(self, returnPointer: bool = False) -> Union[SetOfTiltSeries, Pointer]:
        inTsPointer = getattr(self, IN_TS_SET)
        return inTsPointer if returnPointer else inTsPointer.get()

    def _getCurrentItem(self, tsId: str, doLock: bool = True) -> TiltSeries:
        if doLock:
            with self._lock:
                return self._getInTsSet().getItem(TiltSeries.TS_ID_FIELD, tsId)
        else:
            return self._getInTsSet().getItem(TiltSeries.TS_ID_FIELD, tsId)

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

    def createOutputFailedSet(self, item):
        """ Just copy input item to the failed output set. """
        logger.info(f'Failed TS ---> {item.getTsId()}')
        inputSetPointer = self._getInTsSet(returnPointer=True)
        output = self.getOutputFailedSet(inputSetPointer)
        newItem = item.clone()
        newItem.copyInfo(item)
        output.append(newItem)

        if isinstance(item, TiltSeries):
            newItem.copyItems(item)
            newItem.write(properties=False)

        output.update(newItem)
        output.write()
        self._store(output)

    def getOutputFailedSet(self, inputPtr: Pointer):
        """ Create output set for failed TS or tomograms. """
        inputSet = inputPtr.get()
        if isinstance(inputSet, SetOfTiltSeries):
            failedTs = getattr(self, OUTPUT_TS_FAILED_NAME, None)

            if failedTs:
                failedTs.enableAppend()
            else:
                logger.info('Create the set of failed TS')
                failedTs = SetOfTiltSeries.create(self._getPath(), template='tiltseries', suffix='Failed')
                failedTs.copyInfo(inputSet)
                failedTs.setStreamState(Set.STREAM_OPEN)
                self._defineOutputs(**{OUTPUT_TS_FAILED_NAME: failedTs})
                self._defineSourceRelation(inputPtr, failedTs)

            return failedTs

    # --------------------------- INFO functions ------------------------------
    def _validate(self) -> List[str]:
        errors = []
        if self.doEvenOdd.get() and not self._getInTsSet().hasOddEven():
            errors.append('The even/odd tilt-series cannot be processed as no even/odd tilt-series '
                          'are found in the metadata of the introduced tilt-series.')

        return errors
