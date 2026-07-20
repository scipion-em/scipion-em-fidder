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
import json
import logging
import shutil
import time
from enum import Enum
from os.path import join, basename
from typing import Union, List, Counter
import mrcfile
import numpy as np
from typing_extensions import Tuple
from fidder import Plugin
from pwem.emlib.image import ImageHandler
from pwem.emlib.image.image_readers import ImageReadersRegistry
from pwem.protocols import EMProtocol
from pyworkflow.object import Set, Pointer
from pyworkflow.protocol import PointerParam, FloatParam, GT, LE, GPU_LIST, StringParam, BooleanParam, LEVEL_ADVANCED, \
    STEPS_PARALLEL, ProtStreamingBase
from pyworkflow.utils import Message, makePath, cyanStr, redStr
from pyworkflow.utils.retry_streaming import retry_on_sqlite_lock
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
            with self._lock:
                inTsIds = set(inTsSet.getTSIds())

            # In the if statement below, Counter is used because in the tsId comparison the order doesn’t matter
            # but duplicates do. With a direct comparison, the closing step may not be inserted because of the order:
            # ['ts_a', 'ts_b'] != ['ts_b', 'ts_a'], but they are the same with Counter.
            if not inTsSet.isStreamOpen() and Counter(self.itemTsIdReadList) == Counter(inTsIds):
                logger.info(cyanStr('Input set closed.\n'))
                self._insertFunctionStep(self._closeOutputSet,
                                         prerequisites=closeSetStepDeps,
                                         needsGPU=False)
                break

            nonProcessedTsIds = inTsIds - set(self.itemTsIdReadList)
            tsToProcessDict = {tsId: ts.clone() for ts in inTsSet.iterItems()
                               if (tsId := ts.getTsId()) in nonProcessedTsIds  # Only not processed tsIds
                               and ts.getSize() > 0}  # Avoid processing empty TS
            for tsId, ts in tsToProcessDict.items():
                cInputId = self._insertFunctionStep(self.convertInputStep, ts,
                                                    prerequisites=[],
                                                    needsGPU=False)
                predFidId = self._insertFunctionStep(self.predictAndEraseFiducialMaskStep, tsId,
                                                     prerequisites=cInputId,
                                                     needsGPU=True)
                cOutId = self._insertFunctionStep(self.createOutputStep, ts,
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
    def convertInputStep(self, ts: TiltSeries):
        tsId = ts.getTsId()
        logger.info(cyanStr(f'===> tsId = {tsId}: Unstacking...'))
        # Create the necessary directories in tmp
        self._createTmpDirs(tsId, doEvenOdd=self.doEvenOdd.get())
        # Fidder works with individual MRC images --> the tilt-series must be un-stacked
        self._unstackTiltSeries(ts)
        if self.doEvenOdd.get():
            self._unstackTiltSeries(ts, suffix=EVEN_SUFFIX)
            self._unstackTiltSeries(ts, suffix=ODD_SUFFIX)

    def predictAndEraseFiducialMaskStep(self, tsId: str):
        logger.info(cyanStr(f'===> tsId = {tsId}: Predicting the fiducial mask and erasing them...'))
        try:
            # All images of this tilt-series (main + even + odd) are processed
            # by a single worker process so the U-Net checkpoint and the CUDA
            # context are loaded ONCE per tilt-series instead of once per image
            # (the per-image `fidder` CLI launches were the dominant cost).
            manifestPath = self._buildManifest(tsId)
            Plugin.runFidderBatch(self, manifestPath)
        except Exception as e:
            self.failedItems.append(tsId)
            logger.error(redStr(f'Fidder execution failed for tsId {tsId} -> {e}'))

    def createOutputStep(self, inTs: TiltSeries):
        tsId = inTs.getTsId()
        logger.info(cyanStr(f'===> tsId = {tsId}: Creating the resulting tilt-series...'))
        if tsId in self.failedItems:
            self.createOutputFailedSet(inTs)
            return

        if self.saveMaskStack.get():
            # Mount the segmented stack
            self._mountSegmentedStack(tsId)
        tsFName, tsFnameEven, tsFnameOdd = self._mountTiltSeries(tsId, doEvenOdd=self.doEvenOdd.get())
        self._registerOutput(inTs, tsFName, tsFnameEven, tsFnameOdd)

        # Clean the current ts folder/s in /tmp
        tsIdTmpDir = self._getTmpPath(tsId)
        if tsIdTmpDir:
            shutil.rmtree(tsIdTmpDir)


    @retry_on_sqlite_lock(log=logger)
    def _registerOutput(self,
                        inTs: TiltSeries,
                        tsFName: str,
                        tsFnameEven: str,
                        tsFnameOdd: str):
        with self._lock:
            # Mount the resulting tilt-series
            outTsSet = self._getOutputTsSet()
            newTs = TiltSeries()
            newTs.copyInfo(inTs)
            outTsSet.append(newTs)

            for inTi in inTs.iterItems(orderBy=TiltImage.INDEX_FIELD):
                newTi = TiltImage()
                newTi.copyInfo(inTi)
                newTi.setFileName(tsFName)
                if self.doEvenOdd.get():
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
                    time.sleep(10)

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

    def _getCurrentTsTmpDir(self, tsId: str) -> str:
        return self._getTmpPath(tsId)

    def _getUnstackedImgsDir(self, tsId: str, suffix: str = '') -> str:
        return join(self._getCurrentTsTmpDir(tsId), 'unstackedImgs' + suffix)

    def _getUnstackedMasksDir(self, tsId: str, suffix: str = '') -> str:
        return join(self._getCurrentTsTmpDir(tsId), 'unstackedMasks' + suffix)

    def _getUnstackedErasedImgsDir(self, tsId: str, suffix: str = '') -> str:
        return join(self._getCurrentTsTmpDir(tsId), 'unstackedResults' + suffix)

    def _getOutputMaskFileName(self, tsId: str, inImageFileName: str, suffix: str = '') -> str:
        return join(self._getUnstackedMasksDir(tsId), basename(inImageFileName.replace(suffix, '')))

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
            inImgsDirEven = self._getUnstackedImgsDir(tsId, suffix=EVEN_SUFFIX)
            inImgsDirOdd = self._getUnstackedImgsDir(tsId, suffix=ODD_SUFFIX)
            outImgsDirEven = self._getUnstackedErasedImgsDir(tsId, suffix=EVEN_SUFFIX)
            outImgsDirOdd = self._getUnstackedErasedImgsDir(tsId, suffix=ODD_SUFFIX)
            evenOdddirList = [outImgsDirEven,
                              outImgsDirOdd,
                              inImgsDirEven,
                              inImgsDirOdd]
            dirList.extend(evenOdddirList)
        makePath(*dirList)

    def _buildManifest(self, tsId: str) -> str:
        """Collect every unstacked image of this tilt-series (main + even +
        odd) into a single JSON manifest consumed by the batch worker, which
        loads the model once and processes them all in one process. Returns the
        manifest path.

        The mask is only persisted to disk when the (advanced) "save segmented
        stack" option is on -- it is the sole consumer of the mask MRCs. When
        off, the worker keeps the mask in memory and uses it only to erase,
        avoiding a per-image disk round-trip. Mask paths intentionally collapse
        the even/odd suffix into the same masks dir, matching the previous
        per-image behaviour that `_mountSegmentedStack` relies on.
        """
        saveMask = self.saveMaskStack.get()
        suffixes = [''] + ([EVEN_SUFFIX, ODD_SUFFIX] if self.doEvenOdd.get() else [])
        items = []
        for suffix in suffixes:
            imagesList = sorted(glob.glob(join(self._getUnstackedImgsDir(tsId, suffix=suffix), '*' + MRC)))
            for inImage in imagesList:
                items.append({
                    'input': inImage,
                    'erased_out': self._getOutputImgFileName(tsId, inImage, suffix=suffix),
                    'mask_out': self._getOutputMaskFileName(tsId, inImage, suffix=suffix) if saveMask else None,
                })
        manifest = {
            'pixel_spacing': float(self.sRate),
            'probability_threshold': float(getattr(self, PROB_THRESHOLD).get()),
            'items': items,
        }
        manifestPath = join(self._getCurrentTsTmpDir(tsId), 'fidder_manifest.json')
        with open(manifestPath, 'w') as f:
            json.dump(manifest, f)
        return manifestPath

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
            outImgMask = self._getOutputMaskFileName(tsId, inImage, suffix=suffix)
            outResultImg = self._getOutputImgFileName(tsId, inImage, suffix=suffix)
            # Predict: only for the whole TS
            args = self._getPredictArgs(inImage, outImgMask)
            Plugin.runFidder(self, args)
            # Erase: do always this part, no matter if it's the whole TS, the even or the odd
            args = self._getEraseFidArgs(inImage, outImgMask, outResultImg)
            Plugin.runFidder(self, args)

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

    @retry_on_sqlite_lock(log=logger)
    def createOutputFailedSet(self, item: TiltSeries):
        """ Just copy input item to the failed output set. """
        with self._lock:
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

            # Close explicitly the outputs (for streaming)
            output.close()

    def getOutputFailedSet(self, inputPtr: Pointer):
        """ Create output set for failed TS or tomograms. """
        inputSet = inputPtr.get()
        failedTs = getattr(self, OUTPUT_TS_FAILED_NAME, None)

        if failedTs:
            failedTs.enableAppend()
        else:
            logger.info(cyanStr('Create the set of failed TS'))
            failedTs = SetOfTiltSeries.create(self._getPath(), template='tiltseries', suffix='Failed')
            failedTs.copyInfo(inputSet)
            failedTs.setStreamState(Set.STREAM_OPEN)
            self._defineOutputs(**{OUTPUT_TS_FAILED_NAME: failedTs})
            self._defineSourceRelation(inputPtr, failedTs)

        return failedTs

    def _unstackTiltSeries(self, ts: TiltSeries, suffix: str = '') -> None:
        tsId = ts.getTsId()
        tsFn = ts.getFirstItem().getFileName()
        sRate = ts.getSamplingRate()
        imgStack = ImageReadersRegistry.open(tsFn)
        for i, img in enumerate(imgStack):
            newTiFileName = self._getNewTiTmpFileName(tsId, i + 1, suffix=suffix)
            with mrcfile.new(newTiFileName) as mrc:
                mrc.set_data(img)
                mrc.voxel_size = sRate

    # --------------------------- INFO functions ------------------------------
    def _validate(self) -> List[str]:
        errors = []
        if self.doEvenOdd.get() and not self._getInTsSet().hasOddEven():
            errors.append('The even/odd tilt-series cannot be processed as no even/odd tilt-series '
                          'are found in the metadata of the introduced tilt-series.')

        return errors
