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
import logging
from enum import Enum
from os.path import join
from fidder import Plugin
from pwem.protocols import EMProtocol
from pyworkflow import BETA
from pyworkflow.object import Set
from pyworkflow.protocol import PointerParam, FloatParam, GT, LE, GPU_LIST, StringParam
from pyworkflow.utils import Message, makePath, createLink
from tomo.objects import SetOfTiltSeries, TiltSeries, TiltImage

logger = logging.getLogger(__name__)
# Form variables
IN_TS_SET = 'inTsSet'
PROB_THRESHOLD = 'probThreshold'
# Auxiliar variables
MRC = '.mrc'
IN_TS_DIR = 'inTiltSeries'
OUT_TS_DIR = 'result'


class fidderOutputs(Enum):
    tiltSeries = SetOfTiltSeries


class ProtFidderDetectAndEraseFiducials(EMProtocol):
    """Fidder is a Python package for detecting and erasing gold fiducials in cryo-EM images.
    The fiducials are detected using a pre-trained residual 2D U-Net at 8 Å/px. Segmented regions are replaced
    with white noise matching the local mean and global standard deviation of the image."""

    _label = BETA
    _possibleOutputs = fidderOutputs

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
        form.addHidden(GPU_LIST, StringParam,
                       default='0',
                       label="Choose GPU IDs")
        # TODO: add ood/even functionality

    # --------------------------- INSERT steps functions ----------------------
    def _insertAllSteps(self):
        self._initialize()
        closeSetStepDeps = []
        for ts in self._getInTsSet().iterItems():
            tsId = ts.getTsId()
            tsFileName = ts.getFirstItem().getFileName()
            logging.info(f'===> Inserting the steps for tsId = {tsId}...')
            cInputId = self._insertFunctionStep(self.convertInputStep, tsId, tsFileName,
                                                prerequisites=[],
                                                needsGPU=False)
            predFidId = self._insertFunctionStep(self.predictFiducialMaskStep, tsId,
                                                 prerequisites=cInputId,
                                                 needsGPU=True)
            eraseFidId = self._insertFunctionStep(self.eraseFiducialsStep, tsId,
                                                  prerequisites=predFidId,
                                                  needsGPU=False)
            cOutId = self._insertFunctionStep(self.createOutputStep, tsId,
                                              prerequisites=eraseFidId,
                                              needsGPU=False)
            closeSetStepDeps.append(cOutId)
        self._insertFunctionStep(self._closeOutputSet,
                                 prerequisites=closeSetStepDeps,
                                 needsGPU=False)

    # -------------------------- STEPS functions ------------------------------
    def _initialize(self):
        makePath([self._getInTsSetDir(), self._getOutTsSetDir()])
    #     return {ts.getTsId: ts.clone() for ts in self._getInTsSet()}
    
    def convertInputStep(self, tsId, tsFileName):
        # Fidder works with MRC files
        likedOrConvertedFName = self._getInTsFileName(tsId) 
        if tsFileName.endswith(MRC):
            createLink(tsFileName, likedOrConvertedFName)
        else:
            logger.info(f'===> Converting (tsId, fileName) = ({tsId}, {tsFileName}) into MRC format...')
            from pwem import Domain
            xmipp3 = Domain.importFromPlugin('xmipp3')
            args = '-i %s -o %s -t vol ' % (tsFileName, likedOrConvertedFName)
            xmipp3.Plugin.runXmippProgram('xmipp_image_convert', args)
            
    def predictFiducialMaskStep(self, tsId):
        logger.info(f'===> tsId = {tsId}: Predicting a fiducial mask using a pretrained model...')
        args = self._getPredictArgs(tsId)
        Plugin.runFidder(self, args)

    def eraseFiducialsStep(self, tsId):
        logger.info(f'===> tsId = {tsId}: Erasing the fiducials...')
        args = self._getEraseFidArgs(tsId)
        Plugin.runFidder(self, args)

    def createOutputStep(self, tsId):
        logger.info(f'===> tsId = {tsId}: Creating the resulting tilt-series...')
        inTs = self._getInTsSet().getItem(TiltSeries.TS_ID_FIELD, tsId)
        outTsSet = self._getOutputTsSet()
        resultingTsFileName = self._getResultingTsFileName(tsId)
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
    def _getInTsSet(self, returnPointer=False):
        inTsPointer = getattr(self, IN_TS_SET)
        return inTsPointer if returnPointer else inTsPointer.get()
    
    def _getInTsSetDir(self):
        return self._getExtraPath(IN_TS_DIR)
    
    def _getOutTsSetDir(self):
        return self._getExtraPath(OUT_TS_DIR)
    
    @staticmethod
    def _getTsNewFileName(tsId, suffix=''):
        return f'{tsId}_{suffix}{MRC}'
    
    def _getInTsFileName(self, tsId):
        return join(self._getInTsSetDir(), self._getTsNewFileName(tsId))

    def _getOutputMaskFileName(self, tsId):
        return join(self._getOutTsSetDir(), self._getTsNewFileName(tsId, suffix='mask'))
    
    def _getResultingTsFileName(self, tsId):
        return join(self._getOutTsSetDir(), self._getTsNewFileName(tsId))

    def _getPredictArgs(self, tsId):
        cmd = [
            'predict',
            f'--input-image {self._getInTsFileName(tsId)}',
            f'--output-mask {self._getOutputMaskFileName(tsId)}',
            f'--pixel-spacing {self._getInTsSet().getSamplingRate():.3f}'
            f'--probability-threshold {getattr(self, PROB_THRESHOLD).get():.2f}'
        ]
        return ' '.join(cmd)

    def _getEraseFidArgs(self, tsId):
        cmd = [
            'erase',
            f'--input-image {self._getInTsFileName(tsId)}',
            f'--input-mask {self._getOutputMaskFileName(tsId)}'
            f'--output-image {self._getResultingTsFileName(tsId)}'
        ]
        return ' '.join(cmd)

    def _getOutputTsSet(self):
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