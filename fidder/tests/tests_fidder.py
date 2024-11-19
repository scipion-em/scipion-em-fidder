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
from typing import Union

from fidder.protocols import ProtFidderDetectAndEraseFiducials
from imod.constants import OUTPUT_TILTSERIES_NAME
from imod.protocols import ProtImodTsNormalization, ProtImodImportTransformationMatrix
from pwem import ALIGN_2D
from pyworkflow.tests import setupTestProject, DataSet
from pyworkflow.utils import cyanStr, magentaStr
from tomo.objects import SetOfTiltSeries
from tomo.protocols import ProtImportTs, ProtImportTsBase
from tomo.tests import RE4_STA_TUTO, DataSetRe4STATuto, TS_03, TS_54
from tomo.tests.test_base_centralized_layer import TestBaseCentralizedLayer


class TestFidder(TestBaseCentralizedLayer):
    alignedTs = None
    unbinnedSRate = DataSetRe4STATuto.unbinnedPixSize.value
    binFactor = 4
    expectedTsSetSize = 2
    # Excluded views stuff
    excludedViewsDict = {
        TS_03: [0, 38, 39],
        TS_54: [0, 1, 38, 39, 40]
    }

    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        cls.ds = DataSet.getDataSet(RE4_STA_TUTO)
        testTsIds = (TS_03, TS_54)
        cls.testAcqObjDict, cls.expectedDimsDict, cls.anglesCountDict = (
            DataSetRe4STATuto.genTestTsDicts(testTsIds,
                                             binFactor=cls.binFactor,
                                             isImod=True))  # The TS are binned using IMOD in these tests
        cls.expectedSetSize = len(testTsIds)
        cls.runPrevProtocols()

    @classmethod
    def runPrevProtocols(cls):
        print(cyanStr('--------------------------------- RUNNING PREVIOUS PROTOCOLS ---------------------------------'))
        cls._runPreviousProtocols()
        print(
            cyanStr('\n-------------------------------- PREVIOUS PROTOCOLS FINISHED ---------------------------------'))

    @classmethod
    def _runPreviousProtocols(cls):
        importedTs = cls._runImportTs()
        cls.alignedTs = cls._runImportTrMatrix(importedTs)

    @classmethod
    def _runImportTs(cls, filesPattern=DataSetRe4STATuto.tsPattern.value,
                     exclusionWords=DataSetRe4STATuto.exclusionWordsTs03ts54.value):
        print(magentaStr("\n==> Importing the tilt series:"))
        protImportTs = cls.newProtocol(ProtImportTs,
                                       filesPath=cls.ds.getFile(DataSetRe4STATuto.tsPath.value),
                                       filesPattern=filesPattern,
                                       exclusionWords=exclusionWords,
                                       anglesFrom=2,  # From tlt file
                                       voltage=DataSetRe4STATuto.voltage.value,
                                       magnification=DataSetRe4STATuto.magnification.value,
                                       sphericalAberration=DataSetRe4STATuto.sphericalAb.value,
                                       amplitudeContrast=DataSetRe4STATuto.amplitudeContrast.value,
                                       samplingRate=cls.unbinnedSRate,
                                       doseInitial=DataSetRe4STATuto.initialDose.value,
                                       dosePerFrame=DataSetRe4STATuto.dosePerTiltImgWithTltFile.value,
                                       tiltAxisAngle=DataSetRe4STATuto.tiltAxisAngle.value)

        cls.launchProtocol(protImportTs)
        tsImported = getattr(protImportTs, ProtImportTsBase.OUTPUT_NAME, None)
        return tsImported

    @classmethod
    def _runImportTrMatrix(cls, inTsSet):
        print(magentaStr("\n==> Importing the TS' transformation matrices with IMOD:"))
        protImportTrMatrix = cls.newProtocol(ProtImodImportTransformationMatrix,
                                             filesPath=cls.ds.getFile(DataSetRe4STATuto.tsPath.value),
                                             filesPattern=DataSetRe4STATuto.transformPattern.value,
                                             inputSetOfTiltSeries=inTsSet)
        cls.launchProtocol(protImportTrMatrix)
        outTsSet = getattr(protImportTrMatrix, OUTPUT_TILTSERIES_NAME, None)
        return outTsSet

    @classmethod
    def _runTsPreprocess(cls, inTsSet, binning=1):
        print(magentaStr(f"\n==> Running the TS preprocessing:"
                         f"\n\t- Binning factor = {binning}"))
        protTsNorm = cls.newProtocol(ProtImodTsNormalization,
                                     inputSetOfTiltSeries=inTsSet,
                                     binning=binning)
        cls.launchProtocol(protTsNorm)
        tsPreprocessed = getattr(protTsNorm, OUTPUT_TILTSERIES_NAME, None)
        return tsPreprocessed

    @classmethod
    def _excludeSetViews(cls, inSet: SetOfTiltSeries,
                         excludedViewsDict: Union[dict, None] = None) -> None:
        if not excludedViewsDict:
            excludedViewsDict = cls.excludedViewsDict
        objList = [obj.clone(ignoreAttrs=[]) for obj in inSet]
        for obj in objList:
            cls._excIntermediateSetViews(inSet, obj, excludedViewsDict[obj.getTsId()])

    @staticmethod
    def _excIntermediateSetViews(inSet, obj, excludedViewsList):
        tiList = [ti.clone() for ti in obj]
        for i, ti in enumerate(tiList):
            if i in excludedViewsList:
                ti._objEnabled = False
                obj.update(ti)
        obj.write()
        inSet.update(obj)
        inSet.write()

    def _checkTiltSeries(self, inTsSet, binningFactor=1, expectedDimensions=None,
                         testAcqObjDict=None, anglesCountDict=None, excludedViewsDict=None):
        if not testAcqObjDict:
            testAcqObjDict = self.testAcqObjDict
        if not anglesCountDict:
            anglesCountDict = self.anglesCountDict
        if not expectedDimensions:
            expectedDimensions = self.expectedDimsDict
        self.checkTiltSeries(inTsSet,
                             expectedSetSize=self.expectedSetSize,
                             expectedSRate=self.unbinnedSRate * binningFactor,
                             hasAlignment=True,  # TS alignment was imported
                             alignment=ALIGN_2D,
                             expectedDimensions=expectedDimensions,
                             testAcqObj=testAcqObjDict,
                             anglesCount=anglesCountDict,
                             isHeterogeneousSet=True,
                             excludedViewsDict=excludedViewsDict)

    def _runFidderTest(self, inTsSet, excludedViewsDict=None):
        if excludedViewsDict:
            evMsg = ' with excluded views.'
            evLabel = ', eV'
        else:
            evMsg = ''
            evLabel = ''

        print(magentaStr(f"\n==> Running fidder{evMsg}:"))
        protFidder = self.newProtocol(ProtFidderDetectAndEraseFiducials,
                                      inTsSet=inTsSet)
        protFidder.setObjLabel(f'fidder{evLabel}')
        self.launchProtocol(protFidder)
        outTsSet = getattr(protFidder, protFidder._possibleOutputs.tiltSeries.name, None)
        self._checkTiltSeries(outTsSet,
                              binningFactor=self.binFactor,
                              excludedViewsDict=excludedViewsDict)

    def testFidder(self):
        binnedAlignedTs = self._runTsPreprocess(self.alignedTs,
                                                binning=self.binFactor)
        self._runFidderTest(binnedAlignedTs)

    def testFidderEv(self):
        binnedAlignedTs = self._runTsPreprocess(self.alignedTs,
                                                binning=self.binFactor)
        # Exclude some views at metadata level
        self._excludeSetViews(binnedAlignedTs)
        self._runFidderTest(binnedAlignedTs,
                            excludedViewsDict=self.excludedViewsDict)



