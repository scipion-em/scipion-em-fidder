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
import os
import pwem
from fidder.constants import FIDDER_ENV_ACTIVATION, FIDDER_DEFAULT_ACTIVATION_CMD, FIDDER_DEFAULT_VERSION, FIDDER, \
    FIDDER_CUDA_LIB, V0_0_8, FIDDER_HOME, FIDDER_ENV_NAME
from pyworkflow.utils import Environ

__version__ = '3.0.0'
_logo = "icon.png"
# _references = ['']


class Plugin(pwem.Plugin):
    _pathVars = [FIDDER_CUDA_LIB]
    _supportedVersions = [V0_0_8]
    _url = "https://github.com/scipion-em/scipion-em-fidder"

    @classmethod
    def _defineVariables(cls):
        cls._defineVar(FIDDER_ENV_ACTIVATION, FIDDER_DEFAULT_ACTIVATION_CMD)
        cls._defineVar(FIDDER_CUDA_LIB, pwem.Config.CUDA_LIB)
        
    @classmethod
    def getFidderEnvActivation(cls):
        return cls.getVar(FIDDER_ENV_ACTIVATION)

    @classmethod
    def getEnviron(cls):
        """ Setup the environment variables needed to launch fidder. """
        environ = Environ(os.environ)
        if 'PYTHONPATH' in environ:
            # this is required for python virtual env to work
            del environ['PYTHONPATH']
        cudaLib = cls.getVar(FIDDER_CUDA_LIB, pwem.Config.CUDA_LIB)
        environ.addLibrary(cudaLib)

    @classmethod
    def defineBinaries(cls, env):
        FIDDER_INSTALLED = '%s_%s_installed' % (FIDDER, FIDDER_DEFAULT_VERSION)
        installationCmd = cls.getCondaActivationCmd()
        # Create the environment
        installationCmd += ' conda create -y -n %s python=3.8 && ' % FIDDER_ENV_NAME

        # Activate new the environment
        installationCmd += 'conda activate %s && ' % FIDDER_ENV_NAME

        # Install fidder
        installationCmd += f'pip install {FIDDER}=={FIDDER_DEFAULT_VERSION} && '

        # Flag installation finished
        installationCmd += 'touch %s' % FIDDER_INSTALLED

        FIDDER_commands = [(installationCmd, FIDDER_INSTALLED)]
        envPath = os.environ.get('PATH', "")  # keep path since conda likely in there
        installEnvVars = {'PATH': envPath} if envPath else None

        env.addPackage(FIDDER,
                       version=FIDDER_DEFAULT_VERSION,
                       tar='void.tgz',
                       commands=FIDDER_commands,
                       neededProgs=cls.getDependencies(),
                       vars=installEnvVars,
                       default=True)

    @classmethod
    def getDependencies(cls):
        # try to get CONDA activation command
        condaActivationCmd = cls.getCondaActivationCmd()
        neededProgs = []
        if not condaActivationCmd:
            neededProgs.append('conda')
        return neededProgs

    @classmethod
    def runFidder(cls, protocol, args, cwd=None, numberOfMpi=1):
        """ Run fidder command from a given protocol. """
        cmd = cls.getCondaActivationCmd() + " "
        cmd += cls.getFidderEnvActivation()
        cmd += f" && CUDA_VISIBLE_DEVICES=%(GPU)s {FIDDER} "
        protocol.runJob(cmd, args, env=cls.getEnviron(), cwd=cwd, numberOfMpi=numberOfMpi)

