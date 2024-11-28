=========================
Scipion plugin for fidder
=========================

.. image:: https://img.shields.io/pypi/v/scipion-em-fidder.svg
        :target: https://pypi.python.org/pypi/scipion-em-fidder
        :alt: PyPI release

.. image:: https://img.shields.io/pypi/l/scipion-em-fidder.svg
        :target: https://pypi.python.org/pypi/scipion-em-fidder
        :alt: License

.. image:: https://img.shields.io/pypi/pyversions/scipion-em-fidder.svg
        :target: https://pypi.python.org/pypi/scipion-em-fidder
        :alt: Supported Python versions

.. image:: https://img.shields.io/sonar/quality_gate/scipion-em_scipion-em-fidder?server=https%3A%2F%2Fsonarcloud.io
        :target: https://sonarcloud.io/dashboard?id=scipion-em_scipion-em-fidder
        :alt: SonarCloud quality gate

.. image:: https://img.shields.io/pypi/dm/scipion-em-fidder
        :target: https://pypi.python.org/pypi/scipion-em-fidder
        :alt: Downloads

This plugin provide a wrapper around the program `fidder <https://teamtomo.org/fidder/>`_ to use it within 
`Scipion <https://scipion-em.github.io/docs/release-3.0.0/index.html>`_ framework.

Installation
------------

You will need to use `3.0 <https://scipion-em.github.io/docs/release-3.0.0/docs/scipion-modes/how-to-install.html>`_ 
version of Scipion to be able to run these protocols. To install the plugin, you have two options:


a) Stable version:

.. code-block::

    scipion3 installp -p scipion-em-fidder

b) Developer's version

    * download the repository from github:

    .. code-block::

        git clone -b devel https://github.com/scipion-em/scipion-em-fidder.git

    * install:

    .. code-block::

        scipion3 installp -p /path/to/scipion-em-fidder --devel

To check the installation, simply run the following Scipion test for the plugin:

    .. code-block::

        scipion3 tests fidder.tests.tests_fidder.TestFidder

Licensing
---------

fidder software package is available under `BSD-3-Clause license <https://opensource.org/license/BSD-3-Clause>`_

Protocols
---------

* **Tilt-series: detect and erase fiducials.**

Latest plugin versions
----------------------

If you want to check the latest version and release history go to `CHANGES <https://github.com/scipion-em-fidder/fidder/blob/master/CHANGES.txt>`_