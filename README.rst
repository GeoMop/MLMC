====
MLMC
====
|circleci| |version| |supported-versions|


..  |docs| image:: https://readthedocs.org/projects/mlmc/badge/?style=flat
    :target: https://readthedocs.org/projects/python-mlmc
    :alt: Documentation Status

..  |circleci| image:: https://circleci.com/gh/GeoMop/MLMC/tree/master.svg?style=svg
    :alt: CircleCI Build Status
    :target: https://circleci.com/gh/GeoMop/MLMC/tree/master

..  |appveyor| image:: https://ci.appveyor.com/api/projects/status/github/ionelmc/python-nameless?branch=master&svg=true
    :alt: AppVeyor Build Status
    :target: https://ci.appveyor.com/project/ionelmc/python-nameless

..  |requires| image:: https://requires.io/github/ionelmc/python-nameless/requirements.svg?branch=master
    :alt: Requirements Status
    :target: https://requires.io/github/ionelmc/python-nameless/requirements/?branch=master

..  |codecov| image:: https://codecov.io/github/ionelmc/python-nameless/coverage.svg?branch=master
    :alt: Coverage Status
    :target: https://codecov.io/github/ionelmc/python-nameless

..  |version| image:: https://img.shields.io/pypi/v/mlmc.svg
    :alt: PyPI Package latest release
    :target: https://pypi.org/project/mlmc/

..  |commits-since| image:: https://img.shields.io/github/commits-since/ionelmc/python-mlmc/v0.1.0.svg
    :alt: Commits since latest release
    :target: https://github.com/ionelmc/python-mlmc/compare/v0.1.0...master

..  |wheel| image:: https://img.shields.io/pypi/wheel/mlmc.svg
    :alt: PyPI Wheel
    :target: https://pypi.org/project/mlmc/

..  |supported-versions| image:: https://img.shields.io/pypi/pyversions/mlmc.svg
    :alt: Supported versions
    :target: https://pypi.org/project/mlmc/

..  |supported-implementations| image:: https://img.shields.io/pypi/implementation/mlmc.svg
    :alt: Supported implementations
    :target: https://pypi.org/project/mlmc/


.. end-badges

Multi-level Monte Carlo method with approximation of distribution function and quantiles.
It is meant as part of GeoMop project in particular Analysis component.

Installation
============
Package can be installed via pip.

::

    pip install mlmc

Documentation
=============
https://mlmc.readthedocs.io/

Development
===========

.. _description: https://docs.google.com/document/d/1u3KTiCAEduGTqbpd9hojn-cw0BiB7mKblrMj1A911FE/edit#

Follow `description`_ of continuous integration practices. In particular use `tox` to run tests. Tox
installs necessary requirements as well as the developed package itself into clear virtual environment
and call pytest to search in the `test` folder for tests to execute.


.. Rules:

.. - indent by 4 spaces
.. - use docstrings to document function parameters
.. - initialize and document class attributes in the constructor
.. -

Requirements
============
- NumPy_
- SciPy_
- h5py_
- pyyaml_
- sklearn_

.. _NumPy: https://pypi.org/project/numpy/
.. _SciPy: https://pypi.org/project/scipy/
.. _h5py: https://pypi.org/project/h5py/
.. _pyyaml: https://pypi.org/project/PyYAML/
.. _sklearn: https://pypi.org/project/scikit-learn/

Licence
=======
* Free software: GPL 3.0  License