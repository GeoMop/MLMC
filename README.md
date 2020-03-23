MLMC
====

![](https://github.com/GeoMop/MLMC/workflows/.github/workflows/pythonpackage.yml/badge.svg)
[![PyPI version shields.io](https://img.shields.io/pypi/v/ansicolortags.svg)](https://pypi.org/project/mlmc/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/ansicolortags.svg)](https://pypi.org/project/mlmc/)

Multi-level Monte Carlo method with approximation of distribution function and quantiles.
It is meant as part of GeoMop project in particular Analysis component.

Installation
----------------------------------
Package can be installed via pip.

    pip install mlmc

Documentation
=============
https://mlmc.readthedocs.io/

Development
===========

MLMC is beta software. Provided that you want to contribute, create a pull request and make sure you run `tox` before. Tox
installs necessary requirements as well as the developed package itself into clear virtual environment
and call pytest to search in the `test` folder for tests to execute.


Requirements
============
- [NumPy](https://pypi.org/project/numpy/)
- [SciPy](https://pypi.org/project/scipy/)
- [h5py](https://pypi.org/project/h5py/)
- [pyyaml](https://pypi.org/project/PyYAML/)
- [sklearn](https://pypi.org/project/scikit-learn/)

Licence
=======
* Free software: GPL 3.0  License