
MLMC
----

.. image:: https://github.com/GeoMop/MLMC/workflows/package/badge.svg
    :target: https://github.com/GeoMop/MLMC/actions
.. image:: https://img.shields.io/pypi/v/mlmc.svg
    :target: https://pypi.org/project/mlmc/
.. image:: https://img.shields.io/pypi/pyversions/mlmc.svg
    :target: https://pypi.org/project/mlmc/

MLMC provides tools for the multilevel Monte Carlo method.

mlmc package includes:

- samples scheduling
- estimation of generalized moment functions
- probability density function approximation
- advanced post-processing with Quantity structure


It is meant as part of the `GeoMop  <http://geomop.github.io/>`_ project in particular Analysis component.


Installation
----------------------------------
Package can be installed via pip.

.. code-block::

    pip install mlmc


Documentation
-------------
You can find the documentation including tutorials under https://mlmc.readthedocs.io/


Development
-----------

Provided that you want to contribute, create a pull request and make sure you run `tox` before. Tox
installs necessary requirements as well as the developed package itself into clear virtual environment
and call pytest to search in the `test` folder for tests to execute.


Requirements
------------
- `NumPy  <https://pypi.org/project/numpy/>`_
- `SciPy  <https://pypi.org/project/scipy/>`_
- `h5py  <https://pypi.org/project/h5py/>`_
- `attrs  <https://pypi.org/project/attrs/>`_
- `ruamel.yaml  <https://pypi.org/project/ruamel.yaml/>`_
- `gstools  <https://pypi.org/project/gstools/>`_
- `memoization  <https://pypi.org/project/memoization/>`_
- `sklearn  <https://pypi.org/project/sklearn/>`_


Licence
-------
* Free software: GPL 3.0  License


