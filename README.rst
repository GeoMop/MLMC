MLMC
====

.. image:: https://github.com/GeoMop/MLMC/workflows/package/badge.svg
    :target: https://github.com/GeoMop/MLMC/actions
.. image:: https://img.shields.io/pypi/v/mlmc.svg
    :target: https://pypi.org/project/mlmc/
.. image:: https://img.shields.io/pypi/pyversions/mlmc.svg
    :target: https://pypi.org/project/mlmc/
.. image:: https://img.shields.io/badge/License-GPLv3-blue.svg
    :target: https://www.gnu.org/licenses/gpl-3.0.html

Overview
--------

**MLMC** is a Python library implementing the **Multilevel Monte Carlo (MLMC)** method.
It provides tools for sampling, moment estimation, density reconstruction, and
statistical post-processing, all built around the flexible ``Quantity`` abstraction.

Originally developed as part of the `GeoMop <http://geomop.github.io/>`_ project,
MLMC aims to simplify stochastic analysis, uncertainty quantification,
and convergence studies in scientific computing workflows.

Features
--------

* Sample scheduling
* Estimation of generalized moment functions
* Advanced post-processing with the ``Quantity`` structure
* Approximation of probability density functions (PDFs)
* Bootstrap and regression-based variance estimation
* Diagnostic tools (consistency check, ...)

Installation
------------

The package is available on PyPI and can be installed via pip:

.. code-block:: bash

    pip install mlmc

To install the latest development version:

.. code-block:: bash

    git clone https://github.com/GeoMop/MLMC.git
    cd MLMC
    pip install -e .

Documentation
-------------

Full documentation including tutorials and examples is available at:
`https://mlmc.readthedocs.io/ <https://mlmc.readthedocs.io/>`_

Topics covered include:

* Basic MLMC workflow and examples
* Definition and composition of ``Quantity`` objects
* Moment and covariance estimation
* PDF and density reconstruction

Example
-------

A minimal example illustrating the MLMC workflow:

.. code-block:: python

    import mlmc
    from mlmc import sample_storage, quantity

    # Define your quantity and storage
    storage = sample_storage.Memory()
    q = quantity.SomeQuantityDefinition(...)

    # Create an MLMC estimator
    estimator = mlmc.Estimate(quantity=q, sample_storage=storage)

    # Estimate statistical moments
    means, variances = estimator.estimate_moments()

    # Construct an approximate probability density
    distribution, info, result, moments = estimator.construct_density()

    # Visualize variance breakdown
    estimator.plot_variances()

Development
-----------

Contributions are welcome!
To contribute, please fork the repository and create a pull request.

Before submitting, make sure all tests pass by running ``tox``:

.. code-block:: bash

    pip install tox
    tox

``tox`` creates a clean virtual environment, installs all dependencies,
runs unit tests via ``pytest``, and checks that the package installs correctly.

Requirements
------------

MLMC depends on the following Python packages:

* `NumPy <https://pypi.org/project/numpy/>`_
* `SciPy <https://pypi.org/project/scipy/>`_
* `h5py <https://pypi.org/project/h5py/>`_
* `attrs <https://pypi.org/project/attrs/>`_
* `ruamel.yaml <https://pypi.org/project/ruamel.yaml/>`_
* `gstools <https://pypi.org/project/gstools/>`_
* `memoization <https://pypi.org/project/memoization/>`_
* `scikit-learn <https://pypi.org/project/scikit-learn/>`_

License
-------

* Free software: **GNU General Public License v3.0**
