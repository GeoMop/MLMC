==================
MLMC
==================

.. image:: https://github.com/GeoMop/MLMC/workflows/package/badge.svg
    :target: https://github.com/GeoMop/MLMC/actions
.. image:: https://img.shields.io/pypi/v/mlmc.svg
    :target: https://pypi.org/project/mlmc/
.. image:: https://img.shields.io/pypi/pyversions/mlmc.svg
    :target: https://pypi.org/project/mlmc/

MLMC provides tools for the Multilevel Monte Carlo method:

- samples scheduling
- moments calculation
- probability density function approximation
- advanced post-processing


Installation
============
mlmc can be installed via `pip <https://pypi.org/project/mlmc/>`_

.. code-block:: none

    pip install mlmc


Quickstart
==========

Create MLMC Sampler
^^^^^^^^^^^^^^^^^^^^^^^^^


.. code-block:: python

    import mlmc
    # Prescribe basic MLMC parameters
    n_levels = 3 # number of MLMC levels
    step_range = [0.5, 0.005] # simulation steps at the coarsest and finest levels
    # determine each level simulation steps
    level_parameters = mlmc.estimator.determine_level_parameters(n_levels, step_range)
    # level_parameters can be manually prescribed as a list of lists
    # Create sampling pool
    sampling_pool = mlmc.sampling_pool.OneProcessPool()
    # There is another option mlmc.sampling_pool.ProcessPool()
    #           - supports local parallel sample simulation run
    # sampling_pool = ProcessPool(n),
    #           n - number of parallel simulations, depends on computer architecture

    # Create simulation, instance of class that inherits from mlmc.sim.simulation.Simulation
    sim_configuration = dict(distr=distr, complexity=2)
    simulation_factory = mlmc.sim.synth_simulation.SynthSimulation(config=sim_configuration)
    # Create simple sample storage
    # Memory() storage keeps samples in the computer main memory
    sample_storage = mlmc.sample_storage.Memory()
    # We support also HDF file storage mlmc.sample_storage_hdf.SampleStorageHDF()
    # sample_storage = SampleStorageHDF(file_path=path_to_HDF5_file)

    # Create sampler, it controls the execution of MLMC samples
    sampler = mlmc.sampler.Sampler(sample_storage=sample_storage,
                                   sampling_pool=sampling_pool,
                                   sim_factory=simulation_factory,
                                   level_parameters=level_parameters)


Schedule MLMC samples
^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python

    # Schedule initial number of samples
    sampler