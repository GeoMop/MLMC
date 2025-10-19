Sampler Creation
=================

The **Sampler** controls the execution and management of MLMC (Multilevel Monte Carlo) samples.
This example demonstrates how to configure all essential components for an MLMC simulation.



Basic Setup
------------

First, import the :mod:`mlmc` package and define basic MLMC parameters.

.. testcode::

    import mlmc

    # Define number of MLMC levels
    n_levels = 3

    # Simulation step sizes at the coarsest and finest levels
    step_range = [0.5, 0.005]

    # Compute level parameters (simulation steps per level)
    level_parameters = mlmc.estimator.determine_level_parameters(n_levels, step_range)

    # Alternatively, you can specify level_parameters manually as a list of lists.



Simulation Definition
----------------------

Prepare a simulation instance.
The simulation class must inherit from :class:`mlmc.sim.simulation.Simulation`.

.. testcode::

    simulation_factory = mlmc.SynthSimulation()

This factory will be used by the sampler to create individual simulation runs.



Sampling Pool
--------------

Next, create a sampling pool that controls how samples are executed.

.. testcode::

    sampling_pool = mlmc.OneProcessPool()

The :class:`mlmc.sampling_pool.OneProcessPool` executes samples sequentially within a single process.

You can also use:

- :class:`mlmc.sampling_pool.ProcessPool` — executes samples in parallel across multiple processes.
- :class:`mlmc.sampling_pool_pbs.SamplingPoolPBS` — submits jobs to a PBS (Portable Batch System) cluster for distributed computation.



Sample Storage
---------------

The **sample storage** keeps all data related to simulation results.

.. testcode::

    # Memory storage keeps samples in main memory
    sample_storage = mlmc.Memory()

Alternatively, use persistent file-based storage:

- :class:`mlmc.sample_storage_hdf.SampleStorageHDF` — stores results in an HDF5 file for long-term reuse and analysis.



Sampler Initialization
-----------------------

Finally, create the **Sampler** instance.
It coordinates sample scheduling, simulation execution, and result collection.

.. testcode::

    sampler = mlmc.Sampler(
        sample_storage=sample_storage,
        sampling_pool=sampling_pool,
        sim_factory=simulation_factory,
        level_parameters=level_parameters
    )

The sampler is now ready to generate and manage MLMC samples.



Next Steps
-----------

Proceed to the next example:
:ref:`examples samples scheduling`
