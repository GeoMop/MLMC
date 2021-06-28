Sampler creation
=================
Sampler controls the execution of MLMC samples


First, import mlmc package and define basic MLMC parameters

.. testcode::

    import mlmc
    n_levels = 3 # number of MLMC levels
    step_range = [0.5, 0.005] # simulation steps at the coarsest and finest levels
    level_parameters = mlmc.estimator.determine_level_parameters(n_levels, step_range)
    # level_parameters determine each level simulation steps
    # level_parameters can be manually prescribed as a list of lists


Prepare a simulation, it must be instance of class that inherits from :any:`mlmc.sim.simulation.Simulation`

.. testcode::

    simulation_factory = mlmc.SynthSimulation()

Create a sampling pool

.. testcode::

    sampling_pool = mlmc.OneProcessPool()


You can also use :any:`mlmc.sampling_pool.ProcessPool` which supports parallel execution of MLMC samples.
In order to use PBS (portable batch system), employ :any:`mlmc.sampling_pool_pbs.SamplingPoolPBS`


Create a sample storage

.. testcode::

    # Memory() storage keeps samples in the computer main memory
    sample_storage = mlmc.Memory()

We support also HDF5 file storage :any:`mlmc.sample_storage_hdf.SampleStorageHDF`


Finally, create a sampler that manages scheduling MLMC samples and also saves the results

.. testcode::

    sampler = mlmc.Sampler(sample_storage=sample_storage,
                                   sampling_pool=sampling_pool,
                                   sim_factory=simulation_factory,
                                   level_parameters=level_parameters)



:ref:`examples samples scheduling`