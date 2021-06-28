.. _examples samples scheduling:

Samples scheduling
==================

Once you create a sampler you can schedule samples.


1. Prescribe the exact number of samples
----------------------------------------------------------------

.. testcode::
    :hide:

    import mlmc
    n_levels = 3 # number of MLMC levels
    step_range = [0.5, 0.005] # simulation steps at the coarsest and finest levels
    level_parameters = mlmc.estimator.determine_level_parameters(n_levels, step_range)
    # level_parameters determine each level simulation steps
    # level_parameters can be manually prescribed as a list of lists

    simulation_factory = mlmc.SynthSimulation()
    sampling_pool = mlmc.OneProcessPool()
    # Memory() storage keeps samples in the computer main memory
    sample_storage = mlmc.Memory()

    sampler = mlmc.Sampler(sample_storage=sample_storage,
                                   sampling_pool=sampling_pool,
                                   sim_factory=simulation_factory,
                                   level_parameters=level_parameters)


.. testcode::

    n_samples = [100, 75, 50]
    sampler.set_initial_n_samples(n_samples)

Schedule set samples.

.. testcode::

    sampler.schedule_samples()

You can wait until all samples are finished.

.. testcode::

        running = 1
        while running > 0:
            running = 0
            running += sampler.ask_sampling_pool_for_samples()


2. Prescribe a target variance
-------------------------------------------------------------

Set target variance and number of random variable moments that must meet this variance.

.. testcode::

     target_var = 1e-4
     n_moments = 10

The first phase is the same as the first approach, but the initial samples are automatically determined
as a sequence from 100 samples at the coarsest level to 10 samples at the finest level.

.. testcode::

    sampler.set_initial_n_samples()
    sampler.schedule_samples()
    running = 1
    while running > 0:
        running = 0
        running += sampler.ask_sampling_pool_for_samples()


The :py:class:`mlmc.quantity.quantity.Quantity` instance is created, for details see :ref:`examples quantity`

.. testcode::

    root_quantity = mlmc.make_root_quantity(storage=sampler.sample_storage,
                                   q_specs=sampler.sample_storage.load_result_format())

:code:`root_quantity` contains the structure of sample results and also allows access to their values.

In order to estimate moment values including variance, moment functions class (in this case Legendre polynomials) instance
and :py:class:`mlmc.estimator.Estimate` instance are created.

.. testcode::

    true_domain = mlmc.Estimate.estimate_domain(root_quantity, sample_storage)
    moments_fn = mlmc.Legendre(n_moments, true_domain)

    estimate_obj = mlmc.Estimate(root_quantity, sample_storage=sampler.sample_storage,
                                           moments_fn=moments_fn)


Firstly, the variance of moments and average execution time per sample at each level are estimated from already finished samples.

.. testcode::

    variances, n_ops = estimate_obj.estimate_diff_vars_regression(sampler.n_finished_samples)

Then, an initial estimate of number of MLMC samples that should meet prescribed target variance is conducted.

.. testcode::

    from mlmc.estimator import estimate_n_samples_for_target_variance
    n_estimated = estimate_n_samples_for_target_variance(target_var, variances, n_ops,
                                                         n_levels=sampler.n_levels)


Now it is time for our sampling algorithm that gradually schedules samples and refines the total number of samples
until the number of estimated samples is greater than the number of scheduled samples.

.. testcode::

    while not sampler.process_adding_samples(n_estimated):
        # New estimation according to already finished samples
        variances, n_ops = estimate_obj.estimate_diff_vars_regression(sampler._n_scheduled_samples)
        n_estimated = estimate_n_samples_for_target_variance(target_var, variances, n_ops,
                                                             n_levels=sampler.n_levels)


Finally, wait until all samples are finished.

.. testcode::

    running = 1
    while running > 0:
        running = 0
        running += sampler.ask_sampling_pool_for_samples()

Since our sampling algorithm determines number of samples according to moment variances,
type of moment functions (Legendre by default) might affect total number of MLMC samples

