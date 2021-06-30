Results postprocessing
======================

If you already know how to create a sampler, schedule samples and handle quantities,
postprocessing will be easy for you. Otherwise, see the previous tutorials before.


First, schedule samples and estimate moments for a particular quantity

.. testcode::

    import mlmc
    n_levels = 3 # number of MLMC levels
    step_range = [0.5, 0.005] # simulation steps at the coarsest and finest levels
    target_var = 1e-4
    n_moments = 10
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

    sampler.set_initial_n_samples()
    sampler.schedule_samples()
    running = 1
    while running > 0:
        running = 0
        running += sampler.ask_sampling_pool_for_samples()

    # Get particular quantity
    root_quantity = mlmc.make_root_quantity(sampler.sample_storage, simulation_factory.result_format())
    length = root_quantity['length']
    time = length[1]
    location = time['10']
    q_value = location[0]


    true_domain = mlmc.Estimate.estimate_domain(q_value, sample_storage)
    moments_fn = mlmc.Legendre(n_moments, true_domain)
    estimate_obj = mlmc.Estimate(q_value, sample_storage=sampler.sample_storage,
                                           moments_fn=moments_fn)

    variances, n_ops = estimate_obj.estimate_diff_vars_regression(sampler.n_finished_samples)

    from mlmc.estimator import estimate_n_samples_for_target_variance
    n_estimated = estimate_n_samples_for_target_variance(target_var, variances, n_ops,
                                                         n_levels=sampler.n_levels)

    while not sampler.process_adding_samples(n_estimated):
        # New estimation according to already finished samples
        variances, n_ops = estimate_obj.estimate_diff_vars_regression(sampler._n_scheduled_samples)
        n_estimated = estimate_n_samples_for_target_variance(target_var, variances, n_ops,
                                                             n_levels=sampler.n_levels)

    running = 1
    while running > 0:
        running = 0
        running += sampler.ask_sampling_pool_for_samples()


Probability density function approximation
---------------------

.. testcode::

    from mlmc.plot.plots import Distribution
    distr_obj, result, _, _ = estimate_obj.construct_density()
    distr_plot = Distribution(title="distributions", error_plot=None)
    distr_plot.add_distribution(distr_obj)

    if n_levels == 1:
        samples = estimate_obj.get_level_samples(level_id=0)[..., 0]
        distr_plot.add_raw_samples(np.squeeze(samples)) # add histogram
    distr_plot.show()
