.. _examples results postprocessing:
Results Postprocessing
======================

Once you know how to **create a sampler**, **schedule samples**, and **work with quantities**,
postprocessing becomes straightforward.
If you haven’t gone through those steps yet, please review the earlier tutorials first.

Estimating Moments
------------------

We start by scheduling samples and estimating moments for a specific quantity.

.. testcode::

    import mlmc

    n_levels = 3                     # Number of MLMC levels
    step_range = [0.5, 0.005]        # Simulation steps for coarsest and finest levels
    target_var = 1e-4                # Desired target variance
    n_moments = 10                   # Number of moment functions

    # Compute level parameters (simulation steps per level)
    level_parameters = mlmc.estimator.determine_level_parameters(n_levels, step_range)

    # Initialize components
    simulation_factory = mlmc.SynthSimulation()
    sampling_pool = mlmc.OneProcessPool()
    sample_storage = mlmc.Memory()   # In-memory sample storage

    # Create the sampler
    sampler = mlmc.Sampler(
        sample_storage=sample_storage,
        sampling_pool=sampling_pool,
        sim_factory=simulation_factory,
        level_parameters=level_parameters
    )

    # Schedule and run initial samples
    sampler.set_initial_n_samples()
    sampler.schedule_samples()

    running = 1
    while running > 0:
        running = 0
        running += sampler.ask_sampling_pool_for_samples()

Accessing Quantities
--------------------

Now we extract a specific **Quantity** from the results and set up the estimation.

.. testcode::

    # Obtain the root quantity representing all simulation outputs
    root_quantity = mlmc.make_root_quantity(sampler.sample_storage, simulation_factory.result_format())

    # Select a sub-quantity
    length = root_quantity["length"]
    time = length[1]
    location = time["10"]
    q_value = location[0]


Defining the Domain and Estimator
---------------------------------

Before estimating higher-order statistics, we determine the valid domain and define the estimator.

.. testcode::

    true_domain = mlmc.Estimate.estimate_domain(q_value, sample_storage)
    moments_fn = mlmc.Legendre(n_moments, true_domain)

    estimate_obj = mlmc.Estimate(
        q_value,
        sample_storage=sampler.sample_storage,
        moments_fn=moments_fn
    )


Variance and Sample Estimation
------------------------------

We can now estimate variances and determine how many samples are required
to achieve the target variance.

.. testcode::

    variances, n_ops = estimate_obj.estimate_diff_vars_regression(sampler.n_finished_samples)

    from mlmc.estimator import estimate_n_samples_for_target_variance
    n_estimated = estimate_n_samples_for_target_variance(
        target_var, variances, n_ops, n_levels=sampler.n_levels
    )

    # Add and process samples iteratively until convergence
    while not sampler.process_adding_samples(n_estimated):
        variances, n_ops = estimate_obj.estimate_diff_vars_regression(sampler._n_scheduled_samples)
        n_estimated = estimate_n_samples_for_target_variance(
            target_var, variances, n_ops, n_levels=sampler.n_levels
        )

    running = 1
    while running > 0:
        running = 0
        running += sampler.ask_sampling_pool_for_samples()


Probability Density Function Approximation
------------------------------------------

Finally, we can construct and visualize an approximation of the **probability density function (PDF)**.

.. testcode::

    from mlmc.plot.plots import Distribution

    distr_obj, result, _, _ = estimate_obj.construct_density()

    distr_plot = Distribution(title="Distributions", error_plot=None)
    distr_plot.add_distribution(distr_obj)

    # For single-level simulations, add a histogram of raw samples
    if n_levels == 1:
        samples = estimate_obj.get_level_samples(level_id=0)[..., 0]
        distr_plot.add_raw_samples(np.squeeze(samples))

    distr_plot.show()


**Summary**
-----------

In this tutorial, you learned how to:
- Estimate statistical moments from MLMC results.
- Automatically determine the required number of samples to reach a target variance.
- Construct and visualize an estimated probability density function.

This completes the **postprocessing** workflow of the MLMC pipeline.


