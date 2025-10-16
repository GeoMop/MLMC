.. _examples samples scheduling:

Samples Scheduling
==================

Once you have created a **Sampler**, you can schedule the execution of MLMC samples in different ways.

This tutorial demonstrates two approaches:
1. Scheduling with a prescribed number of samples.
2. Scheduling to reach a target variance automatically.


1. Prescribing an Exact Number of Samples
-----------------------------------------

In this approach, you explicitly set how many samples should be created at each MLMC level.

.. testcode::
    :hide:

    import mlmc
    n_levels = 3  # number of MLMC levels
    step_range = [0.5, 0.005]  # simulation steps at the coarsest and finest levels
    level_parameters = mlmc.estimator.determine_level_parameters(n_levels, step_range)

    simulation_factory = mlmc.SynthSimulation()
    sampling_pool = mlmc.OneProcessPool()
    sample_storage = mlmc.Memory()

    sampler = mlmc.Sampler(
        sample_storage=sample_storage,
        sampling_pool=sampling_pool,
        sim_factory=simulation_factory,
        level_parameters=level_parameters
    )


Set the number of samples for each level:

.. testcode::

    n_samples = [100, 75, 50]
    sampler.set_initial_n_samples(n_samples)

Schedule the prescribed samples:

.. testcode::

    sampler.schedule_samples()

Wait until all samples have finished:

.. testcode::

    running = 1
    while running > 0:
        running = 0
        running += sampler.ask_sampling_pool_for_samples()


2. Prescribing a Target Variance
--------------------------------

This approach automatically determines the number of samples needed to achieve a desired **target variance**.

Define the target variance and the number of **moment functions** used for estimation:

.. testcode::

    target_var = 1e-4
    n_moments = 10

As before, initialize and run a first batch of samples (automatically ranging from 100 samples on the coarsest level
to 10 on the finest level):

.. testcode::

    sampler.set_initial_n_samples()
    sampler.schedule_samples()

    running = 1
    while running > 0:
        running = 0
        running += sampler.ask_sampling_pool_for_samples()


Creating the Quantity and Moment Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Next, create a :py:class:`mlmc.quantity.quantity.Quantity` object representing the MLMC results.

.. testcode::

    root_quantity = mlmc.make_root_quantity(
        storage=sampler.sample_storage,
        q_specs=sampler.sample_storage.load_result_format()
    )

The :code:`root_quantity` contains both the data structure and the sampled values.

Now we create the **moment functions** and an :py:class:`mlmc.estimator.Estimate` instance to perform statistical estimation.

.. testcode::

    true_domain = mlmc.Estimate.estimate_domain(root_quantity, sample_storage)
    moments_fn = mlmc.Legendre(n_moments, true_domain)

    estimate_obj = mlmc.Estimate(
        root_quantity,
        sample_storage=sampler.sample_storage,
        moments_fn=moments_fn
    )


Estimating Variances and Computational Cost
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

From the finished samples, estimate the variance of moments and the average computational cost per sample for each level:

.. testcode::

    variances, n_ops = estimate_obj.estimate_diff_vars_regression(sampler.n_finished_samples)


Estimating the Required Number of Samples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Using the estimated variances and costs, determine the number of samples required to meet the **target variance**:

.. testcode::

    from mlmc.estimator import estimate_n_samples_for_target_variance

    n_estimated = estimate_n_samples_for_target_variance(
        target_var, variances, n_ops, n_levels=sampler.n_levels
    )


Iterative Sampling Process
~~~~~~~~~~~~~~~~~~~~~~~~~~

The sampling algorithm incrementally schedules additional samples until the estimated number of required samples is reached.

.. testcode::

    while not sampler.process_adding_samples(n_estimated):
        # Recalculate estimates based on completed samples
        variances, n_ops = estimate_obj.estimate_diff_vars_regression(sampler._n_scheduled_samples)
        n_estimated = estimate_n_samples_for_target_variance(
            target_var, variances, n_ops, n_levels=sampler.n_levels
        )

Finally, wait until all samples are completed:

.. testcode::

    running = 1
    while running > 0:
        running = 0
        running += sampler.ask_sampling_pool_for_samples()


Notes
-----

- The sampling algorithm automatically adjusts the number of samples per level based on estimated **moment variances**.
- The choice of moment functions (e.g. :class:`mlmc.Legendre`) influences the total number of samples needed.
- For most cases, the default **Legendre polynomials** provide a good balance between accuracy and computational cost.


**Summary**
-----------

In this tutorial, you learned how to:
- Schedule MLMC samples with a fixed number of samples per level.
- Automatically adapt the number of samples to achieve a target variance.
- Use moment functions and variance regression for optimal sample allocation.

Continue to the next section for :ref:`examples results postprocessing`.
