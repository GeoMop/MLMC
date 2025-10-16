.. _examples quantity:

Quantity Tutorial
=================

This tutorial provides an overview of basic :any:`mlmc.quantity.quantity.Quantity` operations.

The :mod:`mlmc.quantity` module and its related classes allow you to:
- Estimate means and variances of MLMC sample results.
- Derive new quantities from existing ones.
- Perform arithmetic and NumPy-based operations on quantities.


Setup
-----

Before exploring `Quantity` operations, we first set up a simple synthetic MLMC sampler.

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

    n_samples = [100, 75, 50]
    sampler.set_initial_n_samples(n_samples)

    running = 1
    while running > 0:
        running = 0
        running += sampler.ask_sampling_pool_for_samples()


.. testcode::

    import numpy as np
    import mlmc.quantity.quantity_estimate
    from examples.synthetic_quantity import create_sampler


Creating a Synthetic Quantity
-----------------------------

We begin by creating a synthetic :class:`mlmc.quantity.quantity.Quantity` with a predefined ``result_format``:

.. testcode::

    # result_format = [
    #     mlmc.QuantitySpec(name="length", unit="m", shape=(2, 1), times=[1, 2, 3], locations=['10', '20']),
    #     mlmc.QuantitySpec(name="width", unit="mm", shape=(2, 1), times=[1, 2, 3], locations=['30', '40']),
    # ]

    sampler, simulation_factory, moments_fn = create_sampler()
    root_quantity = mlmc.make_root_quantity(sampler.sample_storage, simulation_factory.result_format())

This means the sample results contain data for two quantities (`length` and `width`) at three time steps `[1, 2, 3]` and two locations each.

:code:`root_quantity` is an instance of :class:`mlmc.quantity.quantity.Quantity`, representing the full result data structure.


Mean Estimates
--------------

To compute the estimated mean of a quantity:

.. testcode::

    root_quantity_mean = mlmc.quantity.quantity_estimate.estimate_mean(root_quantity)

The returned object, :code:`root_quantity_mean`, is a :class:`mlmc.quantity.quantity.QuantityMean` instance.

To retrieve statistical values:

.. testcode::

    # Total mean and variance
    root_quantity_mean.mean
    root_quantity_mean.var

    # Means and variances at each level
    root_quantity_mean.l_means
    root_quantity_mean.l_vars


Moments and Covariance Estimation
---------------------------------

To create and estimate statistical moments:

.. testcode::

    moments_quantity = mlmc.quantity.quantity_estimate.moments(root_quantity, moments_fn=moments_fn)
    moments_mean = mlmc.quantity.quantity_estimate.estimate_mean(moments_quantity)

To obtain **central moments**, first subtract the mean:

.. testcode::

    central_root_quantity = root_quantity - root_quantity_mean.mean
    central_moments_quantity = mlmc.quantity.quantity_estimate.moments(
        central_root_quantity, moments_fn=moments_fn
    )
    central_moments_mean = mlmc.quantity.quantity_estimate.estimate_mean(central_moments_quantity)

To estimate a **covariance matrix**:

.. testcode::

    covariance_quantity = mlmc.quantity.quantity_estimate.covariance(root_quantity, moments_fn=moments_fn)
    cov_mean = mlmc.quantity.quantity_estimate.estimate_mean(covariance_quantity)


Quantity Selection
------------------

You can access and manipulate sub-quantities directly using the structure defined by `result_format`:

.. testcode::

    length = root_quantity["length"]  # Get quantity with name="length"
    width = root_quantity["width"]    # Get quantity with name="width"

Both are still :class:`mlmc.quantity.quantity.Quantity` instances.

Selecting by **time**:

.. testcode::

    length_locations = length.time_interpolation(2.5)

Selecting by **location**:

.. testcode::

    length_result = length_locations['10']

Now, :code:`length_result` behaves like a NumPy array:

.. testcode::

    length_result[1, 0]
    length_result[:, 0]
    length_result[:, :]
    length_result[:1, :1]
    length_result[:2, ...]

.. note::

   - All derived quantities (like :code:`length_locations` or :code:`length_result`) remain `Quantity` instances.
   - Selecting a **location before time** is not supported.


Binary Operations
-----------------

`Quantity` supports standard arithmetic operations:

**Between quantities:**

.. testcode::

    quantity = root_quantity + root_quantity
    quantity = root_quantity + root_quantity + root_quantity

**With constants:**

.. testcode::

    const = 5
    quantity_const_add = root_quantity + const
    quantity_const_sub = root_quantity - const
    quantity_const_mult = root_quantity * const
    quantity_const_div = root_quantity / const
    quantity_const_mod = root_quantity % const
    quantity_add_mult = root_quantity + root_quantity * const


NumPy Universal Functions
--------------------------

`Quantity` objects are compatible with many NumPy universal functions (`ufuncs`):

.. testcode::

    quantity_np_add = np.add(root_quantity, root_quantity)
    quantity_np_max = np.max(root_quantity, axis=0, keepdims=True)
    quantity_np_sin = np.sin(root_quantity)
    quantity_np_sum = np.sum(root_quantity, axis=0, keepdims=True)
    quantity_np_maximum = np.maximum(root_quantity, root_quantity)

    x = np.ones(24)
    quantity_np_divide_const = np.divide(x, root_quantity)
    quantity_np_add_const = np.add(x, root_quantity)
    quantity_np_arctan2_const = np.arctan2(x, root_quantity)


Conditional Selection
----------------------

You can select parts of a quantity using logical conditions via the :code:`select()` method.

.. testcode::

    selected_quantity = root_quantity.select(0 < root_quantity)

Or using comparisons between quantities:

.. testcode::

    quantity_add = root_quantity + root_quantity
    quantity_add_select = quantity_add.select(root_quantity < quantity_add)
    root_quantity_selected = root_quantity.select(-1 != root_quantity)

Multiple conditions are combined using logical **AND**:

.. testcode::

    quantity_add.select(root_quantity < quantity_add, root_quantity < 10)

Use NumPy logical functions for more complex conditions:

.. testcode::

    selected_quantity_or = root_quantity.select(np.logical_or(0 < root_quantity, root_quantity < 10))

You can also explicitly define the selection mask:

.. testcode::

    mask = np.logical_and(0 < root_quantity, root_quantity < 10)
    q_bounded = root_quantity.select(mask)
