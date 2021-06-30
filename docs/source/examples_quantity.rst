.. _examples quantity:

Quantity tutorial
=================

An overview of basic :any:`mlmc.quantity.quantity.Quantity` operations.
Quantity related classes and functions allow estimate mean and variance of MLMC samples results,
derive other quantities from original ones and much more.

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


First, the synthetic Quantity with the following :code:`result_format` is created

.. testcode::

    # result_format = [
    #     mlmc.QuantitySpec(name="length", unit="m", shape=(2, 1), times=[1, 2, 3], locations=['10', '20']),
    #     mlmc.QuantitySpec(name="width", unit="mm", shape=(2, 1), times=[1, 2, 3], locations=['30', '40']),
    # ]
    # Meaning: sample results contain data on two quantities in three time steps [1, 2, 3] and in two locations,
    #          each quantity can have different shape

    sampler, simulation_factory, moments_fn = create_sampler()
    root_quantity = mlmc.make_root_quantity(sampler.sample_storage, simulation_factory.result_format())

:code:`root_quantity` is :py:class:`mlmc.quantity.quantity.Quantity` instance and represents the whole result data.
According to :code:`result_format` it contains two sub-quantities named "length" and "width".


Mean estimates
---------------
To get estimated mean of a quantity:

.. testcode::

    root_quantity_mean = mlmc.quantity.quantity_estimate.estimate_mean(root_quantity)

:code:`root_quantity_mean` is an instance of :py:class:`mlmc.quantity.quantity.QuantityMean`

To get the total mean value:

.. testcode::

    root_quantity_mean.mean

To get the total variance value:

.. testcode::

    root_quantity_mean.var

To get means at each level:

.. testcode::

    root_quantity_mean.l_means

To get variances at each level:

.. testcode::

    root_quantity_mean.l_vars


Estimate moments and covariance matrix
--------------------------------------

Create a quantity representing moments and get their estimates

.. testcode::

    moments_quantity = mlmc.quantity.quantity_estimate.moments(root_quantity, moments_fn=moments_fn)
    moments_mean = mlmc.quantity.quantity_estimate.estimate_mean(moments_quantity)

To obtain central moments, use:

.. testcode::

    central_root_quantity = root_quantity - root_quantity_mean.mean
    central_moments_quantity = mlmc.quantity.quantity_estimate.moments(central_root_quantity,
                                                                            moments_fn=moments_fn)
    central_moments_mean = mlmc.quantity.quantity_estimate.estimate_mean(central_moments_quantity)


Create a quantity representing a covariance matrix

.. testcode::

    covariance_quantity = mlmc.quantity.quantity_estimate.covariance(root_quantity, moments_fn=moments_fn)
    cov_mean = mlmc.quantity.quantity_estimate.estimate_mean(covariance_quantity)



Quantity selection
------------------

According to the result_format, it is possible to select items from a quantity

.. testcode::

    length = root_quantity["length"]  # Get quantity with name="length"
    width = root_quantity["width"]  # Get quantity with name="width"

:code:`length` and :code:`width` are still :py:class:`mlmc.quantity.quantity.Quantity` instances

To get a quantity at particular time:

.. testcode::

    length_locations = length.time_interpolation(2.5)

:code:`length_locations` represents results for all locations of quantity named "length" at the time 2.5

To get quantity at particular location:

.. testcode::

    length_result = length_locations['10']

:code:`length_result` represents results shape=(2, 1) of quantity named "length" at the time 2,5 and location '10'

Now it is possible to slice Quantity :code:`length_result` the same way as :code:`np.ndarray`. For example:

.. testcode::

    length_result[1, 0]
    length_result[:, 0]
    length_result[:, :]
    length_result[:1, :1]
    length_result[:2, ...]

Keep in mind:
    - all derived quantities such as :code:`length_locations` and :code:`length_result`, ... are still :py:class:`mlmc.quantity.quantity.Quantity` instances
    - selecting location before time is not supported!


Binary operations
-----------------
Following operations are supported

 - Addition, subtraction, ... of compatible quantities

    .. testcode::

        quantity = root_quantity + root_quantity
        quantity = root_quantity + root_quantity + root_quantity

 -  Operations with Quantity and a constant

     .. testcode::

        const = 5
        quantity_const_add = root_quantity + const
        quantity_const_sub = root_quantity - const
        quantity_const_mult = root_quantity * const
        quantity_const_div = root_quantity / const
        quantity_const_mod = root_quantity % const
        quantity_add_mult = root_quantity + root_quantity * const


NumPy universal functions
--------------------------

Examples of tested NumPy universal functions:

.. testcode::

    quantity_np_add = np.add(root_quantity, root_quantity)
    quantity_np_max = np.max(root_quantity, axis=0, keepdims=True)
    quantity_np_sin = np.sin(root_quantity)
    quantity_np_sum = np.sum(root_quantity, axis=0, keepdims=True)
    quantity_np_maximum = np.maximum(root_quantity, root_quantity)

    x = np.ones(24)
    quantity_np_divide_const = np.divide(x, root_quantity)
    quantity_np_add_const = np.add(x, root_quantity)
    quantity_np_arctan2_cosnt = np.arctan2(x, root_quantity)


Quantity selection by conditions
---------------------------------

Method :code:`select` returns :py:class:`mlmc.quantity.quantity.Quantity` instance

.. testcode::

    selected_quantity = root_quantity.select(0 < root_quantity)

.. testcode::

    quantity_add = root_quantity + root_quantity
    quantity_add_select = quantity_add.select(root_quantity < quantity_add)
    root_quantity_selected = root_quantity.select(-1 != root_quantity)

Logical operation among more provided conditions is AND

.. testcode::

    quantity_add.select(root_quantity < quantity_add, root_quantity < 10)

User can use one of the logical NumPy universal functions

.. testcode::

    selected_quantity_or = root_quantity.select(np.logical_or(0 < root_quantity, root_quantity < 10))

It is possible to explicitly define the selection condition of one quantity by another quantity

.. testcode::

    mask = np.logical_and(0 < root_quantity, root_quantity < 10)  # mask is Quantity instance
    q_bounded = root_quantity.select(mask)


