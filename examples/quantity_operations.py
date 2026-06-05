"""
Quantity examples and quick reference.

This module demonstrates basic operations with mlmc Quantity objects.
It shows how to:
 - create a synthetic root quantity (result format specified below),
 - compute mean estimates,
 - estimate moments and covariance,
 - select sub-quantities, time-interpolate and slice,
 - perform arithmetic and NumPy ufuncs with Quantity objects,
 - perform selections using conditions and masks.

Result format used in the synthetic example:

result_format = [
    mlmc.quantity.quantity_spec.QuantitySpec(
        name="length", unit="m", shape=(2, 1), times=[1, 2, 3], locations=['10', '20']
    ),
    mlmc.quantity.quantity_spec.QuantitySpec(
        name="width", unit="mm", shape=(2, 1), times=[1, 2, 3], locations=['30', '40']
    ),
]

Meaning:
 - sample results contain data on two quantities ("length" and "width"),
 - each quantity is evaluated at three times [1,2,3] and two locations,
 - each quantity can also have its own internal shape.
"""
import numpy as np
import mlmc.quantity.quantity_spec
from mlmc.quantity.quantity import make_root_quantity
import mlmc.quantity.quantity_estimate
from examples.synthetic_quantity import create_sampler

# -----------------------------------------------------------------------------
# Create synthetic quantity
# -----------------------------------------------------------------------------
"""
Create a synthetic sampler + factory and produce a root Quantity instance.

- create_sampler() returns: (sampler, simulation_factory, moments_fn)
- make_root_quantity(storage, q_specs) builds a Quantity object that represents the
  whole result data structure (root with named sub-quantities).
"""
sampler, simulation_factory, moments_fn = create_sampler()
root_quantity = make_root_quantity(sampler.sample_storage, simulation_factory.result_format())
# root_quantity is an mlmc.quantity.quantity.Quantity instance and represents the whole result data,
# it contains two sub-quantities named "length" and "width"

# -----------------------------------------------------------------------------
# Mean estimates
# -----------------------------------------------------------------------------
"""
Compute and inspect mean estimates for a Quantity.

- estimate_mean(quantity) returns a QuantityMean instance that contains:
    - .mean : mean Quantity
    - .var  : variance information
    - .l_vars : level variances (if available)
"""
root_quantity_mean = mlmc.quantity.quantity_estimate.estimate_mean(root_quantity)
# root_quantity_mean is an instance of mlmc.quantity.QuantityMean
# To get overall mean value:
root_quantity_mean.mean
# To get overall variance value:
root_quantity_mean.var
# To get level variance value:
root_quantity_mean.l_vars

# -----------------------------------------------------------------------------
# Estimate moments and covariance matrix
# -----------------------------------------------------------------------------
"""
Construct moments and covariance quantities from a root Quantity.

- moments(root_quantity, moments_fn=moments_fn) returns a Quantity of moments.
- estimate_mean(moments_quantity) computes means for those moments.
- covariance(root_quantity, moments_fn=moments_fn) returns a Quantity describing covariance.
"""
moments_quantity = mlmc.quantity.quantity_estimate.moments(root_quantity, moments_fn=moments_fn)
moments_mean = mlmc.quantity.quantity_estimate.estimate_mean(moments_quantity)
# Central moments:
central_root_quantity = root_quantity - root_quantity_mean.mean
central_moments_quantity = mlmc.quantity.quantity_estimate.moments(
    central_root_quantity, moments_fn=moments_fn
)
central_moments_mean = mlmc.quantity.quantity_estimate.estimate_mean(central_moments_quantity)
# Create a quantity representing covariance matrix
covariance_quantity = mlmc.quantity.quantity_estimate.covariance(root_quantity, moments_fn=moments_fn)
cov_mean = mlmc.quantity.quantity_estimate.estimate_mean(covariance_quantity)

# Both moments() and covariance() calls return mlmc.quantity.quantity.Quantity instances

# -----------------------------------------------------------------------------
# Quantity selection
# -----------------------------------------------------------------------------
"""
Examples of indexing and time/ location selection on a Quantity.

According to the result_format you can select by name, then time, then location.
Selecting location before time is not supported.
"""
length = root_quantity["length"]  # Get quantity with name="length"
width = root_quantity["width"]  # Get quantity with name="width"

# To get a quantity at a particular (interpolated) time:
length_locations = length.time_interpolation(2.5)
# length_locations represents results for all locations of quantity named "length" at the time 2.5

# To get quantity at particular location:
length_result = length_locations['10']
# length_result represents shape=(2, 1) data of "length" at time 2.5 and location '10'

# You can slice Quantity like an ndarray:
#   length_result[1, 0], length_result[:, 0], length_result[:2, ...], etc.

# -----------------------------------------------------------------------------
# Binary operations
# -----------------------------------------------------------------------------
"""
Supported arithmetic operations between compatible Quantity objects and scalars.
The result is a Quantity instance with the same result_format structure.
"""
quantity = root_quantity + root_quantity
quantity = root_quantity + root_quantity + root_quantity

# Operations with Quantity and constant
const = 5
quantity_const_add = root_quantity + const
quantity_const_sub = root_quantity - const
quantity_const_mult = root_quantity * const
quantity_const_div = root_quantity / const
quantity_const_mod = root_quantity % const
quantity_add_mult = root_quantity + root_quantity * const

# -----------------------------------------------------------------------------
# NumPy universal functions (ufuncs)
# -----------------------------------------------------------------------------
"""
Many NumPy ufuncs are supported and return Quantity instances:
- np.add, np.max, np.sin, np.sum, np.maximum, np.divide, np.arctan2, ...
"""
quantity_np_add = np.add(root_quantity, root_quantity)
quantity_np_max = np.max(root_quantity, axis=0, keepdims=True)
quantity_np_sin = np.sin(root_quantity)
quantity_np_sum = np.sum(root_quantity, axis=0, keepdims=True)
quantity_np_maximum = np.maximum(root_quantity, root_quantity)

x = np.ones(24)
quantity_np_divide_const = np.divide(x, root_quantity)
quantity_np_add_const = np.add(x, root_quantity)
quantity_np_arctan2_cosnt = np.arctan2(x, root_quantity)

# -----------------------------------------------------------------------------
# Quantity selection by a condition
# -----------------------------------------------------------------------------
"""
The select(condition) method extracts elements of a Quantity according to a boolean condition;
it returns a Quantity (masking is internal and shape-aware).

Examples:
 - selected_quantity = root_quantity.select(0 < root_quantity)
 - quantity_add_select = quantity_add.select(root_quantity < quantity_add)
 - root_quantity_selected = root_quantity.select(-1 != root_quantity)

You can combine conditions via logical ufuncs (np.logical_or / np.logical_and), and
you can explicitly pass a Quantity mask (mask is a Quantity instance).
"""
selected_quantity = root_quantity.select(0 < root_quantity)

quantity_add = root_quantity + root_quantity
quantity_add_select = quantity_add.select(root_quantity < quantity_add)
root_quantity_selected = root_quantity.select(-1 != root_quantity)

# Logical operation for more conditions is AND
quantity_add.select(root_quantity < quantity_add, root_quantity < 10)

# Use NumPy logical ufuncs for complex masks
selected_quantity_or = root_quantity.select(
    np.logical_or(0 < root_quantity, root_quantity < 10)
)

# Explicit mask Quantity
mask = np.logical_and(0 < root_quantity, root_quantity < 10)  # mask is a Quantity instance
q_bounded = root_quantity.select(mask)
