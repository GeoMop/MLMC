import numpy as np
import mlmc.quantity.quantity_spec
from mlmc.quantity.quantity import make_root_quantity
import mlmc.quantity.quantity_estimate
from examples.synthetic_quantity import create_sampler

# An overview of basic Quantity operations

######################################
###    Create synthetic quantity   ###
######################################
# Synthetic Quantity with the following result format
# result_format = [
#     mlmc.quantity.quantity_spec.QuantitySpec(name="length", unit="m", shape=(2, 1), times=[1, 2, 3], locations=['10', '20']),
#     mlmc.quantity.quantity_spec.QuantitySpec(name="width", unit="mm", shape=(2, 1), times=[1, 2, 3], locations=['30', '40']),
# ]
# Meaning: sample results contain data on three quantities in three time steps [1, 2, 3] and in two locations,
#          each quantity can have different shape

sampler, simulation_factory, moments_fn = create_sampler()
root_quantity = make_root_quantity(sampler.sample_storage, simulation_factory.result_format())
# root_quantity is mlmc.quantity.quantity.Quantity instance and represents the whole result data,
# it contains two sub-quantities named "length" and "width"

###################################
####      Mean estimates      #####
###################################
# To get estimated mean of a quantity:
root_quantity_mean = mlmc.quantity.quantity_estimate.estimate_mean(root_quantity)
# root_quantity_mean is an instance of mlmc.quantity.QuantityMean
# To get overall mean value:
root_quantity_mean.mean
# To get overall variance value:
root_quantity_mean.var
# To get level variance value:
root_quantity_mean.l_vars

#########################################################
####     Estimate moments and covariance matrix     #####
#########################################################
# Create a quantity representing moments
moments_quantity = mlmc.quantity.quantity_estimate.moments(root_quantity, moments_fn=moments_fn)
moments_mean = mlmc.quantity.quantity_estimate.estimate_mean(moments_quantity)
# Central moments:
central_root_quantity = root_quantity - root_quantity_mean.mean
central_moments_quantity = mlmc.quantity.quantity_estimate.moments(central_root_quantity, moments_fn=moments_fn)
central_moments_mean = mlmc.quantity.quantity_estimate.estimate_mean(central_moments_quantity)
# Create a quantity representing covariance matrix
covariance_quantity = mlmc.quantity.quantity_estimate.covariance(root_quantity, moments_fn=moments_fn)
cov_mean = mlmc.quantity.quantity_estimate.estimate_mean(covariance_quantity)

# Both moments() and covariance() calls return mlmc.quantity.quantity.Quantity instance

##################################
###    Quantity selection     ####
##################################
# According to the result_format, tt is possible to select items from a quantity
length = root_quantity["length"]  # Get quantity with name="length"
width = root_quantity["width"]  # Get quantity with name="width"
# length and width are still mlmc.quantity.quantity.Quantity instances

# To get a quantity at particular time:
length_locations = length.time_interpolation(2.5)
# length_locations represents results for all locations of quantity named "length" at the time 2.5

# To get quantity at particular location
length_result = length_locations['10']
# length_result represents results shape=(2, 1) of quantity named "length" at the time 2,5 and location '10'

# Now it is possible to slice Quantity length_result the same way as np.ndarray
# For example:
#   length_result[1, 0]
#   length_result[:, 0]
#   length_result[:, :]
#   length_result[:1, :1]
#   length_result[:2, ...]

# Keep in mind:
# - all derived quantities such as length_locations and length_result, ... are still mlmc.quantity.quantity.Quantity instances
# - selecting location before time is not supported!

###################################
####    Binary operations     #####
###################################
# Following operations are supported
# Addition of compatible quantities
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


###################################
#### NumPy universal functions ####
###################################
# Examples of tested NumPy universal functions:
quantity_np_add = np.add(root_quantity, root_quantity)
quantity_np_max = np.max(root_quantity, axis=0, keepdims=True)
quantity_np_sin = np.sin(root_quantity)
quantity_np_sum = np.sum(root_quantity, axis=0, keepdims=True)
quantity_np_maximum = np.maximum(root_quantity, root_quantity)

x = np.ones(24)
quantity_np_divide_const = np.divide(x, root_quantity)
quantity_np_add_const = np.add(x, root_quantity)
quantity_np_arctan2_cosnt = np.arctan2(x, root_quantity)

################################################
####   Quantity selection by a condition    ####
################################################
# Method select returns mlmc.quantity.quantity.Quantity instance
selected_quantity = root_quantity.select(0 < root_quantity)

quantity_add = root_quantity + root_quantity
quantity_add_select = quantity_add.select(root_quantity < quantity_add)
root_quantity_selected = root_quantity.select(-1 != root_quantity)

# Logical operation for more conditions is AND
quantity_add.select(root_quantity < quantity_add, root_quantity < 10)

# User can use one of the logical NumPy universal functions
selected_quantity_or = root_quantity.select(np.logical_or(0 < root_quantity, root_quantity < 10))

# It is possible to explicitly define the selection condition of one quantity by another quantity
mask = np.logical_and(0 < root_quantity, root_quantity < 10)  # mask is Quantity instance
q_bounded = root_quantity.select(mask)
