import numpy as np
from mlmc.quantity.quantity_spec import QuantitySpec
from mlmc.sample_storage import Memory
from mlmc.quantity.quantity import make_root_quantity
from mlmc.quantity.quantity_estimate import estimate_mean, moment, moments, covariance, cache_clear
from examples.synthetic_quantity import fill_sample_storage


######################################
###    Create synthetic quantity   ###
######################################
# Quantity result format
# Meaning: sample results contain data about three quantities in three time steps [1, 2, 3] and at two locations
#          each quantity can have different shape
result_format = [
        QuantitySpec(name="depth", unit="mm", shape=(2, 2), times=[1, 2, 3], locations=['30', '40']),
        QuantitySpec(name="length", unit="m", shape=(2, 3), times=[1, 2, 3], locations=['10', '20']),
        QuantitySpec(name="width", unit="mm", shape=(2, 4), times=[1, 2, 3], locations=['30', '40'])
]

sample_storage = Memory()
result_format, size = fill_sample_storage(sample_storage, result_format)  # Auxiliary method
root_quantity = make_root_quantity(sample_storage, result_format)
# root_quantity is mlmc.quantity.Quantity instance and represents the whole result data,
# it contains all three sub-quantities data

###################################
####    Mean estimates     #####
###################################
# To get means of any quantity call:
means_root = estimate_mean(root_quantity)
# now means_root and means_depth are instances of mlmc.quantity.QuantityMean
# To get overall mean call:
means_root.mean
# To get overall variance call:
means_root.var
# To get level variance call:
means_root.l_vars

##################################
####   Selecting items     #######
##################################
# According to the result_format, tt is possible to select items from the root quantity
depth = root_quantity['depth']  # Get quantity with name="depth"
length = root_quantity["length"]  # Get quantity with name="length"
width = root_quantity["width"]  # Get quantity with name="length"
# depth, length and width are still mlmc.quantity.Quantity instances

# To get a quantity at particular time:
length_locations = length.time_interpolation(2.5)
# length_locations represents results for all locations of quantity named "length" at the time 2,5

# To get quantity at particular location
length_result = length_locations['10']
# length_result represents results shape=(2, 2) of quantity named "length" at the time 2,5 and location '10'

# Now it is possible to slice Quantity length_result the same way as np.ndarray
# For example:
#   length_result[1, 2]
#   length_result[:, 2]
#   length_result[:, :]
#   length_result[:1, 1:2]
#   length_result[:2, ...]

# Keep in mind:
# - all derived quantities such as length_locations and length_result, ... are still mlmc.quantity.Quantity instances
# - selecting location before time is not supported


##################################
####   Select by condition    ####
##################################


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
quanttity_add_mult = root_quantity + root_quantity * const


###################################
#### NumPy universal functions ####
###################################
# Examples of tested NumPy universal functions:
quantity_np_add = np.add(root_quantity, root_quantity)
quantity_np_max = np.max(root_quantity, axis=0, keepdims=True)
quantity_np_sin = np.sin(root_quantity)
quantity_np_sum = np.sum(root_quantity, axis=0, keepdims=True)
quantity_np_maximum = np.maximum(root_quantity, root_quantity)

x = np.ones(108)
quantity_np_divide_const = np.divide(x, root_quantity)
quantity_np_add_const = np.add(x, root_quantity)
quantity_np_arctan2_cosnt = np.arctan2(x, root_quantity)
quantity_np_logical_and = np.logical_and(True, root_quantity)







