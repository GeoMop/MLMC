import numpy as np
import importlib
import correlated_field 
import test_convergence 
from test_convergence import TestSpatialCorrelatedField, make_points_grid


importlib.import_module('test_convergence')
importlib.import_module('correlated_field')
ncells = (32,32)
bounds = (32,32)
points = make_points_grid(bounds, ncells)

test_g = TestSpatialCorrelatedField()
mu_err, cum_sigma = test_g.perform_test_on_point_set(ncells, points, (200,np.inf),'exp')

