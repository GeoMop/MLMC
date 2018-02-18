import flow_mc as fsim
#import importlib
import pytest
import os
import flow_mc
import mlmc
import numpy as np
import matplotlib.pyplot as plt
#from scipy.stats.mstats import mquantiles

"""
JS TODO:
This is a proposed test of the flow_mc using a flow123d_mock.py instead of Flow123d.
"""
@pytest.mark.skip(reason="no way of currently testing this")
def test_flow_mc():
    # Get directory of this test script.
    file_dir = os.path.dirname(os.path.realpath(__file__))

    # Make flow123 wrapper script.
    flow123d = os.path.join(file_dir, "flow123d_mock.sh")
    flow_mock = os.path.join(file_dir, "flow123d_mock.py")
    with open(flow123d, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("python3 %s"%(flow_mock))
    os.chmod(flow123d, 0o770)

    # GMSH (make empty mesh)
    gmsh = None

    # pbs (run localy)
    pbs = None

    # Charon setting:
    # pbs = dict(
    #         n_cpu =1,
    #         n_nodes =1,
    #         mem= '4gb',
    #         queue='charon')

    env = flow_mc.Environment(flow123d, gmsh, pbs)
    cond_field = mlmc.correlated_field.SpatialCorrelatedField(corr_exp='gauss', dim=2, corr_length=0.5, )
    fields = mlmc.correlated_field.FieldSet("conductivity", cond_field)
    yaml_path = os.path.join(file_dir, '01_cond_field', '01_cond.yaml')
    geo_path = 'square.geo'

    sim = flow_mc.FlowMC(env, fields, yaml_path, geo_path)
    sim.set_range((0.2, 0.01))

    n_levels = 3
    moments = mlmc.Monomials()  # JS TODO: This should set a single moment function corresponding to the mean value.
    mc = mlmc.mlmc.MLMC(n_levels, sim, moments)
    mc.set_target_variance(0.01)
    mc.refill_samples()

    # process samples, whch should be moved from Result into MLMC
