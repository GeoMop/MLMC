import src.flow_mc as fsim
#import importlib
import pytest
import os
import src.flow_mc as flow_mc
import src.mlmc
import src.mlmc.correlated_field
import src.mlmc.moments
import src.mlmc.mlmc
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
    flow123d = "/storage/praha1/home/jan_brezina/local/flow123d_2.2.0/"


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
    cond_field = src.mlmc.correlated_field.SpatialCorrelatedField(corr_exp='gauss', dim=2, corr_length=0.5, )
    fields = src.mlmc.correlated_field.FieldSet("conductivity", cond_field)
    yaml_path = os.path.join(file_dir, '01_cond_field', '01_conductivity.yaml')
    geo_path = 'square.geo'

    sim = flow_mc.FlowSimGeneric(env, fields, yaml_path, (10, 100), geo_path)

    n_levels = 3
    n_moments = 5
    moments = src.mlmc.moments.Monomials(n_moments)  # JS TODO: This should set a single moment function corresponding to the mean value.
    mc = src.mlmc.mlmc.MLMC(n_levels, sim, moments)
    mc.set_target_variance(0.01)
    mc.refill_samples()
    # process samples, whch should be moved from Result into MLMC


test_flow_mc()