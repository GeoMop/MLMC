import flow_mc as fsim
#import importlib
import pytest
import os
import flow_mc
from mlmc.correlated_field import FieldSet, SpatialCorrelatedField
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
#from scipy.stats.mstats import mquantiles

def make_abs(*pathlist):
    this_path = os.path.dirname(os.path.realpath(__file__))
    abs_pathlist = (this_path,) + pathlist
    return os.path.join(*abs_pathlist)


def test_flow_mc():
    env = dict(
        flow123d = make_abs("mocks", "flow123d_mock.sh"),
        gmsh = make_abs("mocks", "gmsh_mock.sh")
    )
    cond_field = SpatialCorrelatedField(corr_exp='gauss', dim=2, corr_length=0.5, )
    flow_conf = dict(
        env=env,
        fields=FieldSet("conductivity", cond_field),
        yaml_file=make_abs("01_cond_field", "01_conductivity.yaml"),
        geo_file=make_abs("01_cond_field", "square_1x1.geo")
    )

    flow_sim0 = flow_mc.FlowSim(flow_conf, 0.1)
    dir = "sim_0_step_0.100000"
    assert os.path.isdir(make_abs("01_cond_field", dir))
    flow_input = make_abs("01_cond_field", dir, "flow_input.yaml")
    assert os.path.exists(flow_input)
    mesh_file = make_abs("01_cond_field", dir, "mesh.msh")
    assert os.path.exists(mesh_file)
    with open(flow_input, "r") as f:
        content = " ".join(f.readlines())
        assert content.find(mesh_file) != -1
        assert content.find("fields_sample.msh") != -1
        assert '<' not in content
        assert '>' not in content


    flow_sim1 = flow_mc.FlowSim(flow_conf, 0.01)
    assert os.path.isdir(make_abs("01_cond_field/sim_1_step_0.010000"))


