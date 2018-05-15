import os
import sys
import shutil
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../src/')

import flow_mc as flow_mc
import mlmc.correlated_field as correlated_field
from mlmc.moments import Monomials, FourierFunctions
import mlmc.mlmc
from result import Result
from mlmc.distribution import Distribution
import numpy as np
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from simulation_test import SimulationTest as SimT
from scipy.stats.mstats import mquantiles
from flow_pbs import FlowPbs
import scipy as sc
#import subprocess as sp

def make_abs(*pathlist):
    this_path = os.path.dirname(os.path.realpath(__file__))
    abs_pathlist = (this_path,) + pathlist
    return os.path.join(*abs_pathlist)


"""
JS TODO:
This is a proposed test of the flow_mc using a flow123d_mock.py instead of Flow123d.
"""
def runCmd(exe):
    p = sp.Popen(exe,stdout=sp.PIPE, stderr=sp.STDOUT, shell=True)
    #p = sp.check_output(exe, shell=True).decode("UTF-8")
    #print(p)
    while True:
        retcode = p.poll()
        line = p.stdout.readline()
        yield line
        if retcode is not None:
            break


# @pytest.mark.metacentrum
def test_mlmc_flow():
    # Get directory of this test script.
    file_dir = os.path.dirname(os.path.realpath(__file__))

    # Make flow123 wrapper script.
    flow123d = "/storage/praha1/home/jan_brezina/local/flow123d_2.2.0/flow123d"

    # GMSH (make empty mesh)
    # gmsh = "/usr/bin/gmsh"
    gmsh = "/storage/liberec1-tul/home/martin_spetlik/astra/gmsh/bin/gmsh"

    # Charon setting:
    pbs = FlowPbs("scripts", qsub=True)
    pbs.pbs_common_setting(n_cores=1,
        n_nodes=1,
        mem='4gb',
        queue='charon')

    env = dict(
        flow123d=flow123d,
        gmsh=gmsh,
        pbs=pbs
    )
    corr_field_dict = dict(corr_exp='gauss',
                          dim=2,
                          corr_length=0.5, 
                          log=True,
                          names=["conductivity"])

    yaml_path = os.path.join(file_dir, '01_cond_field', '01_conductivity.yaml')
    geo_path = "/storage/liberec1-tul/home/martin_spetlik/MLMC/test/01_cond_field/square_1x1.geo"

    yaml_path = os.path.abspath(yaml_path)
    if geo_path is None:
        pass
        # geo_path = os.path.splitext(yaml_path)[0] + '.geo'
        # geo_path = "/storage/01_cond_field/square_1x1.geo"
    
    step_range = (1, 0.08) 
    my_config = {
        'env': env,  # The Environment.
        'field_name': corr_field_dict,  # correlated_field.FieldSet object
        'yaml_file': yaml_path,  # The template with a mesh and field placeholders
        'sim_param_range': step_range,  # Range of MLMC simulation parametr. Here the mesh step.
        'geo_file': geo_path  # The file with simulation geometry (independent of the step)
    }
         
    sim_factory = lambda t_level=None: flow_mc.FlowSim.make_sim(my_config, step_range, t_level)
 
    n_levels=5
    n_moments=20
    result = Result(n_moments)
    moments_function = FourierFunctions(n_moments)  # JS TODO: This should set a single moment function corresponding to the mean value. 
    mc = mlmc.mlmc.MLMC(n_levels, sim_factory, moments_function, pbs)
    #mc.set_target_variance([1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5,
    #                        1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5,
    #                        1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5,
    #                        1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5,
    #                        1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5
    #                       ])
    mc.num_of_simulations = [20, 20, 20, 20, 20]
    mc.refill_samples()

test_mlmc_flow()
