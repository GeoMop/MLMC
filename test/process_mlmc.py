import os
import sys
import shutil

src_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(src_path, '..', 'src'))

import flow_mc as flow_mc
import mlmc.correlated_field as correlated_field

import mlmc.moments
import mlmc.mlmc
# from result import Result
import mlmc.distribution
from flow_pbs import FlowPbs



def test_mlmc_subsample():
    # Get directory of this test script.
    file_dir = os.path.dirname(os.path.realpath(__file__))
    input_dir = os.path.join(file_dir, '01_cond_field')
    output_dir = os.path.join(input_dir, 'output')
    scripts_dir = os.path.join(output_dir, 'scripts')

    # Make flow123 wrapper script.
    # flow123d = "/storage/praha1/home/jan_brezina/local/flow123d_2.2.0/flow123d"
    #flow123d = "/home/jb/workspace/flow123d/bin/fterm flow123d dbg"
    flow123d = os.path.join(src_path, 'mocks', 'flow_mock')

    # GMSH (make empty mesh)
    # gmsh = "/usr/bin/gmsh"
    # gmsh = "/storage/liberec1-tul/home/martin_spetlik/astra/gmsh/bin/gmsh"
    gmsh = "/home/jb/local/gmsh-3.0.5-git-Linux/bin/gmsh"
    # Charon setting:
    pbs = FlowPbs(scripts_dir,
                  reload=True)

    env = dict(
        flow123d=flow123d,
        gmsh=gmsh,
        pbs=pbs
    )
    corr_field_dict = dict(
        conductivity=dict(
            corr_exp='gauss',
            dim=2,
            corr_length=0.5,
            log=True
        ))

    yaml_path = os.path.join(file_dir, '01_cond_field', '01_conductivity.yaml')
    geo_path = os.path.join(file_dir, '01_cond_field', 'square_1x1.geo')

    step_range = (1, 0.1)
    simulation_config = {
        'env': env,  # The Environment.
        'field_name': corr_field_dict,  # correlated_field.FieldSet object
        'yaml_file': yaml_path,  # The template with a mesh and field placeholders
        'sim_param_range': step_range,  # Range of MLMC simulation parametr. Here the mesh step.
        'geo_file': geo_path  # The file with simulation geometry (independent of the step)
    }

    flow_mc.FlowSim.total_sim_id = 0
    simultion_factory = lambda t_level: flow_mc.FlowSim.make_sim(simulation_config, step_range, t_level)

    n_levels = 9
    mc = mlmc.mlmc.MLMC(n_levels, simultion_factory, pbs)
    domain = (0, 6)
    n_moments = 5
    moments_fn = lambda x,n=n_moments, a=domain[0], b=domain[1]:   mlmc.moments.legendre_moments(x, n, a, b)

    means, vars = mc.estimate_moments(moments_fn)
    print( "Means: {}\nVars: {}\n".format(means, vars))
    N = 9*[5]
    mc.subsample(N)
    print("Subsample:", N)
    means, vars = mc.estimate_moments(moments_fn)
    print( "Means: {}\nVars: {}\n".format(means, vars))
    mc.subsample()
    print("Subsample full")
    means, vars = mc.estimate_moments(moments_fn)
    print( "Means: {}\nVars: {}\n".format(means, vars))



test_mlmc_subsample()