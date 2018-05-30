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


class TstFlowPbs:
    def __init__(self):
        """
        Create some samples.
        """
        # Get directory of this test script.
        file_dir = os.path.dirname(os.path.realpath(__file__))
        input_dir = os.path.join(file_dir, '01_cond_field')
        output_dir = os.path.join(input_dir, 'output')
        scripts_dir = os.path.join(output_dir, 'scripts')

        # Make flow123 wrapper script.
        # flow123d = "/storage/praha1/home/jan_brezina/local/flow123d_2.2.0/flow123d"
        # flow123d = "/home/jb/workspace/flow123d/bin/fterm flow123d dbg"
        flow123d = os.path.join(src_path, 'mocks', 'flow_mock')

        # GMSH (make empty mesh)
        # gmsh = "/usr/bin/gmsh"
        # gmsh = "/storage/liberec1-tul/home/martin_spetlik/astra/gmsh/bin/gmsh"
        gmsh = "/home/jb/local/gmsh-3.0.5-git-Linux/bin/gmsh"
        # Charon setting:
        self.pbs = FlowPbs(scripts_dir,
                      qsub=os.path.join(src_path, 'mocks', 'qsub'),
                      clean=True)
        # pbs = FlowPbs(scripts_dir,
        #               package_weight=25000,  # max number of elements per package
        #               qsub=None)
        self.pbs.pbs_common_setting(n_cores=1,
                               n_nodes=1,
                               mem='4gb',
                               queue='charon')

        env = dict(
            flow123d=flow123d,
            gmsh=gmsh,
            pbs=self.pbs
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
            'geo_file': geo_path,  # The file with simulation geometry (independent of the step)
            'remove_old': True
        }

        flow_mc.FlowSim.total_sim_id = 0
        simultion_factory = lambda t_level: flow_mc.FlowSim.make_sim(simulation_config, step_range, t_level)

        n_levels = 5
        mc = mlmc.mlmc.MLMC(n_levels, simultion_factory, self.pbs)
        mc.set_initial_n_samples(n_levels * [10])
        mc.refill_samples()
        mc.wait_for_simulations()
        self.mc = mc


def test_flow_pbs_base():
    pbs_test = TstFlowPbs()


