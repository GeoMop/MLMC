# import os
# import sys
# import numpy as np
# import scipy.stats as stat
#
# src_path = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.join(src_path, '..', 'src'))
#
# import flow_mc as flow_mc
# import mlmc.correlated_field as correlated_field
# import mlmc.moments
# import mlmc.mlmc
# # from result import Result
# import mlmc.distribution
# from pbs import Pbs
#
#
# class TstFlowPbs:
#     def __init__(self):
#         """
#         Create some samples.
#         """
#         # Get directory of this test script.
#         file_dir = os.path.dirname(os.path.realpath(__file__))
#         input_dir = os.path.join(file_dir, '01_cond_field')
#         output_dir = os.path.join(input_dir, 'output')
#         self.scripts_dir = os.path.join(output_dir, 'scripts')
#
#         # Make flow123 wrapper script.
#         # flow123d = "/storage/praha1/home/jan_brezina/local/flow123d_2.2.0/flow123d"
#         # flow123d = "/home/jb/workspace/flow123d/bin/fterm flow123d dbg"
#         flow123d = os.path.join(src_path, 'mocks', 'flow_mock')
#
#         # GMSH (make empty mesh)
#         # gmsh = "/usr/bin/gmsh"
#         # gmsh = "/storage/liberec1-tul/home/martin_spetlik/astra/gmsh/bin/gmsh"
#         gmsh = "/home/jb/local/gmsh-3.0.5-git-Linux/bin/gmsh"
#         # Charon setting:
#         self.pbs = Pbs(self.scripts_dir, qsub=os.path.join(src_path, 'mocks', 'qsub'), clean=True)
#         self.pbs.pbs_common_setting(n_cores=1, n_nodes=1, mem='4gb', queue='charon')
#
#         env = dict(
#             flow123d=flow123d,
#             gmsh=gmsh,
#             pbs=self.pbs
#         )
#         conductivity = dict(
#             mu=0.0,
#             sigma=1.0,
#             corr_exp='gauss',
#             dim=2,
#             corr_length=0.5,
#             log=True
#             )
#         cond_field = correlated_field.SpatialCorrelatedField(**conductivity)
#         fields = correlated_field.Fields([correlated_field.Field("conductivity", cond_field)])
#         yaml_path = os.path.join(file_dir, '01_cond_field', '01_conductivity.yaml')
#         geo_path = os.path.join(file_dir, '01_cond_field', 'square_1x1.geo')
#
#         step_range = (1, 0.1)
#         simulation_config = {
#             'env': env,  # The Environment.
#             'fields': fields,
#             'output_dir': os.path.join(file_dir, '01_cond_field', 'output'),
#             'yaml_file': yaml_path,  # The template with a mesh and field placeholders
#             'sim_param_range': step_range,  # Range of MLMC simulation parametr. Here the mesh step.
#             'geo_file': geo_path,  # The file with simulation geometry (independent of the step)
#             'remove_old': True
#         }
#
#         flow_mc.FlowSim.total_sim_id = 0
#         self.simulation_factory = flow_mc.FlowSim.factory(step_range, config=simulation_config)
#
#         self.n_levels = 3
#         mlmc_options = {'output_dir': None,
#                         'keep_collected': True,
#                         'regen_failed': False}
#         mc = mlmc.mlmc.MLMC(self.n_levels, self.simulation_factory, self.pbs, mlmc_options)
#         mc.set_initial_n_samples(self.n_levels * [6])
#         mc.refill_samples()
#         mc.wait_for_simulations()
#         self.mc = mc
#         self.n_moments = 5
#         true_domain = stat.norm.ppf([0.001, 0.999])
#         self.moments_fn = mlmc.moments.Legendre(self.n_moments, true_domain)
#
#
#     def test_load_levels(self):
#         other_pbs = Pbs(self.scripts_dir,
#                         qsub=None)
#         flow_mc.FlowSim.total_sim_id = 0
#
#         mlmc_options = {'output_dir': None,
#                         'keep_collected': True,
#                         'regen_failed': False}
#         other_mc = mlmc.mlmc.MLMC(self.mc.n_levels, self.simulation_factory, other_pbs, mlmc_options)
#         #other_mc.subsample(self.n_levels * [4])
#
#         means_full, _ = self.mc.estimate_moments(self.moments_fn)
#         means, _ = other_mc.estimate_moments(self.moments_fn)
#
#         assert np.allclose(means, means_full)
#
#
# def _test_flow_pbs_base():
#     pbs_test = TstFlowPbs()
#     pbs_test.test_load_levels()
