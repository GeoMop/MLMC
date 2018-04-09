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


def make_abs(*pathlist):
    this_path = os.path.dirname(os.path.realpath(__file__))
    abs_pathlist = (this_path,) + pathlist
    return os.path.join(*abs_pathlist)


"""
JS TODO:
This is a proposed test of the flow_mc using a flow123d_mock.py instead of Flow123d.
"""


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
    pbs = FlowPbs("scripts", qsub=False)
    pbs.pbs_common_setting(n_cores=1,
                           n_nodes=1,
                           mem='4gb',
                           queue='charon')

    env = dict(
        flow123d=flow123d,
        gmsh=gmsh,
        pbs=pbs
    )
    cond_field = correlated_field.SpatialCorrelatedField(corr_exp='gauss', dim=2, corr_length=0.5, log=True)
    fields = correlated_field.FieldSet("conductivity", cond_field)
    yaml_path = os.path.join(file_dir, '01_cond_field', '01_conductivity.yaml')
    geo_path = "/storage/liberec1-tul/home/martin_spetlik/MLMC/test/01_cond_field/square_1x1.geo"
    step_range = (0.2, 0.01)

    # '/home/martin/Documents/MLMC/test/01_cond_field/square_1x1.geo'

    yaml_path = os.path.abspath(yaml_path)
    if geo_path is None:
        pass
        # geo_path = os.path.splitext(yaml_path)[0] + '.geo'
        # geo_path = "/storage/01_cond_field/square_1x1.geo"

    my_config = {
        'env': env,  # The Environment.
        'fields': fields,  # correlated_field.FieldSet object
        'yaml_file': yaml_path,  # The template with a mesh and field placeholders
        'sim_param_range': step_range,  # Range of MLMC simulation parametr. Here the mesh step.
        'geo_file': geo_path  # The file with simulation geometry (independent of the step)
    }

    sim_factory = lambda t_level=None: flow_mc.FlowSim.make_sim(my_config, (0.9, 0.8), t_level)
    # sim_factory = lambda t_level=None: SimT.make_sim(my_config, (10, 100), t_level)

    n_levels = 3
    n_moments = 4
    result = Result(n_moments)
    function = moments_function = Monomials(
        n_moments)  # JS TODO: This should set a single moment function corresponding to the mean value.
    mc = mlmc.mlmc.MLMC(n_levels, sim_factory, moments_function, pbs)
    # mc.set_target_variance([0.01, 0.01, 0.01, 0.01])
    mc.num_of_simulations = [11, 11, 11]
    mc.refill_samples()

    result.mc_levels = mc.levels
    result.process_data()
    result.format_result()
    mean = result.average
    moments_function.mean = mean

    moments_result = result.level_moments()
    print("Moments", moments_result)

    function.mean = mean
    mc_data = result.levels_data[0]
    toleration = 1e-14
    eps = 1e-10

    mc_data = [data for data in result.levels_data[0]]
    bounds = sc.stats.mstats.mquantiles(mc_data, prob=[eps, 1 - eps])
    print(bounds)
    bounds[0] = bounds[0] - 5
    bounds[1] = bounds[1] + 5
    function.bounds = bounds
    function.fixed_quad_n = n_moments * 5
    # Run distribution
    distribution = Distribution(function, n_moments, moments_result, toleration)
    distribution.newton_method()

    # Empirical distribution function
    samples = np.linspace(bounds[0], bounds[1], 10000)
    print(samples)
    distribution_function = []
    ## Set approximate density values
    approximate_density = []
    print("final lagrangians", distribution.lagrangian_parameters)

    def integrand(x):
        return distribution.density(x)

    for step, value in enumerate(samples):
        integral = sc.integrate.quad(integrand, bounds[0], value)
        approximate_density.append(distribution.density(value))
        distribution_function.append(integral[0])

    print("Aproximované momenty", distribution.approximation_moments)
    print("Původní momenty", moments_result)

    path = "Result"
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.mkdir(path, 0o755);
    if isinstance(function, Monomials):
        m_func = "Monomialy"
    else:
        m_func = "Fourierovy funkce"
    result.save_result(path, m_func)
    # print(distribution_function)

    plt.figure(1)
    # plt.plot(samples, distribution_function, color='blue', linestyle='dashed', label="aproximovaná distribuční funkce")
    # plt.legend()
    # plt.savefig(path + "/distribuce.png")
    ## Show approximate and exact density

    # plt.figure(2)
    plt.plot(samples, approximate_density, color='blue', linestyle='dashed', label="aproximovaná hustota")
    # plt.plot(samples, [sc.stats.norm.pdf(sample, 2, 1) for sample in samples],  'r', label="přesná hustota")
    # plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
    #   ncol=2, mode="expand", borderaxespad=0.)
    # plt.savefig(path + "/hustota.png")
    plt.xlim(-10, 10)
    plt.show()


test_mlmc_flow()