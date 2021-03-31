import os
import shutil
import numpy as np
from scipy import stats
import pytest
from mlmc.sim.synth_simulation import SynthSimulationWorkspace
from test.synth_sim_for_tests import SynthSimulationForTests
from mlmc.sampler import Sampler
from mlmc.sample_storage import Memory
from mlmc.sample_storage_hdf import SampleStorageHDF
from mlmc.sampling_pool import OneProcessPool, ProcessPool
from mlmc.moments import Legendre
from mlmc.quantity.quantity import make_root_quantity
import mlmc.estimator

# Set work dir
os.chdir(os.path.dirname(os.path.realpath(__file__)))
work_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '_test_tmp')
if os.path.exists(work_dir):
    shutil.rmtree(work_dir)
os.makedirs(work_dir)

# Create simulations
failed_fraction = 0.1
distr = stats.norm()
simulation_config = dict(distr=distr, complexity=2, nan_fraction=failed_fraction, sim_method='_sample_fn')

shutil.copyfile('synth_sim_config.yaml', os.path.join(work_dir, 'synth_sim_config.yaml'))
simulation_config_workspace = {"config_yaml": os.path.join(work_dir, 'synth_sim_config.yaml')}


def hdf_storage_factory():
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    work_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '_test_tmp')
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir)

    # Create sample storages
    return SampleStorageHDF(file_path=os.path.join(work_dir, "mlmc_test.hdf5"))


@pytest.mark.parametrize("test_case", [(SynthSimulationForTests(simulation_config), Memory(), OneProcessPool()),
                                       (SynthSimulationForTests(simulation_config), Memory(), ProcessPool(4)),
                                       # (SynthSimulationForTests(simulation_config), Memory(), ThreadPool(4)),
                                       (SynthSimulationForTests(simulation_config), hdf_storage_factory(), OneProcessPool()),
                                       (SynthSimulationForTests(simulation_config), hdf_storage_factory(), ProcessPool(4)),
                                       #(SynthSimulationForTests(simulation_config), hdf_storage_factory(), ThreadPool(4)),
                                       (SynthSimulationWorkspace(simulation_config_workspace), Memory(), OneProcessPool(work_dir=work_dir)),
                                       (SynthSimulationWorkspace(simulation_config_workspace), Memory(), ProcessPool(4, work_dir=work_dir)),
                                       #(SynthSimulationWorkspace(simulation_config), storage_memory,  ThreadPool(4, work_dir=work_dir)),
                                       (SynthSimulationWorkspace(simulation_config_workspace), hdf_storage_factory(), OneProcessPool(work_dir=work_dir)),
                                       (SynthSimulationWorkspace(simulation_config_workspace), hdf_storage_factory(), ProcessPool(4, work_dir=work_dir)),
                                       #(SynthSimulationWorkspace(simulation_config), hdf_storage_factory(),  ThreadPool(4, work_dir=work_dir))
                                       ]
                         )
def test_mlmc(test_case):
    np.random.seed(1234)
    n_moments = 5
    step_range = [[0.1], [0.001]]

    simulation_factory, sample_storage, sampling_pool = test_case

    if simulation_factory.need_workspace:
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        shutil.copyfile('synth_sim_config.yaml', os.path.join(work_dir, 'synth_sim_config.yaml'))

    sampler = Sampler(sample_storage=sample_storage, sampling_pool=sampling_pool, sim_factory=simulation_factory,
                      level_parameters=step_range)

    true_domain = distr.ppf([0.0001, 0.9999])
    moments_fn = Legendre(n_moments, true_domain)
    # moments_fn = Monomial(n_moments, true_domain)

    sampler.set_initial_n_samples([10, 10])
    # sampler.set_initial_n_samples([10000])
    sampler.schedule_samples()
    sampler.ask_sampling_pool_for_samples()

    target_var = 1e-3
    sleep = 0
    add_coef = 0.1

    quantity = make_root_quantity(sample_storage, q_specs=simulation_factory.result_format())

    length = quantity['length']
    time = length[1]
    location = time['10']
    value_quantity = location[0]

    estimator = mlmc.estimator.Estimate(value_quantity, sample_storage, moments_fn)

    # New estimation according to already finished samples
    variances, n_ops = estimator.estimate_diff_vars_regression(sampler._n_scheduled_samples)
    n_estimated = mlmc.estimator.estimate_n_samples_for_target_variance(target_var, variances, n_ops,
                                                                       n_levels=sampler.n_levels)

    # Loop until number of estimated samples is greater than the number of scheduled samples
    while not sampler.process_adding_samples(n_estimated, sleep, add_coef):
        # New estimation according to already finished samples
        variances, n_ops = estimator.estimate_diff_vars_regression(sampler._n_scheduled_samples)
        n_estimated = mlmc.estimator.estimate_n_samples_for_target_variance(target_var, variances, n_ops,
                                                                           n_levels=sampler.n_levels)

    means, vars = estimator.estimate_moments(moments_fn)
    assert means[0] == 1
    assert vars[0] == 0


if __name__ == "__main__":
    test_mlmc((SynthSimulationForTests(simulation_config), Memory(), OneProcessPool()))
    #multiproces_sampler_test()
    #threads_sampler_test()

