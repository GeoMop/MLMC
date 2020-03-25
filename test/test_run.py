import os
import shutil
import numpy as np
from scipy import stats
import pytest
from mlmc.sim.synth_simulation import SynthSimulation, SynthSimulationWorkspace
from mlmc.sampler import Sampler
from mlmc.sample_storage import Memory
from mlmc.sample_storage_hdf import SampleStorageHDF
from mlmc.sampling_pool import OneProcessPool, ProcessPool, ThreadPool
from mlmc.moments import Legendre
from mlmc.quantity_estimate import QuantityEstimate
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
simulation = SynthSimulation(simulation_config)
shutil.copyfile('synth_sim_config.yaml', os.path.join(work_dir, 'synth_sim_config.yaml'))
simulation_config = {"config_yaml": os.path.join(work_dir, 'synth_sim_config.yaml')}
simulation_workspace = SynthSimulationWorkspace(simulation_config)

# Create sample storages
storage_hdf = SampleStorageHDF(file_path=os.path.join(work_dir, "mlmc_test.hdf5"))
storage_memory = Memory()

# Create sampling pools
sampling_pool_single_process = OneProcessPool()
sampling_pool_processes = ProcessPool(4)
sampling_pool_thread = ThreadPool(4)
sampling_pool_single_process_dir = OneProcessPool(work_dir=work_dir)
sampling_pool_processes_dir = ProcessPool(4, work_dir=work_dir)
sampling_pool_thread_dir = ThreadPool(4, work_dir=work_dir)


@pytest.mark.parametrize("test_case", [(simulation, storage_memory, sampling_pool_single_process),
                                       (simulation, storage_memory, sampling_pool_processes),
                                       (simulation, storage_memory, sampling_pool_thread),
                                       (simulation, storage_hdf, sampling_pool_single_process),
                                       (simulation, storage_hdf, sampling_pool_processes),
                                       (simulation, storage_hdf, sampling_pool_thread),
                                       (simulation_workspace, storage_memory, sampling_pool_single_process_dir),
                                       (simulation_workspace, storage_memory, sampling_pool_processes_dir),
                                       (simulation_workspace, storage_memory, sampling_pool_thread_dir),
                                       (simulation_workspace, storage_hdf, sampling_pool_single_process_dir),
                                       (simulation_workspace, storage_hdf, sampling_pool_processes_dir),
                                       (simulation_workspace, storage_hdf, sampling_pool_thread_dir)])
def test_mlmc(test_case):
    np.random.seed(1234)
    n_moments = 5
    step_range = [0.1, 0.001]

    simulation_factory, sample_storage, sampling_pool = test_case
    if sampling_pool._work_dir is not None:
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)
        os.makedirs(work_dir)
    if simulation_factory.need_workspace:
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        shutil.copyfile('synth_sim_config.yaml', os.path.join(work_dir, 'synth_sim_config.yaml'))

    sampler = Sampler(sample_storage=sample_storage, sampling_pool=sampling_pool, sim_factory=simulation_factory,
                      step_range=step_range)

    true_domain = distr.ppf([0.0001, 0.9999])
    moments_fn = Legendre(n_moments, true_domain)
    # moments_fn = Monomial(n_moments, true_domain)

    sampler.set_initial_n_samples([10, 10])
    # sampler.set_initial_n_samples([10000])
    sampler.schedule_samples()
    sampler.ask_sampling_pool_for_samples()

    q_estimator = QuantityEstimate(sample_storage=sample_storage, moments_fn=moments_fn, sim_steps=step_range)
    #
    # target_var = 1e-4
    # sleep = 0
    # add_coef = 0.1
    #
    # # @TODO: test
    # # New estimation according to already finished samples
    # variances, n_ops = q_estimator.estimate_diff_vars_regression(sampler._n_scheduled_samples)
    # n_estimated = mlmc.new_estimator.estimate_n_samples_for_target_variance(target_var, variances, n_ops,
    #                                                                    n_levels=sampler.n_levels)
    #
    # # Loop until number of estimated samples is greater than the number of scheduled samples
    # while not sampler.process_adding_samples(n_estimated, sleep, add_coef):
    #     # New estimation according to already finished samples
    #     variances, n_ops = q_estimator.estimate_diff_vars_regression(sampler._n_scheduled_samples)
    #     n_estimated = mlmc.new_estimator.estimate_n_samples_for_target_variance(target_var, variances, n_ops,
    #                                                                        n_levels=sampler.n_levels)

    print("collected samples ", sampler._n_scheduled_samples)
    means, vars = q_estimator.estimate_moments(moments_fn)

    print("means ", means)
    print("vars ", vars)
    assert means[0] == 1
    assert vars[0] == 0
