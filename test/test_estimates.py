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
failed_fraction = 0.0
distr = stats.norm()
simulation_config = dict(distr=distr, complexity=2, nan_fraction=failed_fraction, sim_method='_sample_fn')

shutil.copyfile('synth_sim_config.yaml', os.path.join(work_dir, 'synth_sim_config.yaml'))
simulation_config_workspace = {"config_yaml": os.path.join(work_dir, 'synth_sim_config.yaml')}


def hdf_storage_factory(file_name="mlmc_test.hdf5"):
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    work_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '_test_tmp')
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir)

    # Create sample storages
    return SampleStorageHDF(file_path=os.path.join(work_dir, file_name))


def mlmc_test(test_case):
    #np.random.seed(1234)
    n_moments = 20
    step_range = [[0.1], [0.005], [0.00025]]

    simulation_factory, sample_storage, sampling_pool = test_case

    if simulation_factory.need_workspace:
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        shutil.copyfile('synth_sim_config.yaml', os.path.join(work_dir, 'synth_sim_config.yaml'))

    sampler = Sampler(sample_storage=sample_storage, sampling_pool=sampling_pool, sim_factory=simulation_factory,
                      level_parameters=step_range)

    true_domain = distr.ppf([0.0001, 0.9999])
    moments_fn = Legendre(n_moments, true_domain)
    # moments_fn = Monomial(n_moments, true_domain)

    sampler.set_initial_n_samples([100, 50, 25])
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
        print("n estimated ", n_estimated)
        # New estimation according to already finished samples
        variances, n_ops = estimator.estimate_diff_vars_regression(sampler._n_scheduled_samples)
        n_estimated = mlmc.estimator.estimate_n_samples_for_target_variance(target_var, variances, n_ops,
                                                                           n_levels=sampler.n_levels)

    means, vars = estimator.estimate_moments(moments_fn)
    assert means[0] == 1
    assert vars[0] == 0

    return estimator.moments_mean_obj, sample_storage.get_n_collected(), sample_storage.get_n_ops()


def mlmc_test_mcqmc(test_case):
    # np.random.seed(1234)
    n_moments = 10
    step_range = [[0.1], [0.005], [0.00025]]

    simulation_factory, sample_storage, sampling_pool = test_case

    if simulation_factory.need_workspace:
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        shutil.copyfile('synth_sim_config.yaml', os.path.join(work_dir, 'synth_sim_config.yaml'))

    sampler = Sampler(sample_storage=sample_storage, sampling_pool=sampling_pool, sim_factory=simulation_factory,
                      level_parameters=step_range)

    true_domain = distr.ppf([0.0001, 0.9999])
    moments_fn = Legendre(n_moments, true_domain)
    # moments_fn = Monomial(n_moments, true_domain)

    sampler.set_initial_n_samples([100, 50, 25])
    # sampler.set_initial_n_samples([10000])
    sampler.schedule_samples()
    sampler.ask_sampling_pool_for_samples()

    target_var = 1e-4
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
    n_estimated = mlmc.estimator.estimate_n_samples_for_target_variance_giles(target_var, variances, n_ops,
                                                                       n_levels=sampler.n_levels, theta=0.25)


    # Loop until number of estimated samples is greater than the number of scheduled samples
    while not sampler.process_adding_samples(n_estimated, sleep, add_coef):
        # New estimation according to already finished samples
        kurtosis = estimator.kurtosis_check()
        variances, n_ops = estimator.estimate_diff_vars_regression(sampler._n_scheduled_samples)
        n_estimated = mlmc.estimator.estimate_n_samples_for_target_variance_giles(target_var, variances, n_ops,
                                                                           n_levels=sampler.n_levels, theta=0.25,
                                                                                  kurtosis=kurtosis)

    means, vars = estimator.estimate_moments(moments_fn)

    estimator.kurtosis_check(value_quantity)

    assert means[0] == 1
    assert vars[0] == 0

    return estimator.moments_mean_obj, sample_storage.get_n_collected(), sample_storage.get_n_ops()


def mlmc_test_data(method):
    from mlmc.plot.diagnostic_plots import log_var_per_level, log_mean_per_level, sample_cost_per_level, kurtosis_per_level
    means = []
    l_means = []
    vars = []
    l_vars = []
    n_collected = []
    n_ops = []
    for _ in range(2):
        moments_mean_obj, n_col, n_op = method((SynthSimulationForTests(simulation_config), hdf_storage_factory(file_name="mlmc_test.hdf5"), OneProcessPool()))
        means.append(moments_mean_obj.mean)
        vars.append(moments_mean_obj.var)
        l_means.append(moments_mean_obj.l_means)
        l_vars.append(moments_mean_obj.l_vars)
        n_collected.append(n_col)
        n_ops.append(n_op)

        print("means ", np.mean(means, axis=0))
        print("vars ", np.mean(vars, axis=0))

        log_var_per_level(l_vars=np.mean(l_vars, axis=0), moments=[1, 2, 5])
        log_mean_per_level(l_means=np.mean(l_means, axis=0), moments=[1, 2, 5])
        #sample_cost_level(costs=np.mean(n_ops, axis=0))

        print("n collected ", np.mean(n_collected, axis=0))
        print("cost on level ", np.mean(n_ops, axis=0))
        print("total cost ", np.sum(np.mean(n_ops * np.array(n_collected), axis=0)))


if __name__ == "__main__":
    # means_mlmc = []
    # means_mlmc_giles = []
    # n_collected_mlmc = []
    # n_collected_mlmc_giles = []
    # vars_mlmc = []
    # vars_mlmc_giles = []
    for i in range(3):
        # print("############### Original estimator ###################")
        # mlmc_test_data(mlmc_test)
        print("######################################################")
        print("############### Improved estimator ###################")
        mlmc_test_data(mlmc_test_mcqmc)
        # mean, var, n_estimated, n_ops = mlmc_test((SynthSimulationForTests(simulation_config),
        #                                            hdf_storage_factory(file_name="mlmc_test.hdf5"),
        #                                            OneProcessPool()))
        # means_mlmc.append(mean)
        # vars_mlmc.append(var)
        # n_collected_mlmc.append(n_estimated)
        # mean, var, n_estimated, n_ops = mlmc_test_giles((SynthSimulationForTests(simulation_config),
        #                                                  hdf_storage_factory(file_name="mlmc_giles_test.hdf5"),
        #                                                  OneProcessPool()))
        # means_mlmc_giles.append(mean)
        # vars_mlmc_giles.append(var)
        # n_collected_mlmc_giles.append(n_estimated)

    #
    # print("mlmc means ", np.mean(means_mlmc, axis=0))
    # print("mlmc vars ", np.mean(vars_mlmc, axis=0))
    # print("mlmc n collected ", np.mean(n_collected_mlmc, axis=0))
    #
    # print("mlmc giles means ", np.mean(means_mlmc_giles, axis=0))
    # print("mlmc giles vars ", np.mean(vars_mlmc_giles, axis=0))
    # print("mlmc giles n collected ", np.mean(n_collected_mlmc_giles, axis=0))











