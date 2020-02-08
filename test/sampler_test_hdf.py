import os
import sys
import shutil
import numpy as np
from scipy import stats

src_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(src_path, '..', 'src/mlmc'))
from synth_simulation import SynthSimulation
from synth_simulation_workspace import SynthSimulationWorkspace
from sampler import Sampler
from sample_storage_hdf import SampleStorageHDF
from sampling_pool import OneProcessPool
from mlmc.moments import Legendre
from quantity_estimate import QuantityEstimate
import new_estimator


def sampler_hdf_test():
    np.random.seed(3)
    n_moments = 5
    failed_fraction = 0.1

    work_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '_test_tmp')
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir)

    distr = stats.norm()
    step_range = [0.1, 0.001]

    # User configure and create simulation instance
    simulation_config = dict(distr=distr, complexity=2, nan_fraction=failed_fraction, sim_method='_sample_fn')
    #simulation_config = {"config_yaml": 'synth_sim_config.yaml'}
    simulation_factory = SynthSimulation(simulation_config)

    sample_storage = SampleStorageHDF(file_path=os.path.join(work_dir, "mlmc_{}.hdf5".format(len(step_range))))
    sampling_pool = OneProcessPool()

    # Plan and compute samples
    sampler = Sampler(sample_storage=sample_storage, sampling_pool=sampling_pool, sim_factory=simulation_factory,
                      step_range=step_range)

    true_domain = distr.ppf([0.0001, 0.9999])
    moments_fn = Legendre(n_moments, true_domain)
    # moments_fn = Monomial(n_moments, true_domain)

    sampler.set_initial_n_samples()
    #sampler.set_initial_n_samples([10000])
    sampler.schedule_samples()
    sampler.ask_sampling_pool_for_samples()

    q_estimator = QuantityEstimate(sample_storage=sample_storage, moments_fn=moments_fn, sim_steps=step_range)
    #
    target_var = 1e-4
    sleep = 0
    add_coef = 0.1

    # @TODO: test
    # New estimation according to already finished samples
    variances, n_ops = q_estimator.estimate_diff_vars_regression(sampler._n_created_samples)
    n_estimated = new_estimator.estimate_n_samples_for_target_variance(target_var, variances, n_ops,
                                                                       n_levels=sampler.n_levels)

    # Loop until number of estimated samples is greater than the number of scheduled samples
    while not sampler.process_adding_samples(n_estimated, sleep, add_coef):
        # New estimation according to already finished samples
        variances, n_ops = q_estimator.estimate_diff_vars_regression(sampler._n_created_samples)
        n_estimated = new_estimator.estimate_n_samples_for_target_variance(target_var, variances, n_ops,
                                                                           n_levels=sampler.n_levels)

    print("collected samples ", sampler._n_created_samples)
    means, vars = q_estimator.estimate_moments(moments_fn)

    print("means ", means)
    print("vars ", vars)
    assert means[0] == 1
    assert np.isclose(means[1], 0, atol=1e-2)
    assert vars[0] == 0
    # sampler.schedule_samples()
    # sampler.ask_sampling_pool_for_samples()
    #
    # storage = sampler.sample_storage
    # results = storage.sample_pairs()


if __name__ == "__main__":
    sampler_hdf_test()
