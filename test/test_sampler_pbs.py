import os
import sys
import shutil
import numpy as np
from scipy import stats

src_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(src_path, '..', 'src/mlmc'))
from synth_simulation_workspace import SynthSimulationWorkspace

from moments import Legendre
from sampler import Sampler
from sample_storage import Memory
from sample_storage_hdf import SampleStorageHDF
from sampling_pool_pbs import SamplingPoolPBS
from quantity_estimate import QuantityEstimate
import new_estimator


def sampler_test_pbs():
    np.random.seed(3)
    n_moments = 5

    distr = stats.norm(loc=1, scale=2)
    step_range = [0.01, 0.001]

    # Set work dir
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    work_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '_test_tmp')
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir)
    shutil.copyfile('synth_sim_config.yaml', os.path.join(work_dir, 'synth_sim_config.yaml'))

    simulation_config = {"config_yaml": os.path.join(work_dir, 'synth_sim_config.yaml')}
    simulation_factory = SynthSimulationWorkspace(simulation_config)

    sample_storage = SampleStorageHDF(file_path=os.path.join(work_dir, "mlmc_{}.hdf5".format(len(step_range))))
    sampling_pool = SamplingPoolPBS(job_weight=20000000, job_count=0, work_dir=work_dir)

    pbs_config = dict(
        n_cores=1,
        n_nodes=1,
        select_flags=['cgroups=cpuacct'],
        mem='128mb',
        queue='charon',
        home_dir='/storage/liberec3-tul/home/martin_spetlik/',
        pbs_process_file_dir='/auto/liberec3-tul/home/martin_spetlik/MLMC_new_design/src/mlmc')

    sampling_pool.pbs_common_setting(flow_3=True, **pbs_config)

    # Plan and compute samples
    sampler = Sampler(sample_storage=sample_storage, sampling_pool=sampling_pool, sim_factory=simulation_factory,
                      step_range=step_range)

    true_domain = distr.ppf([0.0001, 0.9999])
    moments_fn = Legendre(n_moments, true_domain)

    sampler.set_initial_n_samples([10, 4])
    # sampler.set_initial_n_samples([1000])
    sampler.schedule_samples()
    sampler.ask_sampling_pool_for_samples()

    q_estimator = QuantityEstimate(sample_storage=sample_storage, moments_fn=moments_fn, sim_steps=step_range)

    target_var = 1e-3
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

    #print("collected samples ", sampler._n_created_samples)
    means, vars = q_estimator.estimate_moments(moments_fn)

    print("means ", means)
    print("vars ", vars)
    assert means[0] == 1
    assert np.isclose(means[1], 0, atol=1e-2)
    assert vars[0] == 0
    sampler.schedule_samples()
    sampler.ask_sampling_pool_for_samples()

    storage = sampler.sample_storage
    results = storage.sample_pairs()


if __name__ == "__main__":
    sampler_test_pbs()
