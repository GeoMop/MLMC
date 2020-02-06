import os
import sys
import shutil
import numpy as np
from scipy import stats

src_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(src_path, '..', 'src/mlmc'))
from synth_simulation_workspace import SynthSimulationWorkspace

from mlmc.moments import Legendre
from sampler import Sampler
from sample_storage import Memory
from sampling_pool_pbs import SamplingPoolPBS


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

    sample_storage = Memory()
    sampling_pool = SamplingPoolPBS(job_weight=200000, job_count=0, work_dir=work_dir)

    pbs_config = dict(
        job_weight=250000,  # max number of elements per job
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

    sampler.set_initial_n_samples()
    # sampler.set_initial_n_samples([1000])
    sampler.schedule_samples()
    sampler.ask_sampling_pool_for_samples()

    sampler.target_var_adding_samples(1e-4, moments_fn, sleep=20)
    print("collected samples ", sampler._n_created_samples)

    means, vars = sampler.estimate_moments(moments_fn)

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
