import os
import shutil
import numpy as np
from scipy import stats

from test.synth_sim_for_tests import SynthSimulationForTests
from mlmc.sampler import Sampler
from mlmc.sample_storage_hdf import SampleStorageHDF
from mlmc.sampling_pool import ProcessPool, ThreadPool, OneProcessPool
from mlmc.moments import Legendre
from mlmc.quantity_estimate import QuantityEstimate
from time import time


def test_sampling_pools():
    n_moments = 5

    distr = stats.norm(loc=1, scale=2)
    step_range = [[0.01], [0.001], [0.0001]]
    failed_fraction = 0

    # Set work dir
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    work_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '_test_tmp')
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir)

    simulation_config = dict(distr=distr, complexity=2, nan_fraction=failed_fraction, sim_method='_sample_fn')
    simulation_factory = SynthSimulationForTests(simulation_config)

    single_process_pool = OneProcessPool(work_dir=work_dir)
    multiprocess_pool = ProcessPool(4, work_dir=work_dir)
    thread_pool = ThreadPool(4, work_dir=work_dir)

    pools = [single_process_pool, multiprocess_pool] #, thread_pool]

    all_data = []
    times = []
    for sampling_pool in pools:
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)
        os.makedirs(work_dir)

        shutil.copyfile('synth_sim_config.yaml', os.path.join(work_dir, 'synth_sim_config.yaml'))
        sample_storage = SampleStorageHDF(file_path=os.path.join(work_dir, "mlmc_{}.hdf5".format(len(step_range))))
        # Plan and compute samples
        sampler = Sampler(sample_storage=sample_storage, sampling_pool=sampling_pool, sim_factory=simulation_factory,
                          level_parameters=step_range)

        true_domain = distr.ppf([0.0001, 0.9999])
        moments_fn = Legendre(n_moments, true_domain)

        start = time()

        sampler.set_initial_n_samples([10, 10, 10])
        sampler.schedule_samples()
        sampler.ask_sampling_pool_for_samples()

        times.append(time() - start)
        q_estimator = QuantityEstimate(sample_storage=sample_storage, moments_fn=moments_fn, sim_steps=step_range)

        means, vars = q_estimator.estimate_moments(moments_fn)
        all_data.append(np.array([means, vars]))

        assert means[0] == 1
        assert vars[0] == 0

    assert times[1] < times[0]
    assert np.allclose(all_data[0], all_data[1])
    #assert np.allclose(all_data[1], all_data[2])
