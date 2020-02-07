import os
import sys
import shutil
from scipy import stats

src_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(src_path, '..', 'src/mlmc'))
from synth_simulation import SynthSimulation
from sampler import Sampler
from sample_storage import Memory
from sampling_pool import ProcessPool, ThreadPool, OneProcessPool
from mlmc.moments import Legendre

def multiproces_sampler_test(hdf=False):

    n_levels = 2
    failed_fraction = 0#0.2

    work_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '_test_tmp')
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir)

    distr = stats.norm()
    step_range = (0.1, 0.006)

    # User configure and create simulation instance
    simulation_config = dict(distr=distr, complexity=2, nan_fraction=failed_fraction, sim_method='_sample_fn')
    #simulation_config = {"config_yaml": 'synth_sim_config.yaml'}
    simulation_factory = SynthSimulation(simulation_config)

    sample_storage = Memory()
    sampling_pool = ProcessPool(4)

    # Plan and compute samples
    sampler = Sampler(sample_storage=sample_storage, sampling_pool=sampling_pool, sim_factory=simulation_factory,
                      step_range=step_range)

    sampler.determine_level_n_samples()
    sampler.schedule_samples()
    sampler.ask_sampling_pool_for_samples()

    # After crash
    # sampler = Sampler(sample_storage=sample_storage, sampling_pool=sampling_pool, config=mlv)
    # sampler.load_from_storage()
    #
    # This should happen automatically
    # sampler.determine_level_n_samples()
    # sampler.create_simulations()


if __name__ == "__main__":
    multiproces_sampler_test()
