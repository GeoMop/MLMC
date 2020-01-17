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


def one_process_sampler_test():
    """
    Test sampler, simulations are running in same process, artificial simulation is used
    :return:
    """
    n_levels = 2
    failed_fraction = 0.2

    distr = stats.norm()
    step_range = (0.1, 0.006)

    # Create simulation instance
    simulation_config = dict(distr=distr, complexity=2, nan_fraction=failed_fraction, sim_method='_sample_fn')
    simulation_factory = SynthSimulation(simulation_config)

    sample_storage = Memory()
    sampling_pool = OneProcessPool()

    # Plan and compute samples
    sampler = Sampler(sample_storage=sample_storage, sampling_pool=sampling_pool, sim_factory=simulation_factory,
                      n_levels=n_levels, step_range=step_range)

    sampler.determine_level_n_samples()
    sampler.create_simulations()
    sampler.ask_simulations_for_samples()

    storage = sampler.sample_storage()
    results = storage.sample_pairs()
    print("results ", results)


if __name__ == "__main__":
    one_process_sampler_test()
