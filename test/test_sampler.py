import numpy as np
from scipy import stats
from mlmc.sample_storage import Memory
from mlmc.sim.synth_simulation import SynthSimulation
from mlmc.sampling_pool import OneProcessPool
from mlmc.sampler import Sampler


def test_sampler():
    # Create simulations
    failed_fraction = 0.1
    distr = stats.norm()
    simulation_config = dict(distr=distr, complexity=2, nan_fraction=failed_fraction, sim_method='_sample_fn')
    simulation = SynthSimulation(simulation_config)
    storage = Memory()
    sampling_pool = OneProcessPool()

    step_range = [[0.1], [0.01], [0.001]]

    sampler = Sampler(sample_storage=storage, sampling_pool=sampling_pool, sim_factory=simulation, level_parameters=step_range)

    assert len(sampler._level_sim_objects) == len(step_range)
    for step, level_sim in zip(step_range, sampler._level_sim_objects):
        assert step[0] == level_sim.config_dict['fine']['step']

    init_samples = list(np.ones(len(step_range)) * 10)

    sampler.set_initial_n_samples(init_samples)
    assert np.allclose(sampler._n_target_samples, init_samples)
    assert 0 == sampler.ask_sampling_pool_for_samples()
    sampler.schedule_samples()
    assert np.allclose(sampler._n_scheduled_samples, init_samples)

    n_estimated = np.array([100, 50, 20])
    sampler.process_adding_samples(n_estimated, 0, 0.1)
    assert np.allclose(sampler._n_target_samples, init_samples + (n_estimated * 0.1), atol=1)
