"""
Tests for mlmc.mc_level
"""
import os
import shutil
import sys
import scipy.stats as stats
import types
import numpy as np
src_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, src_path + '/../src/')
from test.fixtures.synth_simulation import SimulationTest
import mlmc.mlmc
import mlmc.sample
import mlmc.estimate
import pytest


@pytest.mark.parametrize("n_levels, n_samples, failed_fraction", [
    (1, [100], 0.2),
    (2, [200, 100], 0.5),
    (5, [300, 250, 200, 150, 100], 0.3)
])
def test_level(n_levels, n_samples, failed_fraction):
    """
    Test mc_level.Level
    :param n_levels: number of levels
    :param n_samples: list, number of samples on each level
    :param failed_fraction: ratio of failed samples
    :return: None
    """
    mc = create_mc(n_levels, n_samples, failed_fraction)

    # Test methods corresponds with methods names in Level class
    enlarge_samples(mc)
    add_samples(mc)
    make_sample_pair(mc)

    mc = create_mc(n_levels, n_samples, failed_fraction)
    reload_samples(mc)
    load_samples(mc, n_samples, failed_fraction, False)
    mc.clean_levels()
    load_samples(mc, n_samples, failed_fraction, True)
    collect_samples(mc)
    fill_samples(mc)
    estimate_covariance(mc)
    subsample(mc)


def create_mc(n_levels, n_samples, failed_fraction=0.2):
    """
    Create MLMC instance
    :param n_levels: number of levels
    :param n_samples: list, samples on each level
    :param failed_fraction: ratio of simulation failed samples (NaN)
    :return:
    """

    assert n_levels == len(n_samples)

    work_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '_test_tmp')
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir)

    distr = stats.norm()
    step_range = (0.1, 0.006)

    simulation_config = dict(
        distr=distr, complexity=2, nan_fraction=failed_fraction, sim_method='_sample_fn')
    simulation_factory = SimulationTest.factory(step_range, config=simulation_config)

    mlmc_options = {'output_dir': work_dir,
                    'keep_collected': True,
                    'regen_failed': False}

    mc = mlmc.mlmc.MLMC(n_levels, simulation_factory, step_range, mlmc_options)

    mc.create_new_execution()
    mc.set_initial_n_samples(n_samples)
    mc.refill_samples()
    mc.wait_for_simulations()

    return mc


def enlarge_samples(mc):
    """
    Enlarge existing samples
    :param mc: MLMC instance
    :return: None
    """
    # Number of new samples
    enlargement = 25
    for level in mc.levels:
        # Current number of collected samples
        size = len(level._sample_values)
        level.enlarge_samples(size=len(level._sample_values) + enlargement)
        assert len(level._sample_values) == (size + enlargement)


def make_sample_pair(mc):
    """
    Create sample pair
    :param mc:
    :return: None
    """
    for level in mc.levels:
        # Use default sample id
        sample_id = level._n_total_samples
        sample_pair = level._make_sample_pair()
        assert sample_id == np.squeeze(sample_pair)[0]

        # Use explicit sample id
        sample_id = 100
        sample_pair = level._make_sample_pair(sample_id)
        assert sample_id == np.squeeze(sample_pair)[0]


def add_samples(mc):
    """
    Test adding samples
    :param mc: MLMC instance
    :return: None
    """
    for level in mc.levels:
        # Default length of sample values
        len_sample_val = len(level.sample_values)

        # Add correct sample
        level._add_sample('1', (-10.5, 10))
        assert len(level.nan_samples) == 0
        assert len_sample_val + 1 == len(level.sample_values) == level._n_collected_samples

        # Add NaN samples
        level._add_sample('1', (np.nan, 10))
        level._add_sample('1', (-10.5, np.nan))
        assert len(level.nan_samples) == 2
        assert len_sample_val + 1 == len(level.sample_values) == level._n_collected_samples


def reload_samples(mc):
    """
    Test reload samples method
    :param mc: MLMC instance
    :return: None
    """
    for level in mc.levels:
        # Both returned values are generators
        scheduled, collected = level._reload_samples()
        assert isinstance(scheduled, types.GeneratorType)
        assert isinstance(collected, types.GeneratorType)


def load_samples(mc, n_samples, failed_fraction, regen_failed):
    """
    Test load samples method
    :param mc: MLMC instance
    :param n_samples: number of samples, list
    :param failed_fraction: ratio of failed samples
    :param regen_failed: bool, if True regenerate failed samples
    :return: None
    """
    for level in mc.levels:
        # Reset level
        level.reset()
        failed_samples = len(level.failed_samples)
        level.load_samples(regen_failed)

        # Number of samples should be equal to sum of all types of level samples
        assert n_samples[level._level_idx] == len(level.collected_samples) + len(level.scheduled_samples) +\
                                              len(level.failed_samples)

        # There are failed samples
        if regen_failed is False:
            assert len(level.failed_samples) == n_samples[level._level_idx] * failed_fraction
        else:
            # Check new failed samples
            assert len(level.failed_samples) <= failed_samples * failed_fraction

        # We also store times for each sample
        assert len(level.coarse_times) == len(level.fine_times) == len(level.collected_samples)


def collect_samples(mc):
    """
    Test collecting samples
    :param mc: MLMC instance
    :return: None
    """
    for level in mc.levels:
        all_samples = len(level.scheduled_samples) + len(level.collected_samples) + len(level.failed_samples)

        # Number of scheduled samples after collecting -> should be zero because we don't use pbs for tests
        new_scheduled = level.collect_samples()

        assert len(level._not_queued_sample_ids()) == 0
        assert len(level.scheduled_samples) == new_scheduled == 0
        assert len(level.collected_samples) + len(level.failed_samples) == all_samples


def fill_samples(mc):
    """
    Fill samples - test adding samples
    :param mc: MLMC instance
    :return: None
    """
    n_new_samples = 20
    for level in mc.levels:
        level.fill_samples()
        # Run samples
        level.collect_samples()
        # Scheduled samples should be equal to 0
        if level.target_n_samples < level._n_total_samples:
            assert len(level.scheduled_samples) == 0

        # Increase number of target samples
        level.target_n_samples = level._n_total_samples + n_new_samples
        level.fill_samples()
        # Check if new samples are also scheduled
        assert len(level.scheduled_samples) == n_new_samples


def subsample(mc):
    """
    Test subsample
    :param mc: MLMC instance
    :return: None
    """
    size = 20
    for level in mc.levels:
        # Sample indices are default None
        assert level.sample_indices is None

        # Get subsample indices
        level.subsample(size)
        sample_indices = level.sample_indices
        level.subsample(size)

        assert len(sample_indices) == size
        # Subsample indices should be random
        assert not np.array_equal(level.sample_indices, sample_indices)


def estimate_covariance(mc):
    """
    Basic tests for covariance matrix
    :param mc: MLMC instance
    :return: None
    """
    estimator = mlmc.estimate.Estimate(mc)

    n_moments = 15
    moments_fn = mlmc.moments.Legendre(n_moments, estimator.estimate_domain(mc), safe_eval=True, log=False)

    for level in mc.levels:
        cov = level.estimate_covariance(moments_fn)
        assert np.allclose(cov, cov.T, atol=1e-6)
        # We don't have enough samples for more levels
        if mc.n_levels == 1:
            assert np.all(np.linalg.eigvals(cov) > 0)


if __name__ == "__main__":
    test_level(n_levels=1, n_samples=[100], failed_fraction=0.1)
