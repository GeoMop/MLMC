import os
import shutil
import unittest
import numpy as np
import random
from scipy import stats
from mlmc.quantity.quantity_spec import QuantitySpec
from mlmc.sample_storage import Memory
from mlmc.sample_storage_hdf import SampleStorageHDF
from mlmc.quantity.quantity import make_root_quantity
from mlmc.quantity.quantity_estimate import estimate_mean, moment, moments, covariance, cache_clear
from mlmc import Quantity, QuantityConst
from mlmc import ScalarType
from mlmc.sampler import Sampler
from mlmc.moments import Monomial
from mlmc.sampling_pool import OneProcessPool
from test.synth_sim_for_tests import SynthSimulationForTests
import mlmc.estimator


def fill_sample_storage(sample_storage, result_format):
    """
    Populate a sample storage with synthetic scheduled and finished samples for tests.

    This saves:
      - global meta-data (result_format, level_parameters)
      - scheduled samples per level
      - finished (successful) samples per level
      - number-of-operations records per level

    Parameters
    ----------
    sample_storage : mlmc.sample_storage.* instance
        Storage object to be filled (Memory or HDF).
    result_format : list[QuantitySpec]
        Result format used to determine sizes of flattened sample vectors.

    Returns
    -------
    tuple
        (result_format, sizes) where `sizes` is the last computed per-quantity sizes list.
    """
    np.random.seed(123)
    n_levels = 3

    sample_storage.save_global_data(result_format=result_format, level_parameters=np.ones(n_levels))

    successful_samples = {}
    failed_samples = {}
    n_ops = {}
    n_successful = 150
    sizes = []
    for l_id in range(n_levels):
        sizes = []
        for quantity_spec in result_format:
            sizes.append(np.prod(quantity_spec.shape) * len(quantity_spec.times) * len(quantity_spec.locations))

        # Dict[level_id, List[Tuple[sample_id:str, Tuple[fine_result: ndarray, coarse_result: ndarray]]]]
        successful_samples[l_id] = []
        for sample_id in range(n_successful):
            fine_result = np.random.randint(5 + 5 * sample_id, high=5 + 5 * (1 + sample_id),
                                            size=(np.sum(sizes),))

            if l_id == 0:
                coarse_result = (np.zeros((np.sum(sizes),)))
            else:
                coarse_result = (np.random.randint(5 + 5 * sample_id, high=5 + 5 * (1 + sample_id),
                                                   size=(np.sum(sizes),)))

            successful_samples[l_id].append((str(sample_id), (fine_result, coarse_result)))
        n_ops[l_id] = [random.random(), n_successful]
        sample_storage.save_scheduled_samples(l_id, samples=["S{:07d}".format(i) for i in range(n_successful)])

    sample_storage.save_samples(successful_samples, failed_samples)
    sample_storage.save_n_ops(list(n_ops.items()))

    return result_format, sizes


def create_sampler(clean=True, memory=True, n_moments=5):
    """
    Create and prepare a Sampler with synthetic simulation factory and initial scheduled samples.

    The function:
      - creates a temporary working directory (when clean=True),
      - constructs a SynthSimulationForTests factory,
      - chooses Memory or HDF sample storage,
      - creates a OneProcessPool sampling_pool,
      - constructs a Sampler instance,
      - prepares a Monomial moments function,
      - schedules an initial set of samples and triggers immediate sample execution.

    Parameters
    ----------
    clean : bool, optional
        If True, removes and recreates the test working directory before use (default True).
    memory : bool, optional
        If True use Memory() storage; otherwise use SampleStorageHDF (default True).
    n_moments : int, optional
        Number of monomial moments (unused in this helper besides constructing moments_fn) (default 5).

    Returns
    -------
    tuple
        (sampler, simulation_factory, moments_fn)
        - sampler: mlmc.sampler.Sampler instance with initial samples scheduled/executed
        - simulation_factory: SynthSimulationForTests instance used by the sampler
        - moments_fn: Monomial moments object constructed for the true_domain
    """
    # Set work dir
    np.random.seed(1234)
    n_levels = 3
    step_range = [0.5, 0.01]

    level_parameters = mlmc.estimator.determine_level_parameters(n_levels=n_levels, step_range=step_range)

    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    work_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '_test_tmp')
    if clean:
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)
        os.makedirs(work_dir)

    # Create simulations
    failed_fraction = 0.0
    distr = stats.norm()
    simulation_config = dict(distr=distr, complexity=2, nan_fraction=failed_fraction, sim_method='_sample_fn')
    simulation_factory = SynthSimulationForTests(simulation_config)

    # Create sample storages
    if memory:
        sample_storage = Memory()
    else:
        sample_storage = SampleStorageHDF(file_path=os.path.join(work_dir, "mlmc_test.hdf5"))
    # Create sampling pools
    sampling_pool = OneProcessPool()

    if clean:
        if sampling_pool._output_dir is not None:
            if os.path.exists(work_dir):
                shutil.rmtree(work_dir)
            os.makedirs(work_dir)
        if simulation_factory.need_workspace:
            os.chdir(os.path.dirname(os.path.realpath(__file__)))
            shutil.copyfile('synth_sim_config.yaml', os.path.join(work_dir, 'synth_sim_config.yaml'))

    sampler = Sampler(sample_storage=sample_storage, sampling_pool=sampling_pool, sim_factory=simulation_factory,
                      level_parameters=level_parameters)

    distr = stats.norm()
    true_domain = distr.ppf([0.0001, 0.9999])
    # moments_fn = Legendre(n_moments, true_domain)
    moments_fn = Monomial(n_moments, true_domain)

    sampler.set_initial_n_samples([80, 40, 20])
    sampler.schedule_samples()
    sampler.ask_sampling_pool_for_samples()

    return sampler, simulation_factory, moments_fn
