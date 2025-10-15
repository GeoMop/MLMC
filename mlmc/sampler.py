import time
import numpy as np
from typing import List
from mlmc.sample_storage import SampleStorage
from mlmc.sampling_pool import SamplingPool
from mlmc.sim.simulation import Simulation


class Sampler:
    """
    Manages sample scheduling, result collection, and persistent storage.

    Coordinates the sampling pool, simulation factory, and sample storage:
    - schedules new samples according to target counts,
    - collects finished samples and writes them to storage,
    - handles failed samples and runtime (n_ops) bookkeeping.
    """

    ADDING_SAMPLES_TIMEOUT = 1e-15

    def __init__(self, sample_storage: SampleStorage, sampling_pool: SamplingPool, sim_factory: Simulation,
                 level_parameters: List[List[float]], seed=1234):
        """
        Initialize sampler and prepare per-level simulation objects.

        :param sample_storage: store scheduled samples, results and result structure
        :param sampling_pool: sampling pool responsible for executing simulations
        :param sim_factory: factory that creates level Simulation instances and provides result_format()
        :param level_parameters: List of per-level parameters (e.g. simulation steps)
        :param seed: global RNG seed used to seed NumPy's RNG
        """
        np.random.seed(seed)
        self.sample_storage = sample_storage
        self._sampling_pool = sampling_pool

        # Target number of samples per level (may be updated later)
        self._n_target_samples = np.zeros(len(level_parameters))

        # Create LevelSimulation objects for each level using the provided factory
        self._level_sim_objects = []
        self._create_level_sim_objects(level_parameters, sim_factory)

        # Persist global data (level parameters and result format) into storage
        sample_storage.save_global_data(level_parameters=level_parameters,
                                        result_format=sim_factory.result_format())

        # Load already scheduled samples (if any) from storage
        self._n_scheduled_samples = [len(level_scheduled) for level_id, level_scheduled in
                                     sample_storage.load_scheduled_samples().items()]

        # If there are no scheduled samples yet, initialize to zeros
        if not self._n_scheduled_samples:
            self._n_scheduled_samples = np.zeros(len(level_parameters))

        # Check for unfinished samples and inform the sampling pool
        self._check_failed_samples()

        # @TODO: If sampler is restarted, collect any samples finished while offline:
        #  - add permanent samples into pool queues,
        #  - before scheduling new samples, call get_finished to know how many are already done.

    @property
    def n_levels(self):
        """Return number of MLMC levels managed by this sampler."""
        return len(self._level_sim_objects)

    @property
    def n_finished_samples(self):
        """
        Retrieve numbers of finished samples for all levels.

        :return: array-like containing finished counts per level
        """
        return self.sample_storage.n_finished()

    def _create_level_sim_objects(self, level_parameters, sim_factory):
        """
        Create LevelSimulation object for each level via the simulation factory.

        :param level_parameters: List of per-level parameters
        :param sim_factory: Simulation factory providing level_instance and calculate methods
        :return: None
        """
        n_levels = len(level_parameters)
        for level_id in range(n_levels):
            if level_id == 0:
                level_sim = sim_factory.level_instance(level_parameters[level_id], [0])
            else:
                level_sim = sim_factory.level_instance(level_parameters[level_id], level_parameters[level_id - 1])

            # Attach factory methods and metadata to the LevelSimulation
            level_sim._calculate = sim_factory.calculate
            level_sim._result_format = sim_factory.result_format
            level_sim._level_id = level_id
            self._level_sim_objects.append(level_sim)

    def sample_range(self, n0, nL):
        """
        Generate a geometric sequence of length L decreasing from n0 to nL.

        Useful to generate a set of target sample counts across levels.

        :param n0: int, number of samples at finest level
        :param nL: int, number of samples at coarsest level
        :return: np.ndarray of length self.n_levels with integer sample counts
        """
        return np.round(np.exp2(np.linspace(np.log2(n0), np.log2(nL), self.n_levels))).astype(int)

    def set_initial_n_samples(self, n_samples=None):
        """
        Set initial target number of samples for each level.

        Accepts:
          - None (defaults to [100, 10]),
          - single integer (interpreted as n0, with default nL=10),
          - two-element list [n0, nL] (geometric interpolation across levels).

        :param n_samples: scalar, length-2 list, or array specifying target counts
        :return: None
        """
        if n_samples is None:
            n_samples = [100, 10]
        n_samples = np.atleast_1d(n_samples)

        # Single value -> treat as n0 with default nL
        if len(n_samples) == 1:
            n_samples = np.array([n_samples[0], 10])

        # Two values -> create geometric progression across levels
        if len(n_samples) == 2:
            n0, nL = n_samples
            n_samples = self.sample_range(n0, nL)

        self._n_target_samples = n_samples

    def _get_sample_tag(self, level_id):
        """
        Create a unique sample tag for a given level.

        :param level_id: identifier of current level
        :return: str unique sample tag (e.g. 'L00_S0000123')
        """
        return "L{:02d}_S{:07d}".format(level_id, int(self._n_scheduled_samples[level_id]))

    def schedule_samples(self, timeout=None, level_id=None, n_samples=None):
        """
        Schedule new simulation samples in the sampling pool and record them in storage.

        For each scheduled sample:
         1) generate a unique sample id shared by fine and coarse tasks,
         2) obtain the LevelSimulation instance for the level,
         3) schedule the sample with SamplingPool,
         4) store scheduled sample ids in SampleStorage.

        :param timeout: float or None, passed to ask_sampling_pool_for_samples() before scheduling
        :param level_id: int or None, if provided schedule only for this level (default: highest level)
        :param n_samples: int or None, if provided schedule exactly this many samples for the specified level
        :return: None
        """
        # First, collect any finished samples
        self.ask_sampling_pool_for_samples(timeout=timeout)
        plan_samples = self._n_target_samples - self._n_scheduled_samples

        # Default to the coarsest level if not specified
        if level_id is None:
            level_id = len(plan_samples) - 1

        # If a specific number of samples for one level is requested
        if n_samples is not None:
            samples = []
            for _ in range(int(n_samples)):
                sample_id = self._get_sample_tag(level_id)
                level_sim = self._level_sim_objects[level_id]

                self._sampling_pool.schedule_sample(sample_id, level_sim)
                self._n_scheduled_samples[level_id] += 1
                samples.append(sample_id)

            self.sample_storage.save_scheduled_samples(level_id, samples)
        else:
            # Iterate levels from coarsest to finest and schedule required samples
            for n_samples in np.flip(plan_samples):
                samples = []
                for _ in range(int(n_samples)):
                    sample_id = self._get_sample_tag(level_id)
                    level_sim = self._level_sim_objects[level_id]

                    self._sampling_pool.schedule_sample(sample_id, level_sim)
                    self._n_scheduled_samples[level_id] += 1
                    samples.append(sample_id)

                self.sample_storage.save_scheduled_samples(level_id, samples)
                level_id -= 1

    def _check_failed_samples(self):
        """
        Query storage for unfinished sample IDs and inform the sampling pool.

        This allows the sampling pool to reattach or handle 'permanent' samples
        that may have been started previously.
        :return: None
        """
        unfinished_sample_ids = self.sample_storage.unfinished_ids()
        self._sampling_pool.have_permanent_samples(unfinished_sample_ids)

    def ask_sampling_pool_for_samples(self, sleep=0, timeout=None):
        """
        Poll the sampling pool for finished simulations and store their results.

        :param sleep: float, time to sleep between polls (seconds)
        :param timeout: float or None, maximum time to wait; if <= 0 returns immediately
        :return: int, number of running simulations remaining after the call
        """
        if timeout is None:
            timeout = 0
        elif timeout <= 0:
            return 1

        n_running = 1
        t0 = time.perf_counter()
        while n_running > 0:
            successful_samples, failed_samples, n_running, n_ops = self._sampling_pool.get_finished()
            # Persist finished samples and operation counts
            self._store_samples(successful_samples, failed_samples, n_ops)
            time.sleep(sleep)
            if 0 < timeout < (time.perf_counter() - t0):
                break

        return n_running

    def _store_samples(self, successful_samples, failed_samples, n_ops):
        """
        Persist finished samples and operation time estimates to storage.

        :param successful_samples: Dict[level_id, List[Tuple[sample_id:str, (fine, coarse)]]]
        :param failed_samples: Dict[level_id, List[Tuple[sample_id:str, error_message:str]]]
        :param n_ops: Dict[level_id, Tuple[total_time:float, n_success_samples:int]]
        :return: None
        """
        self.sample_storage.save_samples(successful_samples, failed_samples)
        self.sample_storage.save_n_ops(n_ops)

    def process_adding_samples(self, n_estimated, sleep=0, add_coeff=0.1, timeout=ADDING_SAMPLES_TIMEOUT):
        """
        Add newly estimated samples in batches, scheduling a fraction of the difference
        between current scheduled and newly estimated targets.

        Note: n_estimated may be unreliable if per-level n_ops are similar across levels.

        :param n_estimated: array-like, estimated target samples per level
        :param sleep: float, time to sleep while waiting for results
        :param add_coeff: float in (0,1], fraction of the difference to schedule each iteration (default 0.1)
        :param timeout: float, timeout passed to ask_sampling_pool_for_samples()
        :return: bool, True if scheduled counts reached the estimates for all levels
        """
        # Ensure storage reflects any finished work
        self.ask_sampling_pool_for_samples(timeout=timeout)

        # Currently scheduled samples per level
        n_scheduled = self.l_scheduled_samples()

        # Compute new scheduled values (add_coeff fraction of the remaining difference)
        new_scheduled = np.where((n_estimated * add_coeff) > (n_estimated - n_scheduled),
                                 n_estimated,
                                 n_scheduled + (n_estimated - n_scheduled) * add_coeff)

        n_scheduled = np.ceil(np.where(n_estimated < n_scheduled,
                                       n_scheduled,
                                       new_scheduled))

        # Levels where estimated > scheduled
        greater_items = np.where(np.greater(n_estimated, n_scheduled))[0]

        # Schedule and wait until at least a fraction of newly scheduled samples finish
        self.set_scheduled_and_wait(n_scheduled, greater_items, sleep, timeout=timeout)

        return np.all(n_estimated[greater_items] == n_scheduled[greater_items])

    def set_scheduled_and_wait(self, n_scheduled, greater_items, sleep, fin_sample_coef=0.5, timeout=1e-7):
        """
        Set scheduled sample targets and wait until a proportion of those samples finish.

        :param n_scheduled: ndarray, target number of scheduled samples per level
        :param greater_items: iterable of indices where targets were increased
        :param sleep: float, time to sleep between polls
        :param fin_sample_coef: float in (0,1], fraction of scheduled samples that should finish before continuing
        :param timeout: float, timeout passed to ask_sampling_pool_for_samples()
        :return: None
        """
        # Update internal targets and schedule required samples
        self.set_level_target_n_samples(n_scheduled)
        self.schedule_samples(timeout=timeout)

        # Current finished counts
        n_finished = self.n_finished_samples

        # Wait until at least fin_sample_coef fraction of scheduled samples are finished for affected levels
        while np.any(n_finished[greater_items] < fin_sample_coef * n_scheduled[greater_items]):
            time.sleep(sleep)
            self.ask_sampling_pool_for_samples(timeout=timeout)
            n_finished = self.n_finished_samples

    def set_level_target_n_samples(self, n_samples):
        """
        Update the per-level target sample counts to at least the provided values.

        :param n_samples: iterable of new target samples per level
        :return: None
        """
        for level, n in enumerate(n_samples):
            self._n_target_samples[level] = max(self._n_target_samples[level], n)

    def l_scheduled_samples(self):
        """
        Return the currently scheduled sample counts per level.

        :return: list or array-like of scheduled sample counts
        """
        return self._n_scheduled_samples

    def renew_failed_samples(self):
        """
        Reschedule previously failed samples.

        Retrieves failed sample IDs from storage, re-schedules them in the sampling pool,
        and clears failed records from storage.
        :return: None
        """
        failed_samples = self.sample_storage.failed_samples()

        for level_id, sample_ids in failed_samples.items():
            samples = []
            level_id = int(level_id)
            for sample_id in sample_ids:
                level_sim = self._level_sim_objects[level_id]
                self._sampling_pool.schedule_sample(sample_id, level_sim)
                samples.append(sample_id)

        # Clear failed sample records after rescheduling
        self.sample_storage.clear_failed()
