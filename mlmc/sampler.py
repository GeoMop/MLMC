import time
import numpy as np
from typing import List
from mlmc.sample_storage import SampleStorage
from mlmc.sampling_pool import SamplingPool
from mlmc.sim.simulation import Simulation


class Sampler:
    """
    Manages samples scheduling, results collection, and result storage.
    """

    ADDING_SAMPLES_TIMEOUT = 1e-15

    def __init__(self, sample_storage: SampleStorage, sampling_pool: SamplingPool, sim_factory: Simulation,
                 level_parameters: List[List[float]], seed=1234):
        """
        :param sample_storage: store scheduled samples, results and result structure
        :param sampling_pool: calculate samples
        :param sim_factory: generate samples
        :param level_parameters: List of e.g. simulation steps, ...
        :param seed: global random seed
        """
        np.random.seed(seed)
        self.sample_storage = sample_storage
        self._sampling_pool = sampling_pool

        self._n_target_samples = np.zeros(len(level_parameters))
        # Number of target samples
        self._level_sim_objects = []
        self._create_level_sim_objects(level_parameters, sim_factory)

        sample_storage.save_global_data(level_parameters=level_parameters,
                                        result_format=sim_factory.result_format())

        self._n_scheduled_samples = [len(level_scheduled) for level_id, level_scheduled in
                                     sample_storage.load_scheduled_samples().items()]
        # Number of created samples

        if not self._n_scheduled_samples:
            self._n_scheduled_samples = np.zeros(len(level_parameters))

        # Are there any unfinished samples which have already finished?
        self._check_failed_samples()

        # @TODO: get unfinished samples from sampler and call have permanent samples -> add results to pool's queues,
        # before scheduled samples call, call get_finished - we need to know how many samples is finished

    @property
    def n_levels(self):
        return len(self._level_sim_objects)

    @property
    def n_finished_samples(self):
        """
        Retrieve number of all finished samples
        :return:
        """
        return self.sample_storage.n_finished()

    def _create_level_sim_objects(self, level_parameters, sim_factory):
        """
        Create LevelSimulation object for each level, use simulation factory
        :param: level_parameters: List, simulation steps, ...
        :param: sim_factory: Simulation instance
        :return: None
        """
        n_levels = len(level_parameters)
        for level_id in range(n_levels):
            if level_id == 0:
                level_sim = sim_factory.level_instance(level_parameters[level_id], [0])

            else:
                level_sim = sim_factory.level_instance(level_parameters[level_id], level_parameters[level_id - 1])

            level_sim._calculate = sim_factory.calculate
            level_sim._result_format = sim_factory.result_format
            level_sim._level_id = level_id
            self._level_sim_objects.append(level_sim)

    def sample_range(self, n0, nL):
        """
        Geometric sequence of L elements decreasing from n0 to nL.
        Useful to set number of samples explicitly.
        :param n0: int
        :param nL: int
        :return: np.array of length L = n_levels.
        """
        return np.round(np.exp2(np.linspace(np.log2(n0), np.log2(nL), self.n_levels))).astype(int)

    def set_initial_n_samples(self, n_samples=None):
        """
        Set target number of samples for each level
        :param n_samples: array of number of samples
        :return: None
        """
        if n_samples is None:
            n_samples = [100, 10]
        # Num of samples to ndarray
        n_samples = np.atleast_1d(n_samples)

        # Just maximal number of samples is set
        if len(n_samples) == 1:
            n_samples = np.array([n_samples[0], 10])

        # Create number of samples for all levels
        if len(n_samples) == 2:
            n0, nL = n_samples
            n_samples = self.sample_range(n0, nL)

        self._n_target_samples = n_samples

    def _get_sample_tag(self, level_id):
        """
        Create sample tag
        :param level_id: identifier of current level
        :return: str
        """
        return "L{:02d}_S{:07d}".format(level_id, int(self._n_scheduled_samples[level_id]))

    def schedule_samples(self, timeout=None):
        """
        Create simulation samples, loop through "levels" and its samples (given the number of target samples):
            1) generate sample tag (same for fine and coarse simulation)
            2) get LevelSimulation instance by simulation factory
            3) schedule sample via sampling pool
            4) store scheduled samples in sample storage, separately for each level
        :param timeout: int, get_finished - while break timeout in seconds
        :return: None
        """
        self.ask_sampling_pool_for_samples(timeout=timeout)
        plan_samples = self._n_target_samples - self._n_scheduled_samples

        for level_id, n_samples in enumerate(plan_samples):
            samples = []
            for _ in range(int(n_samples)):
                # Unique sample id
                sample_id = self._get_sample_tag(level_id)
                level_sim = self._level_sim_objects[level_id]

                # Schedule current sample
                self._sampling_pool.schedule_sample(sample_id, level_sim)
                # Increment number of created samples at current level
                self._n_scheduled_samples[level_id] += 1

                samples.append(sample_id)

            # Store scheduled samples
            self.sample_storage.save_scheduled_samples(level_id, samples)

    def _check_failed_samples(self):
        """
        Get unfinished samples and check if failed samples have saved results then collect them
        :return:
        """
        unfinished_sample_ids = self.sample_storage.unfinished_ids()
        self._sampling_pool.have_permanent_samples(unfinished_sample_ids)

    def ask_sampling_pool_for_samples(self, sleep=0, timeout=None):
        """
        Waiting for running simulations
        :param sleep: time for doing nothing
        :param timeout: maximum time for waiting on running simulations
        :return: int, number of running simulations
        """
        if timeout is None:
            timeout = 0
        elif timeout <= 0:
            return 1

        n_running = 1
        t0 = time.perf_counter()
        while n_running > 0:
            successful_samples, failed_samples, n_running, n_ops = self._sampling_pool.get_finished()
            # Store finished samples
            self._store_samples(successful_samples, failed_samples, n_ops)
            time.sleep(sleep)
            if 0 < timeout < (time.perf_counter() - t0):
                break

        return n_running

    def _store_samples(self, successful_samples, failed_samples, n_ops):
        """
        Store finished samples
        :param successful_samples: Dict[level_id, List[Tuple[sample_id:str, Tuple[ndarray, ndarray]]]]
        :param failed_samples: Dict[level_id, List[Tuple[sample_id: str, error message: str]]]
        :param n_ops: Dict[level_id: int, List[total time: float, number of success samples: int]]
        :return: None
        """
        self.sample_storage.save_samples(successful_samples, failed_samples)
        self.sample_storage.save_n_ops(n_ops)

    def process_adding_samples(self, n_estimated, sleep=0, add_coeff=0.1, timeout=ADDING_SAMPLES_TIMEOUT):
        """
        Process adding samples
        Note: n_estimated are wrong if n_ops is similar through all levels
        :param n_estimated: Number of estimated samples on each level, list
        :param sleep: Sample waiting time
        :param add_coeff: default value 0.1, The number of scheduled samples would be 'add_coef' fraction of difference
         between current number of target samples and new estimated number of target samples
        :param timeout: ask sampling pool for finished samples timeout
        :return: bool, if True adding samples is complete
        """
        self.ask_sampling_pool_for_samples(timeout=timeout)

        # Get default scheduled samples
        n_scheduled = self.l_scheduled_samples()

        # New scheduled sample will be 10 percent of difference
        # between current number of target samples and new estimated one
        # If 10 percent of estimated samples is greater than difference between estimated and scheduled samples,
        # set scheduled samples to estimated samples
        new_scheduled = np.where((n_estimated * add_coeff) > (n_estimated - n_scheduled),
                                 n_estimated,
                                 n_scheduled + (n_estimated - n_scheduled) * add_coeff)

        n_scheduled = np.ceil(np.where(n_estimated < n_scheduled,
                                       n_scheduled,
                                       new_scheduled))

        # Levels where estimated are greater than scheduled
        greater_items = np.where(np.greater(n_estimated, n_scheduled))[0]

        # Scheduled samples and wait until at least half of the samples are done
        self.set_scheduled_and_wait(n_scheduled, greater_items, sleep, timeout=timeout)

        return np.all(n_estimated[greater_items] == n_scheduled[greater_items])

    def set_scheduled_and_wait(self, n_scheduled, greater_items, sleep, fin_sample_coef=0.5, timeout=1e-7):
        """
        Scheduled samples on each level and wait until at least half of the samples is done
        :param n_scheduled: ndarray, number of scheduled samples on each level
        :param greater_items: Items where n_estimated is greater than n_scheduled
        :param sleep: Time waiting for samples
        :param fin_sample_coef: The proportion of samples to finished for further estimate
        :return: None
        """
        # Set scheduled samples and run simulations
        self.set_level_target_n_samples(n_scheduled)
        self.schedule_samples(timeout=timeout)

        # Finished level samples
        n_finished = self.n_finished_samples

        # Wait until at least half of the scheduled samples are done on each level
        while np.any(n_finished[greater_items] < fin_sample_coef * n_scheduled[greater_items]):
            # Wait a while
            time.sleep(sleep)
            self.ask_sampling_pool_for_samples(timeout=timeout)
            n_finished = self.n_finished_samples

    def set_level_target_n_samples(self, n_samples):
        """
        Set level number of target samples
        :param n_samples: list, each level target samples
        :return: None
        """
        for level, n in enumerate(n_samples):
            self._n_target_samples[level] = max(self._n_target_samples[level], n)

    def l_scheduled_samples(self):
        """
        Get all levels number of scheduled samples
        :return: list
        """
        return self._n_scheduled_samples

    def renew_failed_samples(self):
        """
        Resurrect failed samples
        :return: None
        """
        failed_samples = self.sample_storage.failed_samples()

        for level_id, sample_ids in failed_samples.items():
            samples = []
            level_id = int(level_id)
            for sample_id in sample_ids:
                level_sim = self._level_sim_objects[level_id]
                # Schedule current sample
                self._sampling_pool.schedule_sample(sample_id, level_sim)
                samples.append(sample_id)

        self.sample_storage.clear_failed()
