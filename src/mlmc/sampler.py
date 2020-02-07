import time
import numpy as np
from typing import List
from sample_storage import SampleStorage
from sampling_pool import SamplingPool
from new_simulation import Simulation


class Sampler:

    def __init__(self, sample_storage: SampleStorage, sampling_pool: SamplingPool, sim_factory: Simulation,
                 step_range: List[float]):
        """
        :param sample_storage: store scheduled samples, results and result structure
        :param sampling_pool: calculate samples
        :param sim_factory: generate samples
        :param step_range: simulation step range
        """
        self.sample_storage = sample_storage
        self._sampling_pool = sampling_pool

        self._step_range = step_range

        # Number of created samples
        self._n_created_samples = np.zeros(len(step_range))
        # Number of target samples
        self._n_target_samples = np.zeros(len(step_range))
        self._n_finished_samples = np.zeros(len(step_range))
        self._level_sim_objects = []
        self._create_level_sim_objects(len(step_range), sim_factory)

        sample_storage.save_global_data(step_range=step_range,
                                        result_format=sim_factory.result_format())

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

    def _create_level_sim_objects(self, n_levels, sim_factory):
        """
        Create LevelSimulation object for each level, use simulation factory
        :param: n_levels: int, number of levels
        :param: sim_factory: Simulation instance
        :return: None
        """
        for level_id in range(n_levels):
            if level_id == 0:
                level_sim = sim_factory.level_instance([self._step_range[level_id]], [0])

            else:
                level_sim = sim_factory.level_instance([self._step_range[level_id]], [self._step_range[level_id - 1]])

            level_sim.calculate = sim_factory.calculate
            level_sim.level_id = level_id
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
            n_samples = [100, 3]
        # Num of samples to ndarray
        n_samples = np.atleast_1d(n_samples)

        # Just maximal number of samples is set
        if len(n_samples) == 1:
            n_samples = np.array([n_samples[0], 3])

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
        return "L{:02d}_S{:07d}".format(level_id, int(self._n_created_samples[level_id]))

    def schedule_samples(self):
        """
        Create simulation samples, loop through "levels" and its samples (given the number of target samples):
            1) generate sample tag (same for fine and coarse simulation)
            2) get LevelSimulation instance by simulation factory
            3) schedule sample via sampling pool
            4) store scheduled samples in sample storage, separately for each level
        :return: None
        """
        self.ask_sampling_pool_for_samples()
        plan_samples = self._n_target_samples - self._n_created_samples

        for level_id, n_samples in enumerate(plan_samples):
            samples = []
            for _ in range(int(n_samples)):
                # Unique sample id
                sample_id = self._get_sample_tag(level_id)
                level_sim = self._level_sim_objects[level_id]

                # Schedule current sample
                self._sampling_pool.schedule_sample(sample_id, level_sim)
                # Increment number of created samples at current level
                self._n_created_samples[level_id] += 1

                samples.append(sample_id)

            # Store scheduled samples
            self.sample_storage.save_scheduled_samples(level_id, samples)

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
        t0 = time.clock()
        while n_running > 0:
            successful_samples, failed_samples, n_running, n_ops = self._sampling_pool.get_finished()

            for level_id, s_samples in successful_samples.items():
                self._n_finished_samples[level_id] += len(s_samples)
            for level_id, f_samples in failed_samples.items():
                self._n_finished_samples[level_id] += len(f_samples)

            # Store finished samples
            if len(successful_samples) > 0:
                self._store_samples(successful_samples, failed_samples, n_ops)

            time.sleep(sleep)
            if 0 < timeout < (time.clock() - t0):
                break

        return n_running

    def _store_samples(self, successful_samples, failed_samples, n_ops):
        """
        Store finished samples
        :param successful_samples: List[Tuple[sample_id:str, Tuple[ndarray, ndarray]]]
        :param failed_samples: List[Tuple[sample_id: str, error message: str]]
        :param n_ops: Dict[level_id: int, List[total time: float, number of success samples: int]]
        :return: None
        """
        self.sample_storage.save_samples(successful_samples, failed_samples)
        self.sample_storage.save_n_ops(n_ops)

    def process_adding_samples(self, n_estimated, sleep, add_coef=0.1):
        """
        Process adding samples
        :param n_estimated: Number of estimated samples on each level, list
        :param sleep: Sample waiting time
        :param add_coef: default value 0.1
        :return: bool, if True adding samples is complete
        """
        # Get default scheduled samples
        n_scheduled = self.l_scheduled_samples()

        # New scheduled sample will be 10 percent of difference
        # between current number of target samples and new estimated one
        # If 10 percent of estimated samples is greater than difference between estimated and scheduled samples,
        # set scheduled samples to estimated samples
        new_scheduled = np.where((n_estimated * add_coef) > (n_estimated - n_scheduled),
                                 n_estimated,
                                 n_scheduled + (n_estimated - n_scheduled) * add_coef)

        n_scheduled = np.ceil(np.where(n_estimated < n_scheduled,
                                       n_scheduled,
                                       new_scheduled))

        # Levels where estimated are greater than scheduled
        greater_items = np.where(np.greater(n_estimated, n_scheduled))[0]

        # Scheduled samples and wait until at least half of the samples are done
        self.set_scheduled_and_wait(n_scheduled, greater_items, sleep)

        return np.all(n_estimated[greater_items] == n_scheduled[greater_items])

    def set_scheduled_and_wait(self, n_scheduled, greater_items, sleep, fin_sample_coef=0.5):
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
        self.schedule_samples()

        # Finished level samples
        n_finished = self.n_finished_samples

        # Wait until at least half of the scheduled samples are done on each level
        while np.any(n_finished[greater_items] < fin_sample_coef * n_scheduled[greater_items]):
            # Wait a while
            time.sleep(sleep)
            self.ask_sampling_pool_for_samples()
            n_finished = self.n_finished_samples

    def set_level_target_n_samples(self, n_samples, fraction=1.0):
        """
        Set level number of target samples
        :param n_samples: list, each level target samples
        :param fraction: Use just fraction of total samples
        :return: None
        """
        for level, n in enumerate(n_samples):
            self._n_target_samples[level] += int(n * fraction)

    def l_scheduled_samples(self):
        """
        Get all levels target number of samples
        :return: list
        """
        return self._n_target_samples
