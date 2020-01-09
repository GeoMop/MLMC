from abc import ABC, abstractmethod
from multiprocessing import Pool as ProcPool
from multiprocessing.pool import ThreadPool
from level_simulation import LevelSimulation
import queue
from test.new_synth_simulation import SimulationTest


class SamplingPool(ABC):

    def __init__(self):
        self.need_workspace = False

    @abstractmethod
    def schedule_sample(self, sample_id, level_sim: LevelSimulation):
        """
        Plan samples
        """

    @abstractmethod
    def have_permanent_sample(self, sample_id):
        """
        Is sample serialized?
        """

    @abstractmethod
    def get_finished(self):
        """
        Return finished samples
        """


class ProcessPool(SamplingPool):

    def __init__(self, n_processes):
        self._pool = ProcPool(n_processes)
        self._queue = queue.Queue()

    def schedule_sample(self, sample_id, level_sim):
        print("level sim ", level_sim)
        level_sim.config_dict["sample_id"] = sample_id
        result = self._pool.apply_async(ProcessPool.calculate_sample, args=(sample_id, level_sim, ),
                                        callback=self.result_callback, error_callback=self.error_callback)
        result.get()
        return result

    @staticmethod
    def calculate_sample(sample_id, level_sim):
        """
        Method for calculating results
        :param sample_id:
        :param level_sim:
        :return:
        """
        res = level_sim.calculate(level_sim.config_dict, sample_workspace=level_sim.sample_workspace)
        return sample_id, res

    def result_callback(self, res):
        self._queue.put((res[0], res[1], "message"))

    def error_callback(self, res):
        self._queue.put((res[0], res[1], "There was an error"))

    def have_permanent_sample(self, sample_id):
        """
        List of permanent samples or find per call?
        """
        return False

    def get_finished(self):
        """
        return results from queue - list of (sample_id, pair_of_result_vectors, error_message)
        """
        return self._queue.get()


class ThreadPool(ProcessPool):

    def __init__(self):
        self._pool = ThreadPool()
        self._queue = queue.Queue()
