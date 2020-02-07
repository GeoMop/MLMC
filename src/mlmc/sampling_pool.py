import shutil
import queue
import hashlib
import numpy as np
from typing import List
from abc import ABC, abstractmethod
from multiprocessing import Pool as ProcPool
from multiprocessing import pool
from level_simulation import LevelSimulation


class SamplingPool(ABC):

    def __init__(self):
        self.need_workspace = False

    @abstractmethod
    def schedule_sample(self, sample_id, level_sim: LevelSimulation):
        """
        Method for calculating simulation samples
        :param sample_id: str
        :param level_sim: level_simulation.LevelSimulation instance
        :return: Tuple[str, List]
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
        :return: list of results, number of running samples
        """

    @staticmethod
    def compute_seed(sample_id):
        """
        Calculate seed for given sample id
        :param sample_id: str
        :return: int
        """
        hash = hashlib.md5(sample_id.encode('ascii'))
        seed = np.frombuffer(hash.digest(), dtype='uint32')[0]
        return seed

    @staticmethod
    def calculate_sample(sample_id, level_sim):
        """
        Method for calculating results
        :param sample_id:
        :param level_sim:
        :return:
        """
        seed = SamplingPool.compute_seed(sample_id)
        res = (None, None)
        err_msg = ""
        try:
            res = level_sim.calculate(level_sim.config_dict, seed)
        except Exception as err:
            err_msg = str(err)

        return sample_id, res, err_msg


class OneProcessPool(SamplingPool):

    def __init__(self):
        """
        Everything is running in one process
        """
        self._failed_queues = {}
        self._queues = {}
        self._n_running = 0

    def schedule_sample(self, sample_id, level_sim):
        self._n_running += 1
        sample_id, result, err_msg = SamplingPool.calculate_sample(sample_id, level_sim)

        if not err_msg:
            self._queues.setdefault(level_sim.level_id, queue.Queue()).put((sample_id, (result[0], result[1])))
        else:
            self._failed_queues.setdefault(level_sim.level_id, queue.Queue()).put((sample_id, err_msg))

    def have_permanent_sample(self, sample_id):
        return False

    def get_finished(self):
        """
        return results from queue - list of (sample_id, pair_of_result_vectors, error_message)
        """
        successful = self._queues_to_list(self._queues)
        failed = self._queues_to_list(self._failed_queues)

        return successful, failed, self._n_running

    def _queues_to_list(self, queue_dict):
        results = {}
        for level_id, q in queue_dict.items():
            queue_list = list(q.queue)
            if not queue_list:
                continue
            results[level_id] = queue_list
            # Thread safe clear
            with q.mutex:
                q.queue.clear()

            self._n_running -= len(results[level_id])

        return results

    # def _modify_level_sim_obj(self, level_sim, level_id, sample_dir):
    #     """
    #     Copy simulation common files to sample directory
    #     :param level_sim: LevelSimulation instance
    #     :param level_id: level identifier
    #     :param sample_dir: sample directory
    #     :return: LevelSimulation
    #     """
    #     # Copy simulation common files
    #     if level_sim.common_files:
    #         self.copy_sim_files(level_sim.common_files, sample_dir)
    #
    #         if level_sim.sample_workspace is not None:
    #             raise Exception("Sample workspace must be set in Sampler method")
    #         level_sim.sample_workspace = sample_dir
    #
    #     return level_sim
    #
    # def copy_sim_files(self, files: List[str], sample_dir):
    #     """
    #     Copy simulation common files to current simulation sample directory
    #     :param files: List of files
    #     :return:
    #     """
    #     for file in files:
    #         shutil.copy(file, sample_dir)


# class ProcessPool(OneProcessPool):
#
#     def __init__(self, n_processes):
#         self._pool = ProcPool(n_processes)
#         self._queues = {}
#         self._n_running = 0
#
#         self._queue = queue.Queue()
#
#     def schedule_sample(self, sample_id, level_sim):
#         level_sim.config_dict["sample_id"] = sample_id
#         result = self._pool.apply_async(ProcessPool.calculate_sample, args=(sample_id, level_sim, ))
#         res = result.get()
#         self._queues.setdefault(level_sim.level_id, queue.Queue()).put((res[0], res[1][0], res[1][1]))
#         return result

#
# class ThreadPool(ProcessPool):
#
#     def __init__(self, n_thread):
#         self._pool = pool.ThreadPool(n_thread)
#         self._queue = queue.Queue()
