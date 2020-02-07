import os
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

    def __init__(self, work_dir=None):
        self._work_dir = work_dir

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
        :param sample_id: str
        :param level_sim: LevelSimulation
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

    def _change_to_sample_directory(self, path: str):
        """
        Create sample directory and change working directory
        :param path: str
        :return: None
        """
        sample_dir = os.path.join(self._work_dir, path)
        if not os.path.isdir(sample_dir):
            os.makedirs(sample_dir, mode=0o775, exist_ok=True)
        os.chdir(sample_dir)

    def _copy_sim_files(self, files: List[str]):
        """
        Copy simulation common files to current simulation sample directory
        :param files: List of files
        :return: None
        """
        for file in files:
            shutil.copy(file, os.getcwd())

    def _handle_sim_files(self, sample_id, level_sim):
        """
        Change working directory to sample dir and copy common files
        :param sample_id: str
        :param level_sim: LevelSimulation
        :return: None
        """
        if self._work_dir is None:
            raise FileNotFoundError("Work dir is not set")
        self._change_to_sample_directory(sample_id)
        self._copy_sim_files(level_sim.common_files)

    def _create_failed(self):
        """
        Create directory for all failed samples
        :return: None
        """
        failed_dir = os.path.join(self._work_dir, "failed")
        if not os.path.isdir(failed_dir):
            os.makedirs(failed_dir, mode=0o775, exist_ok=True)

        return failed_dir

    def _move_failed_dir(self, sample_id):
        """
        Move failed sample dir to failed directory
        :param sample_id: str
        :return: None
        """
        if self._work_dir is not None:
            failed_dir = self._create_failed()
            self._change_to_sample_directory(sample_id)
            shutil.copytree(os.getcwd(), os.path.join(failed_dir, sample_id))
            shutil.rmtree(os.getcwd(), ignore_errors=True)

    def _remove_sample_dir(self, sample_id):
        """
        Remove sample directory
        :param sample_id: str
        :return: None
        """
        if self._work_dir is not None:
            self._change_to_sample_directory(sample_id)
            shutil.rmtree(os.getcwd(), ignore_errors=True)


class OneProcessPool(SamplingPool):

    def __init__(self, work_dir=None):
        """
        Everything is running in one process
        """
        self._work_dir = work_dir
        self._failed_queues = {}
        self._queues = {}
        self._n_running = 0

    def schedule_sample(self, sample_id, level_sim):
        self._n_running += 1

        if level_sim.need_sample_workspace:
            self._handle_sim_files(sample_id, level_sim)

        sample_id, result, err_msg = SamplingPool.calculate_sample(sample_id, level_sim)

        if not err_msg:
            self._queues.setdefault(level_sim.level_id, queue.Queue()).put((sample_id, (result[0], result[1])))
            self._remove_sample_dir(sample_id)
        else:
            self._failed_queues.setdefault(level_sim.level_id, queue.Queue()).put((sample_id, err_msg))
            self._move_failed_dir(sample_id)

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


class ProcessPool(OneProcessPool):
    """
    Suitable for local parallel sampling for simulations WITHOUT external program call
    """

    def __init__(self, n_processes, work_dir=None):
        self._pool = ProcPool(n_processes)
        super().__init__(work_dir=work_dir)

    def schedule_sample(self, sample_id, level_sim):
        self._n_running += 1

        if level_sim.need_sample_workspace:
            self._handle_sim_files(sample_id, level_sim)

        res = self._pool.apply_async(SamplingPool.calculate_sample,
                                                            args=(sample_id, level_sim,))

        sample_id, result, err_msg = res.get()

        if not err_msg:
            self._queues.setdefault(level_sim.level_id, queue.Queue()).put((sample_id, (result[0], result[1])))
        else:
            self._failed_queues.setdefault(level_sim.level_id, queue.Queue()).put((sample_id, err_msg))
            self._move_failed_dir(sample_id)


class ThreadPool(ProcessPool):
    """
    Suitable local parallel sampling for simulations WITH external program call
    """

    def __init__(self, n_thread, work_dir=False):
        self._pool = pool.ThreadPool(n_thread)
        self._work_dir = work_dir
        self._failed_queues = {}
        self._queues = {}
        self._n_running = 0
