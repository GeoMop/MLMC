import os
import shutil
import queue
import time
import hashlib
import copy
import numpy as np
from typing import List
from abc import ABC, abstractmethod
from multiprocessing import Pool as ProcPool
from multiprocessing import pool
from mlmc.level_simulation import LevelSimulation


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
    def have_permanent_samples(self, sample_ids):
        """
        Informs the Pool about sample_ids that have been scheduled but not yet finished
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
    def calculate_sample(sample_id, level_sim, work_dir=None):
        """
        Method for calculating results
        :param sample_id: str
        :param level_sim: LevelSimulation
        :param work_dir: working directory
        :return:
        """
        seed = SamplingPool.compute_seed(sample_id)
        res = (None, None)
        err_msg = ""
        running_time = 0

        if level_sim.need_sample_workspace:
            SamplingPool._handle_sim_files(work_dir, sample_id, level_sim)

        try:
            start = time.time()
            res = level_sim.calculate(level_sim.config_dict, seed)
            running_time = time.time() - start
        except Exception as err:
            err_msg = str(err)

        return sample_id, res, err_msg, running_time

    @staticmethod
    def _change_to_sample_directory(work_dir, path: str):
        """
        Create sample directory and change working directory
        :param path: str
        :return: None
        """
        sample_dir = os.path.join(work_dir, path)
        if not os.path.isdir(sample_dir):
            os.makedirs(sample_dir, mode=0o775, exist_ok=True)
        return sample_dir

    @staticmethod
    def _copy_sim_files(files: List[str], sample_dir):
        """
        Copy simulation common files to current simulation sample directory
        :param files: List of files
        :return: None
        """
        for file in files:
            shutil.copy(file, sample_dir)

    @staticmethod
    def _handle_sim_files(work_dir, sample_id, level_sim):
        """
        Change working directory to sample dir and copy common files
        :param sample_id: str
        :param level_sim: LevelSimulation
        :return: None
        """
        sample_dir = SamplingPool._change_to_sample_directory(work_dir, sample_id)
        SamplingPool._copy_sim_files(level_sim.common_files, sample_dir)

    def _create_failed(self):
        """
        Create directory for all failed samples
        :return: None
        """
        failed_dir = os.path.join(self._work_dir, "failed")
        if not os.path.isdir(failed_dir):
            os.makedirs(failed_dir, mode=0o775, exist_ok=True)

        return failed_dir

    def _move_failed_dir(self, sample_id, sample_workspace):
        """
        Move failed sample dir to failed directory
        :param sample_id: str
        :param sample_workspace: bool, simulation needs workspace
        :return: None
        """
        if sample_workspace and self._work_dir is not None:
            failed_dir = self._create_failed()
            sample_dir = SamplingPool._change_to_sample_directory(self._work_dir, sample_id)
            shutil.copytree(sample_dir, os.path.join(failed_dir, sample_id))
            shutil.rmtree(sample_dir, ignore_errors=True)

    def _remove_sample_dir(self, sample_id, sample_workspace):
        """
        Remove sample directory
        :param sample_id: str
        :param sample_workspace: bool, simulation needs workspace
        :return: None
        """
        if sample_workspace and self._work_dir is not None:
            sample_dir = SamplingPool._change_to_sample_directory(self._work_dir, sample_id)
            shutil.rmtree(sample_dir, ignore_errors=True)


class OneProcessPool(SamplingPool):

    def __init__(self, work_dir=None):
        """
        Everything is running in one process
        """
        self._work_dir = work_dir
        self._failed_queues = {}
        self._queues = {}
        self._n_running = 0
        self.times = {}

    def schedule_sample(self, sample_id, level_sim):
        self._n_running += 1

        if self._work_dir is None and level_sim.need_sample_workspace:
            self._work_dir = os.getcwd()

        sample_id, result, err_msg, running_time = SamplingPool.calculate_sample(sample_id, level_sim, self._work_dir)

        # Save running time for n_ops
        self._save_running_time(level_sim.level_id, running_time)

        if not err_msg:
            self._queues.setdefault(level_sim.level_id, queue.Queue()).put((sample_id, (result[0], result[1])))
            self._remove_sample_dir(sample_id, level_sim.need_sample_workspace)
        else:
            self._failed_queues.setdefault(level_sim.level_id, queue.Queue()).put((sample_id, err_msg))
            self._move_failed_dir(sample_id, level_sim.need_sample_workspace)

    def _save_running_time(self, level_id, running_time):
        """
        Save running time to dictionary, store total time and number of samples
        :param level_id: int
        :param running_time: float
        :return: None
        """
        # Save sample times [total time, number of samples]
        if level_id not in self.times:
            self.times[level_id] = [0, 0]
        # Failed samples have running time equal 0 by default
        if running_time != 0:
            self.times[level_id][0] += running_time
            self.times[level_id][1] += 1

    def have_permanent_samples(self, sample_ids):
        return False

    def get_finished(self):
        """
        return results from queue - list of (sample_id, pair_of_result_vectors, error_message)
        """
        successful = self._queues_to_list(self._queues)
        failed = self._queues_to_list(self._failed_queues)

        return successful, failed, self._n_running, copy.deepcopy(self.times)

    def _queues_to_list(self, queue_dict):
        results = {}
        for level_id, q in list(queue_dict.items()):
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

    def res_callback(self, result, level_sim):
        """
        Process simulation results
        :param result: tuple
        :param level_sim: LevelSimulation instance
        :return: None
        """
        sample_id, result, err_msg, running_time = result

        # Save sample running time
        self._save_running_time(level_sim.level_id, running_time)

        if not err_msg:
            self._queues.setdefault(level_sim.level_id, queue.Queue()).put((sample_id, (result[0], result[1])))
            self._remove_sample_dir(sample_id, level_sim.need_sample_workspace)
        else:
            self._failed_queues.setdefault(level_sim.level_id, queue.Queue()).put((sample_id, err_msg))
            self._move_failed_dir(sample_id, level_sim.need_sample_workspace)

    def schedule_sample(self, sample_id, level_sim):
        self._n_running += 1

        if self._work_dir is None and level_sim.need_sample_workspace:
            self._work_dir = os.getcwd()

        self._pool.apply_async(SamplingPool.calculate_sample, args=(sample_id, level_sim, self._work_dir),
                               callback=lambda res: self.res_callback(res, level_sim),
                               error_callback=lambda res: self.res_callback(res, level_sim))


class ThreadPool(ProcessPool):
    """
    Suitable local parallel sampling for simulations WITH external program call
    """

    def __init__(self, n_thread, work_dir=None):
        self._pool = pool.ThreadPool(n_thread)
        self._work_dir = work_dir
        self._failed_queues = {}
        self._queues = {}
        self._n_running = 0
        self.times = {}


# class change_cwd:
#     """
#     Context manager that change CWD, to given relative or absolute path.
#     """
#     def __init__(self, path: str):
#         self.path = path
#         self.orig_cwd = ""
#
#     def __enter__(self):
#         if self.path:
#             self.orig_cwd = os.getcwd()
#             os.chdir(self.path)
#
#     def __exit__(self, exc_type, exc_value, traceback):
#         if self.orig_cwd:
#             os.chdir(self.orig_cwd)
