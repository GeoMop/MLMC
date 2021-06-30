import os
import sys
import shutil
import queue
import time
import hashlib
import numpy as np
from typing import List
import traceback
from abc import ABC, abstractmethod
from multiprocessing import Pool as ProcPool
from multiprocessing import pool
from mlmc.level_simulation import LevelSimulation


class SamplingPool(ABC):
    """
    Determining the runtime environment of samples, eg single process, multiple processes, running PBS, ...
    """

    FAILED_DIR = 'failed'
    SEVERAL_SUCCESSFUL_DIR = 'several_successful'
    N_SUCCESSFUL = 5
    # Number of successful samples to store

    def __init__(self, work_dir=None, debug=False):
        """
        :param work_dir: Path to working directory
        :param debug: bool, if True keep sample directories
        """
        self._output_dir = None
        if work_dir is not None:
            work_dir = os.path.abspath(work_dir)
            self._output_dir = os.path.join(work_dir, "output")
        self._debug = debug

        self._create_dir()  # prepare output dir
        self._create_dir(SamplingPool.FAILED_DIR)  # prepare failed dir
        self._successful_dir = self._create_dir(SamplingPool.SEVERAL_SUCCESSFUL_DIR)  # prepare several successful dir

    def _create_dir(self, directory=""):
        """
        Create output directory, in 'debug' mode not remove existing output_dir
        :return: None
        """
        if self._output_dir is not None:
            directory = os.path.join(self._output_dir, directory)
            if os.path.exists(directory) and not self._debug:
                shutil.rmtree(directory)
            os.makedirs(directory, mode=0o775, exist_ok=True)
            return directory

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
    def calculate_sample(sample_id, level_sim, work_dir=None, seed=None):
        """
        Method for calculating results
        :param sample_id: str
        :param level_sim: LevelSimulation
        :param work_dir: working directory
        :param seed: random seed
        :return: sample id, sample result, error message with traceback, running time
        """
        if seed is None:
            seed = SamplingPool.compute_seed(sample_id)
        res = (None, None)
        err_msg = ""
        running_time = 0

        if level_sim.need_sample_workspace:
            SamplingPool.handle_sim_files(work_dir, sample_id, level_sim)
        try:
            start = time.time()
            res = level_sim._calculate(level_sim.config_dict, seed)
            running_time = time.time() - start

            # Check result format
            if type(res[0]) is np.ndarray and type(res[1]) is np.ndarray:
                flatten_fine_res = res[0].flatten()
                flatten_coarse_res = res[1].flatten()

                res_expected_len = np.sum(
                    [np.prod(quantity_spec.shape) * len(quantity_spec.times) * len(quantity_spec.locations)
                     for quantity_spec in level_sim._result_format()])

                assert len(flatten_fine_res) == len(flatten_coarse_res) == res_expected_len, \
                    "Unexpected result format, expected length: {}, resultf length: {}".format(res_expected_len,
                                                                                               len(flatten_fine_res))

        except Exception:
            str_list = traceback.format_exception(*sys.exc_info())
            err_msg = "".join(str_list)

        return sample_id, res, err_msg, running_time

    @staticmethod
    def change_to_sample_directory(work_dir, path: str):
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
    def copy_sim_files(files: List[str], sample_dir):
        """
        Copy simulation common files to current simulation sample directory
        :param files: List of files
        :return: None
        """
        for file in files:
            shutil.copy(file, sample_dir)

    @staticmethod
    def handle_sim_files(work_dir, sample_id, level_sim):
        """
        Change working directory to sample dir and copy common files
        :param sample_id: str
        :param level_sim: LevelSimulation
        :return: None
        """
        if level_sim.need_sample_workspace:
            sample_dir = SamplingPool.change_to_sample_directory(work_dir, sample_id)

            if level_sim.common_files is not None:
                SamplingPool.copy_sim_files(level_sim.common_files, sample_dir)
            os.chdir(sample_dir)

    @staticmethod
    def move_successful_rm(sample_id, level_sim, output_dir, dest_dir):
        if int(sample_id[-7:]) < SamplingPool.N_SUCCESSFUL:
            SamplingPool.move_dir(sample_id, level_sim.need_sample_workspace, output_dir, dest_dir=dest_dir)
        SamplingPool.remove_sample_dir(sample_id, level_sim.need_sample_workspace, output_dir)

    @staticmethod
    def move_failed_rm(sample_id, level_sim, output_dir, dest_dir):
        SamplingPool.move_dir(sample_id, level_sim.need_sample_workspace, output_dir, dest_dir=dest_dir)
        SamplingPool.remove_sample_dir(sample_id, level_sim.need_sample_workspace, output_dir)

    @staticmethod
    def move_dir(sample_id, sample_workspace, work_dir, dest_dir):
        """
        Move failed sample dir to failed directory
        :param sample_id: str
        :param sample_workspace: bool, simulation needs workspace
        :param work_dir: str
        :param dest_dir: destination
        :return: None
        """
        if sample_workspace and work_dir is not None and dest_dir is not None:
            destination_dir = os.path.join(work_dir, dest_dir)
            sample_dir = SamplingPool.change_to_sample_directory(work_dir, sample_id)
            if os.path.exists(os.path.join(destination_dir, sample_id)):
                shutil.rmtree(os.path.join(destination_dir, sample_id), ignore_errors=True)
            shutil.copytree(sample_dir, os.path.join(destination_dir, sample_id))

    @staticmethod
    def remove_sample_dir(sample_id, sample_workspace, work_dir):
        """
        Remove sample directory
        :param sample_id: str
        :param sample_workspace: bool, simulation needs workspace
        :param work_dir: str
        :return: None
        """
        if sample_workspace and work_dir is not None:
            sample_dir = SamplingPool.change_to_sample_directory(work_dir, sample_id)
            shutil.rmtree(sample_dir, ignore_errors=True)


class OneProcessPool(SamplingPool):

    def __init__(self, work_dir=None, debug=False):
        """
        Everything is running in one process
        """
        super().__init__(work_dir=work_dir, debug=debug)
        self._failed_queues = {}
        self._queues = {}
        self._n_running = 0
        self.times = {}

    def schedule_sample(self, sample_id, level_sim):
        self._n_running += 1

        if self._output_dir is None and level_sim.need_sample_workspace:
            self._output_dir = os.getcwd()

        sample_id, result, err_msg, running_time = SamplingPool.calculate_sample(sample_id, level_sim,
                                                                                 work_dir=self._output_dir)

        self._process_result(sample_id, result, err_msg, running_time, level_sim)

    def _process_result(self, sample_id, result, err_msg, running_time, level_sim):
        """
        Save sample result
        :param sample_id: sample identifier from calculate_sample()
        :param result: sample result from calculate_sample()
        :param err_msg: sample error message from calculate_sample()
        :param running_time: running time for sample from calculate_sample()
        :param level_sim: level_simulation instance
        :return: None
        """
        # Save running time for n_ops
        self._save_running_time(level_sim._level_id, running_time)

        if not err_msg:
            self._queues.setdefault(level_sim._level_id, queue.Queue()).put((sample_id, (result[0], result[1])))
            if not self._debug:
                SamplingPool.move_successful_rm(sample_id, level_sim, output_dir=self._output_dir, dest_dir=self._successful_dir)
        else:
            if not level_sim.need_sample_workspace:
                print("Sample {} error: {}".format(sample_id, err_msg))
            else:
                SamplingPool.move_failed_rm(sample_id, level_sim, output_dir=self._output_dir, dest_dir=SamplingPool.FAILED_DIR)
            self._failed_queues.setdefault(level_sim._level_id, queue.Queue()).put((sample_id, err_msg))

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
        successful = self._queues_to_list(list(self._queues.items()))
        failed = self._queues_to_list(list(self._failed_queues.items()))

        return successful, failed, self._n_running, list(self.times.items())

    def _queues_to_list(self, queue_dict_list):
        results = {}
        for level_id, q in queue_dict_list:
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

    def __init__(self, n_processes, work_dir=None, debug=False):
        self._pool = ProcPool(n_processes)
        super().__init__(work_dir=work_dir, debug=debug)

    def res_callback(self, result, level_sim):
        """
        Process simulation results
        :param result: tuple
        :param level_sim: LevelSimulation instance
        :return: None
        """
        self._process_result(*result, level_sim)

    def schedule_sample(self, sample_id, level_sim):
        self._n_running += 1

        if self._output_dir is None and level_sim.need_sample_workspace:
            self._output_dir = os.getcwd()

        self._pool.apply_async(SamplingPool.calculate_sample, args=(sample_id, level_sim, self._output_dir),
                               callback=lambda res: self.res_callback(res, level_sim),
                               error_callback=lambda res: self.res_callback(res, level_sim))


class ThreadPool(ProcessPool):
    """
    Suitable local parallel sampling for simulations WITH external program call
    """

    def __init__(self, n_thread, work_dir=None, debug=False):
        super().__init__(n_thread, work_dir=work_dir, debug=debug)
        self._pool = pool.ThreadPool(n_thread)
        self._failed_queues = {}
        self._queues = {}
        self._n_running = 0
        self.times = {}
