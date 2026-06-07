import os
import sys
import shutil
import queue
import time
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
import traceback
from abc import ABC, abstractmethod
from multiprocessing import Pool as ProcPool
from multiprocessing import pool
from mlmc.level_simulation import LevelSimulation, SampleInput


class SamplingPool(ABC):
    """
    Abstract base class defining the runtime environment for sample simulations.
    It manages sample execution across different backends (single process,
    multiprocessing, PBS, etc.).
    """

    FAILED_DIR = 'failed'
    SEVERAL_SUCCESSFUL_DIR = 'several_successful'
    N_SUCCESSFUL = 5
    # Number of successful samples to store.

    def __init__(self, work_dir: Optional[str] = None, debug: bool = False):
        """
        Initialize the sampling pool environment.

        :param work_dir: Path to an existing working directory where outputs are
            stored. If omitted, the current working directory is used.
        :param debug: If True, keep sample directories for debugging.
        """
        work_dir = os.getcwd() if work_dir is None else os.path.abspath(work_dir)
        if not os.path.isdir(work_dir):
            raise FileNotFoundError("SamplingPool work_dir does not exist: {}".format(work_dir))

        self._output_dir = os.path.join(work_dir, "output")
        self._debug = debug

        # Prepare main output, failed, and successful directories.
        self._create_dir()
        self._create_dir(SamplingPool.FAILED_DIR)
        self._successful_dir = self._create_dir(SamplingPool.SEVERAL_SUCCESSFUL_DIR)

    def _create_dir(self, directory: str = "") -> Optional[str]:
        """
        Create the output directory if it does not exist.

        In debug mode, existing directories are preserved.
        """
        directory = os.path.join(self._output_dir, directory)
        if os.path.exists(directory) and not self._debug:
            shutil.rmtree(directory)
        os.makedirs(directory, mode=0o775, exist_ok=True)
        return directory

    # --- Abstract methods to be implemented by subclasses ---

    @abstractmethod
    def schedule_sample(self, sample_input: SampleInput, level_sim: LevelSimulation):
        """
        Schedule a simulation sample for execution.

        :param sample_input: Tuple ``(sample_id, input_value)`` prepared by
            ``LevelSimulation.prepare_samples()``.
        :param level_sim: LevelSimulation instance.
        :return: Tuple[str, List]
        """

    @abstractmethod
    def have_permanent_samples(self, sample_inputs: List[Any]) -> bool:
        """
        Inform the pool about samples that have been scheduled but not yet finished.
        """

    @abstractmethod
    def get_finished(self):
        """
        Retrieve finished sample results.

        :return: Tuple containing (successful samples, failed samples, number of running samples)
        """

    # --- Utility methods shared across subclasses ---

    @staticmethod
    def compute_seed(sample_id: str) -> int:
        """
        Compute a deterministic seed for a given sample ID.

        :param sample_id: Unique sample identifier.
        :return: Integer seed value.
        DEPRECATED,use LevelSimulation.compute_seed instead
        """
        return LevelSimulation.compute_seed(sample_id)

    @staticmethod
    def calculate_sample(sample_input: SampleInput, level_sim: LevelSimulation,
                         work_dir: Optional[str] = None) -> Tuple[str, Any, str, float]:
        """
        Execute a single simulation sample.

        :param sample_input: Tuple ``(sample_id, input_value)``. The default
            ``LevelSimulation.prepare_samples()`` uses the deterministic seed
            as ``input_value``; specialized simulations may provide richer
            input values.
        :param level_sim: LevelSimulation instance.
        :param work_dir: Working directory for the sample.
        :return: Tuple(sample_id, result, error_message, running_time)
        """
        sample_id, input_value = sample_input

        res = (None, None)
        err_msg = ""
        running_time = 0.0

        if level_sim.need_sample_workspace:
            SamplingPool.handle_sim_files(work_dir, sample_id, level_sim)

        try:
            start = time.time()
            res = level_sim._calculate(level_sim.config_dict, input_value)
            running_time = time.time() - start

            # Validate result format.
            if isinstance(res[0], np.ndarray) and isinstance(res[1], np.ndarray):
                flatten_fine_res = res[0].flatten()
                flatten_coarse_res = res[1].flatten()

                expected_len = np.sum([
                    np.prod(q.shape) * len(q.times) * len(q.locations)
                    for q in level_sim._result_format()
                ])

                assert len(flatten_fine_res) == len(flatten_coarse_res) == expected_len, \
                    f"Unexpected result format. Expected length: {expected_len}, got: {len(flatten_fine_res)}"

        except Exception:
            err_msg = "".join(traceback.format_exception(*sys.exc_info()))
            print("Error msg:", err_msg)

        return sample_id, res, err_msg, running_time

    # --- File handling helpers ---

    @staticmethod
    def change_to_sample_directory(work_dir: str, path: str) -> str:
        """
        Create and switch to the sample-specific directory.

        :param work_dir: Base working directory.
        :param path: Sample subdirectory name.
        :return: Absolute path to the created sample directory.
        """
        sample_dir = os.path.join(work_dir, path)
        os.makedirs(sample_dir, mode=0o775, exist_ok=True)

        # We have observed possible problems with directory creation on the 
        # network filesystem. However it is not sure that was the cause of the problem.
        # This code would make sure that we have write access to the created directory.
        
        # Commented out in order to try to reproduce the problem.
        #for i in range(30):
            #try:
                #with open(os.path.join(sample_dir, "_test_file.txt"), "w") as f:
                    #f.write("test")
            #except:
                #time.sleep(1)
                #continue
            #print("Workdir ready after {i} sleep seconds.")
            #break
        #else:
            #print("Workdir still not ready ready after {i} sleep seconds.")
    
        return sample_dir

    @staticmethod
    def copy_sim_files(files: List[str], sample_dir: str):
        """
        Copy shared simulation files to the sample directory.

        :param files: List of file paths to copy.
        :param sample_dir: Destination sample directory.
        """
        for file in files:
            shutil.copy(file, sample_dir)

    @staticmethod
    def handle_sim_files(work_dir: str, sample_id: str, level_sim: LevelSimulation):
        """
        Prepare the sample workspace (create directory, copy common files, set cwd).

        :param work_dir: Base working directory.
        :param sample_id: Sample identifier.
        :param level_sim: LevelSimulation instance.
        """
        if level_sim.need_sample_workspace:
            sample_dir = SamplingPool.change_to_sample_directory(work_dir, sample_id)
            if level_sim.common_files is not None:
                SamplingPool.copy_sim_files(level_sim.common_files, sample_dir)
            os.chdir(sample_dir)

    @staticmethod
    def move_successful_rm(sample_id: str, level_sim: LevelSimulation,
                           output_dir: str, dest_dir: str):
        """
        Move successful sample directories and remove originals.
        """
        if int(sample_id[-7:]) < SamplingPool.N_SUCCESSFUL:
            SamplingPool.move_dir(sample_id, level_sim.need_sample_workspace, output_dir, dest_dir)
        SamplingPool.remove_sample_dir(sample_id, level_sim.need_sample_workspace, output_dir)

    @staticmethod
    def move_failed_rm(sample_id: str, level_sim: LevelSimulation,
                       output_dir: str, dest_dir: str):
        """
        Move failed sample directories and remove originals.
        """
        SamplingPool.move_dir(sample_id, level_sim.need_sample_workspace, output_dir, dest_dir)
        SamplingPool.remove_sample_dir(sample_id, level_sim.need_sample_workspace, output_dir)

    @staticmethod
    def move_dir(sample_id: str, sample_workspace: bool,
                 work_dir: str, dest_dir: str):
        """
        Move a sample directory to another location (e.g., failed or successful).

        :param sample_id: Sample identifier.
        :param sample_workspace: Whether the sample uses its own workspace.
        :param work_dir: Base working directory.
        :param dest_dir: Destination subdirectory name.
        """
        if sample_workspace and work_dir and dest_dir:
            destination_dir = os.path.join(work_dir, dest_dir)
            sample_dir = SamplingPool.change_to_sample_directory(work_dir, sample_id)
            target_dir = os.path.join(destination_dir, sample_id)
            if os.path.exists(target_dir):
                shutil.rmtree(target_dir, ignore_errors=True)
            shutil.copytree(sample_dir, target_dir)

    @staticmethod
    def remove_sample_dir(sample_id: str, sample_workspace: bool, work_dir: str):
        """
        Remove the directory for a completed or failed sample.

        :param sample_id: Sample identifier.
        :param sample_workspace: Whether the sample uses its own workspace.
        :param work_dir: Base working directory.
        """
        if sample_workspace and work_dir:
            sample_dir = SamplingPool.change_to_sample_directory(work_dir, sample_id)
            shutil.rmtree(sample_dir, ignore_errors=True)


class OneProcessPool(SamplingPool):
    """
    Sampling pool implementation that executes all samples sequentially in a single process.
    Used primarily for debugging or lightweight simulations.
    """

    def __init__(self, work_dir=None, debug=False):
        """
        Initialize the one-process pool.

        Parameters
        ----------
        work_dir : str, optional
            Working directory for storing sample outputs.
        debug : bool, default=False
            If True, disables moving/removing files after successful execution.
        """
        super().__init__(work_dir=work_dir, debug=debug)
        self._failed_queues = {}  # Stores failed sample queues per level
        self._queues = {}  # Stores successful sample queues per level
        self._n_running = 0  # Tracks number of currently running samples
        self.times = {}  # Stores total runtime and count per level

    def schedule_sample(self, sample_input: SampleInput, level_sim):
        """
        Execute a single sample synchronously (in the current process).

        Parameters
        ----------
        sample_input
            Tuple ``(sample_id, input_value)`` prepared by LevelSimulation.
        level_sim : LevelSimulation
            Simulation instance containing configuration for the sample.
        """
        self._n_running += 1  # Increment running sample counter

        # Run the sample and collect result, error message, and runtime
        sample_id, result, err_msg, running_time = SamplingPool.calculate_sample(
            sample_input, level_sim, work_dir=self._output_dir
        )

        # Process result (successful or failed)
        self._process_result(sample_id, result, err_msg, running_time, level_sim)

    def _process_result(self, sample_id, result, err_msg, running_time, level_sim):
        """
        Process result from a sample execution and store it in the appropriate queue.

        Parameters
        ----------
        sample_id : int
            Identifier of the executed sample.
        result : tuple
            Pair of fine and coarse results (numpy arrays).
        err_msg : str
            Error message if the sample failed, empty string otherwise.
        running_time : float
            Runtime of the sample execution in seconds.
        level_sim : LevelSimulation
            Simulation instance used to produce the sample.
        """
        # Record runtime for this level
        self._save_running_time(level_sim._level_id, running_time)

        # If no error occurred, store successful result
        if not err_msg:
            level_queue = self._queues.setdefault(level_sim._level_id, queue.Queue())
            level_queue.put((sample_id, (result[0], result[1])))
            # Move successful sample to its permanent directory unless debugging
            if not self._debug:
                SamplingPool.move_successful_rm(
                    sample_id, level_sim, output_dir=self._output_dir, dest_dir=self._successful_dir
                )
        else:
            # If the simulation failed
            if not level_sim.need_sample_workspace:
                print(f"Sample {sample_id} error: {err_msg}")
            else:
                SamplingPool.move_failed_rm(
                    sample_id, level_sim, output_dir=self._output_dir, dest_dir=SamplingPool.FAILED_DIR
                )
            self._failed_queues.setdefault(level_sim._level_id, queue.Queue()).put((sample_id, err_msg))

    def _save_running_time(self, level_id, running_time):
        """
        Save sample execution time in the tracking dictionary.

        Parameters
        ----------
        level_id : int
            Identifier of the simulation level.
        running_time : float
            Execution time of the sample.
        """
        # Initialize level entry if missing
        if level_id not in self.times:
            self.times[level_id] = [0, 0]
        # Only count successful samples with nonzero runtime
        if running_time != 0:
            self.times[level_id][0] += running_time  # Accumulate total runtime
            self.times[level_id][1] += 1  # Increment sample count

    def have_permanent_samples(self, sample_inputs):
        """
        Return False, indicating that no samples are stored permanently.

        Parameters
        ----------
        sample_inputs : list
            List of scheduled sample inputs (ignored).

        Returns
        -------
        bool
            Always False.
        """
        return False

    def get_finished(self):
        """
        Retrieve all completed (successful and failed) samples.

        Returns
        -------
        successful : dict
            Dictionary of successful samples by level.
        failed : dict
            Dictionary of failed samples by level.
        n_running : int
            Number of currently running samples.
        times : list
            List of (level_id, [total_time, n_samples]) pairs.
        """
        successful = self._queues_to_list(list(self._queues.items()))
        failed = self._queues_to_list(list(self._failed_queues.items()))
        return successful, failed, self._n_running, list(self.times.items())

    def _queues_to_list(self, queue_dict_list):
        """
        Convert queues to lists and clear them safely.

        Parameters
        ----------
        queue_dict_list : list
            List of (level_id, queue.Queue) pairs.

        Returns
        -------
        results : dict
            Dictionary mapping level_id to list of queue entries.
        """
        results = {}
        for level_id, q in queue_dict_list:
            queue_list = list(q.queue)
            if not queue_list:
                continue
            results[level_id] = queue_list

            # Thread-safe queue clearing
            with q.mutex:
                q.queue.clear()

            # Update running sample counter
            self._n_running -= len(results[level_id])
        return results


# ==============================================================================

class ProcessPool(OneProcessPool):
    """
    Sampling pool using multiprocessing for parallel sample execution.
    Suitable for simulations without external program calls.
    """

    def __init__(self, n_processes, work_dir=None, debug=False):
        """
        Initialize process-based parallel sampling pool.

        Parameters
        ----------
        n_processes : int
            Number of worker processes to use.
        work_dir : str, optional
            Working directory for samples.
        debug : bool, default=False
            If True, disables moving/removing sample outputs.
        """
        super().__init__(work_dir=work_dir, debug=debug)
        self._pool = ProcPool(n_processes)  # Multiprocessing pool

    def res_callback(self, result, level_sim):
        """
        Callback for handling results from asynchronous execution.

        Parameters
        ----------
        result : tuple
            Returned result from SamplingPool.calculate_sample().
        level_sim : LevelSimulation
            Simulation level instance.
        """
        self._process_result(*result, level_sim)

    def schedule_sample(self, sample_input: SampleInput, level_sim):
        """
        Schedule a sample for parallel execution in a separate process.

        Parameters
        ----------
        sample_input
            Tuple ``(sample_id, input_value)`` prepared by LevelSimulation.
        level_sim : LevelSimulation
            Simulation configuration instance.
        """
        self._n_running += 1

        # Submit task asynchronously to process pool
        self._pool.apply_async(
            SamplingPool.calculate_sample,
            args=(sample_input, level_sim, self._output_dir),
            callback=lambda res: self.res_callback(res, level_sim),
            error_callback=lambda res: self.res_callback(res, level_sim)
        )


# ==============================================================================

class ThreadPool(ProcessPool):
    """
    Sampling pool using threading for local parallel sampling.
    Suitable for simulations with external program calls (I/O-bound).
    """

    def __init__(self, n_thread, work_dir=None, debug=False):
        """
        Initialize thread-based parallel sampling pool.

        Parameters
        ----------
        n_thread : int
            Number of threads to use.
        work_dir : str, optional
            Working directory for samples.
        debug : bool, default=False
            If True, disables moving/removing sample outputs.
        """
        super().__init__(n_thread, work_dir=work_dir, debug=debug)
        self._pool = pool.ThreadPool(n_thread)  # Thread-based pool instead of process-based
        self._failed_queues = {}
        self._queues = {}
        self._n_running = 0
        self.times = {}
