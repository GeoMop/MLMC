import os
import shutil
import sys
import yaml
import time
import pickle
import json
from typing import List
from pathlib import Path


src_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(src_path, '..', '..', 'src'))
sys.path.append(os.path.join(src_path, '..', '..', 'test'))
sys.path.append(os.path.join(src_path))


class PbsProcess:

    def __init__(self):
        self._job_id = None
        self._files = None
        self._level_simulations = None
        self._level_paths = None
        self._scheduled_file_path = None
        self._failed_result_file = None
        self._successful_result_file = None
        self._time_file = None

        self._command_params()

    def _command_params(self):
        """
        Read command parameters - job identifier and file with necessary files
        :return: None
        """
        self._read_file_structure(sys.argv[1])
        self._job_id = sys.argv[2]

    def get_files(self):
        """
        Get path and name of some crucial files
        :return:
        """
        # Get LevelSimulations anf paths to each level directory
        self._get_level_config()

        self._scheduled_file_path = self._files['scheduled'].format(self._job_id)
        self._failed_result_file = self._files['failed_results'].format(self._job_id)
        self._successful_result_file = self._files['successful_results'].format(self._job_id)
        self._time_file = self._files['time'].format(self._job_id)

    def _get_level_config(self):
        """
        Deserialize LevelSimulation object
        :return: None
        """
        with open(self._files['levels_config'], "r") as reader:
            level_config_files = reader.read().splitlines()

        self._level_simulations = {}
        self._level_paths = {}
        for level_config_file_path in level_config_files:
            with open(level_config_file_path, "rb") as f:
                l_sim = pickle.load(f)
                self._level_simulations[l_sim.level_id] = l_sim

            path = Path(level_config_file_path)
            self._level_paths[l_sim.level_id] = path.parent

    def _read_file_structure(self, file_path):
        """
        Read file structure - necessary files
        :param file_path: str
        :return: None
        """
        with open(file_path, "r") as reader:
            self._files = json.load(reader)

    def _get_level_id_sample_id_seed(self):
        """
        Get scheduled samples
        :return: List[Tuple[level_id: int, sample_id: str, seed: int]]
        """
        with open(self._scheduled_file_path) as file:
            level_id_sample_id_seed = yaml.load(file, yaml.Loader)

        return level_id_sample_id_seed

    def calculate_samples(self):
        """
        Calculate scheduled samples
        :return:
        """
        # List of Tuple[level id, sample id, random seed]
        level_id_sample_id_seed = self._get_level_id_sample_id_seed()

        failed = {}
        success = {}
        times = {}
        for level_id, sample_id, seed in level_id_sample_id_seed:
            level_sim = self._level_simulations[level_id]
            assert level_sim.level_id == level_id
            self._handle_sim_files(sample_id, level_sim)

            # Go to sample directory
            #os.chdir(os.path.join(self._level_paths[level_id], sample_id))

            # Calculate sample
            res = (None, None)
            err_msg = ""
            running_time = 0
            try:
                start = time.time()
                res = level_sim.calculate(level_sim.config_dict, seed)
                running_time = time.time() - start
            except Exception as err:
                err_msg = str(err)

            # Save sample times [total time, number of samples]
            if level_id not in times:
                times[level_id] = [0, 0]
            # Failed samples have running time equal 0 by default
            if running_time != 0:
                times[level_id][0] += running_time
                times[level_id][1] += 1

            if not err_msg:
                success.setdefault(level_sim.level_id, []).append((sample_id, (res[0], res[1])))
            else:
                failed.setdefault(level_sim.level_id, []).append((sample_id, err_msg))

        # Write all results at one go otherwise there is duplicate key error
        self._write_results_to_file(success, failed, times)

    def _write_results_to_file(self, success, failed, times):
        """
        Write success samples, failed samples and average time
        :param success: successful samples
        :param failed: failed samples
        :param times: n ops estimated
        :return: None
        """

        for level_id, results in success.items():
            self._append_file(results, os.path.join(self._level_paths[level_id], self._successful_result_file))

        for level_id, results in failed.items():
            self._append_file(results, os.path.join(self._level_paths[level_id], self._failed_result_file))

        for level_id, time_n_samples in times.items():
            self._append_file(time_n_samples, os.path.join(self._level_paths[level_id], self._time_file))

    def _append_file(self, data, path):
        with open(path, "a") as f:
            yaml.dump(data, f)

    def _change_to_sample_directory(self, level_id, sample_id):
        """
        Create sample directory and change working directory
        :param level_id: str
        :param sample_id: str
        :return: None
        """
        sample_dir = os.path.join(self._level_paths[level_id], sample_id)
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
        self._change_to_sample_directory(level_sim.level_id, sample_id)
        self._copy_sim_files(level_sim.common_files)


if __name__ == "__main__":
    pbs_process = PbsProcess()
    pbs_process.get_files()
    pbs_process.calculate_samples()
