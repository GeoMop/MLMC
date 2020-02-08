import os
import sys
import yaml
import time
import pickle
import json
from pathlib import Path


src_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(src_path, '..', '..', 'src'))
sys.path.append(os.path.join(src_path, '..', '..', 'test'))
sys.path.append(os.path.join(src_path))

print("sys.path ", sys.path)


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
        # Get LevelSimulations anf paths to each level directory
        self._get_level_config()

        print("level sims ", self._level_simulations)
        print("l paths ", self._level_paths)
        self._scheduled_file_path = self._files['scheduled'].format(self._job_id)
        print("scheduled file path ", self._scheduled_file_path)
        self._failed_result_file = self._files['failed_results'].format(self._job_id)
        self._successful_result_file = self._files['successful_results'].format(self._job_id)
        self._time_file = self._files['time'].format(self._job_id)

    def _get_level_config(self):
        """
        Deserialize LevelSimulation object
        :return:
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
        with open(file_path, "r") as reader:
            self._files = json.load(reader)

    def _get_level_id_sample_id_seed(self):
        with open(self._scheduled_file_path) as file:
            level_id_sample_id_seed = yaml.load(file)

        return level_id_sample_id_seed

    def schedule_samples(self):
        """

        :param level_id_sample_id:
        :param level_sims:
        :param result_file_path:
        :return:
        """

        level_id_sample_id_seed = self._get_level_id_sample_id_seed()
        print("level id sample id seed", level_id_sample_id_seed)

        failed = {}
        success = {}
        times = {}
        for level_id, sample_id, seed in level_id_sample_id_seed:
            print("level id ", level_id)
            print("sample id ", sample_id)
            print("seed ", seed)
            level_sim = self._level_simulations[level_id]

            assert level_sim.level_id == level_id

            print("l_paths[level_id] ", self._level_paths[level_id])
            os.chdir(os.path.join(self._level_paths[level_id], sample_id))

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

            # results.append([sample_id, res, err_msg, running_time])

        # Write all results at one go otherwise there is duplicate key error
        self._write_results_to_file(success, failed, times)

    def _write_results_to_file(self, success, failed, times):
        """
        Write success samples, failed samples and average time
        :param success:
        :param failed:
        :param times:
        :return:
        """

        for level_id, results in success.items():
            self._append_file(results, os.path.join(self._level_paths[level_id], self._successful_result_file))

        for level_id, results in failed.items():
            self._append_file(results, os.path.join(self._level_paths[level_id], self._failed_result_file))

        for level_id, (time, n_samples) in times.items():
            self._append_file(time/n_samples, os.path.join(self._level_paths[level_id], self._time_file))

        # f = {}
        # s = {}
        # t = 0
        # if os.path.exists(os.path.join(self._level_paths[level_id], self._failed_result_file)):
        #     with open(os.path.join(self._level_paths[level_id], self._failed_result_file), "r") as reader:
        #         f = yaml.load(reader)
        #
        # if os.path.join(self._level_paths[level_id], self._successful_result_file):
        #     with open(os.path.join(self._level_paths[level_id], self._successful_result_file), "r") as reader:
        #         s = yaml.load(reader)
        #
        # if os.path.join(self._level_paths[level_id], self._time_file):
        #     with open(os.path.join(self._level_paths[level_id], self._time_file), "r") as reader:
        #         t = yaml.load(reader)
        #
        # print("failed ", f)
        # print("successful ", s)
        # print("times" , t)

    def _append_file(self, data, path):
        with open(path, "a") as f:
            yaml.dump(data, f)


if __name__ == "__main__":
    pbs_process = PbsProcess()
    pbs_process.get_files()
    pbs_process.schedule_samples()
