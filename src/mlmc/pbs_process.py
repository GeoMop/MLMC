import os
import shutil
import sys
import yaml
import time
import pickle
import json
from typing import List


src_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(src_path, '..', '..', 'src'))
sys.path.append(os.path.join(src_path, '..', '..', 'test'))
sys.path.append(os.path.join(src_path))


class PbsProcess:

    SCHEDULED = "{}_scheduled.yaml"  # Store scheduled samples as List[(level_sim.level_id, sample_id, seed)]
    SUCCESSFUL_RESULTS = "{}_successful_results.yaml"  # Simulation results as Dict[level_id, List[Tuple[sample_id, (fine result, coarse result)]]]
    FAILED_RESULTS = "{}_failed_results.yaml"  # Failed samples as Dict[level_id, List[Tuple[sample id, error messae]]]
    TIME = "{}_times.yaml"  # Dict[level_id, List[time, finished samples]]
    PBS_ID = "{}_"  # File which name assign our job id to pbs jobs id 'JobID_Pbs_ID'
    CLASS_FILE = "pbs_process_serialized.txt"  # Serialized data which are "passed" from sampling pool to pbs process

    def __init__(self, output_dir, jobs_dir, job_id, level_sim_file_format):
        self._output_dir = output_dir
        self._jobs_dir = jobs_dir
        self._job_id = job_id
        self._level_sim_file_format = level_sim_file_format

        self._files = None
        self._level_simulations = {}
        self._scheduled_file_path = None
        self._failed_result_file = None
        self._successful_result_file = None
        self._time_file = None


    @classmethod
    def create_job(cls, output_dir, jobs_dir, job_id, level_sim_file_format):
        pbs_process = cls(output_dir, jobs_dir, job_id, level_sim_file_format)
        PbsProcess._serialize_pbs_process(pbs_process)

        return pbs_process


    @classmethod
    def create_process(cls):
        job_id, output_dir = PbsProcess.command_params()
        jobs_dir, level_sim_file_format = PbsProcess._deserialize_pbs_process(output_dir)

        return cls(output_dir, jobs_dir, job_id, level_sim_file_format)

    @staticmethod
    def _serialize_pbs_process(pbs_process):
        if not os.path.exists(os.path.join(pbs_process._output_dir, PbsProcess.CLASS_FILE)):
            with open(os.path.join(pbs_process._output_dir, PbsProcess.CLASS_FILE), "w") as writer:
                writer.write(pbs_process._jobs_dir + ";")
                writer.write(pbs_process._level_sim_file_format + ";")

        PbsProcess._deserialize_pbs_process(pbs_process._output_dir)

    @staticmethod
    def _deserialize_pbs_process(output_dir):
        with open(os.path.join(output_dir, PbsProcess.CLASS_FILE), "r") as reader:
            line = reader.readline().split(';')

            return line[0], line[1]

    @staticmethod
    def command_params():
        """
        Read command parameters - job identifier and file with necessary files
        :return: None
        """
        output_dir = sys.argv[1]
        job_id = sys.argv[2]

        return job_id, output_dir

    def get_files(self):
        """
        Get path and name of some crucial files
        :return:
        """
        self._scheduled_file_path = os.path.join(self._jobs_dir, PbsProcess.SCHEDULED.format(self._job_id))
        self._failed_result_file = os.path.join(self._jobs_dir, PbsProcess.FAILED_RESULTS.format(self._job_id))
        self._successful_result_file = os.path.join(self._jobs_dir, PbsProcess.SUCCESSFUL_RESULTS.format(self._job_id))
        self._time_file = os.path.join(self._jobs_dir, PbsProcess.TIME.format(self._job_id))

    def _get_level_sim(self, level_id):
        """
        Deserialize LevelSimulation object
        :return: None
        """
        with open(os.path.join(self._output_dir, self._level_sim_file_format.format(level_id)), "rb") as reader:
            l_sim = pickle.load(reader)
            self._level_simulations[l_sim.level_id] = l_sim

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
        :return: List[Tuple[level_id: int, sample_id: str, seed: int]] sorted by level_id ASC
        """
        with open(self._scheduled_file_path) as file:
            level_id_sample_id_seed = yaml.load(file, yaml.Loader)

        level_id_sample_id_seed.sort(key=lambda tup: tup[0])
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
        current_level = 0
        start_time = time.time()
        times = {0: [0, 0]}
        chunk_to_file = 2  # Number of samples which are saved to file at one time
        for level_id, sample_id, seed in level_id_sample_id_seed:
            # Deserialize level simulation config
            if level_id not in self._level_simulations:
                self._get_level_sim(level_id)

            # Start measuring time
            if current_level != level_id:
                times[current_level][0] = time.time() - start_time
                times[level_id] = [0, 0]
                start_time = time.time()
                current_level = level_id
                #self._write_new_level(level_id)

            level_sim = self._level_simulations[level_id]
            assert level_sim.level_id == level_id
            self._handle_sim_files(sample_id, level_sim)
            # Go to sample directory
            #os.chdir(os.path.join(self._level_paths[level_id], sample_id))

            # Calculate sample
            res = (None, None)
            err_msg = ""
            try:
                res = level_sim.calculate(level_sim.config_dict, seed)
            except Exception as err:
                err_msg = str(err)

            times[level_id][1] += 1
            chunk_to_file -= 1

            if not err_msg:
                success.setdefault(level_id, []).append((sample_id, (res[0], res[1])))
                #self._remove_sample_dir(sample_id, level_sim.need_sample_workspace)
            else:
                failed.setdefault(level_id, []).append((sample_id, err_msg))
                #self._move_failed_dir(sample_id, level_sim.need_sample_workspace)

            if chunk_to_file == 0:
                chunk_to_file = 2

                times[level_id][0] = time.time() - start_time

                # Write results to files
                print("success ", success)
                print("failed ", failed)
                print("times ", times)
                if success:
                    self._append_file(success, self._successful_result_file, level_id)
                if failed:
                    self._append_file(failed, self._failed_result_file, level_id)
                if times:
                    self._append_times_file(times, self._time_file, level_id)

                success = {}
                failed = {}
                #times = {level_id: [0, 0]}

        self._add_end_mark()

    def _add_end_mark(self):
        # end_mark = {"end": "#"}
        # self._append_file(end_mark, self._successful_result_file)
        # self._append_file(end_mark, self._failed_result_file)
        # self._append_file(end_mark, self._time_file)

        if os.path.exists(self._successful_result_file):
            with open(self._successful_result_file, "r") as reader:
                successful = yaml.load(reader, yaml.Loader)

            print("successful ", successful)

    def _append_times_file(self, data, path, level_id):
        print("path ", path)
        print("given data ", data)
        if os.path.exists(path):
            with open(path, "r") as reader:
                file_data = yaml.load(reader, yaml.Loader)
                file_data[level_id] = data[level_id]
                data = file_data

        print("data ", data)

        with open(path, "w") as f:
            yaml.dump(data, f)

    def _append_file(self, data, path, level_id):
        print("path ", path)
        print("given data ", data)
        if os.path.exists(path):
            with open(path, "r") as reader:
                file_data = yaml.load(reader, yaml.Loader)
                print("file data ", file_data)
                if level_id not in file_data:
                    file_data[level_id] = []

                file_data[level_id].extend(data[level_id])

                data = file_data

        print("data ", data)

        with open(path, "w") as f:
            yaml.dump(data, f)

    def _change_to_sample_directory(self, sample_id):
        """
        Create sample directory and change working directory
        :param sample_id: str
        :return: None
        """
        sample_dir = os.path.join(self._output_dir, sample_id)
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
        if level_sim.need_sample_workspace:
            self._change_to_sample_directory(sample_id)
            self._copy_sim_files(level_sim.common_files)

    @classmethod
    def _read_results(self, job_id):
        """
        Read result file for given job id
        :param job_id: str
        :return: successful: Dict[level_id, List[Tuple[sample_id:str, Tuple[ndarray, ndarray]]]]
                 failed: Dict[level_id, List[Tuple[sample_id: str, error message: str]]]
                time: Dict[level_id: int, List[total time: float, number of success samples: int]]
        """
        successful = {}
        failed = {}
        time = [0, 0]

        if os.path.exists(os.path.join(self._jobs_dir, PbsProcess.SUCCESSFUL_RESULTS.format(job_id))):
            with open(os.path.join(self._jobs_dir, PbsProcess.SUCCESSFUL_RESULTS.format(job_id)), "r") as reader:
                successful = yaml.load(reader, yaml.Loader)

        if os.path.exists(os.path.join(self._jobs_dir, PbsProcess.FAILED_RESULTS.format(job_id))):
            with open(os.path.join(self._jobs_dir, PbsProcess.FAILED_RESULTS.format(job_id)), "r") as reader:
                failed = yaml.load(reader, yaml.Loader)

        if os.path.exists(os.path.join(self._jobs_dir, PbsProcess.TIME.format(job_id))):
            with open(os.path.join(self._jobs_dir, PbsProcess.TIME.format(job_id)), "r") as reader:
                time = yaml.load(reader, yaml.Loader)

        return successful, failed, time

    def create_files(self):
        self._scheduled_file = os.path.join(self._jobs_dir, PbsProcess.SCHEDULED.format(self._job_id))
        self._pbs_id_file = os.path.join(self._jobs_dir, PbsProcess.PBS_ID.format(self._job_id))

    def write_pbs_id(self, pbs_job_id):
        """
        Create empty file name contains pbs jobID and our jobID
        :param pbs_job_id: str
        :return: None
        """
        self._pbs_id_file += pbs_job_id
        print("PBS ID FILE ", self._pbs_id_file)
        with open(self._pbs_id_file, 'w') as w:
            pass

    def save_scheduled(self, scheduled):
        """
        Save scheduled samples to yaml file
        format: List[Tuple[level_id, sample_id]]
        :return: None
        """
        try:
            with open(self._scheduled_file, "w") as file:
                yaml.dump(scheduled, file)
        except FileNotFoundError:
            print("Make sure you call _create_files method previously")






if __name__ == "__main__":
    pbs_process = PbsProcess.create_process()
    pbs_process.get_files()
    pbs_process.calculate_samples()
