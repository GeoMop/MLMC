import os
import shutil
import subprocess
import yaml
import re
from typing import List
import pickle
import json
import glob
from level_simulation import LevelSimulation
from sampling_pool import SamplingPool


class SamplingPoolPBS(SamplingPool):

    OUTPUT_DIR = "output"
    LEVEL_DIR = "level_{}"
    JOBS_DIR = "jobs"
    SCHEDULED = "{}_scheduled.yaml"
    SUCCESSFUL_RESULTS = "{}_successful_results.yaml"
    FAILED_RESULTS = "{}_failed_results.yaml"
    TIME = "{}_times.yaml"
    PBS_ID = "{}_"
    JOB = "{}_job.sh"
    LEVEL_SIM_CONFIG = "level_simulation_config"
    LEVELS_CONFIG = "levels_config.txt"
    STRUCTURE = "structure.json"

    def __init__(self, job_weight=200000, job_count=0, work_dir=None):
        """
        :param job_weight: Number of simulation elements per job script
        :param job_count: Number of created jobs
        """
        self._work_dir = work_dir
        # Weight of the single PBS script (putting more small jobs into single PBS job).
        self.job_weight = job_weight
        # Current collected weight.
        self._current_job_weight = 0
        # Number of executed jobs.
        self._job_count = job_count
        self.max_realizations = 10
        self._number_of_realizations = 0
        # Lines to put at the beginning of the PBS script.
        self.pbs_script_heading = None
        self.pbs_script = None
        # Set q sub command or direct execution.
        # self.qsub_cmd = qsub
        self._pbs_config = None
        self._pbs_header_template = None

        self.qsub_cmd = None
        self._scheduled = []
        
        self._level_dir = {}

        self._output_dir = None
        self._jobs_dir = None
        self._scheduled_file = None
        self._pbs_id_file = None
        self._job_file = None
        self._levels_config = None
        self._files_structure = None

        self._create_output_dir()
        self._create_job_dir()

    def _create_output_dir(self):
        if self._work_dir is None:
            raise NotADirectoryError("Working directory must be pass to Sampler init")

        if self._output_dir is None:
            self._output_dir = os.path.join(self._work_dir, SamplingPoolPBS.OUTPUT_DIR)

            #  @TODO: remove
            if os.path.isdir(self._output_dir):
                shutil.rmtree(self._output_dir)

            os.makedirs(self._output_dir, mode=0o775, exist_ok=True)

    def _create_level_workspace(self, level_id):
        """
        Create level directory
        :param level_id: Level identifier
        :return: None
        """
        if self._output_dir is not None:
            if not os.path.isdir(os.path.join(self._output_dir, SamplingPoolPBS.LEVEL_DIR.format(level_id))):
                level_dir = os.path.join(self._output_dir, SamplingPoolPBS.LEVEL_DIR.format(level_id))
                os.makedirs(level_dir, mode=0o775, exist_ok=True)
                self._level_dir[level_id] = level_dir
        else:
            raise NotADirectoryError("Create output directory at the first place")

    def change_to_sample_directory(self, path: str, level_id=None):
        sample_dir = os.path.join(self._level_dir[level_id], path)
        os.makedirs(sample_dir, mode=0o775, exist_ok=True)
        return sample_dir

    def copy_sim_files(self, files: List[str], sample_dir):
        """
        Copy simulation common files to current simulation sample directory
        :param files: List of files
        :return:
        """
        for file in files:
            shutil.copy(file, sample_dir)

    def pbs_common_setting(self, flow_3=False, **kwargs):
        """
        Values for common header of script
        :param flow_3: use flow123d version 3.0.0
        :param kwargs: dict with params vales
        :return: None
        """
        # Script header
        select_flags_list = kwargs.get('select_flags', [])
        if select_flags_list:
            kwargs['select_flags'] = ":" + ":".join(select_flags_list)
        else:
            kwargs['select_flags'] = ""

        self._pbs_header_template = ["#!/bin/bash",
                                     '#PBS -S /bin/bash',
                                     '#PBS -l select={n_nodes}:ncpus={n_cores}:mem={mem}{select_flags}',
                                     '#PBS -l walltime=1:00:00',
                                     '#PBS -q {queue}',
                                     '#PBS -N MLMC_sim',
                                     '#PBS -j oe',
                                     '#PBS -o {pbs_output_dir}/{job_name}.OU',
                                     '#PBS -e {pbs_output_dir}/{job_name}.ER',
                                     '']
        #  @TODO: prepare environment
        if flow_3:
            self._pbs_header_template.extend(['module use /storage/praha1/home/jan-hybs/modules',
                                              'source /storage/liberec3-tul/home/martin_spetlik/MLMC_new_design/test/load_modules.sh',
                                              'source env/bin/activate', ''])

        self._pbs_header_template.extend((
                                         'python3 {pbs_process_file_dir}/pbs_process.py {files_structure} {job_name} >{pbs_output_dir}/{job_name}_STDOUT 2>&1',))
        self._pbs_config = kwargs

    def _change_to_sample_directory(self, level_id, sample_id):
        """
        Create sample directory and change working directory
        :param path: str
        :return: None
        """
        sample_dir = os.path.join(self._level_dir[level_id], sample_id)
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
        self._change_to_sample_directory(level_sim.level_id, sample_id)
        self._copy_sim_files(level_sim.common_files)

    def schedule_sample(self, sample_id, level_sim):
        """
        Add sample to current PBS package
        :param sample_id: unique sample id from Sampler
        :param level_sim: LevelSimulation instance
        :return: None
        """
        self._create_level_workspace(level_sim.level_id)
        self.serialize_level_sim(level_sim)
        print("scheduled sample_id: {}, level_sim: {}".format(sample_id, level_sim))

        self._handle_sim_files(sample_id, level_sim)

        seed = self.compute_seed(sample_id)
        self._scheduled.append((level_sim.level_id, sample_id, seed))

        print("level_sim taks size ", level_sim.task_size)

        self._number_of_realizations += 1
        self._current_job_weight += level_sim.task_size
        if self._current_job_weight > self.job_weight or self._number_of_realizations > self.max_realizations:
            print("self.current job weight ", self._current_job_weight)

            print("self job weight ", self.job_weight)
            print("self number of realization ", self._number_of_realizations)
            print("self mas realizations ", self.max_realizations)
            self.execute()

    def have_permanent_sample(self, sample_id):
        """
        List of permanent samples or find per call?
        """
        
    def _qstat_pbs_job(self):
        """
        Parse qstat output and get all unfinished job ids
        :return: list of jobs ids (str)
        """
        # current user
        user = subprocess.run(["whoami"], stderr=subprocess.PIPE, stdout=subprocess.PIPE).stdout.decode("ascii").strip()

        # qstat call
        process = subprocess.run(["qstat", "-u", user], stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        if process.returncode != 0:
            raise Exception(process.stderr.decode("ascii"))

        output = process.stdout.decode("ascii")
        pbs_jobs = re.findall("(\d+)\.", output)

        return pbs_jobs

    def _parse_qsub_output(self, output):
        """
        Write pbs job id into file
        :param output: subprocess.CompletedProcess instance
        :return: None
        """
        stdout = output.stdout.decode("ascii")
        job_id = stdout.split(".")[0]
        self.write_pbs_id(job_id)

        self._qstat_pbs_job()

    def _create_script(self):
        # Job output with similar name to job
        self._pbs_config['job_name'] = "{:04d}".format(self._job_count)
        self._pbs_config['pbs_output_dir'] = self.jobs_dir
        self._pbs_config['files_structure'] = self.files_structure

        self.pbs_script = [line.format(**self._pbs_config) for line in self._pbs_header_template]

    def execute(self):
        """
        Execute pbs script
        :return: None
        """
        if len(self._scheduled) > 0:
            job_id = "{:04d}".format(self._job_count)

            print("execute scheduled ", self._scheduled)
            self.create_files(job_id)
            self.save_scheduled(self._scheduled)

            self._create_script()

            if self.pbs_script is None or self._number_of_realizations == 0:
                return

            # Write pbs script
            script_content = "\n".join(self.pbs_script)
            self.write_pbs_job_file(script_content)
            self._job_count += 1

            print("self job count ", self._job_count)

            #  @TODO: qsub command is required for PBS

            #subprocess.call(self.pbs_job_file)

            process = subprocess.run(['qsub', self.pbs_job_file], stderr=subprocess.PIPE, stdout=subprocess.PIPE)
            if process.returncode != 0:
                raise Exception(process.stderr.decode('ascii'))
            self._parse_qsub_output(process)

        # Clean script for other usage
        # self.clean_script()
        self._current_job_weight = 0
        self._number_of_realizations = 0
        self._scheduled = []

    def get_finished(self):
        """
        Get results
        :return:
        """
        self.execute()
        job_ids = self._qstat_pbs_job()
        successful, failed, n_running, times = self.get_result_files(job_ids)

        return successful, failed, n_running, times

    @property
    def jobs_dir(self):
        if self._jobs_dir is None:
            self._create_output_dir()
            self._create_job_dir()
        return self._jobs_dir

    @property
    def pbs_job_file(self):
        if self._job_file is None:
            raise FileNotFoundError
        return self._job_file

    # @property
    # def levels_config_file(self):
    #     if self._levels_config is None:
    #         raise FileNotFoundError
    #     return self._levels_config

    @property
    def files_structure(self):
        if self._files_structure is None:
            raise FileNotFoundError
        return self._files_structure

    def _create_job_dir(self):
        if self._work_dir is None:
            raise NotADirectoryError("Working directory must be pass to Sampler init")

        if self._jobs_dir is None:
            self._jobs_dir = os.path.join(self._output_dir, SamplingPoolPBS.JOBS_DIR)
            os.makedirs(self._jobs_dir, mode=0o775, exist_ok=True)

        self._levels_config = os.path.join(self._output_dir, SamplingPoolPBS.LEVELS_CONFIG)
        self._files_structure = os.path.join(self._output_dir, SamplingPoolPBS.STRUCTURE)
        self._save_structure()

    def serialize_level_sim(self, level_sim: LevelSimulation):
        file_path = os.path.join(self._level_dir[level_sim.level_id], SamplingPoolPBS.LEVEL_SIM_CONFIG)

        if not os.path.exists(file_path):
            with open(file_path, "wb") as f:
                pickle.dump(level_sim, f)

            with open(self._levels_config, "a") as w:
                w.write(file_path + '\n')

    def create_files(self, job_id):
        self._scheduled_file = os.path.join(self._jobs_dir, SamplingPoolPBS.SCHEDULED.format(job_id))
        self._pbs_id_file = os.path.join(self._jobs_dir, SamplingPoolPBS.PBS_ID.format(job_id))
        self._job_file = os.path.join(self._jobs_dir, SamplingPoolPBS.JOB.format(job_id))

    def write_pbs_id(self, pbs_job_id):
        self._pbs_id_file += pbs_job_id
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

    def write_pbs_job_file(self, content):
        with open(self._job_file, "w") as f:
            f.write(content)
        os.chmod(self._job_file, 0o774)

    def delete_pbs_id_file(self, file_path):
        """
        Delete jobId_pbsId file - it indicates finished job
        :param file_path: str
        :return: None
        """
        try:
            os.remove(file_path)
        except FileNotFoundError:
            print("Failed to remove PBS id file, file not found")

    def read_results(self, job_id):
        with open(os.path.join(self._jobs_dir, SamplingPoolPBS.RESULTS.format(job_id))) as reader:
            reader.readlines()

    def _save_structure(self):
        """
        Save structure of files needed for pbs process run
        :return: None
        """
        files_structure = {"scheduled": os.path.join(self._jobs_dir, SamplingPoolPBS.SCHEDULED),
                           "successful_results": SamplingPoolPBS.SUCCESSFUL_RESULTS,
                           "failed_results": SamplingPoolPBS.FAILED_RESULTS,
                           "time": SamplingPoolPBS.TIME,
                           "levels_config": os.path.join(self._output_dir, SamplingPoolPBS.LEVELS_CONFIG)}

        with open(self._files_structure, "w") as writer:
            json.dump(files_structure, writer)

    def get_result_files(self, unfinished_pbs_jobs):
        """
        Get results from files
        :param unfinished_pbs_jobs: PBS jobs id from qstat
        :return:
        """
        os.chdir(self.jobs_dir)
        # Find all files "jobId_pbsId"
        reg = "*_*[0-9]"
        file = glob.glob(reg)
        
        n_running = 0
        successful_results = {}
        failed_results = {}
        times = {}
        for f in file:
            job_id, pbs_id = re.findall("(\d+)_(\d+)", f)[0]

            # Get number of running samples
            if pbs_id in unfinished_pbs_jobs:
                with open(os.path.join(self._jobs_dir, SamplingPoolPBS.SCHEDULED.format(job_id))) as file:
                    lines = yaml.load(file, yaml.Loader)
                    n_running += len(lines)
                continue

            # Get sample results
            for level_id, path in self._level_dir.items():
                successful, failed, time = self._read_results(level_id, job_id)
                successful_results.setdefault(level_id, []).extend(successful)
                failed_results.setdefault(level_id, []).extend(failed)

                times[level_id] = time

            self.delete_pbs_id_file(f)

        return successful_results, failed_results, n_running, times

    def _read_results(self, level_id, job_id):
        """
        Read result file for given job id
        :param level_id: int
        :param job_id: str
        :return: successful: Dict[level_id, List[Tuple[sample_id:str, Tuple[ndarray, ndarray]]]]
                 failed: Dict[level_id, List[Tuple[sample_id: str, error message: str]]]
                time: Dict[level_id: int, List[total time: float, number of success samples: int]]
        """
        successful = {}
        failed = {}
        time = [0, 0]

        if os.path.exists(os.path.join(self._level_dir[level_id], SamplingPoolPBS.SUCCESSFUL_RESULTS.format(job_id))):
            with open(os.path.join(self._level_dir[level_id], SamplingPoolPBS.SUCCESSFUL_RESULTS.format(job_id)), "r") as reader:
                successful = yaml.load(reader, yaml.Loader)

        if os.path.exists(os.path.join(self._level_dir[level_id], SamplingPoolPBS.FAILED_RESULTS.format(job_id))):
            with open(os.path.join(self._level_dir[level_id], SamplingPoolPBS.FAILED_RESULTS.format(job_id)), "r") as reader:
                failed = yaml.load(reader, yaml.Loader)

        if os.path.exists(os.path.join(self._level_dir[level_id], SamplingPoolPBS.TIME.format(job_id))):
            with open(os.path.join(self._level_dir[level_id], SamplingPoolPBS.TIME.format(job_id)), "r") as reader:
                time = yaml.load(reader, yaml.Loader)

        return successful, failed, time

    # def save_to_storage(self):
    #     return {"work_dir": self._work_dir,
    #             "job_dir": self._jobs_dir}
    #
    # def get_from_storage(self):
    #     pass
