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
    # LEVEL_DIR = "level_{}"
    JOBS_DIR = "jobs"
    SCHEDULED = "{}_scheduled.yaml"
    SUCCESSFUL_RESULTS = "{}_successful_results.yaml"
    FAILED_RESULTS = "{}_failed_results.yaml"
    TIME = "{}_times.yaml"
    PBS_ID = "{}_"
    JOB = "{}_job.sh"
    LEVEL_SIM_CONFIG = "level_{}_simulation_config"
    #LEVELS_CONFIG = "levels_config.txt"
    STRUCTURE = "structure.json"

    def __init__(self, work_dir, job_weight=200000, job_count=0):
        """
        :param work_dir: Path to working directory
        :param job_weight: Maximum number of task sizes summation in single one job, if this value is exceeded then the job is executed
        :param job_count: Number of created jobs, it holds the current jobID, useful for regenerating of failed samples (jobs)
        """
        self._work_dir = work_dir
        # Working directory, other subdirectories are created in this one
        self.job_weight = job_weight
        # Weight of the single PBS script (putting more small jobs into single PBS job).
        self._current_job_weight = 0
        # Current collected weight.
        self._job_count = job_count
        # Number of executed jobs.
        self.max_realizations = 10
        # It is an alternative to job weight only that, this is maximum number of samples in job
        self._number_of_realizations = 0
        self.pbs_script = None
        # Lines to put at the beginning of the PBS script.
        self._pbs_config = None
        self._pbs_header_template = None
        self._scheduled = []
        self._level_dir = {}
        # Level directories

        self._pbs_ids = []
        # List of pbs job ids which should run

        self._output_dir = None
        self._jobs_dir = None
        self._scheduled_file = None
        self._pbs_id_file = None
        self._job_file = None
        #self._levels_config = None
        self._files_structure = None

        self._create_output_dir()
        self._create_job_dir()

    def _create_output_dir(self):
        """
        Create output dir in working directory, remove existing one
        :return: None
        """
        self._output_dir = os.path.join(self._work_dir, SamplingPoolPBS.OUTPUT_DIR)

        #  @TODO: remove
        if os.path.isdir(self._output_dir):
            shutil.rmtree(self._output_dir)

        os.makedirs(self._output_dir, mode=0o775, exist_ok=True)

    # def _create_level_workspace(self, level_id):
    #     """
    #     Create level directory
    #     :param level_id: Level identifier
    #     :return: None
    #     """
    #     if self._output_dir is not None:
    #         if not os.path.isdir(os.path.join(self._output_dir, SamplingPoolPBS.LEVEL_DIR.format(level_id))):
    #             level_dir = os.path.join(self._output_dir, SamplingPoolPBS.LEVEL_DIR.format(level_id))
    #             os.makedirs(level_dir, mode=0o775, exist_ok=True)
    #             self._level_dir[level_id] = level_dir
    #     else:
    #         raise NotADirectoryError("Create output directory at the first place")

    def pbs_common_setting(self, **kwargs):
        """
        Values for common header of script
        :param kwargs: dict with params vales
        :return: None
        """
        # Script header
        select_flags_list = kwargs.get('select_flags', [])
        if select_flags_list:
            kwargs['select_flags'] = ":" + ":".join(select_flags_list)
        else:
            kwargs['select_flags'] = ""

        print("kwargs ", kwargs)

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

        # @TODO: move to simulation
        self._pbs_header_template.extend(kwargs['modules'])

        self._pbs_header_template.extend((
                                         '{python} {pbs_process_file_dir}/pbs_process.py {files_structure} {job_name} >{pbs_output_dir}/{job_name}_STDOUT 2>&1',))
        self._pbs_config = kwargs

    def schedule_sample(self, sample_id, level_sim):
        """
        Add sample to current PBS package
        :param sample_id: unique sample id from Sampler
        :param level_sim: LevelSimulation instance
        :return: None
        """
        #self._create_level_workspace(level_sim.level_id)
        self.serialize_level_sim(level_sim)
        self._handle_sim_files(sample_id, level_sim)

        seed = self.compute_seed(sample_id)
        self._scheduled.append((level_sim.level_id, sample_id, seed))

        self._number_of_realizations += 1
        self._current_job_weight += level_sim.task_size
        if self._current_job_weight > self.job_weight or self._number_of_realizations > self.max_realizations:
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
        finished_pbs_jobs = []
        if len(self._pbs_ids) > 0:
            pbs_ids_str = ' '.join([str(id) for id in self._pbs_ids]).strip()

            # Get PBS id's status,
            # '-x' - displays status information for finished and moved jobs in addition to queued and running jobs.
            qstat_call = ["qstat", "-x"]
            qstat_call.extend(self._pbs_ids)

            # qstat call
            process = subprocess.run(qstat_call, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
            if process.returncode != 0:
                raise Exception(process.stderr.decode("ascii"))

            output = process.stdout.decode("ascii")
            finished_pbs_jobs = re.findall(r"(\d+)\..*\d+ F", output)

        unfinished_pbs_jobs = []
        for pbs_id in self._pbs_ids:
            if pbs_id not in finished_pbs_jobs:
                unfinished_pbs_jobs.append(pbs_id)

        return finished_pbs_jobs, unfinished_pbs_jobs

    def _parse_qsub_output(self, output):
        """
        Write pbs job id into file
        :param output: subprocess.CompletedProcess instance
        :return: None
        """
        stdout = output.stdout.decode("ascii")
        pbs_id = stdout.split(".")[0]

        # Store pbs id for qstat call
        self._pbs_ids.append(pbs_id)

        self.write_pbs_id(pbs_id)

    def _create_script(self):
        """
        Format pbs script
        :return: None
        """
        # Job output with similar name to job
        self._pbs_config['job_name'] = "{:04d}".format(self._job_count)
        self._pbs_config['pbs_output_dir'] = self._jobs_dir
        self._pbs_config['files_structure'] = self._files_structure

        self.pbs_script = [line.format(**self._pbs_config) for line in self._pbs_header_template]

    def execute(self):
        """
        Execute pbs script
        :return: None
        """
        if len(self._scheduled) > 0:
            job_id = "{:04d}".format(self._job_count)
            self.create_files(job_id)
            self.save_scheduled(self._scheduled)

            self._create_script()

            if self.pbs_script is None or self._number_of_realizations == 0:
                return

            # Write pbs script
            script_content = "\n".join(self.pbs_script)
            self.write_pbs_job_file(script_content)
            self._job_count += 1

            #  @TODO: qsub command is required for PBS


            #subprocess.call(self.pbs_job_file)

            process = subprocess.run(['qsub', self._job_file], stderr=subprocess.PIPE, stdout=subprocess.PIPE)
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
        finished_pbs_jobs, unfinished_pbs_jobs = self._qstat_pbs_job()
        successful, failed, n_running, times = self.get_result_files(finished_pbs_jobs, unfinished_pbs_jobs)

        return successful, failed, n_running, times

    def _create_job_dir(self):
        self._jobs_dir = os.path.join(self._output_dir, SamplingPoolPBS.JOBS_DIR)
        os.makedirs(self._jobs_dir, mode=0o775, exist_ok=True)

        #self._levels_config = os.path.join(self._output_dir, SamplingPoolPBS.LEVELS_CONFIG)
        self._files_structure = os.path.join(self._output_dir, SamplingPoolPBS.STRUCTURE)
        self._save_structure()

    def serialize_level_sim(self, level_sim: LevelSimulation):
        file_path = os.path.join(self._output_dir, SamplingPoolPBS.LEVEL_SIM_CONFIG.format(level_sim.level_id))

        if not os.path.exists(file_path):
            with open(file_path, "wb") as f:
                pickle.dump(level_sim, f)

            # with open(self._levels_config, "a") as w:
            #     w.write(file_path + '\n')

    def create_files(self, job_id):
        self._scheduled_file = os.path.join(self._jobs_dir, SamplingPoolPBS.SCHEDULED.format(job_id))
        self._pbs_id_file = os.path.join(self._jobs_dir, SamplingPoolPBS.PBS_ID.format(job_id))
        self._job_file = os.path.join(self._jobs_dir, SamplingPoolPBS.JOB.format(job_id))

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
                           "successful_results": os.path.join(self._jobs_dir, SamplingPoolPBS.SUCCESSFUL_RESULTS),
                           "failed_results": os.path.join(self._jobs_dir, SamplingPoolPBS.FAILED_RESULTS),
                           "time": os.path.join(self._jobs_dir, SamplingPoolPBS.TIME),
                           "level_sim_config": os.path.join(self._output_dir, SamplingPoolPBS.LEVEL_SIM_CONFIG)
                           }

        with open(self._files_structure, "w") as writer:
            json.dump(files_structure, writer)

    def get_result_files(self, finished_pbs_jobs, unfinished_pbs_jobs):
        """
        Get results from files
        :param finished_pbs_jobs: PBS jobs id from qstat
        :return:
        """
        os.chdir(self._jobs_dir)
        # Find all files "jobId_pbsId"
        # reg = "*_*[0-9]"
        # file = glob.glob(reg)

        # Get number of running samples
        print("UNFINISHED")

        n_running = 0
        for pbs_id in unfinished_pbs_jobs:
            reg = "*_{}".format(pbs_id)
            file = glob.glob(reg)
            print("file ", file)
            job_id = re.findall(r'(\d+)_\d+', file[0])[0]
            print("job id")
            with open(os.path.join(self._jobs_dir, SamplingPoolPBS.SCHEDULED.format(job_id))) as file:
                lines = yaml.load(file, yaml.Loader)
                n_running += len(lines)

        print("n running ", n_running)

        successful_results = {}
        failed_results = {}
        times = {}
        for pbs_id in finished_pbs_jobs:
            reg = "*_{}".format(pbs_id)
            file = glob.glob(reg)

            print("os.getcwd ", os.getcwd())
            print("os.lstat ", os.listdir(os.getcwd()))

            print("file ", file)

            if file:
                file = file[0]

                print("file", file)

                job_id = re.findall(r'(\d+)_\d+', file)[0]

                print("job_id ", job_id)

                # Get sample results
                
                successful, failed, time = self._read_results(job_id)
                successful_results.setdefault(level_id, []).extend(successful)
                failed_results.setdefault(level_id, []).extend(failed)

                times[level_id] = time

                self.delete_pbs_id_file(file)

        # @TODO: get sample ids for failed pbs jobs and add these samples to failed ones with particular error message

        return successful_results, failed_results, n_running, times

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

        if os.path.exists(os.path.join(self._jobs_dir, SamplingPoolPBS.SUCCESSFUL_RESULTS.format(job_id))):
            with open(os.path.join(self._jobs_dir, SamplingPoolPBS.SUCCESSFUL_RESULTS.format(job_id)), "r") as reader:
                successful = yaml.load(reader, yaml.Loader)

        if os.path.exists(os.path.join(self._jobs_dir, SamplingPoolPBS.FAILED_RESULTS.format(job_id))):
            with open(os.path.join(self._jobs_dir, SamplingPoolPBS.FAILED_RESULTS.format(job_id)), "r") as reader:
                failed = yaml.load(reader, yaml.Loader)

        if os.path.exists(os.path.join(self._jobs_dir, SamplingPoolPBS.TIME.format(job_id))):
            with open(os.path.join(self._jobs_dir, SamplingPoolPBS.TIME.format(job_id)), "r") as reader:
                time = yaml.load(reader, yaml.Loader)

        return successful, failed, time

    # def save_to_storage(self):
    #     return {"work_dir": self._work_dir,
    #             "job_dir": self._jobs_dir}
    #
    # def get_from_storage(self):
    #     pass
