import os
import shutil
import subprocess
import re
import glob
import json
import yaml
from typing import List
import pickle
from mlmc.level_simulation import LevelSimulation
from sampling_pool import SamplingPool


class SamplingPoolPBS(SamplingPool):

    def __init__(self, job_weight=200000, job_count=0):
        """
        :param job_weight: Number of simulation elements per job script
        :param job_count: Number of created jobs
        """
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

        self.need_workspace = True
        self.need_jobs = True

        self.workspace = None

        self.qsub_cmd = None

        self._scheduled = []

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
                                     '#PBS -l walltime=4:00:00',
                                     '#PBS -q {queue}',
                                     '#PBS -N Flow123d',
                                     '#PBS -j oe',
                                     '#PBS -o {pbs_output_dir}/{job_name}.OU',
                                     '#PBS -e {pbs_output_dir}/{job_name}.ER',
                                     '']
        #  @TODO: prepare environment
        if flow_3:
            self._pbs_header_template.extend(('module use /storage/praha1/home/jan-hybs/modules',
                                              'module load python36-modules-gcc'
                                              'module load flow123d', ''))

        self._pbs_header_template.extend((
                                         'python3 {pbs_process_file_dir}/pbs_process.py {files_structure} {job_name} >{pbs_output_dir}/{job_name}_STDOUT 2>&1',))
        self._pbs_config = kwargs

    def schedule_sample(self, sample_id, level_sim):
        """
        Add sample to current PBS package
        :param sample_id: unique sample id from Sampler
        :param level_sim: LevelSimulation instance
        :return: None
        """
        # @TODO: try to not pass sample id
        level_sim.config_dict["sample_id"] = sample_id

        self._sampling_pool.serialize_level_sim(level_sim)

        self._scheduled.append((level_sim.level_id, sample_id))
        self._number_of_realizations += 1
        self._current_job_weight += level_sim.task_size
        if self._current_job_weight > self.job_weight or self._number_of_realizations > self.max_realizations:
            self.execute()

    def have_permanent_sample(self, sample_id):
        """
        List of permanent samples or find per call?
        """
        pass

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
        self.workspace.write_pbs_id(job_id)

        # TODO: remove
        self._qstat_pbs_job()

    def _create_script(self):
        # Job output with similar name to job
        self._pbs_config['job_name'] = "{:04d}".format(self._job_count)
        self._pbs_config['pbs_output_dir'] = self.workspace.jobs_dir
        self._pbs_config['files_structure'] = self.workspace.files_structure

        self.pbs_script = [line.format(**self._pbs_config) for line in self._pbs_header_template]

    def execute(self):
        """
        Execute pbs script
        :return: None
        """
        job_id = "{:04d}".format(self._job_count)
        self.workspace.create_files(job_id)
        self.workspace.save_scheduled(self._scheduled)

        if self.pbs_script is None:
            self._create_script()

        if self.pbs_script is None or self._number_of_realizations == 0:
            return

        # Write pbs script
        script_content = "\n".join(self.pbs_script)
        self.workspace.write_pbs_job_file(script_content)
        self._job_count += 1

        #  @TODO: qsub command is required for PBS

        process = subprocess.run(['qsub', self.workspace.pbs_job_file], stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        if process.returncode != 0:
            raise Exception(process.stderr.decode('ascii'))

        self._parse_qsub_output(process)

        # Clean script for other usage
        # self.clean_script()
        self._current_job_weight = 0
        self._number_of_realizations = 0

    def get_finished(self):
        """
        Get results
        :return:
        """
        self.execute()
        job_ids = self._qstat_pbs_job()
        results, n_running = self.workspace.get_result_files(job_ids)
        return results, n_running


class WholeWorkspace:
    OUTPUT_DIR = "output_{}"
    LEVEL_DIR = "level_{}"

    def __init__(self, work_dir, n_levels):
        self._work_dir = work_dir
        self._output_dir = None
        self._n_levels = n_levels
        self._level_dir = []

        self._create_output_dir()
        self._create_level_workspace()

    def _create_output_dir(self):
        if self._work_dir is None:
            raise NotADirectoryError("Working directory must be pass to Sampler init")

        if self._output_dir is None:
            self._output_dir = os.path.join(self._work_dir, WholeWorkspace.OUTPUT_DIR.format(self._n_levels))
            os.makedirs(self._output_dir, mode=0o775, exist_ok=True)

    def _create_level_workspace(self):
        if self._output_dir is not None:
            for level_id in range(self._n_levels):
                level_dir = os.path.join(self._output_dir, WholeWorkspace.LEVEL_DIR.format(level_id))
                os.makedirs(level_dir, mode=0o775, exist_ok=True)
                self._level_dir.append(level_dir)
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

    def serialize_level_sim(self, level_sim: LevelSimulation):
        pass

    def save_to_storage(self):
        pass




class PBSWorkspace(WholeWorkspace):
    JOBS_DIR = "jobs"
    SCHEDULED = "{}_scheduled.yaml"
    RESULTS = "{}_results.yaml"
    PBS_ID = "{}_"
    JOB = "{}_job.sh"
    LEVEL_SIM_CONFIG = "level_simulation_config"
    LEVELS_CONFIG = "levels_config.txt"
    STRUCTURE = "structure.json"

    def __init__(self, work_dir, n_levels):
        super().__init__(work_dir, n_levels)

        self._jobs_dir = None
        self._scheduled_file = None
        self._results_file = None
        self._pbs_id_file = None
        self._job_file = None
        self._levels_config = None
        self._files_structure = None

        self._create_job_dir()

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

    def _create_output_dir(self):
        if self._work_dir is None:
            raise NotADirectoryError("Working directory must be pass to Sampler init")

        if self._output_dir is None:

            self._output_dir = os.path.join(self._work_dir, PBSWorkspace.OUTPUT_DIR.format(self._n_levels))

            #  @TODO: remove
            if os.path.isdir(self._output_dir):
                shutil.rmtree(self._output_dir)

            os.makedirs(self._output_dir, mode=0o775, exist_ok=True)

    def _create_job_dir(self):
        if self._work_dir is None:
            raise NotADirectoryError("Working directory must be pass to Sampler init")

        if self._jobs_dir is None:
            self._jobs_dir = os.path.join(self._output_dir, PBSWorkspace.JOBS_DIR)
            os.makedirs(self._jobs_dir, mode=0o775, exist_ok=True)

        self._levels_config = os.path.join(self._output_dir, PBSWorkspace.LEVELS_CONFIG)
        self._files_structure = os.path.join(self._output_dir, PBSWorkspace.STRUCTURE)
        self._save_structure()

    def serialize_level_sim(self, level_sim: LevelSimulation):
        file_path = os.path.join(self._level_dir[level_sim.level_id], PBSWorkspace.LEVEL_SIM_CONFIG)

        if not os.path.exists(file_path):
            with open(file_path, "wb") as f:
                pickle.dump(level_sim, f)

            with open(self._levels_config, "a") as w:
                w.write(file_path + '\n')

    def create_files(self, job_id):
        self._scheduled_file = os.path.join(self._jobs_dir, PBSWorkspace.SCHEDULED.format(job_id))
        self._results_file = os.path.join(self._jobs_dir, PBSWorkspace.RESULTS.format(job_id))
        self._pbs_id_file = os.path.join(self._jobs_dir, PBSWorkspace.PBS_ID.format(job_id))
        self._job_file = os.path.join(self._jobs_dir, PBSWorkspace.JOB.format(job_id))

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

        # with open(self._scheduled_file) as file:
        #     documents = yaml.load(file)
        #     print("doc ", documents)

    def write_pbs_job_file(self, content):
        with open(self._job_file, "w") as f:
            f.write(content)

        os.chmod(self._job_file, 0o774)

    def delete_pbs(self):
        try:
            os.remove(self._pbs_id_file)
        except FileNotFoundError:
            print("Failed to remove PBS id file, file not found")

    def read_results(self, job_id):
        with open(os.path.join(self._jobs_dir, PBSWorkspace.RESULTS.format(job_id))) as reader:
            reader.readlines()

    def _save_structure(self):
        """
        Save structure of files needed for pbs process run
        :return: None
        """
        files_structure = {"scheduled": os.path.join(self._jobs_dir, PBSWorkspace.SCHEDULED),
                           "results": os.path.join(self._jobs_dir, PBSWorkspace.RESULTS),
                           "levels_config": os.path.join(self._output_dir, PBSWorkspace.LEVELS_CONFIG)}

        with open(self._files_structure, "w") as writer:
            json.dump(files_structure, writer)

    def get_result_files(self, unfinished_pbs_jobs):
        """
        Get results from files
        :param unfinished_pbs_jobs: PBS jobs id from qstat
        :return:
        """
        os.chdir(self.jobs_dir)

        reg = "*_*[0-9]"
        file = glob.glob(reg)

        n_running = 0
        results = []
        for f in file:
            job_id, pbs_id = re.findall("(\d+)_(\d+)", f)[0]

            if pbs_id in unfinished_pbs_jobs:
                print("scheduled file ", self._scheduled_file)
                with open(self._scheduled_file) as file:
                    lines = yaml.load(file)
                    n_running += len(lines)
                continue

            res = self._read_results(job_id)
            results.extend(res)

        return results, n_running

    def _read_results(self, job_id):
        """
        Read result file for given job id
        :param job_id: str
        :return:
        """
        with open(os.path.join(self._jobs_dir, PBSWorkspace.RESULTS.format(job_id)), "r") as reader:
            results = yaml.load(reader)

        return results

    def save_to_storage(self):
        return {"work_dir": self._work_dir,
                "job_dir": self._jobs_dir}

    def get_from_storage(self):
        pass
