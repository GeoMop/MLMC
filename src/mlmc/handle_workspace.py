import os
import shutil
from abc import ABC, abstractmethod
from typing import List
import yaml
import pickle
import json
import glob
import re
from level_simulation import LevelSimulation


class Workspace(ABC):

    def __init__(self, work_dir=None):
        self._work_dir = work_dir

    @abstractmethod
    def change_to_sample_directory(self, sample_id, level_id):
        pass

    @abstractmethod
    def copy_sim_files(self, files: List[str], sample_dir: str):
        """
        Copy simulation common files to current simulation sample directory
        :param files: List of files
        :return:
        """
        pass

    @abstractmethod
    def serialize_level_sim(self, level_sim: LevelSimulation):
        pass


class WithoutWorkspace(Workspace):

    def change_to_sample_directory(self, sample_id, level_id=None):
        pass

    def copy_sim_files(self, files: List[str]):
        """
        Copy simulation common files to current simulation sample directory
        :param files: List of files
        :return:
        """

    def serialize_level_sim(self, level_sim: LevelSimulation):
        pass


class SimulationWorkspace(Workspace):

    def change_to_sample_directory(self, path: str, level_id=None):
        sample_dir = os.path.join(self._work_dir, path)
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


class WholeWorkspace(Workspace):

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

            # with open(file_path, "rb") as f:
            #     l_sim = pickle.load(f)
            #     print("L SIM DES ", l_sim)

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

        for pbs_id in unfinished_pbs_jobs:
            reg = '*_{}'.format(pbs_id)
            file = glob.glob(reg)

            if not file:
                continue

            if len(file) > 1:
                raise Exception

            job_id = re.findall("(\d+)\_", file[0])

            if len(job_id) == 1:
                self._read_results(job_id[0])


    def _read_results(self, job_id):
        """
        Read result file for given job id
        :param job_id: str
        :return:
        """
        with open(PBSWorkspace.RESULTS.format(job_id), "r") as reader:
            results = reader.readlines()


