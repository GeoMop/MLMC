import os
import shutil
from abc import ABC, abstractmethod
from typing import List
import yaml
import pickle
from src.mlmc.level_simulation import LevelSimulation


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
            self._output_dir = os.path.join(self._work_dir, "output_{}".format(self._n_levels))
            os.makedirs(self._output_dir, mode=0o775, exist_ok=True)

    def _create_level_workspace(self):
        if self._output_dir is not None:
            for level_id in range(self._n_levels):
                level_dir = os.path.join(self._output_dir, "level_{}".format(level_id))
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

    SCHEDULED = "{}_scheduled.yaml"
    RESULTS = "{}_results.yaml"
    PBS_ID = "{}_pbs_id"
    JOB = "{}_job.sh"
    LEVEL_SIM_CONFIG = "level_simulation_config"

    def __init__(self, work_dir, n_levels):
        super().__init__(work_dir, n_levels)

        self._jobs_dir = None
        self._scheduled_file = None
        self._results_file = None
        self._pbs_id_file = None
        self._job_file = None

        self._create_job_dir()

    def _create_output_dir(self):
        if self._work_dir is None:
            raise NotADirectoryError("Working directory must be pass to Sampler init")

        if self._output_dir is None:

            self._output_dir = os.path.join(self._work_dir, "output_{}".format(self._n_levels))

            #  @TODO: remove
            if os.path.isdir(self._output_dir):
                shutil.rmtree(self._output_dir)

            os.makedirs(self._output_dir, mode=0o775, exist_ok=True)

    def _create_job_dir(self):
        if self._work_dir is None:
            raise NotADirectoryError("Working directory must be pass to Sampler init")

        if self._jobs_dir is None:
            self._jobs_dir = os.path.join(self._output_dir, "jobs")
            os.makedirs(self._jobs_dir, mode=0o775, exist_ok=True)

    def serialize_level_sim(self, level_sim: LevelSimulation):
        file_path = os.path.join(self._level_dir[level_sim.level_id], PBSWorkspace.LEVEL_SIM_CONFIG)

        print("file_path ", file_path)

        if not os.path.exists(file_path):
            print("PICKLE")
            with open(file_path, "wb") as f:
                pickle.dump(level_sim, f)

            with open(file_path, "rb") as f:
                l_sim = pickle.load(f)

    def create_files(self, job_id):
        self._scheduled_file = os.path.join(self._jobs_dir, PBSWorkspace.SCHEDULED.format(job_id))
        self._results_file = os.path.join(self._jobs_dir, PBSWorkspace.RESULTS.format(job_id))
        self._pbs_id_file = os.path.join(self._jobs_dir, PBSWorkspace.PBS_ID.format(job_id))
        self._job_file = os.path.join(self._jobs_dir, PBSWorkspace.JOB.format(job_id))

    def _save_scheduled(self, scheduled):
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

