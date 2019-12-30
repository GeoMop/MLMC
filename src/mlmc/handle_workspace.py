import os
import shutil
from abc import ABC, abstractmethod
from typing import List


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


class WithoutWorkspace(Workspace):

    def change_to_sample_directory(self, sample_id, level_id=None):
        pass

    def copy_sim_files(self, files: List[str]):
        """
        Copy simulation common files to current simulation sample directory
        :param files: List of files
        :return:
        """


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


class PBSWorkspace(WholeWorkspace):

    def __init__(self, work_dir, n_levels):
        super().__init__(work_dir, n_levels)
        self._job_dir = None

        self._create_job_dir()

    def _create_job_dir(self):
        if self._work_dir is None:
            raise NotADirectoryError("Working directory must be pass to Sampler init")

        if self._job_dir is None:
            self._job_dir = os.path.join(self._output_dir, "jobs")
            os.makedirs(self._job_dir, mode=0o775, exist_ok=True)
