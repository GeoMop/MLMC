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
        :param sample_dir: Sample directory
        :return:
        """
        pass

    @abstractmethod
    def serialize_level_sim(self, level_sim: LevelSimulation):
        pass

    @abstractmethod
    def save_to_storage(self):
        """
        Define dictionary of variables which should by saved in sample storage
        :return:
        """


class WithoutWorkspace(Workspace):

    def change_to_sample_directory(self, sample_id, level_id=None):
        pass

    def copy_sim_files(self, files: List[str], ):
        """
        Copy simulation common files to current simulation sample directory
        :param files: List of files
        :return:
        """

    def serialize_level_sim(self, level_sim: LevelSimulation):
        pass

    def save_to_storage(self):
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

    def save_to_storage(self):
        pass




