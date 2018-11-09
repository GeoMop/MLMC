import h5py
import numpy as np
import os
from mlmc.hdf.level_group import LevelGroup


class HDF5:
    # @TODO: comment hdf structure
    def __init__(self, work_dir, file_name="mlmc.hdf5", job_dir="scripts"):
        # Read/write if exists, create otherwise
        self._hdf_file = h5py.File(os.path.join(work_dir, file_name), 'a')

        self.work_dir = work_dir
        # Job directroy relative path
        self.job_dir = job_dir
        # Job directory absolute path
        self.job_dir_abs_path = os.path.join(work_dir, job_dir)

        self.n_levels = None
        self.step_range = None

    def load_from_file(self):
        # Set class attributes from hdf file
        for attr_name, value in self._hdf_file.attrs.items():
            self.__dict__[attr_name] = value

    def init_header(self, step_range, n_levels):
        """
        Create hdf5 file with basic structure
        :param step_range: MLMC level range of steps
        :param n_levels: Number of MLMC levels
        :return: None
        """
        self.step_range = step_range
        self.n_levels = n_levels

        # Set global attributes
        self._hdf_file.attrs['version'] = '1.0.1'
        self._hdf_file.attrs['work_dir'] = self.work_dir
        self._hdf_file.attrs['job_dir'] = self.job_dir
        self._hdf_file.attrs['step_range'] = step_range
        self._hdf_file.attrs.create("n_levels", n_levels, dtype=np.int8)
        # Create group Levels
        self._hdf_file.create_group("Levels")

    def add_level_group(self, level_id):
        """
        Create group for particular level, parent group is 'Levels'
        :param level_id: str, Level id
        :return: str, HDF5 path to level group
        """
        level_group_hdf_path = '/Levels/' + level_id
        # Create group if it has not yet been created
        if level_group_hdf_path not in self._hdf_file:
            # Create group for each level named by level id (e.g. 0, 1, 2, ...)
            self._hdf_file['Levels'].create_group(level_id)
            # Set level id as group attribute
        # @TODO: level step as attribute
        return LevelGroup(self._hdf_file[level_group_hdf_path], level_id, job_dir=self.job_dir_abs_path)

    # @property
    # def job_dir(self):
    #     """
    #     Get os path to job scripts directory
    #     :return: str; path
    #     """
    #     return os.path.join(self.work_dir, self._job_dir)
    #
    # @job_dir.setter
    # def job_dir(self, job_dir):
    #     self._job_dir = job_dir
