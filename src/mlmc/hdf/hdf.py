import h5py
import numpy as np
import os
from mlmc.hdf.level_group import LevelGroup


class HDF5:
    """
    HDF5 file is organized into groups (h5py.Group objects)
    which is somewhat like dictionaries in python terminology - 'keys' are names of group members
    'values' are members (groups (h5py.Group objects) and datasets (h5py.Dataset objects - similar to NumPy arrays)).
    Each group and dataset (including root group) can store metadata in 'attributes' (h5py.AttributeManager objects)
    HDF5 files (h5py.File) work generally like standard Python file objects
    
    Our HDF5 file strucutre:
        Main Group:
            Attributes:
                work_dir: directory where HDF5 file was created (other paths are relative to this one)
                job_dir: path containing all pbs_scripts of individual jobs, relative to work_dir
                n_levels: number of levels (dtype=numpy.int8)
        Keys:
            Levels: h5py.Group
                Attributes:
                    step_range: [a, b]
                Keys:
                    <N>: h5py.Group (N - level id, start with 0)
                        Attributes:
                            id: str
                            n_ops_estimate: float
                        Keys:
                            Jobs: h5py.Group
                                Keys:
                                    <job_id>: h5py.Dataset
                                    dtype: int
                                    maxshape: (None, 1)
                                    chunks: True
                            scheduled: h5py.Dataset
                                dtype: structured array 
                                    Struct: (fine: SampleStruct, coarse: SampleStruct)
                                        SampleStruct:
                                                dir - simulation directory
                                                job - job_ID
                                                prepare_time - duration of creating the sample
                                                queued_time - absolute time when the sample was planned             
                                shape: (N, 1), N - number of scheduled values
                                maxshape: (None, 1)
                                chunks: True
                            collected_values: h5py.Dataset
                                dtype: numpy.float64
                                shape: (Nc, 2, M) double… TODO: table of values
                                maxshape: (None, 2, None)
                                chunks: True
                            collected_ids: h5py.Dataset
                                dtype: numpy.int16  index into scheduled
                                shape: (Nc, 1) 
                                maxshape: (None, 1)
                                chunks: True
                            collected_times: h5py.Dataset
                                dtype: numpy.float64
                                shape (Nc, 2, T), T … different kind of times
                                maxshape (None, 2, None)
                                chunks: True
                            failed_ids: h5py.Dataset 
                                dtype: int … index into scheduled
                                shape: (Nf, 1)
                                mashape: (None, 1)
                                chunks: True 
    """

    def __init__(self, work_dir, file_name="mlmc.hdf5", job_dir="scripts"):
        """
        Create HDF5 class instance
        :param work_dir: absolute path to MLMC work directory
        :param file_name: Name of mlmc HDF5 file
        :param job_dir: pbs job scripts directory, relative path to work_dir
        """
        # Work directory abs path
        self.work_dir = work_dir
        # Absolute path to mlmc HDF5 file
        self.file_name = os.path.join(work_dir, file_name)
        # Job directroy relative path
        self.job_dir = job_dir
        # Job directory absolute path
        self.job_dir_abs_path = os.path.join(work_dir, job_dir)

        # h5py.File object - works generally like standard Python file objects, ppen with mode append
        self._hdf_file = h5py.File(self.file_name, 'a')

        # Class attributes necessary for mlmc
        self.n_levels = None
        self.step_range = None

    def load_from_file(self):
        """
        Load root group attributes from existing HDF5 file
        :return: None
        """
        # Set class attributes from hdf file
        for attr_name, value in self._hdf_file.attrs.items():
            self.__dict__[attr_name] = value

    def clear_groups(self):
        """
        Remove HDF5 group Levels, it allows run same mlmc object more times
        :return: None
        """
        for item in self._hdf_file.keys():
            del self._hdf_file[item]

    def init_header(self, step_range, n_levels):
        """
        Add h5py.File metadata to .attrs (attrs objects are of class h5py.AttributeManager)
        :param step_range: MLMC level range of steps
        :param n_levels: Number of MLMC levels
        :return: None
        """
        # Set mlmc attributes
        self.step_range = step_range
        self.n_levels = n_levels

        # Set global attributes to root group (h5py.Group)
        self._hdf_file.attrs['version'] = '1.0.1'
        self._hdf_file.attrs['work_dir'] = self.work_dir
        self._hdf_file.attrs['job_dir'] = self.job_dir
        self._hdf_file.attrs['step_range'] = step_range
        self._hdf_file.attrs.create("n_levels", n_levels, dtype=np.int8)

        # Create h5py.Group Levels, it contains other groups with mlmc.Level data
        self._hdf_file.create_group("Levels")

    def add_level_group(self, level_id):
        """
        Create group for particular level, parent group is 'Levels'
        :param level_id: str, mlmc.Level identifier
        :return: LevelGroup instance, it is container for h5py.Group instance
        """
        # HDF5 path to particular level group
        level_group_hdf_path = '/Levels/' + level_id
        # Create group (h5py.Group) if it has not yet been created
        if level_group_hdf_path not in self._hdf_file:
            # Create group for level named by level id (e.g. 0, 1, 2, ...)
            self._hdf_file['Levels'].create_group(level_id)

        return LevelGroup(self._hdf_file[level_group_hdf_path], level_id, job_dir=self.job_dir_abs_path)
