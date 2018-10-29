import h5py
import numpy as np
import os

# One sample data type for dataset scheduled
SAMPLE_DTYPE = {'names': ('dir', 'job_id', 'prepare_time', 'queued_time'),
                'formats': ('S100', 'S5', 'f8', 'f8')}
# Row format for dataset scheduled
SCHEDULED_DTYPE = {'names': ('fine_sample', 'coarse_sample', 'status'),
                   'formats': (SAMPLE_DTYPE, SAMPLE_DTYPE, 'S10')}

# Data that is collected, only this data can by saved to datasets
# {attribute name in class Sample : {name: dataset name, maxshape: dataset maxshape, dtype: dataset dtype}}
COLLECTED_ATTRS = {"sample_id": {'name': 'collected_ids', 'maxshape': (None, 1), 'dtype': np.int16},
                   "result": {'name': 'collected_values', 'maxshape': (None, 2, None), 'dtype': np.float16},
                   "time": {'name': 'collected_times',  'maxshape': (None, 2, None), 'dtype': np.float16}}


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class HDF5(metaclass=Singleton):
    #@TODO: base_strucure() params as kwargs
    def __init__(self, work_dir, step_range, n_levels, file_name="mlmc.hdf5", job_dir="scripts"):
        # HDF5 file name
        self._file_name = os.path.join(work_dir, file_name)
        # Create file with basic structure
        self.base_structure(work_dir, step_range, job_dir, n_levels)
        # Hdf file object with mode append
        self._hdf_file_append = h5py.File(self._file_name, 'a')
        self._hdf_file_reader = h5py.File(self._file_name, 'r')

    def base_structure(self, work_dir, step_range, job_dir, n_levels):
        """
        Create hdf5 file with basic structure
        :param work_dir: 
        :param step_range: 
        :param job_dir:
        :param n_levels:
        :return: 
        """
        #TODO: set attributes at once (from dict)
        with h5py.File(self._file_name, "w") as writer:
            # Set global attributes
            writer.attrs['version'] = '1.0.1'
            writer.attrs['work_dir'] = work_dir
            writer.attrs['job_dir'] = job_dir
            writer.attrs.create("n_levels", n_levels, dtype=np.int8)
            # Create groups Levels and set step range attribute
            writer.create_group("Levels")
            writer["Levels"].attrs.create("step_range", step_range, dtype=np.float16)

    def create_level_group(self, level_id):
        """
        Create group for particular level, parent group is 'Levels'
        :param level_id: str, Level id
        :return: str, level group path
        """
        level_group_path = '/Levels/' + level_id
        # Create group if it has not yet been created
        if level_group_path not in self._hdf_file_append:
            # Create group for each level named by level id (e.g. 0, 1, 2, ...)
            self._hdf_file_append['Levels'].create_group(level_id)
            # Set level id as group attribute
            #riter[level_group_path].create_group("Jobs")
            self._hdf_file_append[level_group_path].attrs.create("id", level_id, dtype=np.int8)
        #@TODO: level step as attribute

        return level_group_path

    def set_n_ops_estimate(self, level_group_path, n_ops_estimate):
        """
        Set n ops estimate (simulation attribute)
        :param level_group_path: Path to level group 
        :param n_ops_estimate: float
        :return: None
        """
        self._hdf_file_append[level_group_path].attrs['n_ops_estimate'] = float(n_ops_estimate)

    # def set_attributes(self, attributes):
    #     """
    #     Create hdf file with default metadata
    #     :param hdf_file:
    #     :param attributes:
    #     :return:
    #     """
    #     with h5py.File(self._file_name, 'w') as writer:
    #         for name, value in attributes.items():
    #             writer.attrs[name] = value

    def save_scheduled(self, level_group_path, scheduled_samples):
        """
        Save scheduled sample to dataset
        :param level_group_path: Path to level group in hdf file
        :param scheduled_samples: list of sample objects [[fine_sample, coarse_sample]]
        :return: None
        """
        # Creates a scheduled samples format acceptable for hdf
        sim_hdf_form = []
        # Jobs {job_id: [sample_id, ...]}
        jobs = {}
        # Go through scheduled samples and format them
        for fine_sample, coarse_sample in scheduled_samples:
            sim_hdf_form.append((fine_sample.scheduled_saved_data(), coarse_sample.scheduled_saved_data(), "RUN"))

            # Append sample id to job id
            jobs.setdefault(fine_sample.job_id, []).append(fine_sample.sample_id)
            jobs.setdefault(coarse_sample.job_id, []).append(coarse_sample.sample_id)

        #  Create dataset scheduled
        if not level_group_path + '/scheduled' in self._hdf_file_append:
            self._create_scheduled_dataset(level_group_path, sim_hdf_form)
        # Dataset scheduled already exists
        else:
            self._append_dataset(level_group_path + '/scheduled', sim_hdf_form)

        # Save to jobs to datasets
        self._save_jobs(level_group_path, jobs)

    def _save_jobs(self, level_group_path, jobs):
        """
        Create jobs datasets and save there sample id
        :param level_group_path: Path to level dataset
        :param jobs: dict, {job_id: [sample_id,...]}
        :return: None
        """
        for job_name, job_samples in jobs.items():
            job_dataset_path = level_group_path + '/Jobs/' + job_name
            if job_dataset_path not in self._hdf_file_append:
                self._hdf_file_append.create_dataset(job_dataset_path, data=job_samples, maxshape=(None,), chunks=True)
            else:
                self._append_dataset(job_dataset_path, job_samples)

    def _create_scheduled_dataset(self, level_group_path, scheduled_samples):
        """
        Create dataset for scheduled samples
        :param level_group_path: Path to level group in hdf file
        :param scheduled_samples: list of sample objects [[fine_sample, coarse_sample]]
        :return: None
        """
        # Create dataset with SCHEDULED_DTYPE and with unlimited max shape (required for resize)
        scheduled_dset = self._hdf_file_append[level_group_path].\
            create_dataset("scheduled", shape=(len(scheduled_samples),), maxshape=(None,), dtype=SCHEDULED_DTYPE,
                           chunks=True)

        # Set values to dataset
        scheduled_dset[:] = scheduled_samples

    def save_collected(self, level_group_path, collected_samples):
        """
        Save level collected data to datasets corresponding to COLLECTED_ATTRS
        :param level_group_path: Path to particular level
        :param collected_samples: Level sample [[fine sample, coarse sample]], both are Sample() object instances
        :return: None
        """
        # Init collected_data as dict {attribute_name: [], ...}
        collected_data = {attr: [] for attr in COLLECTED_ATTRS.keys()}

        # Format fine sample and coarse sample values as tuple (fine value, coarse value)
        # Add tuples to corresponding collected attributes
        for fine_sample, coarse_sample in collected_samples:
            fine_coarse_data = self.sample_pairs(fine_sample, coarse_sample)
            [collected_data[key].append(value) for key, value in fine_coarse_data.items()]

        # Create own dataset for each collected attribute
        for attr_name, data in collected_data.items():
            data = np.array(data)
            # Data shape
            shape = data.shape
            # Maxshape as a constant
            maxshape = COLLECTED_ATTRS[attr_name]['maxshape']

            # Data is squeezed, so expand last dimension to 'maxshape' shape
            if len(shape) == len(maxshape) - 1:
                data = np.expand_dims(data, axis=len(maxshape) - 1)
                shape = data.shape

            # Path to current dataset
            dataset_path = os.path.join(level_group_path, COLLECTED_ATTRS[attr_name]['name'])
            # Dataset already exists, append new values
            if dataset_path in self._hdf_file_append:
                self._append_dataset(dataset_path, data)
                continue

            # Create resizable dataset
            dataset = self._hdf_file_append[level_group_path].create_dataset(COLLECTED_ATTRS[attr_name]['name'],
                                                                             shape=shape,
                                                                             maxshape=maxshape,
                                                                             dtype=COLLECTED_ATTRS[attr_name]['dtype'],
                                                                             chunks=True)
            # Set data
            dataset[:] = data

    def sample_pairs(self, fine_sample, coarse_sample):
        """
        Get fine sample and coarse sample collected values
        :param fine_sample: Fine sample
        :param coarse_sample: Coarse sample
        :return: dict {sample_id: id, key: (fine value, coarse value)}
        """
        # Dictionary with attributes names and values for fine and coarse sample
        f_dict = fine_sample.collected_data(COLLECTED_ATTRS.keys())
        c_dict = coarse_sample.collected_data(COLLECTED_ATTRS.keys())

        # Merge fine and coarse key values to tuple (fine, coarse)
        merge = {}
        for k in f_dict:
            # Sample id is the only exception, just one value
            if k == 'sample_id':
                merge[k] = f_dict[k]
            else:
                merge[k] = (f_dict.get(k), c_dict.get(k))

        return merge

    def _append_dataset(self, dataset_path, data):
        """
        Append data to existing dataset
        :param dataset_path: Path to dataset
        :param data: list of data
        :return: None
        """
        # Resize dataset
        self._hdf_file_append[dataset_path].resize(self._hdf_file_append[dataset_path].shape[0] +
                                                   len(data), axis=0)
        # Append new samples
        self._hdf_file_append[dataset_path][-len(data):] = data

    def read_level(self, level_path):
        """
        Read level datasets with collected data
        :param level_path: Path to level group
        :return: 
        """
        if 'collected_ids' not in self._hdf_file_reader[level_path]:
            return

        #@TODO: read datasets according to COLLECTED_ATTRS and create samples
