import h5py
import numpy as np
import os
from mlmc.sample import Sample


class HDF5:
    # One sample data type for dataset scheduled
    SAMPLE_DTYPE = {'names': ('dir', 'job_id', 'prepare_time', 'queued_time'),
                    'formats': ('S100', 'S5', 'f8', 'f8')}
    # Row format for dataset scheduled
    SCHEDULED_DTYPE = {'names': ('fine_sample', 'coarse_sample', 'status'),
                       'formats': (SAMPLE_DTYPE, SAMPLE_DTYPE, 'S10')}

    # Data that is collected, only this data can by saved to datasets
    # @TODO change docstring
    # {attribute name in class Sample : {name: dataset name, maxshape: dataset maxshape, dtype: dataset dtype}}
    COLLECTED_ATTRS = {"sample_id": {'name': 'collected_ids', 'maxshape': (None, 1), 'dtype': np.int16},
                       "result": {'name': 'collected_values', 'maxshape': (None, 2, None), 'dtype': np.float16},
                       "time": {'name': 'collected_times', 'maxshape': (None, 2, None), 'dtype': np.float16}}

    DATASET_NAMES = {"scheduled": "scheduled",
                     "collected_ids": "collected_ids"}

    # @TODO: comment hdf structure
    # @TODO: base structure params in form of dictionary
    def __init__(self, work_dir, step_range, n_levels, file_name="mlmc.hdf5", job_dir="scripts"):
        # HDF5 file name
        self._file_name = os.path.join(work_dir, file_name)
        # Create file with basic structure
        if not os.path.exists(self._file_name):
            self.base_structure(work_dir, step_range, job_dir, n_levels)
        # Hdf file object with mode append
        self._hdf_file_append = h5py.File(self._file_name, 'a')
        # Hdf file object with read mode
        self._hdf_file_reader = h5py.File(self._file_name, 'r')

        # HDF absolute path to level groups {level_id: path in hdf file}
        self._level_group_hdf_paths = {}

    def base_structure(self, work_dir, step_range, job_dir, n_levels):
        """
        Create hdf5 file with basic structure
        :param work_dir: Path to working directory
        :param step_range: MLMC level range of steps
        :param job_dir: Name of directory with pbs jobs data, abs path is work_dir/job_dir
        :param n_levels: Number of MLMC levels
        :return: None
        """
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
        :return: str, HDF5 path to level group
        """
        level_group_hdf_path = '/Levels/' + level_id
        self._level_group_hdf_paths[level_id] = level_group_hdf_path
        # Create group if it has not yet been created
        if level_group_hdf_path not in self._hdf_file_append:
            # Create group for each level named by level id (e.g. 0, 1, 2, ...)
            self._hdf_file_append['Levels'].create_group(level_id)
            # Set level id as group attribute
            self._hdf_file_append[level_group_hdf_path].attrs.create("id", level_id, dtype=np.int8)
        # @TODO: level step as attribute

        return level_group_hdf_path

    # @TODO: create property and setter for n_ops_estimate, or rather general method for getting and setting level group attributes
    def set_n_ops_estimate(self, level_id, n_ops_estimate):
        """
        Set n ops estimate (simulation attribute)
        :param level_id: Level id 
        :param n_ops_estimate: float
        :return: None
        """
        self._hdf_file_append[self._level_group_hdf_paths[level_id]].attrs['n_ops_estimate'] = float(n_ops_estimate)

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

    def save_scheduled(self, level_id, scheduled_samples):
        """
        Save scheduled sample to dataset
        :param level_id: Level id
        :param scheduled_samples: list of sample objects [(fine_sample, coarse_sample)]
        :return: None
        """
        level_group_hdf_path = self._level_group_hdf_paths[level_id]
        # Creates a scheduled samples format acceptable for hdf
        sim_hdf_form = []
        # Jobs {job_id: [sample_id, ...]}
        jobs = {}

        # Go through scheduled samples and format them
        for sample_id, (fine_sample, coarse_sample) in scheduled_samples.items():
            # @TODO: status ("RUN") is not needed now - discuss
            sim_hdf_form.append((fine_sample.scheduled_data(), coarse_sample.scheduled_data(), "RUN"))
            # Append sample id to job id
            jobs.setdefault(fine_sample.job_id, set()).add(fine_sample.sample_id)
            if int(level_id) != 0:
                jobs.setdefault(coarse_sample.job_id, set()).add(coarse_sample.sample_id)

        # Create dataset scheduled
        scheduled_dset_hdf_path = "/".join([level_group_hdf_path, HDF5.DATASET_NAMES['scheduled']])
        if scheduled_dset_hdf_path not in self._hdf_file_append:
            self._create_scheduled_dataset(level_group_hdf_path, sim_hdf_form)
        # Dataset scheduled already exists
        else:
            self._append_dataset(scheduled_dset_hdf_path, sim_hdf_form)

        # Save to jobs to datasets
        self._save_jobs(level_group_hdf_path, jobs)

    def _save_jobs(self, level_group_hdf_path, jobs):
        """
        Save jobs to datasets
        :param level_group_hdf_path: HDF5 path to level group
        :param jobs: dict, {job_id: [sample_id,...], ...}
        :return: None
        """
        for job_name, job_samples in jobs.items():
            job_dataset_path = '/'.join([level_group_hdf_path, 'Jobs', job_name])
            if job_dataset_path not in self._hdf_file_append:
                self._hdf_file_append.create_dataset(job_dataset_path, data=list(job_samples), maxshape=(None,), chunks=True)
            else:
                self._append_dataset(job_dataset_path, list(job_samples))

    def _create_scheduled_dataset(self, level_hdf_path, scheduled_samples):
        """
        Create dataset for scheduled samples
        :param level_hdf_path: HDF5 path to dataset scheduled
        :param scheduled_samples: list in form [((fine sample data, coarse sample data), status), ...]
        :return: None
        """
        # Create dataset with SCHEDULED_DTYPE and with unlimited max shape (required for resize)
        scheduled_dset = self._hdf_file_append[level_hdf_path].create_dataset(
            HDF5.DATASET_NAMES['scheduled'],
            shape=(len(scheduled_samples),),
            maxshape=(None,),
            dtype=HDF5.SCHEDULED_DTYPE,
            chunks=True)

        # Set values to dataset
        scheduled_dset[()] = scheduled_samples

    def change_scheduled_sample_status(self, level_id, sample_id, status):
        """
        Change scheduled sample status, it is common for fine sample and coarse sample too
        :param level_id: Level id
        :param sample_id: Sample id
        :param status: status, e.g. "RUN"
        :return: None
        """
        scheduled_dataset = '/'.join([self._level_group_hdf_paths[level_id], HDF5.DATASET_NAMES['scheduled']])
        if scheduled_dataset not in self._hdf_file_append:
            raise Exception("Dataset is not in HDF5 file")
        else:
            sample_row = self._hdf_file_append[scheduled_dataset][sample_id]
            sample_row[-1] = status
            self._hdf_file_append[scheduled_dataset][sample_id] = sample_row

    def save_collected(self, level_id, collected_samples, rewrite=False):
        """
        Save level collected data to datasets corresponding to COLLECTED_ATTRS
        :param level_id: Level id
        :param collected_samples: Level sample [(fine sample, coarse sample)], both are Sample() object instances
        :param rewrite: bool; if True rewrite datasets else append to existing dataset
        :return: None
        """
        collected_data = {}
        # Get sample attributes pairs as zip (fine attribute values, coarse attribute values)
        samples_attr_pairs = self.sample_attributes_pairs(collected_samples)
        # Attribute values with attribute names
        for key, values in zip(HDF5.COLLECTED_ATTRS.keys(), samples_attr_pairs):
            collected_data[key] = values

        # Create own dataset for each collected attribute
        for attr_name, data in collected_data.items():
            data = np.array(data)
            # Sample id is same for fine and coarse sample
            if attr_name == 'sample_id':
                data = data[:, 0]
            # Data shape
            shape = data.shape
            # Maxshape as a constant
            maxshape = HDF5.COLLECTED_ATTRS[attr_name]['maxshape']

            # Data is squeezed, so expand last dimension to 'maxshape' shape
            if len(shape) == len(maxshape) - 1:
                data = np.expand_dims(data, axis=len(maxshape) - 1)
                shape = data.shape

            # Path to current dataset
            dataset_hdf_path = "/".join([self._level_group_hdf_paths[level_id], HDF5.COLLECTED_ATTRS[attr_name]['name']])
            # Dataset already exists, append new values
            if dataset_hdf_path in self._hdf_file_append and not rewrite:
                self._append_dataset(dataset_hdf_path, data)
                continue

            # Create resizable dataset
            dataset = self._hdf_file_append[self._level_group_hdf_paths[level_id]].create_dataset(
                HDF5.COLLECTED_ATTRS[attr_name]['name'],
                shape=shape,
                maxshape=maxshape,
                dtype=HDF5.COLLECTED_ATTRS[attr_name]['dtype'],
                chunks=True)
            # Set data
            dataset[()] = data

    def sample_attributes_pairs(self, fine_coarse_samples):
        """
        Merge fine sample and coarse sample collected values to one dictionary
        :param fine_coarse_samples: list of tuples; [(Sample(), Sample()), ...]
        :return: zip
        """
        fine_coarse_data = np.array([(f_sample.collected_data_array(HDF5.COLLECTED_ATTRS),
                                      coarse_sample.collected_data_array(HDF5.COLLECTED_ATTRS))
                                     for f_sample, coarse_sample in fine_coarse_samples])

        samples_attr_values = (zip(*[zip(f, c) for f, c in zip(fine_coarse_data[:, 0], fine_coarse_data[:, 1])]))
        return samples_attr_values

    def _append_dataset(self, dataset_hdf_path, values):
        """
        Append values to existing dataset
        :param dataset_hdf_path: Path to dataset
        :param values: list of values (tuple or single value)
        :return: None
        """
        if dataset_hdf_path not in self._hdf_file_reader:
            raise Exception("Dataset doesn't exist")
        # Resize dataset
        self._hdf_file_append[dataset_hdf_path].resize(self._hdf_file_append[dataset_hdf_path].shape[0] +
                                                       len(values), axis=0)
        # Append new values to the end of dataset
        self._hdf_file_append[dataset_hdf_path][-len(values):] = values

    def read_scheduled(self, level_id):
        """
        Read level dataset with scheduled samples
        :param level_id: Level id
        :return: generator, each item is in form (Sample(), Sample())
        """
        # HDF path to datset scheduled
        scheduled_dset_hdf_path = '/'.join([self._level_group_hdf_paths[level_id], HDF5.DATASET_NAMES['scheduled']])
        # Generator of scheduled dataset values
        scheduled_dset = self._read_dataset(scheduled_dset_hdf_path)

        # Create fine and coarse samples
        for sample_id, (fine, coarse, status) in enumerate(scheduled_dset):
            yield (Sample(sample_id=sample_id,
                          directory=fine[0].decode('UTF-8'),
                          job_id=fine[1].decode('UTF-8'),
                          prepare_time=fine[2], queued_time=fine[3]),
                   Sample(sample_id=sample_id,
                          directory=coarse[0].decode('UTF-8'),
                          job_id=coarse[1].decode('UTF-8'),
                          prepare_time=coarse[2], queued_time=coarse[3]))

    def _read_dataset(self, dataset_hdf_path):
        """
        Read dataset values
        :param dataset_hdf_path: HDF path to dataset
        :return: generator
        """
        # Dataset doesn't exist
        if dataset_hdf_path not in self._hdf_file_reader:
            return
        else:
            # Scheduled dataset
            dataset = self._hdf_file_reader[dataset_hdf_path]

            # Return dataset items
            for dset_value in dataset:
                yield dset_value

    def read_collected(self, level_id):
        """
        Read all level datasets with collected data (reading dataset values via self._read_dataset() method),
        create fine and coarse samples as Sample() instances
        :param level_id: Level id
        :return: generator; one item is tuple (Sample(), Sample())
        """
        # HDF5 path to Level group
        l_group_hdf_path = self._level_group_hdf_paths[level_id]
        # Collected ids dataset exists in Level group
        if HDF5.DATASET_NAMES['collected_ids'] not in self._hdf_file_reader[l_group_hdf_path]:
            return

        # Number of collected samples
        num_collected_samples = self._hdf_file_reader[l_group_hdf_path][HDF5.DATASET_NAMES['collected_ids']].len()

        # Collected data as dictionary according to lookup table COLLECTED_ATTRS
        # dictionary format -> {COLLECTED_ATTRS key: corresponding dataset values generator, ...}
        collected_data = {sample_attr_name: self._read_dataset('/'.join([l_group_hdf_path, dset_params['name']]))
                          for sample_attr_name, dset_params in HDF5.COLLECTED_ATTRS.items()}

        # For each item create two Sample() instances - fine sample and coarse sample
        for _ in range(num_collected_samples):
            fine_values = {}
            coarse_values = {}
            # Loop through all datasets (in form of generator) with collected data
            for sample_attr_name, values_generator in collected_data.items():
                value = next(values_generator)
                # Sample id is store like single value for both samples
                if sample_attr_name == 'sample_id':
                    fine_values[sample_attr_name] = value[0]
                    coarse_values[sample_attr_name] = value[0]
                else:
                    fine_values[sample_attr_name] = value[0]
                    coarse_values[sample_attr_name] = value[1]
            # Create tuple of Sample() instances
            yield (Sample(**fine_values), Sample(**coarse_values))


    def level_metadata(self, level_id):
        """
        
        :return: dict {n_ops_estimate, level_id} 
        """


    def mlmc_setup(self):
        """
        
        :return: dict {n_levels, step_range}
        """

    def get_n_ops(self, level_id):
        """
        Get level n ops estimate from HDF group attribute
        :param level_id: Level id
        :return: int
        """
        return self._hdf_file_reader[self._level_group_hdf_paths[level_id]].attrs.get('n_ops_estimate')

    def get_scripts_dir(self):
        """
        Get os path to job scripts directory
        :return: str; path
        """
        return os.path.join(self._hdf_file_reader.attrs.get("work_dir"), self._hdf_file_reader.attrs.get("job_dir"))

    def get_level_jobs(self, level_id):
        """
        Get level job ids
        :param level_id: Level id
        :return: list of job ids
        """
        # HDF path to jobs group
        jobs_group_hdf_path = "/".join([self._level_group_hdf_paths[level_id], 'Jobs'])
        # Names of datasets in group Jobs, each dataset contains sample ids
        return list(self._hdf_file_reader[jobs_group_hdf_path].keys())

    def get_job_samples(self, level_id, job_dataset_names):
        """
        Get sample ids from job datasets
        :param level_id: Level id
        :param job_dataset_names: Job dataset names
        :return: set; sample ids
        """
        # HDF path to level Jobs group
        jobs_group_hdf_path = "/".join([self._level_group_hdf_paths[level_id], 'Jobs'])

        sample_ids = set()
        # Get job samples
        for dset_name in job_dataset_names:
            dataset = self._hdf_file_reader["/".join([jobs_group_hdf_path, dset_name])]
            sample_ids.update(dataset.value)

        return sample_ids
