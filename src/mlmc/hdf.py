import os
import numpy as np
import h5py
from mlmc.sample import Sample


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
        # Job directory relative path
        self.job_dir = job_dir
        # Job directory absolute path
        self.job_dir_abs_path = os.path.join(work_dir, job_dir)

        # Load from file flag
        self._loaded_from_file = False

        # @TODO: Remove this and solve it complexly
        # if os.path.exists(self.file_name):
        #         #    os.remove(self.file_name)

        # Class attributes necessary for mlmc
        self.n_levels = None
        self.step_range = None

    def load_from_file(self):
        """
        Load root group attributes from existing HDF5 file
        :return: None
        """
        self._loaded_from_file = True
        with h5py.File(self.file_name, "r") as hdf_file:
            # Set class attributes from hdf file
            for attr_name, value in hdf_file.attrs.items():
                self.__dict__[attr_name] = value

    def clear_groups(self):
        """
        Remove HDF5 group Levels, it allows run same mlmc object more times
        :return: None
        """
        with h5py.File(self.file_name, "a") as hdf_file:
            for item in hdf_file.keys():
                del hdf_file[item]

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

        with h5py.File(self.file_name, "a") as hdf_file:
            # Set global attributes to root group (h5py.Group)
            hdf_file.attrs['version'] = '1.0.1'
            hdf_file.attrs['work_dir'] = self.work_dir
            hdf_file.attrs['job_dir'] = self.job_dir
            hdf_file.attrs['step_range'] = step_range
            hdf_file.attrs.create("n_levels", n_levels, dtype=np.int8)

            # Create h5py.Group Levels, it contains other groups with mlmc.Level data
            hdf_file.create_group("Levels")

    def add_level_group(self, level_id):
        """
        Create group for particular level, parent group is 'Levels'
        :param level_id: str, mlmc.Level identifier
        :return: LevelGroup instance, it is container for h5py.Group instance
        """
        # HDF5 path to particular level group
        level_group_hdf_path = '/Levels/' + level_id

        with h5py.File(self.file_name, "a") as hdf_file:
            # Create group (h5py.Group) if it has not yet been created
            if level_group_hdf_path not in hdf_file:
                # Create group for level named by level id (e.g. 0, 1, 2, ...)
                hdf_file['Levels'].create_group(level_id)

        return LevelGroup(self.file_name, level_group_hdf_path, level_id, job_dir=self.job_dir_abs_path,
                          loaded_from_file=self._loaded_from_file)


class LevelGroup:
    # One sample data type for dataset (h5py.Dataset) scheduled
    SAMPLE_DTYPE = {'names': ('dir', 'job_id', 'prepare_time', 'queued_time'),
                    'formats': ('S100', 'S5', 'f8', 'f8')}

    # Row format for dataset (h5py.Dataset) scheduled
    SCHEDULED_DTYPE = {'names': ('fine_sample', 'coarse_sample'),
                       'formats': (SAMPLE_DTYPE, SAMPLE_DTYPE)}

    """
    Data that are collected, only this data can by saved to HDF datasets (h5py.Dataset)
    {attribute name in class Sample : {name: dataset name,
                                       default_shape: dataset shape, used while creating
                                       maxshape: Maximal dataset shape, necessary for resizing
                                       dtype: dataset values dtype}
                                       }
    """
    COLLECTED_ATTRS = {"sample_id": {'name': 'collected_ids', 'default_shape': (0,), 'maxshape': (None,),
                                     'dtype': np.int32},
                       "result": {'name': 'collected_values', 'default_shape': (0, 2, 1), 'maxshape': (None, 2, None),
                                  'dtype': np.float64},
                       "time": {'name': 'collected_times', 'default_shape': (0, 2, 1), 'maxshape': (None, 2, None),
                                'dtype': np.float64},
                       # "running_time": {'name': 'running_times', 'default_shape': (0, 2, 1), 'maxshape': (None, 2, None),
                       #          'dtype': np.float64}
                       }

    def __init__(self, file_name, hdf_group_path, level_id, job_dir, loaded_from_file=False):
        """
        Create LevelGroup instance, each mlmc.Level has access to corresponding LevelGroup to save data
        :param file_name: Name of hdf file
        :param hdf_group_path: h5py.Group path
        :param level_id: Unambiguous identifier of mlmc.Level object 
        :param job_dir: Absolute path to jobs directory which contains pbs scripts of samples
        """
        # HDF file name
        self.file_name = file_name
        # mlmc.Level identifier
        self.level_id = level_id
        # HDF Group object (h5py.Group)
        self.level_group_path = hdf_group_path
        # Absolute path to the job directory
        self.job_dir = job_dir

        # Attribute necessary for mlmc run
        self._n_ops_estimate = None

        # Set group attribute 'level_id'
        with h5py.File(self.file_name, 'a') as hdf_file:
            hdf_file[self.level_group_path].attrs['level_id'] = self.level_id

        # Create necessary datasets (h5py.Dataset) a groups (h5py.Group)
        if not loaded_from_file:
            self._make_groups_datasets()

    def _make_groups_datasets(self):
        """
        Create h5py.Dataset for scheduled samples, collected samples according to COLLECTED_ATTRS and failed samples,
        also create h5py.Group for Jobs - it contains or will contain datasets with sample id,
        so we can find all sample ids which was run in particular pbs job
        :return: None
        """
        # Create dataset for scheduled samples
        self._make_dataset(name='scheduled', shape=(0,), maxshape=(None,), dtype=LevelGroup.SCHEDULED_DTYPE, chunks=True)

        # Create datasets for collected samples by COLLECTED_ATTRS
        for _, attr_properties in LevelGroup.COLLECTED_ATTRS.items():
            self._make_dataset(name=attr_properties['name'], shape=attr_properties['default_shape'],
                               maxshape=attr_properties['maxshape'], dtype=attr_properties['dtype'], chunks=True)

        # Create dataset for failed samples
        self._make_dataset(name='failed_ids', shape=(0,), dtype=np.int32, maxshape=(None,), chunks=True)

    def _make_dataset(self, **kwargs):
        """
        Create h5py.Dataset
        :param kwargs: h5py.Dataset properties
                    name: Dataset name, key in h5py.Group (self._level_group)
                    shape: NumPy-style shape tuple giving dataset dimensions.
                    dtype: NumPy dtype object giving the dataset’s type.
                    maxshape: NumPy-style shape tuple indicating the maxiumum dimensions up to
                              which the dataset may be resized. Axes with None are unlimited.
                    chunks: Tuple giving the chunk shape, or True if we want to use chunks but not specify the size or 
                            None if chunked storage is not used
        :return: Dataset name
        """
        with h5py.File(self.file_name, 'a') as hdf_file:
            # Check if dataset exists
            if kwargs.get('name') not in hdf_file[self.level_group_path]:
                hdf_file[self.level_group_path].create_dataset(
                    kwargs.get('name'),
                    shape=kwargs.get('shape'),
                    dtype=kwargs.get('dtype'),
                    maxshape=kwargs.get('maxshape'),
                    chunks=kwargs.get('chunks'))

        return kwargs.get('name')

    @property
    def collected_ids_dset(self):
        """
        Collected ids dataset
        :return: Dataset name
        """
        return "collected_ids"

    @property
    def scheduled_dset(self):
        """
        Dataset with scheduled samples
        :return: Dataset name
        """
        return "scheduled"

    @property
    def failed_ids_dset(self):
        """
        Dataset of ids of failed samples
        :return: Dataset name
        """
        return "failed_ids"

    def append_scheduled(self, scheduled_samples):
        """
        Save scheduled samples to dataset (h5py.Dataset)
        :param scheduled_samples: list of sample objects [(fine_sample, coarse_sample)]
        :return: None
        """
        # Prepare NumPy array for all scheduled samples
        samples_scheduled_data = np.empty(len(scheduled_samples), dtype=LevelGroup.SCHEDULED_DTYPE)
        # Jobs {job_id: [sample_id, ...]}
        jobs = {}

        # Go through scheduled samples, format them and get samples job ids (pbs script id, see src.Pbs)
        for index, (sample_id, (fine_sample, coarse_sample)) in enumerate(scheduled_samples.items()):
            # Add of fine samples scheduled data and coarse sample scheduled data
            samples_scheduled_data[index] = fine_sample.scheduled_data(), coarse_sample.scheduled_data()
            # Append sample id to job id, default create empty set
            jobs.setdefault(fine_sample.job_id, set()).add(sample_id)
            # For non zero level add also coarse sample job id
            if int(self.level_id) != 0:
                jobs.setdefault(coarse_sample.job_id, set()).add(sample_id)

        # Append samples to existing scheduled dataset
        self._append_dataset(self.scheduled_dset, samples_scheduled_data)

        # Save to jobs to datasets
        self._append_jobs(jobs)

    def _append_jobs(self, jobs):
        """
        Save jobs to datasets (h5py.Dataset)
        :param jobs: dict, {job_id: [sample_id,...], ...}
        :return: None
        """
        # Loop through all jobs
        for job_name, job_samples in jobs.items():
            # HDF path to job dataset
            job_dataset_path = '/'.join(['Jobs', job_name])

            # Get job dataset and direct save sample ids
            job_dset = self._make_dataset(name=job_dataset_path, shape=(0,),
                                          dtype=np.int32, maxshape=(None,), chunks=True)
            # Append sample ids to existing job dataset
            self._append_dataset(job_dset, list(job_samples))

    def append_collected(self, collected_samples):
        """
        Save level collected samples to datasets (h5py.Dataset) corresponding to the COLLECTED_ATTRS
        :param collected_samples: Level sample [(fine sample, coarse sample)], both are Sample() object instances
        :return: None
        """
        # Get sample attributes pairs as NumPy array [num_attrs, num_samples, 2]
        samples_attr_pairs = self._sample_attr_pairs(collected_samples)
        # Append attributes datasets - dataset name matches the attribute name
        for attr_name, data in zip(LevelGroup.COLLECTED_ATTRS.keys(), samples_attr_pairs):
            # Sample id is same for fine and coarse sample, use just one
            if attr_name == 'sample_id':
                data = data[:, 0]

            # Data are squeezed, so expand last dimension to 'maxshape' shape
            if len(data.shape) == len(LevelGroup.COLLECTED_ATTRS[attr_name]['maxshape']) - 1:
                data = np.expand_dims(data, axis=len(LevelGroup.COLLECTED_ATTRS[attr_name]['maxshape']) - 1)

            # Append dataset
            self._append_dataset(LevelGroup.COLLECTED_ATTRS[attr_name]['name'], data)

    def _sample_attr_pairs(self, fine_coarse_samples):
        """
        Merge fine sample and coarse sample collected values to one NumPy array
        :param fine_coarse_samples: list of tuples; [(Sample(), Sample()), ...]
        :return: Fine and coarse samples in array: [n_attrs, N, 2]
        """
        # Number of attributes
        n_attrs = len(Sample().collected_data_array(LevelGroup.COLLECTED_ATTRS))
        # Prepare matrix for fine and coarse data
        fine_coarse_data = np.empty((len(fine_coarse_samples), 2, n_attrs))
        # Set sample's collected data
        for index, (f_sample, c_sample) in enumerate(fine_coarse_samples):
            fine_coarse_data[index, 0, :] = f_sample.collected_data_array(LevelGroup.COLLECTED_ATTRS)
            fine_coarse_data[index, 1, :] = c_sample.collected_data_array(LevelGroup.COLLECTED_ATTRS)

        # Shape: [N, 2, n_attrs] -> [n_attrs, N, 2]
        return fine_coarse_data.transpose([2, 0, 1])

    def save_failed(self, failed_samples):
        """
        Save level failed sample ids (not append samples)
        :param failed_samples: set; Level sample ids
        :return: None
        """
        with h5py.File(self.file_name, 'a') as hdf_file:
            hdf_file[self.level_group_path][self.failed_ids_dset].resize((len(failed_samples), ))
            hdf_file[self.level_group_path][self.failed_ids_dset][:] = list(failed_samples)

    def _append_dataset(self, dataset_name, values):
        """
        Append values to existing dataset
        :param dataset_name: str, dataset name
        :param values: list of values (tuple, NumPy array or single value)
        :return: None
        """
        with h5py.File(self.file_name, 'a') as hdf_file:
            dataset = hdf_file[self.level_group_path][dataset_name]
            # Resize dataset
            dataset.resize(dataset.shape[0] + len(values), axis=0)
            # Append new values to the end of dataset
            dataset[-len(values):] = values

    def scheduled(self):
        """
        Read level dataset with scheduled samples
        :return: generator, each item is in form (Sample(), Sample())
        """
        with h5py.File(self.file_name, 'r') as hdf_file:
            scheduled_dset = hdf_file[self.level_group_path][self.scheduled_dset]
            # Create fine and coarse samples
            for sample_id, (fine, coarse) in enumerate(scheduled_dset[:]):
                yield (Sample(sample_id=sample_id,
                              directory=fine[0].decode('UTF-8'),
                              job_id=fine[1].decode('UTF-8'),
                              prepare_time=fine[2], queued_time=fine[3]),
                       Sample(sample_id=sample_id,
                              directory=coarse[0].decode('UTF-8'),
                              job_id=coarse[1].decode('UTF-8'),
                              prepare_time=coarse[2], queued_time=coarse[3]))

    def collected(self):
        """
        Read all level datasets with collected data, create fine and coarse samples as Sample() instances
        :return: generator; one item is tuple (Sample(), Sample())
        """
        with h5py.File(self.file_name, 'r') as hdf_file:
            # Number of collected samples
            num_samples = hdf_file[self.level_group_path][self.collected_ids_dset].len()

            # Empty datasets cannot be read
            if num_samples == 0:
                return

            # Initialize matrix of all collected values
            sample_matrix = np.empty((num_samples, (len(LevelGroup.COLLECTED_ATTRS) * 2) - 1), dtype=np.float)

            # Auxiliary dictionaries for Sample instance creation
            fine_attribute_column = {}
            coarse_attribute_column = {}

            column_index = 0
            # Loop through all collected datasets and put its values into matrix
            for sample_attr_name, dset_params in LevelGroup.COLLECTED_ATTRS.items():
                # Skip not used datasets
                if dset_params['name'] not in hdf_file[self.level_group_path]:
                    continue
                dataset = hdf_file["/".join([self.level_group_path, dset_params['name']])]

                # Collected ids is used for both - fine and coarse sample
                if dset_params['name'] == 'collected_ids':
                    dataset.read_direct(sample_matrix, np.s_[:, ], np.s_[:, column_index])
                    # Attribute name with corresponding column index in sample matrix
                    fine_attribute_column[sample_attr_name] = column_index
                    coarse_attribute_column[sample_attr_name] = column_index
                else:
                    # Read dataset values to matrix
                    dataset.read_direct(sample_matrix, np.s_[:, 0, 0], np.s_[:, column_index])
                    fine_attribute_column[sample_attr_name] = column_index
                    column_index += 1
                    coarse_attribute_column[sample_attr_name] = column_index
                    dataset.read_direct(sample_matrix, np.s_[:, 1, 0], np.s_[:, column_index])
                column_index += 1

            # Create fine and coarse Sample
            for row in sample_matrix:
                yield Sample(**{attr_name: row[index] for attr_name, index in fine_attribute_column.items()}), \
                      Sample(**{attr_name: row[index] for attr_name, index in coarse_attribute_column.items()})

    def level_jobs(self):
        """
        Get level job ids
        :return: list of job ids - in this case it is equivalent to h5py.Group.keys() (h5py.Dataset names)
        """
        with h5py.File(self.file_name, 'a') as hdf_file:
            return list(hdf_file[self.level_group_path]['Jobs'].keys())

    def job_samples(self, job_dataset_names):
        """
        Get sample ids from job datasets
        :param job_dataset_names: Job dataset names
        :return: NumPy array of unique sample ids
        """
        with h5py.File(self.file_name, 'a') as hdf_file:
            # HDF path to level Jobs group
            if 'Jobs' not in hdf_file[self.level_group_path]:
                hdf_file[self.level_group_path].create_group('Jobs')

            # Path 'Jobs' group
            jobs_group_hdf_path = "/".join([self.level_group_path, 'Jobs'])
            sample_ids = []
            # Get job samples
            for dset_name in job_dataset_names:
                dataset = hdf_file[self.level_group_path]["/".join([jobs_group_hdf_path, dset_name])]
                sample_ids.extend(dataset.value)

        return np.unique(np.array(sample_ids))

    def get_finished_ids(self):
        """
        Get collected and failed samples ids
        :return: NumPy array
        """
        with h5py.File(self.file_name, 'r') as hdf_file:
            failed_ids = hdf_file[self.level_group_path]['failed_ids'].value
            return np.concatenate((hdf_file[self.level_group_path][self.collected_ids_dset].value, np.array(failed_ids)),
                                  axis=0)

    def get_failed_ids(self):
        """
        Failed samples ids
        :return: set() of failed sample ids
        """
        with h5py.File(self.file_name, 'r') as hdf_file:
            # Return NumPy array otherwise return set()
            return set(hdf_file[self.level_group_path]['failed_ids'])

    @property
    def n_ops_estimate(self):
        """
        Get number of operations estimate
        :return: float
        """
        with h5py.File(self.file_name, 'r') as hdf_file:
            if self._n_ops_estimate is None and 'n_ops_estimate' in hdf_file[self.level_group_path].attrs:
                self._n_ops_estimate = hdf_file[self.level_group_path].attrs['n_ops_estimate']

        return self._n_ops_estimate

    @n_ops_estimate.setter
    def n_ops_estimate(self, n_ops_estimate):
        """
        Set property n_ops_estimate
        :param n_ops_estimate: number of operations
        :return: None
        """
        with h5py.File(self.file_name, 'a') as hdf_file:
            self._n_ops_estimate = hdf_file[self.level_group_path].attrs['n_ops_estimate'] = float(n_ops_estimate)
