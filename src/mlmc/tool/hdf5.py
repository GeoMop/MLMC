import numpy as np
import h5py
from mlmc.quantity.quantity_spec import ChunkSpec


class HDF5:
    """
    HDF5 file is organized into groups (h5py.Group objects)
    which is somewhat like dictionaries in python terminology - 'keys' are names of group members
    'values' are members (groups (h5py.Group objects) and datasets (h5py.Dataset objects - similar to NumPy arrays)).
    Each group and dataset (including root group) can store metadata in 'attributes' (h5py.AttributeManager objects)
    HDF5 files (h5py.File) work generally like standard Python file objects

    Our HDF5 file strucutre:
        Main Group:
        Keys:
            Levels: h5py.Group
                Attributes:
                    level_parameters: [[a], [b], [], ...]
                Keys:
                    <N>: h5py.Group (N - level id, start with 0)
                        Attributes:
                            id: str
                            n_ops_estimate: float
                        Keys:
                            scheduled: h5py.Dataset
                                dtype: S100
                                shape: (N,), N - number of scheduled values
                                maxshape: (None,)
                                chunks: True
                            collected_values: h5py.Dataset
                                dtype: numpy.float64
                                shape: (Nc, 2, M) dtype structure is defined in simulation class
                                maxshape: (None, 2, None)
                                chunks: True
                            collected_ids: h5py.Dataset
                                dtype: numpy.int16  index into scheduled
                                shape: (Nc, 1)
                                maxshape: (None, 1)
                                chunks: True
                            failed: h5py.Dataset
                                dtype: ('S100', 'S1000')
                                shape: (Nf, 1)
                                mashape: (None, 1)
                                chunks: True
    """
    def __init__(self, file_path, load_from_file=False):
        """
        Create HDF5 class instance
        :param file_path: hdf5 file path
        """
        # # Absolute path to mlmc HDF5 file
        self.file_name = file_path
        # If True not create file structure from scratch
        self._load_from_file = load_from_file
        if self._load_from_file:
            self.load_from_file()

    def create_file_structure(self, level_parameters):
        """
        Create hdf structure
        :param level_parameters: List[float]
        :return: None
        """
        if self._load_from_file:
            self.load_from_file()
        else:
            self.clear_groups()
            self.init_header(level_parameters=level_parameters)

    def load_from_file(self):
        """
        Load root group attributes from existing HDF5 file
        :return: None
        """
        with h5py.File(self.file_name, "r") as hdf_file:
            # Set class attributes from hdf file
            for attr_name, value in hdf_file.attrs.items():
                self.__dict__[attr_name] = value

        if 'level_parameters' not in self.__dict__:
            raise Exception("'level_parameters' aren't store in HDF file, so unable to create level groups")

    def clear_groups(self):
        """
        Remove HDF5 group Levels, it allows run same mlmc object more times
        :return: None
        """
        with h5py.File(self.file_name, "a") as hdf_file:
            for item in hdf_file.keys():
                del hdf_file[item]

    def init_header(self, level_parameters):
        """
        Add h5py.File metadata to .attrs (attrs objects are of class h5py.AttributeManager)
        :param level_parameters: MLMC level range of steps
        :return: None
        """
        with h5py.File(self.file_name, "a") as hdf_file:
            # Set global attributes to root group (h5py.Group)
            hdf_file.attrs['version'] = '1.0.1'
            hdf_file.attrs['level_parameters'] = level_parameters
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

        return LevelGroup(self.file_name, level_group_hdf_path, level_id, loaded_from_file=self._load_from_file)

    @property
    def result_format_dset_name(self):
        """
        Result format dataset name
        :return: str
        """
        return "result_format"

    def save_result_format(self, result_format, res_dtype):
        """
        Save result format to dataset
        :param result_format: List[QuantitySpec]
        :param res_dtype: result numpy dtype
        :return: None
        """
        result_format_dtype = res_dtype

        # Create data set
        with h5py.File(self.file_name, 'a') as hdf_file:
            # Check if dataset exists
            if self.result_format_dset_name not in hdf_file:
                hdf_file.create_dataset(
                    self.result_format_dset_name,
                    shape=(len(result_format),),
                    dtype=result_format_dtype,
                    maxshape=(None,),
                    chunks=True)

        # Format data
        result_array = np.empty((len(result_format),), dtype=result_format_dtype)
        for res, quantity_spec in zip(result_array, result_format):
            for attribute in list(quantity_spec.__dict__.keys()):
                if isinstance(getattr(quantity_spec, attribute), (tuple, list)):
                    res[attribute][:] = getattr(quantity_spec, attribute)
                else:
                    res[attribute] = getattr(quantity_spec, attribute)

        # Write to file
        with h5py.File(self.file_name, 'a') as hdf_file:
            dataset = hdf_file[self.result_format_dset_name]
            dataset[:] = result_array

    def load_result_format(self):
        """
        Load format result, it just read dataset
        :return:
        """
        with h5py.File(self.file_name, 'r') as hdf_file:
            if self.result_format_dset_name not in hdf_file:
                raise AttributeError

            dataset = hdf_file[self.result_format_dset_name]
            return dataset[()]

    def load_level_parameters(self):
        with h5py.File(self.file_name, "r") as hdf_file:
            # Set global attributes to root group (h5py.Group)
            if 'level_parameters' in hdf_file.attrs:
                return hdf_file.attrs['level_parameters']
            else:
                return []


class LevelGroup:
    # Row format for dataset (h5py.Dataset) scheduled
    SCHEDULED_DTYPE = {'names': ['sample_id'],
                       'formats': ['S100']}

    FAILED_DTYPE = {'names': ('sample_id', 'message'),
                    'formats': ('S100', 'S1000')}

    COLLECTED_ATTRS = {"sample_id": {'name': 'collected_ids', 'default_shape': (0,), 'maxshape': (None,),
                                     'dtype': SCHEDULED_DTYPE}}

    def __init__(self, file_name, hdf_group_path, level_id, loaded_from_file=False):
        """
        Create LevelGroup instance, each mlmc.Level has access to corresponding LevelGroup to save data
        :param file_name: Name of hdf file
        :param hdf_group_path: h5py.Group path
        :param level_id: Unambiguous identifier of mlmc.Level object
        :param loaded_from_file: bool, create new file or loaded existing groups
        """
        self.file_name = file_name
        # HDF file name
        self.level_id = level_id
        # Level identifier
        self.level_group_path = hdf_group_path
        # HDF Group object (h5py.Group)
        self._n_items_in_chunk = None
        # Collected items in one chunk
        self._chunk_size_items = {}
        # Chunk size and corresponding number of items

        # Set group attribute 'level_id'
        with h5py.File(self.file_name, 'a') as hdf_file:
            if 'level_id' not in hdf_file[self.level_group_path].attrs:
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
        self._make_dataset(name=self.scheduled_dset, shape=(0,), maxshape=(None,), dtype=LevelGroup.SCHEDULED_DTYPE,
                           chunks=True)

        # Create datasets for collected samples by COLLECTED_ATTRS
        for _, attr_properties in LevelGroup.COLLECTED_ATTRS.items():
            self._make_dataset(name=attr_properties['name'], shape=attr_properties['default_shape'],
                               maxshape=attr_properties['maxshape'], dtype=attr_properties['dtype'], chunks=True)

        # Create dataset for failed samples
        self._make_dataset(name=self.failed_dset, shape=(0,), dtype=LevelGroup.FAILED_DTYPE, maxshape=(None,), chunks=True)

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
    def failed_dset(self):
        """
        Dataset of ids of failed samples
        :return: Dataset name
        """
        return "failed"

    def append_scheduled(self, scheduled_samples):
        """
        Save scheduled samples to dataset (h5py.Dataset)
        :param scheduled_samples: list of sample ids
        :return: None
        """
        # Append samples to existing scheduled dataset
        if len(scheduled_samples) > 0:
            self._append_dataset(self.scheduled_dset, scheduled_samples)

    def append_successful(self, samples: np.array):
        """
        Save level samples to datasets (h5py.Dataset), save ids of collected samples and their results
        :param samples: np.ndarray
        :return: None
        """
        self._append_dataset(self.collected_ids_dset, samples[:, 0])

        values = samples[:, 1]
        result_type = np.dtype((np.float, np.array(values[0]).shape))

        # Create dataset for failed samples
        self._make_dataset(name='collected_values', shape=(0,),
                           dtype=result_type, maxshape=(None,),
                           chunks=True)

        d_name = 'collected_values'
        self._append_dataset(d_name, [val for val in values])

    def append_failed(self, failed_samples):
        """
        Save level failed sample ids (not append samples)
        :param failed_samples: set; Level sample ids
        :return: None
        """
        self._append_dataset(self.failed_dset, failed_samples)

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
        :return:
        """
        with h5py.File(self.file_name, 'r') as hdf_file:
            scheduled_dset = hdf_file[self.level_group_path][self.scheduled_dset]
            return scheduled_dset[()]

    def chunks(self, n_samples=None):
        with h5py.File(self.file_name, 'r') as hdf_file:
            if 'collected_values' not in hdf_file[self.level_group_path]:
                raise AttributeError("No collected values in level group ".format(self.level_id))
            dataset = hdf_file["/".join([self.level_group_path, "collected_values"])]

            if n_samples is not None:
                yield ChunkSpec(chunk_id=0, chunk_slice=slice(0, n_samples, 1), level_id=int(self.level_id))
            else:
                for chunk_id, chunk in enumerate(dataset.iter_chunks()):
                    yield ChunkSpec(chunk_id=chunk_id, chunk_slice=chunk[0], level_id=int(self.level_id))  # slice, level_id

    def collected(self, chunk_slice):
        """
        Read collected data by chunks,
        number of items in chunk is determined by LevelGroup.chunk_size (number of bytes)
        :param chunk_slice: slice() object
        :return: np.ndarray
        """
        with h5py.File(self.file_name, 'r') as hdf_file:
            if 'collected_values' not in hdf_file[self.level_group_path]:
                return None
            dataset = hdf_file["/".join([self.level_group_path, "collected_values"])]
            return dataset[chunk_slice]

    def collected_n_items(self):
        """
        Number of collected samples
        :return: int
        """
        with h5py.File(self.file_name, 'r') as hdf_file:
            if 'collected_values' not in hdf_file[self.level_group_path]:
                return AttributeError("collected_values dataset not in HDF file for level {}".format(self.level_id))
            dataset = hdf_file["/".join([self.level_group_path, "collected_values"])]
            collected_n_items = len(dataset[()])
        return collected_n_items

    def get_finished_ids(self):
        """
        Get collected and failed samples ids
        :return: NumPy array
        """
        with h5py.File(self.file_name, 'r') as hdf_file:
            failed_ids = [sample[0].decode() for sample in hdf_file[self.level_group_path][self.failed_dset][()]]
            successful_ids = [sample[0].decode() for sample in hdf_file[self.level_group_path][self.collected_ids_dset][()]]
            return np.concatenate((np.array(successful_ids), np.array(failed_ids)), axis=0)

    def get_unfinished_ids(self):
        """
        Get unfinished sample ids as difference between scheduled ids and finished ids
        :return: list
        """
        scheduled_ids = [sample[0].decode() for sample in self.scheduled()]
        return list(set(scheduled_ids) - set(self.get_finished_ids()))

    def get_failed_ids(self):
        """
        Failed samples ids
        :return: list of failed sample ids
        """
        with h5py.File(self.file_name, 'r') as hdf_file:
            failed_ids = [sample[0].decode() for sample in hdf_file[self.level_group_path][self.failed_dset][()]]

        return failed_ids

    def clear_failed_dataset(self):
        """
        Clear failed_ids dataset
        :return: None
        """
        with h5py.File(self.file_name, 'a') as hdf_file:
            if self.failed_dset in hdf_file[self.level_group_path]:
                del hdf_file[self.level_group_path][self.failed_dset]
                # Create dataset for failed samples
                self._make_dataset(name=self.failed_dset, shape=(0,), dtype=LevelGroup.FAILED_DTYPE, maxshape=(None,),
                                   chunks=True)

    @property
    def n_ops_estimate(self):
        """
        Get number of operations estimate
        :return: float
        """
        with h5py.File(self.file_name, 'r') as hdf_file:
            if 'n_ops_estimate' in hdf_file[self.level_group_path].attrs:
                return hdf_file[self.level_group_path].attrs['n_ops_estimate']

    @n_ops_estimate.setter
    def n_ops_estimate(self, n_ops_estimate):
        """
        Set property n_ops_estimate
        :param n_ops_estimate: number of operations (time) per samples
        :return: None
        """
        with h5py.File(self.file_name, 'a') as hdf_file:
            if 'n_ops_estimate' not in hdf_file[self.level_group_path].attrs:
                hdf_file[self.level_group_path].attrs['n_ops_estimate'] = [0., 0.]
            hdf_file[self.level_group_path].attrs['n_ops_estimate'] = n_ops_estimate

