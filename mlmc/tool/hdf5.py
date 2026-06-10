"""
Implementation detail of sample_storage_hdf.py.
Refactor storage structure or replace hdf by zarr completely in particular simplify
storage of samlples to unify code of LevelGroup methods:
- append_*
- scheduled_samples, get_finished_ids, get_unfinished_samples, get_failed_samples
- and other
"""
import numpy as np
import h5py
from mlmc.quantity.quantity_spec import ChunkSpec
import time
import logging

class FileSafe(h5py.File):
    """
    Context manager for openning HDF5 files with some timeout
    amd retrying of getting acces.creation and usage of a workspace dir.
    """
    def __init__(self, filename:str, mode='r', timeout=5, **kwargs):
        """
        :param filename:
        :param timeout: time to try acquire the lock
        """
        end_time = time.time() + timeout
        while time.time() < end_time:
            try:
                super().__init__(filename, mode, **kwargs)
                return
            except BlockingIOError as e:
                time.sleep(0.01)
                continue
            break
        logging.exception(f"Unable to lock access to HDF5 file: {filename}, give up after: {timeout}s.")
        raise BlockingIOError(f"Unable to lock access to HDF5 file: {filename}, give up after: {timeout}s.")


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
        Create HDF5 class instance.
        :param file_path: Path to HDF5 file to use.
        :param load_from_file: If True, load metadata from an existing file instead of initializing a new structure.
        """
        self.file_name = file_path
        self._load_from_file = load_from_file
        if self._load_from_file:
            self.load_from_file()

    def create_file_structure(self, level_parameters):
        """
        Create top-level HDF5 structure for MLMC results or load existing one.

        :param level_parameters: List[float] of level parameters to store in root attributes when initializing new file.
        :return: None
        """
        if self._load_from_file:
            self.load_from_file()
        else:
            self.clear_groups()
            self.init_header(level_parameters=level_parameters)

    def load_from_file(self):
        """
        Load root group attributes from an existing HDF5 file and set them as instance attributes.

        Raises an Exception if required attributes (like 'level_parameters') are not present.

        :return: None
        """
        with h5py.File(self.file_name, "r") as hdf_file:
            # Set class attributes from hdf file root attributes
            for attr_name, value in hdf_file.attrs.items():
                self.__dict__[attr_name] = value

        if 'level_parameters' not in self.__dict__:
            raise Exception("'level_parameters' aren't stored in HDF file, so unable to create level groups")

    def clear_groups(self):
        """
        Remove all top-level groups/datasets from the HDF5 file.

        Useful when reinitializing a new MLMC run into an existing file.

        :return: None
        """
        with h5py.File(self.file_name, "a") as hdf_file:
            for item in list(hdf_file.keys()):
                del hdf_file[item]

    def init_header(self, level_parameters):
        """
        Initialize root attributes and create the top-level 'Levels' group.

        :param level_parameters: Iterable of level parameters to store in root attributes.
        :return: None
        """
        with FileSafe(self.file_name, "a") as hdf_file:
            # Set global attributes on root group
            hdf_file.attrs['version'] = '1.0.1'
            hdf_file.attrs['level_parameters'] = level_parameters
            # Create top-level group 'Levels' to hold per-level groups
            if "Levels" not in hdf_file:
                hdf_file.create_group("Levels")

    def add_level_group(self, level_id):
        """
        Create (if necessary) and return a LevelGroup wrapper for a particular level.

        :param level_id: str, mlmc.Level identifier (e.g. '0', '1', ...)
        :return: LevelGroup instance bound to the '/Levels/{level_id}' HDF5 group
        """
        level_group_hdf_path = '/Levels/' + level_id

        try:
            with FileSafe(self.file_name, "a") as hdf_file:
                # Create group (h5py.Group) if it has not yet been created

                if "Levels" not in hdf_file:
                    hdf_file.create_group('Levels')

                if level_group_hdf_path not in hdf_file:
                    # Create group for level named by level id (e.g. 0, 1, 2, ...)
                    hdf_file['Levels'].create_group(level_id)
        except BlockingIOError as e:
            raise BlockingIOError(f"Unable to lock file: {self.file_name}")
        return LevelGroup(self.file_name, level_group_hdf_path, level_id, loaded_from_file=self._load_from_file)

    @property
    def result_format_dset_name(self):
        """
        Dataset name used to store the simulation result format (QuantitySpec array).

        :return: str dataset name
        """
        return "result_format"

    def single_format(self, spec: "QuantitySpec"):
        """
        Create a one-row structured array and dtype for a QuantitySpec.
        """
        # point or named region
        first_loc = spec.locations[0]
        has_str_locations = isinstance(first_loc, (str, bytes))
        if has_str_locations:
            loc_dtype = 'S30'
        else:
            assert len(first_loc) == 3
            loc_dtype = np.dtype((float, (3,)))
        shape_dtype = np.dtype((np.int32, (len(spec.shape),)))
        locations_dtype = np.dtype((loc_dtype, (len(spec.locations),)))
        result_dtype = {'names': ('name','unit', 'shape', 'times', 'locations'),
                        'formats': ('S50',
                                    'S50',
                                    shape_dtype,
                                    np.dtype((float, (len(spec.times),))),
                                    locations_dtype
                                    )
                        }
        format_items = (spec.name, spec.unit, spec.shape, spec.times, spec.locations)
        res_format = np.array([format_items], dtype=result_dtype)
        return res_format, result_dtype

    def save_result_format(self, result_format):
        """
        Save simulation result format into a group of structured datasets.

        Each QuantitySpec can have different time/location sizes, so every
        spec is stored in a separate one-row dataset with its own dtype.

        :param result_format: List[QuantitySpec] (objects describing output fields)
        :return: None
        """
        format_items = [
            (f"{ispec:04d}", *self.single_format(quantity_spec))
            for ispec, quantity_spec in enumerate(result_format)
        ]

        with h5py.File(self.file_name, 'a') as hdf_file:
            if self.result_format_dset_name not in hdf_file:
                format_group = hdf_file.create_group(self.result_format_dset_name)
            else:
                format_group = hdf_file[self.result_format_dset_name]

            for ispec, q_format, format_dtype in format_items:
                if ispec not in format_group:
                    format_group.create_dataset(
                        name=ispec,
                        shape=(1,),
                        dtype=format_dtype,
                        maxshape=(None,),
                        chunks=True)
                dataset = format_group[ispec]
                dataset[0] = q_format[0]

    def load_result_format(self):
        """
        Load the saved result_format group.

        :return: Dict[str, numpy.ndarray] containing one structured record per QuantitySpec
        :raises AttributeError: if the group is not present
        """
        with h5py.File(self.file_name, 'r') as hdf_file:
            if self.result_format_dset_name not in hdf_file:
                raise AttributeError("Result format dataset not present in HDF file")
            format_group = hdf_file[self.result_format_dset_name]
            return {ispec: np.array(dataset) for ispec, dataset in format_group.items()}

    def load_level_parameters(self):
        """
        Read level_parameters from the HDF5 file root attributes.

        :return: value of 'level_parameters' attribute or empty list if not present
        """
        with h5py.File(self.file_name, "r") as hdf_file:
            if 'level_parameters' in hdf_file.attrs:
                return hdf_file.attrs['level_parameters']
            else:
                return []


class LevelGroup:
    """
    Helper class to manipulate per-level HDF5 group contents.

    It provides convenience methods to append scheduled samples, collected results,
    failed entries and to iterate over collected data in chunks.
    """

    # Structured dtype for sample id rows.
    SCHEDULED_DTYPE = {'names': ['sample_id'],
                       'formats': ['S100']}

    # Structured dtype for failed entries: sample id and message
    FAILED_DTYPE = {'names': ('sample_id', 'message'),
                    'formats': ('S100', 'S1000')}

    # Attributes describing datasets we create for collected ids/values
    COLLECTED_ATTRS = {"sample_id": {'name': 'collected_ids', 'default_shape': (0,), 'maxshape': (None,),
                                     'dtype': SCHEDULED_DTYPE}}

    def __init__(self, file_name, hdf_group_path, level_id, loaded_from_file=False):
        """
        Create LevelGroup instance bound to given HDF5 group path.

        :param file_name: Path to HDF5 file.
        :param hdf_group_path: HDF5 group path string (e.g. '/Levels/0').
        :param level_id: str identifier of the level (used as attribute).
        :param loaded_from_file: If True, assume group already existed and skip creating datasets.
        """
        self.file_name = file_name
        self.level_id = level_id
        self.level_group_path = hdf_group_path
        self._n_items_in_chunk = None
        self._chunk_size_items = {}

        # Ensure HDF group has attribute 'level_id'
        with FileSafe(self.file_name, 'a') as hdf_file:
            if 'level_id' not in hdf_file[self.level_group_path].attrs:
                hdf_file[self.level_group_path].attrs['level_id'] = self.level_id

        # If creating anew, initialize required datasets/groups
        if not loaded_from_file:
            self._make_groups_datasets()

    def _make_groups_datasets(self):
        """
        Create default datasets under the level group:
          - scheduled (resizable structured array of sample ids)
          - scheduled_inputs (optional resizable array of sample inputs)
          - collected_ids (resizable structured array of collected ids)
          - failed (resizable structured array of failed entries)
          - collected_values is created later when first result is appended

        :return: None
        """
        # scheduled dataset (initially empty)
        self._make_dataset(name=self.scheduled_dset, shape=(0,), maxshape=(None,), dtype=LevelGroup.SCHEDULED_DTYPE,
                           chunks=True)

        # collected_ids dataset(s)
        for _, attr_properties in LevelGroup.COLLECTED_ATTRS.items():
            self._make_dataset(name=attr_properties['name'], shape=attr_properties['default_shape'],
                               maxshape=attr_properties['maxshape'], dtype=attr_properties['dtype'], chunks=True)

        # failed dataset (initially empty)
        self._make_dataset(name=self.failed_dset, shape=(0,), dtype=LevelGroup.FAILED_DTYPE, maxshape=(None,), chunks=True)

    def _make_dataset(self, **kwargs):
        """
        Generic helper to create a dataset under the level group if missing.

        :param kwargs: expects keys:
            - name: dataset name (str)
            - shape: initial shape tuple
            - dtype: numpy dtype or structured dtype
            - maxshape: max shape for resizable axes
            - chunks: chunk setting (True or tuple)
        :return: str the dataset name that was created/ensured
        """
        with h5py.File(self.file_name, 'a') as hdf_file:
            # Create dataset only if it does not exist
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
        Name of dataset storing collected ids.

        :return: str
        """
        return "collected_ids"

    @property
    def scheduled_dset(self):
        """
        Name of dataset storing scheduled sample ids.

        :return: str
        """
        return "scheduled"

    @property
    def scheduled_inputs_dset(self):
        """
        Name of dataset storing scheduled sample inputs.

        :return: str
        """
        return "scheduled_inputs"

    @property
    def failed_dset(self):
        """
        Name of dataset storing failed sample rows.

        :return: str
        """
        return "failed"

    def append_scheduled(self, scheduled_samples):
        """
        Append scheduled samples to the scheduled dataset.

        :param scheduled_samples: iterable of sample-id strings or
            (sample_id, sample_input) tuples.
        :return: None
        """
        if len(scheduled_samples) > 0:
            scheduled_ids, sample_inputs = self._split_scheduled_samples(scheduled_samples)
            scheduled_rows = [(sample_id,) for sample_id in scheduled_ids]
            self._append_dataset(self.scheduled_dset, scheduled_rows)
            if sample_inputs is not None:
                self._append_scheduled_inputs(sample_inputs)

    def append_successful(self, sample_ids: np.array, samples: np.array):
        """
        Append successful (collected) samples.

        The method appends sample ids to 'collected_ids' and result values to
        'collected_values'.

        :param sample_ids: numpy.ndarray with collected sample ids.
        :param samples: numpy.ndarray where each row is [sample_id, value], value may be array-like itself.
        :return: None
        """
        assert samples.shape[0] == len(sample_ids)
        assert samples.shape[1] == 2
        # Append collected ids (first column)
        self._append_dataset(self.collected_ids_dset, sample_ids)

        # Determine dtype for stored result values (store as numeric array shape)
        result_type = np.dtype((float, np.array(samples[0]).shape))


        # Ensure collected_values dataset exists (resizable)
        self._make_dataset(name='collected_values', shape=(0,),
                           dtype=result_type, maxshape=(None,),
                           chunks=True)

        # Append values (converted to simple list for h5py)
        d_name = 'collected_values'
        self._append_dataset(d_name, [val for val in samples])

    def append_failed(self, failed_samples):
        """
        Append failed sample rows to the failed dataset.

        :param failed_samples: iterable of failed sample descriptors (e.g. tuples (sample_id, message))
        :return: None
        """
        self._append_dataset(self.failed_dset, failed_samples)

    def _append_dataset(self, dataset_name, values):
        """
        Append new rows to a resizable dataset.

        :param dataset_name: Name of dataset under the level group.
        :param values: Iterable of new entries; for structured dtypes supply tuples.
        :return: None
        """
        with h5py.File(self.file_name, 'a') as hdf_file:
            dataset = hdf_file[self.level_group_path][dataset_name]
            # Resize along first axis to accommodate new rows
            dataset.resize(dataset.shape[0] + len(values), axis=0)
            dataset[-len(values):] = values

    def get_scheduled_ids(self):
        """
        Read scheduled sample ids.
        """
        with FileSafe(self.file_name, 'r') as hdf_file:
            scheduled_dset = hdf_file[self.level_group_path][self.scheduled_dset]
            return [
                sample["sample_id"].decode()
                for sample in scheduled_dset[()]
            ]

    @staticmethod
    def _split_scheduled_samples(scheduled_samples):
        """
        Split scheduled samples into sample ids and optional input values.

        """
        if isinstance(scheduled_samples[0], str):
            assert all(isinstance(sample, str) for sample in scheduled_samples), \
                "Scheduled samples must either all include inputs or all be ids."
            return list(scheduled_samples), None

        sample_ids, sample_inputs = zip(*scheduled_samples)

        input_arrays = [np.asarray(sample_input) for sample_input in sample_inputs]
        input_shape = input_arrays[0].shape
        for input_array in input_arrays:
            assert input_array.shape == input_shape, \
                "Scheduled sample input shape must be constant within a level."

        return sample_ids, np.stack(input_arrays)

    def _append_scheduled_inputs(self, sample_inputs):
        """
        Append sample inputs to the parallel scheduled_inputs dataset.
        """
        with h5py.File(self.file_name, 'a') as hdf_file:
            level_group = hdf_file[self.level_group_path]
            if self.scheduled_inputs_dset not in level_group:
                level_group.create_dataset(
                    self.scheduled_inputs_dset,
                    shape=(0,) + sample_inputs.shape[1:],
                    maxshape=(None,) + sample_inputs.shape[1:],
                    dtype=sample_inputs.dtype,
                    chunks=True,
                )

            dataset = level_group[self.scheduled_inputs_dset]
            assert dataset.shape[1:] == sample_inputs.shape[1:], \
                "Scheduled sample input shape must be constant within a level."
            assert dataset.dtype == sample_inputs.dtype, \
                "Scheduled sample input dtype must be constant within a level."
            dataset.resize(dataset.shape[0] + len(sample_inputs), axis=0)
            dataset[-len(sample_inputs):] = sample_inputs

    def scheduled_samples(self):
        """
        Read scheduled samples with stored inputs when available.
        """
        scheduled_ids = self.get_scheduled_ids()
        with FileSafe(self.file_name, 'r') as hdf_file:
            level_group = hdf_file[self.level_group_path]
            if self.scheduled_inputs_dset not in level_group:
                return scheduled_ids

            scheduled_inputs = level_group[self.scheduled_inputs_dset][()]
            assert len(scheduled_ids) == len(scheduled_inputs), \
                "Scheduled sample ids and inputs have inconsistent lengths."
            return list(zip(scheduled_ids, scheduled_inputs))

    def chunks(self, n_samples=None):
        """
        Iterate over collected_values dataset chunks and yield ChunkSpec descriptors.

        :param n_samples: If provided, yield a single ChunkSpec from 0..n_samples instead of iterating actual chunks.
        :yield: ChunkSpec(chunk_id, chunk_slice, level_id)
        """
        with h5py.File(self.file_name, 'r') as hdf_file:
            if 'collected_values' not in hdf_file[self.level_group_path]:
                raise AttributeError(f"No collected values for level {self.level_id} at {self.file_name}:{self.level_group_path}."
                                     f"Found keys: {list(hdf_file[self.level_group_path].keys())}")
            dataset = hdf_file["/".join([self.level_group_path, "collected_values"])]

            if n_samples is not None:
                yield ChunkSpec(chunk_id=0, chunk_slice=slice(0, n_samples, 1), level_id=int(self.level_id))
            else:
                for chunk_id, chunk in enumerate(dataset.iter_chunks()):
                    yield ChunkSpec(chunk_id=chunk_id, chunk_slice=chunk[0], level_id=int(self.level_id))

    def collected(self, chunk_slice):
        """
        Read a slice (chunk) from the collected_values dataset.

        :param chunk_slice: slice object describing which rows to read
        :return: numpy.ndarray with the chunk rows or None if dataset missing
        """
        with h5py.File(self.file_name, 'r') as hdf_file:
            if 'collected_values' not in hdf_file[self.level_group_path]:
                return None
            dataset = hdf_file["/".join([self.level_group_path, "collected_values"])]
            return dataset[chunk_slice]

    def collected_n_items(self):
        """
        Return the number of collected items (rows) stored for this level.

        :return: int number of collected rows
        """
        with h5py.File(self.file_name, 'r') as hdf_file:
            if 'collected_values' not in hdf_file[self.level_group_path]:
                raise AttributeError("collected_values dataset not in HDF file for level {}".format(self.level_id))
            dataset = hdf_file["/".join([self.level_group_path, "collected_values"])]
            collected_n_items = len(dataset[()])
        return collected_n_items

    def get_finished_ids(self):
        """
        Return concatenated list of successful and failed sample ids for this level.

        :return: numpy.ndarray of sample id strings (successful then failed)
        """
        with h5py.File(self.file_name, 'r') as hdf_file:
            # Extract failed and successful rows and decode bytes to strings
            failed_rows = hdf_file[self.level_group_path][self.failed_dset][()]
            failed_ids = [sample["sample_id"].decode() for sample in failed_rows] if len(failed_rows) > 0 else []

            success_rows = hdf_file[self.level_group_path][self.collected_ids_dset][()]
            successful_ids = [sample["sample_id"].decode() for sample in success_rows] if len(success_rows) > 0 else []

            return np.concatenate((np.array(successful_ids), np.array(failed_ids)), axis=0)

    def get_unfinished_samples(self):
        """
        Compute unfinished scheduled samples = scheduled samples - finished ids.

        :return: list of scheduled sample ids or (sample_id, sample_input) tuples
        """
        return self._scheduled_samples_by_id(set(self.get_finished_ids()), include=False)

    def get_failed_samples(self):
        """
        Get stored scheduled samples that have failed.

        :return: list of scheduled sample ids or (sample_id, sample_input) tuples
        """
        with h5py.File(self.file_name, 'r') as hdf_file:
            failed_rows = hdf_file[self.level_group_path][self.failed_dset][()]
            failed_ids = {sample["sample_id"].decode() for sample in failed_rows}

        failed_samples = self._scheduled_samples_by_id(failed_ids, include=True)
        assert len(failed_samples) == len(failed_ids), \
            "Some failed samples have no stored scheduled input."
        return failed_samples

    def _scheduled_samples_by_id(self, sample_ids, include):
        """
        Filter scheduled samples by sample id membership.
        """
        scheduled_samples = self.scheduled_samples()
        if not scheduled_samples or isinstance(scheduled_samples[0], str):
            scheduled_ids = scheduled_samples
        else:
            scheduled_ids = self.get_scheduled_ids()

        return [
            scheduled_sample
            for scheduled_id, scheduled_sample in zip(scheduled_ids, scheduled_samples)
            if (scheduled_id in sample_ids) == include
        ]

    def clear_failed_dataset(self):
        """
        Remove and recreate the failed dataset (clears all failure records).

        :return: None
        """
        with h5py.File(self.file_name, 'a') as hdf_file:
            if self.failed_dset in hdf_file[self.level_group_path]:
                del hdf_file[self.level_group_path][self.failed_dset]
                # Recreate failed dataset as empty
                self._make_dataset(name=self.failed_dset, shape=(0,), dtype=LevelGroup.FAILED_DTYPE, maxshape=(None,),
                                   chunks=True)

    @property
    def n_ops_estimate(self):
        """
        Number of operations estimate stored as a group attribute.

        :return: float or object stored under 'n_ops_estimate' attribute (if present)
        """
        with h5py.File(self.file_name, 'r') as hdf_file:
            if 'n_ops_estimate' in hdf_file[self.level_group_path].attrs:
                return hdf_file[self.level_group_path].attrs['n_ops_estimate']

    @n_ops_estimate.setter
    def n_ops_estimate(self, n_ops_estimate):
        """
        Set 'n_ops_estimate' attribute for the level group.

        :param n_ops_estimate: numeric estimate (e.g., task weight per sample)
        :return: None
        """
        with h5py.File(self.file_name, 'a') as hdf_file:
            if 'n_ops_estimate' not in hdf_file[self.level_group_path].attrs:
                hdf_file[self.level_group_path].attrs['n_ops_estimate'] = [0., 0.]
            hdf_file[self.level_group_path].attrs['n_ops_estimate'] = n_ops_estimate
