import os
import numpy as np
from typing import List
from mlmc.sample_storage import SampleStorage
from mlmc.quantity.quantity_spec import QuantitySpec, ChunkSpec
import mlmc.tool.hdf5 as hdf


class SampleStorageHDF(SampleStorage):
    """
    Store and manage sample data in an HDF5 file.

    This implementation of the SampleStorage interface provides efficient
    persistent storage for MLMC simulation results using HDF5.
    """

    def __init__(self, file_path):
        """
        Initialize the HDF5 storage and create or load the file structure.

        :param file_path: Absolute path to the HDF5 file.
                          If the file exists, it will be loaded instead of created.
        """
        super().__init__()
        load_from_file = os.path.exists(file_path)

        # HDF5 interface
        self._hdf_object = hdf.HDF5(file_path=file_path, load_from_file=load_from_file)
        self._level_groups = []

        # Load existing level groups if file already contains data
        if load_from_file:
            if len(self._level_groups) != len(self._hdf_object.level_parameters):
                for i_level in range(len(self._hdf_object.level_parameters)):
                    self._level_groups.append(self._hdf_object.add_level_group(str(i_level)))

    def _hdf_result_format(self, locations, times):
        """
        Construct an appropriate dtype for QuantitySpec data representation in HDF5.

        :param locations: List of spatial locations (as coordinates or identifiers).
        :param times: List of time steps.
        :return: Numpy dtype describing the QuantitySpec data structure.
        """
        if len(locations[0]) == 3:
            tuple_dtype = np.dtype((float, (3,)))
            loc_dtype = np.dtype((tuple_dtype, (len(locations),)))
        else:
            loc_dtype = np.dtype(('S50', (len(locations),)))

        result_dtype = {
            'names': ('name', 'unit', 'shape', 'times', 'locations'),
            'formats': (
                'S50',
                'S50',
                np.dtype((np.int32, (2,))),
                np.dtype((float, (len(times),))),
                loc_dtype
            )
        }

        return result_dtype

    def save_global_data(self, level_parameters: List[float], result_format: List[QuantitySpec]):
        """
        Save HDF5 global attributes including simulation parameters and result format.

        :param level_parameters: List of simulation level parameters (e.g., mesh sizes).
        :param result_format: List of QuantitySpec objects describing result quantities.
        :return: None
        """
        res_dtype = self._hdf_result_format(result_format[0].locations, result_format[0].times)
        self._hdf_object.create_file_structure(level_parameters)

        # Create HDF5 groups for each simulation level
        if len(self._level_groups) != len(level_parameters):
            for i_level in range(len(level_parameters)):
                self._level_groups.append(self._hdf_object.add_level_group(str(i_level)))

        self.save_result_format(result_format, res_dtype)

    def load_scheduled_samples(self):
        """
        Load scheduled samples from storage.

        :return: Dict[level_id, List[sample_id: str]]
        """
        scheduled = {}
        for level in self._level_groups:
            scheduled[int(level.level_id)] = [sample[0].decode() for sample in level.scheduled()]
        return scheduled

    def save_result_format(self, result_format: List[QuantitySpec], res_dtype):
        """
        Save result format metadata to HDF5.

        :param result_format: List of QuantitySpec objects defining stored quantities.
        :param res_dtype: Numpy dtype for structured storage.
        :return: None
        """
        try:
            if self.load_result_format() != result_format:
                raise ValueError(
                    "Attempting to overwrite an existing result format with a new incompatible one."
                )
        except AttributeError:
            pass

        self._hdf_object.save_result_format(result_format, res_dtype)

    def load_result_format(self) -> List[QuantitySpec]:
        """
        Load and reconstruct the result format from HDF5.

        :return: List of QuantitySpec objects.
        """
        results_format = self._hdf_object.load_result_format()
        quantities = []
        for res_format in results_format:
            spec = QuantitySpec(
                res_format[0].decode(),
                res_format[1].decode(),
                res_format[2],
                res_format[3],
                [loc.decode() for loc in res_format[4]]
            )
            quantities.append(spec)
        return quantities

    def save_samples(self, successful, failed):
        """
        Save successful and failed samples to the HDF5 storage.

        :param successful: Dict[level_id, List[Tuple[sample_id: str, (fine, coarse)]]]
        :param failed: Dict[level_id, List[Tuple[sample_id: str, error_message: str]]]
        :return: None
        """
        self._save_successful(successful)
        self._save_failed(failed)

    def _save_successful(self, successful_samples):
        """
        Append successful sample results to the appropriate level group.

        :param successful_samples: Dict[level_id, List[Tuple[sample_id, (fine, coarse)]]]
        :return: None
        """
        for level, samples in successful_samples.items():
            if len(samples) > 0:
                self._level_groups[level].append_successful(np.array(samples, dtype=object))

    def _save_failed(self, failed_samples):
        """
        Append failed sample identifiers and messages.

        :param failed_samples: Dict[level_id, List[Tuple[sample_id, error_message]]]
        :return: None
        """
        for level, samples in failed_samples.items():
            if len(samples) > 0:
                self._level_groups[level].append_failed(samples)

    def save_scheduled_samples(self, level_id, samples: List[str]):
        """
        Append scheduled sample identifiers for a specific level.

        :param level_id: Integer level identifier.
        :param samples: List of sample identifiers.
        :return: None
        """
        self._level_groups[level_id].append_scheduled(samples)

    def _level_chunks(self, level_id, n_samples=None):
        """
        Generate chunk specifications for a given level.

        :param level_id: Level identifier.
        :param n_samples: Optional number of samples to include per chunk.
        :return: Generator of ChunkSpec objects.
        """
        return self._level_groups[level_id].chunks(n_samples)

    def sample_pairs(self):
        """
        Retrieve all sample pairs from storage.

        :return: List[np.ndarray[M, N, 2]] where M = number of results, N = number of samples.
        """
        if len(self._level_groups) == 0:
            raise Exception(
                "Level groups are not initialized. "
                "Ensure save_global_data() is called before using SampleStorageHDF."
            )

        levels_results = list(np.empty(len(self._level_groups)))

        for level in self._level_groups:
            chunk_spec = next(
                self.chunks(
                    level_id=int(level.level_id),
                    n_samples=self.get_n_collected()[int(level.level_id)]
                )
            )
            results = self.sample_pairs_level(chunk_spec)
            if results is None or len(results) == 0:
                levels_results[int(level.level_id)] = []
                continue
            levels_results[int(level.level_id)] = results
        return levels_results

    def sample_pairs_level(self, chunk_spec):
        """
        Retrieve samples for a specific level and chunk.

        :param chunk_spec: ChunkSpec containing level ID and slice information.
        :return: np.ndarray of shape [M, chunk size, 2].
        """
        level_id = chunk_spec.level_id or 0
        chunk = self._level_groups[int(level_id)].collected(chunk_spec.chunk_slice)

        # Remove auxiliary zeros from level zero sample pairs
        if level_id == 0:
            chunk = chunk[:, :1, :]

        return chunk.transpose((2, 0, 1))  # [M, chunk size, 2]

    def n_finished(self):
        """
        Count the number of finished samples for each level.

        :return: np.ndarray[int] containing finished sample counts per level.
        """
        n_finished = np.zeros(len(self._level_groups))
        for level in self._level_groups:
            n_finished[int(level.level_id)] += len(level.get_finished_ids())
        return n_finished

    def unfinished_ids(self):
        """
        Return identifiers of all unfinished samples.

        :return: List[str]
        """
        unfinished = []
        for level in self._level_groups:
            unfinished.extend(level.get_unfinished_ids())
        return unfinished

    def failed_samples(self):
        """
        Return dictionary of failed samples for each level.

        :return: Dict[str, List[str]]
        """
        failed_samples = {}
        for level in self._level_groups:
            failed_samples[str(level.level_id)] = list(level.get_failed_ids())
        return failed_samples

    def clear_failed(self):
        """
        Clear all failed sample records from storage.
        """
        for level in self._level_groups:
            level.clear_failed_dataset()

    def save_n_ops(self, n_ops):
        """
        Save the estimated number of operations (e.g., runtime) for each level.

        :param n_ops: Dict[level_id, List[total_time, num_successful_samples]]
        :return: None
        """
        for level_id, (time, n_samples) in n_ops:
            if self._level_groups[level_id].n_ops_estimate is None:
                self._level_groups[level_id].n_ops_estimate = [0., 0.]

            if n_samples > 0:
                n_ops_saved = self._level_groups[level_id].n_ops_estimate
                n_ops_saved[0] += time
                n_ops_saved[1] += n_samples
                self._level_groups[level_id].n_ops_estimate = n_ops_saved

    def get_n_ops(self):
        """
        Get the average number of operations per sample for each level.

        :return: List[float]
        """
        n_ops = list(np.zeros(len(self._level_groups)))
        for level in self._level_groups:
            if level.n_ops_estimate[1] > 0:
                n_ops[int(level.level_id)] = level.n_ops_estimate[0] / level.n_ops_estimate[1]
            else:
                n_ops[int(level.level_id)] = 0
        return n_ops

    def get_level_ids(self):
        """
        Get identifiers of all levels stored in HDF5.

        :return: List[int]
        """
        return [int(level.level_id) for level in self._level_groups]

    def get_level_parameters(self):
        """
        Load stored level parameters (e.g., step sizes or resolutions).

        :return: List[float]
        """
        return self._hdf_object.load_level_parameters()

    def get_n_collected(self):
        """
        Get the number of collected (stored) samples for each level.

        :return: List[int]
        """
        n_collected = list(np.zeros(len(self._level_groups)))
        for level in self._level_groups:
            n_collected[int(level.level_id)] = level.collected_n_items()
        return n_collected

    def get_n_levels(self):
        """
        Get total number of levels present in storage.

        :return: int
        """
        return len(self._level_groups)
