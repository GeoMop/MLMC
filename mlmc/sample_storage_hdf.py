import os
import numpy as np
from typing import List
from mlmc.sample_storage import SampleStorage
from mlmc.quantity.quantity_spec import QuantitySpec, ChunkSpec
import mlmc.tool.hdf5 as hdf
import warnings
warnings.simplefilter("ignore", np.VisibleDeprecationWarning)


class SampleStorageHDF(SampleStorage):
    """
    Sample's data are stored in a HDF5 file
    """

    def __init__(self, file_path):
        """
        HDF5 storage, provide method to interact with storage
        :param file_path: absolute path to hdf file (which not exists at the moment)
        """
        super().__init__()
        # If file exists load not create new file
        load_from_file = True if os.path.exists(file_path) else False

        # HDF5 interface
        self._hdf_object = hdf.HDF5(file_path=file_path, load_from_file=load_from_file)
        self._level_groups = []

        # 'Load' level groups
        if load_from_file:
            # Create level group for each level
            if len(self._level_groups) != len(self._hdf_object.level_parameters):
                for i_level in range(len(self._hdf_object.level_parameters)):
                    self._level_groups.append(self._hdf_object.add_level_group(str(i_level)))

    def _hdf_result_format(self, locations, times):
        """
        QuantitySpec data type, necessary for hdf storage
        :return:
        """
        if len(locations[0]) == 3:
            tuple_dtype = np.dtype((np.float, (3,)))
            loc_dtype = np.dtype((tuple_dtype, (len(locations),)))
        else:
            loc_dtype = np.dtype(('S50', (len(locations),)))

        result_dtype = {'names': ('name', 'unit', 'shape', 'times', 'locations'),
                        'formats': ('S50',
                                    'S50',
                                    np.dtype((np.int32, (2,))),
                                    np.dtype((np.float, (len(times),))),
                                    loc_dtype
                                    )
                        }

        return result_dtype

    def save_global_data(self, level_parameters: List[np.float], result_format: List[QuantitySpec]):
        """
        Save hdf5 file global attributes
        :param level_parameters: list of simulation steps
        :param result_format: simulation result format
        :return: None
        """
        res_dtype = self._hdf_result_format(result_format[0].locations, result_format[0].times)

        # Create file structure
        self._hdf_object.create_file_structure(level_parameters)

        # Create group for each level
        if len(self._level_groups) != len(level_parameters):
            for i_level in range(len(level_parameters)):
                self._level_groups.append(self._hdf_object.add_level_group(str(i_level)))

        # Save result format (QuantitySpec)
        self.save_result_format(result_format, res_dtype)

    def load_scheduled_samples(self):
        """
        Get scheduled samples for each level
        :return:  Dict[level_id, List[sample_id: str]]
        """
        scheduled = {}
        for level in self._level_groups:
            scheduled[int(level.level_id)] = [sample[0].decode() for sample in level.scheduled()]
        return scheduled

    def save_result_format(self, result_format: List[QuantitySpec], res_dtype):
        """
        Save result format to hdf
        :param result_format: List[QuantitySpec]
        :return: None
        """
        try:
            if self.load_result_format() != result_format:
                raise ValueError('You are setting a new different result format for an existing sample storage')
        except AttributeError:
            pass
        self._hdf_object.save_result_format(result_format, res_dtype)

    def load_result_format(self) -> List[QuantitySpec]:
        """
        Load result format
        """
        results_format = self._hdf_object.load_result_format()
        quantities = []
        for res_format in results_format:
            spec = QuantitySpec(res_format[0].decode(), res_format[1].decode(), res_format[2], res_format[3],
                                [loc.decode() for loc in res_format[4]])

            quantities.append(spec)

        return quantities

    def save_samples(self, successful, failed):
        """
        Save successful and failed samples
        :param successful: List[Tuple[sample_id: str, Tuple[ndarray, ndarray]]]
        :param failed: List[Tuple[sample_id: str, error_message: str]]
        :return: None
        """
        self._save_succesful(successful)
        self._save_failed(failed)

    def _save_succesful(self, successful_samples):
        for level, samples in successful_samples.items():
            if len(samples) > 0:
                self._level_groups[level].append_successful(np.array(samples))

    def _save_failed(self, failed_samples):
        for level, samples in failed_samples.items():
            if len(samples) > 0:
                self._level_groups[level].append_failed(samples)

    def save_scheduled_samples(self, level_id, samples: List[str]):
        """
        Append scheduled samples
        :param level_id: int
        :param samples: list of sample identifiers
        :return: None
        """
        self._level_groups[level_id].append_scheduled(samples)

    def _level_chunks(self, level_id, n_samples=None):
        return self._level_groups[level_id].chunks(n_samples)

    def sample_pairs(self):
        """
        Load results from hdf file
        :return: List[Array[M, N, 2]]
        """
        if len(self._level_groups) == 0:
            raise Exception("self._level_groups shouldn't be empty, save_global_data() method should have set it, "
                            "that method is always called from mlmc.sampler.Sampler constructor."
                            " In other cases, call save_global_data() directly")

        levels_results = list(np.empty(len(self._level_groups)))

        for level in self._level_groups:
            chunk_spec = next(self.chunks(level_id=int(level.level_id),
                                          n_samples=self.get_n_collected()[int(level.level_id)]))
            results = self.sample_pairs_level(chunk_spec)  # return all samples no chunks
            if results is None or len(results) == 0:
                levels_results[int(level.level_id)] = []
                continue
            levels_results[int(level.level_id)] = results
        return levels_results

    def sample_pairs_level(self, chunk_spec):
        """
        Get result for particular level and chunk
        :param chunk_spec: object containing chunk identifier level identifier and chunk_slice - slice() object
        :return: np.ndarray
        """
        level_id = chunk_spec.level_id
        if chunk_spec.level_id is None:
            level_id = 0
        chunk = self._level_groups[int(level_id)].collected(chunk_spec.chunk_slice)

        # Remove auxiliary zeros from level zero sample pairs
        if level_id == 0:
            chunk = chunk[:, :1, :]

        return chunk.transpose((2, 0, 1))  # [M, chunk size, 2]

    def n_finished(self):
        """
        Number of finished samples on each level
        :return: List[int]
        """
        n_finished = np.zeros(len(self._level_groups))
        for level in self._level_groups:
            n_finished[int(level.level_id)] += len(level.get_finished_ids())

        return n_finished

    def unfinished_ids(self):
        """
        List of unfinished ids
        :return: list
        """
        unfinished = []

        for level in self._level_groups:
            unfinished.extend(level.get_unfinished_ids())

        return unfinished

    def failed_samples(self):
        """
        Dictionary of failed samples
        :return: dict
        """
        failed_samples = {}
        for level in self._level_groups:
            failed_samples[str(level.level_id)] = list(level.get_failed_ids())
        return failed_samples

    def clear_failed(self):
        for level in self._level_groups:
            level.clear_failed_dataset()

    def save_n_ops(self, n_ops):
        """
        Save number of operations (time) of samples
        :param n_ops: Dict[level_id, List[overall time, number of successful samples]]
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
        Get number of estimated operations on each level
        :return: List
        """
        n_ops = list(np.zeros(len(self._level_groups)))
        for level in self._level_groups:
            if level.n_ops_estimate[1] > 0:
                n_ops[int(level.level_id)] = level.n_ops_estimate[0] / level.n_ops_estimate[1]
            else:
                n_ops[int(level.level_id)] = 0
        return n_ops

    def get_level_ids(self):
        return [int(level.level_id) for level in self._level_groups]

    def get_level_parameters(self):
        return self._hdf_object.load_level_parameters()

    def get_n_collected(self):
        """
        Get number of collected samples at each level
        :return: List
        """
        n_collected = list(np.zeros(len(self._level_groups)))
        for level in self._level_groups:
            n_collected[int(level.level_id)] = level.collected_n_items()
        return n_collected

    def get_n_levels(self):
        """
        Get number of levels
        :return: int
        """
        return len(self._level_groups)
