import os
import numpy as np
from typing import List
from sample_storage import SampleStorage
from simulation import QuantitySpec
import hdf5 as hdf


# Starts from scratch
class SampleStorageHDF(SampleStorage):

    def __init__(self, file_path, append=False):
        """
        HDF5 storage, provide method to interact with storage
        :param file_path: absolute path to hdf file (which not exists at the moment)
        :param append: append to existing hdf5
        """
        # If file exists load not create new file
        load_from_file = False
        if os.path.exists(file_path):
            if append:
                load_from_file = True
            else:
                raise FileExistsError("HDF file {} already exists, use --force to delete it".format(file_path))

        # HDF5 interface
        self._hdf_object = hdf.HDF5(file_path=file_path, load_from_file=load_from_file)
        self._level_groups = []

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

    def save_global_data(self, step_range: List[np.float], result_format: List[QuantitySpec]):
        """
        Save hdf5 file global attributes
        :param step_range: list of simulation steps
        :param result_format: simulation result format
        :return: None
        """
        res_dtype = self._hdf_result_format(result_format[0].locations, result_format[0].times)

        # Create file structure
        self._hdf_object.create_file_structure(step_range)

        # Create group for each level
        for i_level in range(len(step_range)):
            self._level_groups.append(self._hdf_object.add_level_group(str(i_level)))

        # Save result format (QuantitySpec)
        self.save_result_format(result_format, res_dtype)

    def load_scheduled_samples(self):
        """
        Get scheduled samples for each level
        :return: List[List]
        """
        scheduled = list(np.empty(len(self._level_groups)))
        for level in self._level_groups:
            scheduled[int(level.level_id)] = [sample[0].decode() for sample in level.scheduled()]
        return scheduled

    def save_result_format(self, result_format: List[QuantitySpec], res_dtype):
        """
        Save result format to hdf
        :param result_format: List[QuantitySpec]
        :return: None
        """
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

    def sample_pairs(self):
        """
        Load results from hdf file
        :return: List[Array[M, N, 2]]
        """
        levels_results = list(np.empty(len(self._level_groups)))
        for level in self._level_groups:
            results = level.collected()
            if results is None or len(results) == 0:
                levels_results[int(level.level_id)] = []
                continue

            levels_results[int(level.level_id)] = results.transpose((2, 0, 1))

        return levels_results

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
            print("level.get_failed_ids() ", level.get_failed_ids())
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
        for level_id, (time, n_samples) in n_ops.items():
            if n_samples == 0:
                self._level_groups[level_id].n_ops_estimate = 0
            else:
                self._level_groups[level_id].n_ops_estimate = time/n_samples

    def get_n_ops(self):
        """
        Get number of estimated operations on each level
        :return: List
        """
        n_ops = list(np.zeros(len(self._level_groups)))
        for level in self._level_groups:
            n_ops[int(level.level_id)] = level.n_ops_estimate

        return n_ops
