import os
import numpy as np
from typing import List
from sample_storage import SampleStorage
from new_simulation import QuantitySpec
import hdf5 as hdf


# Starts from scratch
class SampleStorageHDF(SampleStorage):

    def __init__(self, file_path):
        """
        HDF5 storage, provide method to interact with storage
        :param file_path: absolute path to hdf file (which not exists at the moment)
        """
        # If file exists load not create new file
        load_from_file = False
        if os.path.exists(file_path):
            load_from_file = True

        # HDF5 interface
        self._hdf_object = hdf.HDF5(file_path=file_path, load_from_file=load_from_file)
        self._level_groups = []

    def save_global_data(self, step_range: List[np.float], result_format: List[QuantitySpec]):
        """
        Save hdf5 file global attributes
        :param step_range: list of simulation steps
        :param result_format: simulation result format
        :return: None
        """
        # Create file structure
        self._hdf_object.create_file_structure(step_range)

        # Create group for each level
        for i_level in range(len(step_range)):
            self._level_groups.append(self._hdf_object.add_level_group(str(i_level)))

        # Save result format (QuantitySpec)
        self.save_result_format(result_format)

    def save_result_format(self, result_format: List[QuantitySpec]):
        """
        Save result format to hdf
        :param result_format: List[QuantitySpec]
        :return: None
        """
        self._hdf_object.save_result_format(result_format)

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
            self._level_groups[level].append_successful(np.array(samples))

    def _save_failed(self, failed_samples):
        for level, samples in failed_samples.items():
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
