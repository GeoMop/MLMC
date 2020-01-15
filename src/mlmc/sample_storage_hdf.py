import numpy as np
import re
from typing import Tuple, List
from sample_storage import SampleStorage
from new_simulation import QuantitySpec
import hdf5 as hdf


# Starts from scratch
class SampleStorageHDF(SampleStorage):

    def __init__(self, file_path):
        """
        Create new HDF5 file and provide method to interact with storage
        :param file_path: absolute path to hdf file (which not exists at the moment)
        """
        print("file path ", file_path)
        self._hdf_object = hdf.HDF5(file_path=file_path)
        # This storage starts from blank file
        self._hdf_object.clear_groups()

        self._sample_dtype = None
        self._level_groups = []

    def save_global_data(self, step_range: Tuple[np.float, np.float], n_levels: np.int, result_format: List[QuantitySpec]):
        self._hdf_object.init_header(step_range=step_range,
                                     n_levels=n_levels)

        # Create group for each level
        for i_level in range(n_levels):
            self._level_groups.append(self._hdf_object.add_level_group(str(i_level)))

        # Save result format (QuantitySpec)
        self.save_result_format(result_format)

    def save_result_format(self, result_format: List[QuantitySpec]):
        """

        :param result_format:
        :return:
        """
        sample_dtypes = []
        for res_format in result_format:
            res_dtype = np.dtype((np.dtype((np.float, (len(res_format.times),
                                                       len(res_format.locations),
                                                       res_format.shape[0],
                                                       res_format.shape[1])
                                            ))
                                  ))

            sample_dtypes.append(res_dtype)

        sample_dtype = np.dtype((tuple(dtype for dtype in sample_dtypes)))

        print("sample dtype ", sample_dtype)
        self._sample_dtype = sample_dtype

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

    def save_results(self, results: List[Tuple[str, Tuple[np.array, np.array], str]]):
        """
        Simulation samples results
        :param results: [(sample_id, (fine result, coarse result), message)]
        :return: None
        """
        #  @TODO: extract failed samples
        #  @TODO: split to levels should be in general class
        level_samples = {}
        for res in results:
            level = re.findall(r'L0?(\d+)_', res[0])[0]
            level_samples.setdefault(level, []).append(res)

        for key, samples in level_samples.items():
            # @TODO: append_collected refactoring
            self._level_groups[int(key)].sample_dtype = self._sample_dtype

            self._level_groups[int(key)].append_collected(np.array(samples))

    def save_scheduled_samples(self, level_id: int, samples: List[str]):
        """
        Append scheduled samples
        :param level_id: int
        :param samples: list of sample identifiers
        :return: None
        """
        self._level_groups[level_id].append_scheduled(samples)

    def save_workspace(self, workspace=None):
        self._hdf_object.save_workspace_attrs("workdir", "job_dir")

    def write_data(self):
        pass

    def sample_pairs(self):
        pass


# It keeps its content
class SampleStorageHDFPreserve(SampleStorage):

    def __init__(self, file_name, work_dir, ):
        self._hdf_object = hdf.HDF5(file_name=file_name,
                                    work_dir=work_dir)

        self._hdf_object.clear_groups()

        self._hdf_object.add_level_group()

    def save_global_data(self, step_range: Tuple[np.float, np.float], result_format: List[QuantitySpec]):
        self._hdf_object.init_header(step_range=self.step_range,
                                     n_levels=self._n_levels)

    def save_results(self, res):
        pass

    def save_result_specification(self, res_spec):
        self._hdf_object.add

    def write_data(self):
        pass

    def sample_pairs(self):
        pass