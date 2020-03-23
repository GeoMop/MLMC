import numpy as np
from abc import ABCMeta
from abc import abstractmethod
from typing import List, Dict
from mlmc.simulation import QuantitySpec


class SampleStorage(metaclass=ABCMeta):

    @abstractmethod
    def save_samples(self, successful_samples, failed_samples):
        """
        Write results to storage
        """

    @abstractmethod
    def save_result_format(self, res_spec: List[QuantitySpec]):
        """
        Save result format
        """

    @abstractmethod
    def load_result_format(self) -> List[QuantitySpec]:
        """
        Load result format
        """

    @abstractmethod
    def save_global_data(self, result_format: List[QuantitySpec], step_range=None):
        """
        Save global data, at the moment: result_format, step_range
        """

    @abstractmethod
    def save_scheduled_samples(self):
        """
        Save scheduled samples ids
        """

    @abstractmethod
    def load_scheduled_samples(self):
        """
        Load scheduled samples
        :return: Dict[level_id, List[sample_id: str]]
        """

    @abstractmethod
    def sample_pairs(self):
        """
        Get results from storage
        :return: List[Array[M, N, 2]]
        """

    @abstractmethod
    def n_finished(self):
        """
        Number of finished samples
        :return: List
        """

    @abstractmethod
    def save_n_ops(self, n_ops: Dict[int, List[float]]):
        """
        Save number of operations (time)
        :param n_ops: Dict[level_id, List[overall time, number of valid samples]]
        """

    @abstractmethod
    def get_n_ops(self):
        """
        Number of operations (time) per sample for each level
        :return: List[float]
        """


class Memory(SampleStorage):

    def __init__(self):
        self._failed = {}
        self._results = {}
        self._successful_sample_ids = {}
        self._scheduled = {}
        self._result_specification = []
        self._n_ops = {}
        self._n_finished = {}

    def save_samples(self, successful_samples, failed_samples):
        """
        Save successful samples - store result pairs
             failed samples - store sample ids and corresponding error messages
        :return:
        """
        self._save_successful(successful_samples)
        self._save_failed(failed_samples)

    def save_global_data(self, result_format, step_range=None):
        self.save_result_format(result_format)

    def _save_successful(self, samples):
        """
        Save successful samples
        :param samples: List[Tuple[sample_id: str, Tuple[ndarray, ndarray]]]
        :return: None
        """
        for level_id, res in samples.items():
            res = np.array(res)
            fine_coarse_res = res[:, 1]

            result_type = np.dtype((np.float, np.array(fine_coarse_res[0]).shape))
            results = np.empty(shape=(len(res),), dtype=result_type)
            results[:] = [val for val in fine_coarse_res]

            # Save sample ids
            self._successful_sample_ids.setdefault(level_id, []).extend(res[:, 0])

            if level_id not in self._n_finished:
                self._n_finished[level_id] = 0

            self._n_finished[level_id] += results.shape[0]

            if level_id not in self._results:
                self._results[level_id] = results
            else:
                self._results[level_id] = np.concatenate((self._results[level_id], results), axis=0)

    def _save_failed(self, samples):
        """
        Save failed ids and error messages
        :param samples: List[Tuple[sample_id: str, error_message: str]]
        :return: None
        """
        for level_id, res in samples.items():
            self._failed.setdefault(level_id, []).extend(res)

            if level_id not in self._n_finished:
                self._n_finished[level_id] = 0
            else:
                self._n_finished[level_id] += len(res)

    def save_result_format(self, res_spec: List[QuantitySpec]):
        """
        Save sample result format
        :param res_spec: List[QuantitySpec]
        :return: None
        """
        self._result_specification = res_spec

    def n_finished(self):
        """
        Number of finished samples on each level
        :return: List
        """
        n_finished = np.empty(max(self._n_finished.items(), key=lambda k: k[0])[0]+1)
        for level_id, n_fin in self._n_finished.items():
            n_finished[level_id] = n_fin

        return n_finished

    def load_result_format(self) -> List[QuantitySpec]:
        """
        Load result format
        """
        return self._result_specification

    def save_scheduled_samples(self, level_id, samples):
        """
        Save scheduled sample ids
        :param level_id: int
        :param samples: List[str]
        :return: None
        """
        self._scheduled.setdefault(level_id, []).extend(samples)

    def load_scheduled_samples(self):
        """
        :return: Dict[level_id, List[sample_id: str]]
        """
        return self._scheduled

    def sample_pairs(self):
        """
        Sample results split to numpy arrays
        :return: List[Array[M, N, 2]]
        """
        levels_results = list(np.empty(len(np.max(self._results.keys()))))
        for level_id, results in self._results.items():
            levels_results[level_id] = results.transpose((2, 0, 1))

        return levels_results

    def save_n_ops(self, n_ops):
        """
        Save number of operations
        :param n_ops: Dict[level_id, List[time, number of valid samples]]
        :return: None
        """
        for level, (time, n_samples)in n_ops.items():
            if level not in self._n_ops or n_samples == 0:
                self._n_ops[level] = 0
  
            self._n_ops[level] += time/n_samples

    def get_n_ops(self):
        """
        Get number of operations on each level
        :return: List[float]
        """
        n_ops = list(np.empty(len(np.max(self._n_ops.keys()))))
        for level, time in self._n_ops.items():
            n_ops[level] = time
        return n_ops
