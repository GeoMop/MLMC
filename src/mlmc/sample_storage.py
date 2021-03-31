import numpy as np
from abc import ABCMeta
from abc import abstractmethod
from typing import List, Dict
from mlmc.quantity_spec import QuantitySpec, ChunkSpec


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
    def save_global_data(self, result_format: List[QuantitySpec], level_parameters=None):
        """
        Save global data, at the moment: _result_format, level_parameters
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
        :return: Dict[_level_id, List[sample_id: str]]
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
        :param n_ops: Dict[_level_id, List[overall time, number of valid samples]]
        """

    @abstractmethod
    def get_n_ops(self):
        """
        Number of operations (time) per sample for each level
        :return: List[float]
        """

    @abstractmethod
    def unfinished_ids(self):
        """
        Get unfinished sample's ids
        :return: list
        """

    @abstractmethod
    def get_level_ids(self):
        """
        Get number of levels
        :return: int
        """

    @abstractmethod
    def get_n_levels(self):
        """
        Get number of levels
        :return: int
        """

    @abstractmethod
    def get_level_parameters(self):
        """
        Get level parameters
        :return: list
        """

    @abstractmethod
    def get_n_collected(self):
        """
        Get number of collected results at each evel
        :return: list
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
        self._level_parameters = []
        super().__init__()

    def save_samples(self, successful_samples, failed_samples):
        """
        Save successful samples - store result pairs
             failed samples - store sample ids and corresponding error messages
        :return:
        """
        self._save_successful(successful_samples)
        self._save_failed(failed_samples)

    def save_global_data(self, result_format, level_parameters=None):
        self.save_result_format(result_format)
        self._level_parameters = level_parameters

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
        :return: Dict[_level_id, List[sample_id: str]]
        """
        return self._scheduled

    def sample_pairs(self):
        """
        Sample results split to numpy arrays
        :return: List[Array[M, N, 2]]
        """
        levels_results = list(np.empty(len(np.max(self._results.keys()))))

        for level_id in self.get_level_ids():
            results = self.sample_pairs_level(ChunkSpec(level_id))
            levels_results[level_id] = results

        return levels_results

    def sample_pairs_level(self, chunk_spec):
        """
        Get samples for given level, chunks does not make sense in Memory storage so all data are retrieved at once
        :param chunk_spec: ChunkSpec instance, contains level_id, chunk_id, possibly n_samples
        :return: np.ndarray
        """
        if chunk_spec.chunk_id != 0:
            raise StopIteration

        results = self._results[int(chunk_spec.level_id)]
        n_samples = chunk_spec.n_samples\
            if chunk_spec.n_samples is not None and chunk_spec.n_samples < results.shape[0]\
            else results.shape[0]

        # Handle scalar simulation result
        #@TODO: think it over again
        if len(results.shape) != 3:
            results = results.reshape(results.shape[0], results.shape[1],
                                      1 if np.prod(results.shape) == results.shape[0] * results.shape[1] else
                                      int(np.prod(results.shape) / results.shape[0] * results.shape[1]))

        # Remove auxiliary zeros from level zero sample pairs
        if chunk_spec.level_id == 0:
            results = results[:, :1, :]

        return results[:n_samples, ...].transpose((2, 0, 1))  # [M, N, 2]

    def save_n_ops(self, n_ops):
        """
        Save number of operations
        :param n_ops: Dict[_level_id, List[time, number of valid samples]]
        :return: None
        """
        for level, (time, n_samples) in n_ops:
            if level not in self._n_ops:
                self._n_ops[level] = 0

            if n_samples != 0:
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

    def unfinished_ids(self):
        """
        We finished all samples in memory
        :return:
        """
        return []

    def get_level_ids(self):
        return list(self._results.keys())

    def get_n_collected(self):
        """
        Number of collected samples at each level
        :return: List
        """
        n_collected = list(np.zeros(len(self._results)))
        for level_id, result in self._results.items():
            n_collected[int(level_id)] = len(result)
        return n_collected

    def get_n_levels(self):
        """
        Get number of levels
        :return: int
        """
        return len(self._results)

    def get_level_parameters(self):
        return self._level_parameters
