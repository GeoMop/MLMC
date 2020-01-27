import numpy as np
from abc import ABCMeta
from abc import abstractmethod
from typing import List
from new_simulation import QuantitySpec
from workspace import Workspace


class SampleStorage(metaclass=ABCMeta):

    @abstractmethod
    def save_results(self, res):
        """
        Write results to storag
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
    def save_workspace(self, workspace: Workspace):
        """
        Save some workspace attributes
        :return: None
        """

    @abstractmethod
    def sample_pairs(self):
        """
        Get results from storage
        :return:
        """


class Memory(SampleStorage):

    def __init__(self):
        self._results = {}
        self._scheduled = {}
        self._result_specification = []

    def save_results(self, results):
        """
        Same result with respect to sample level
        :param results:
        :return:
        """
        print("results ", results)
        for level_id, res in results.items():

            self._results.setdefault(level_id, []).extend(res)

    def save_result_format(self, res_spec):
        self._result_specification = res_spec

    def load_result_format(self) -> List[QuantitySpec]:
        """
        Load result format
        """
        return self._result_specification

    def save_scheduled_samples(self, level_id, samples):
        self._scheduled.setdefault(level_id, []).append(samples)

    def load_scheduled_samples(self):
        return self._scheduled

    def save_workspace(self, workspace: Workspace):
        pass

    def sample_pairs(self):
        levels_results = list(np.empty(len(np.max(self._results.keys()))))
        for level_id, res in self._results.items():
            res = np.array(res)
            fine_coarse_res = res[:, 1]

            result_type = np.dtype((np.float, np.array(fine_coarse_res[0]).shape))
            results = np.empty(shape=(len(res), ), dtype=result_type)
            results[:] = [val for val in fine_coarse_res]

            levels_results[level_id] = results.transpose((2, 0, 1))

        return levels_results
