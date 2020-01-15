from abc import ABCMeta
from abc import abstractmethod
from typing import List
import re
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

    def _get_level(self, sample_id: str):
        return re.findall(r'L0?(\d+)_', sample_id)[0]


class Memory(SampleStorage):

    def __init__(self):
        self._results = {}
        self._result_specification = []

    def save_results(self, results):
        """
        Same result with respect to sample level
        :param results:
        :return:
        """
        for res in results:
            level = self._get_level(res[0])
            self._results.setdefault(level, []).append(res)

    def save_result_format(self, res_spec):
        self._result_specification = res_spec

    def load_result_format(self) -> List[QuantitySpec]:
        """
        Load result format
        """
        return self._result_specification

    def save_workspace(self, workspace: Workspace):
        pass

    def sample_pairs(self, level):

        return self._results
