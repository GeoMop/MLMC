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

        """

    @abstractmethod
    def read_results(self):
        """

        """

    @abstractmethod
    def save_workspace(self, workspace: Workspace):
        """
        Save some workspace attributes
        :return: None
        """


class Memory(SampleStorage):

    def __init__(self):
        self._results = []
        self._result_specification = []

    def save_results(self, res):
        self._results.append(res)

    def save_result_format(self, res_spec):
        self._result_specification = res_spec

    def read_results(self):
        pass

    def save_workspace(self, workspace: Workspace):
        pass
