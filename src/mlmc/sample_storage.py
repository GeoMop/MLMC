from abc import ABCMeta
from abc import abstractmethod


class SampleStorage(metaclass=ABCMeta):

    @abstractmethod
    def write_results(self, res):
        """
        Write results to storag
        """

    @abstractmethod
    def write_data(self):
        """

        """


    @abstractmethod
    def read_data(self):
        """

        """


class Memory(SampleStorage):

    def __init__(self):
        self._results = []

    def write_results(self, res):
        self._results.append(res)

    def write_data(self):
        pass

    def read_data(self):
        pass
