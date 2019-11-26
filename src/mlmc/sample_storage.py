from abc import ABCMeta
from abc import abstractmethod


class SampleStorage(metaclass=ABCMeta):
    @abstractmethod
    def write_data(self):
        pass

    @abstractmethod
    def read_data(self):
        pass


class HDF(SampleStorage):
    def write_data(self):
        pass

    def read_data(self):
        pass


class InMemory(SampleStorage):
    def write_data(self):
        pass

    def read_data(self):
        pass
