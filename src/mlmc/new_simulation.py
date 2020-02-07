import numpy as np
import attr
from abc import ABC, abstractmethod
from typing import List, Tuple, Union, Any
from mlmc.level_simulation import LevelSimulation


@attr.s(auto_attribs=True)
class QuantitySpec:
    name: str
    unit: str
    shape: Tuple[int, int]
    times: List[float]
    locations: Union[List[str], List[Tuple[float, float, float]]]
    used_attributes: List = ["name", "unit", "shape", "times", "locations"]


class Simulation(ABC):

    def __init__(self, config):
        self._config = config

    @abstractmethod
    def level_instance(self, fine_level_params: List[float], coarse_level_params: List[float])-> LevelSimulation:
        """
        Create LevelSimulation object which is farther used for calculation etc.
        :param fine_level_params:
        :param coarse_level_params:
        :return: LevelSimulation
        """

    @staticmethod
    def calculate(config_dict, sample_workspace=None):
        pass

    @staticmethod
    def result_format()-> List[QuantitySpec]:
        pass
