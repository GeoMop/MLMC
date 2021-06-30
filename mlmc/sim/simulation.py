from abc import ABC, abstractmethod
from typing import List
from mlmc.level_simulation import LevelSimulation
from mlmc.quantity.quantity_spec import QuantitySpec


class Simulation(ABC):

    @abstractmethod
    def level_instance(self, fine_level_params: List[float], coarse_level_params: List[float]) -> LevelSimulation:
        """
        Create LevelSimulation object which is farther used for calculation etc.
        :param fine_level_params:
        :param coarse_level_params:
        :return: LevelSimulation
        """

    @abstractmethod
    def result_format(self) -> List[QuantitySpec]:
        """
        Define simulation result format
        :return: List[QuantitySpec, ...]
        """

    @staticmethod
    @abstractmethod
    def calculate(config_dict, seed):
        """
        Method that actually run the calculation, calculate fine and coarse sample and also extract their results
        :param config_dict: dictionary containing simulation configuration, LevelSimulation.config_dict (set in level_instance)
        :param seed: random seed, int
        :return: List[fine result, coarse result], both flatten arrays (see mlmc.sim.synth_simulation._calculate())
        """
