from abc import ABC, abstractmethod
from typing import List
from mlmc.level_simulation import LevelSimulation
from mlmc.quantity.quantity_spec import QuantitySpec


class Simulation(ABC):
    """
    Abstract base class for multi-level Monte Carlo (MLMC) simulations.

    Defines the interface that all concrete simulation classes must implement.
    Provides methods for creating level simulations, specifying result formats, and running calculations.
    """

    @abstractmethod
    def level_instance(self, fine_level_params: List[float], coarse_level_params: List[float]) -> LevelSimulation:
        """
        Create a LevelSimulation object for a given level.

        The LevelSimulation instance is used for sample generation and result extraction
        at both the fine and coarse levels in MLMC.

        :param fine_level_params: List of floats defining parameters for the fine simulation level.
        :param coarse_level_params: List of floats defining parameters for the coarse simulation level.
        :return: LevelSimulation instance configured for the given level parameters.
        """

    @abstractmethod
    def result_format(self) -> List[QuantitySpec]:
        """
        Define the format of the simulation results.

        This method should return a list of QuantitySpec objects, which describe the
        type, shape, and units of each quantity produced by the simulation.

        :return: List of QuantitySpec objects defining the simulation output format.
        """

    @staticmethod
    @abstractmethod
    def calculate(config_dict, seed: int):
        """
        Execute a single simulation calculation.

        This method runs the simulation for both fine and coarse levels, computes
        the results, and returns them in a flattened form suitable for MLMC analysis.

        :param config_dict: Dictionary containing simulation configuration parameters
                            (usually LevelSimulation.config_dict from level_instance).
        :param seed: Random seed (int) to ensure reproducibility of the stochastic simulation.
        :return: List containing two elements:
                 [fine_result, coarse_result], both as flattened arrays.
        """
