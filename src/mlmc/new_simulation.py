import attr
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional, Any


@attr.s
class QuantitySpec:
    name: str
    unit: str
    shape: Tuple[int, int]
    times: List[float]
    locations: List[str]


class Simulation(ABC):

    """
    Previous version:
        mc_level: create fine_simulation - it requires params: precision, level_id


                  make_sample_pair() - set_coarse_sim(): set fine_simulation from previous level as coarse_simulation
                                                         at current level
                                     - generate_random_sample(): make_fields and assign them to fine_sim.input_samples
                                                                 and coarse_sim.input_samples
                                     - call fine_simulation.simulation_sample() and coarse_simulation.simulation_sample()


    """

    def __init__(self, config):
        self._config = config

    def serialize_data(self):
        pass

    def deserialize_data(self):
        pass

    def create_coarse_sample(self, fine_sample):
        pass

    @staticmethod
    def extract_result(sample):
        pass

    def _simulation_sample(self):
        """
        Create simulation samples
        :return:
        """

    def _prepare_work_space(self):
        pass

    @abstractmethod
    def level_instance(self, fine_level_params: List[float], coarse_level_params: List[float])-> \
            Tuple[Dict[Any, Any], Optional[List[str]]]:
        """

        :param fine_level_params:
        :param coarse_level_params:
        :return:
        """

    @staticmethod
    def calculate(config_dict, sample_workspace=None):#-> (coarse_sample_vector, fine_sample_vector)
        pass

    @staticmethod
    def result_format()-> List[QuantitySpec]:
        pass
