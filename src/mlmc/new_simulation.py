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

    def prepare_workspace(self):
        pass

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

    # def fine_simulation(self):
    #     """
    #     Fine simulation object
    #     :return: Simulation object
    #     """
    #     if self._fine_simulation is None:
    #         self._fine_simulation = self._sim_factory(self._precision, int(self._level_idx))
    #     return self._fine_simulation
    #
    # def coarse_simulation(self):
    #     """
    #     Coarse simulation object
    #     :return: Simulations object
    #     """
    #     if self._previous_level is not None and self._coarse_simulation is None:
    #         self._coarse_simulation = self._previous_level.fine_simulation
    #     return self._coarse_simulation

    # @abstractmethod
    # def generate_random_samples(self):
    #     """
    #     Generate random samples
    #     """

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
    def calculate(config_dict):#-> (coarse_sample_vector, fine_sample_vector)
        pass

    @staticmethod
    def result_format()-> List[QuantitySpec]:
        pass


# class Sample:
#     """
#     Would it be enough to serialize the Sample class? Or compute it without serialization?
#     """
#
#     def __init__(self, level_id):
#         self.level_id = level_id,
#         self.id = None  # possibly static method which generates unique ids (from some existing library)
#
#         self.status = None#["Running", "Success", "Failed"]
#
#     def extract_result(self):
#         """
#         Compute result from files - possibly in separate process
#         :return:
#         """
#         result = None
