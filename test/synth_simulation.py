import numpy as np
from typing import List
from new_simulation import Simulation
from level_simulation import LevelSimulation
from new_simulation import QuantitySpec


class SynthSimulation(Simulation):

    n_nans = 0
    nan_fraction = 0
    len_results = 0
    result_dict = {}

    # Artificial simulation. Just random parameter + numerical error."""
    def __init__(self, config):
        """
        :param config: Dict:
            'distr' scipy.stats.XYZ freezed distribution
            'complexity' number of FLOPS as function of step
        :param step:
        """
        super().__init__(config)
        self.config = config
        #self.nan_fraction =
        SynthSimulation.n_nans = 0
        SynthSimulation.nan_fraction = config.get('nan_fraction', 0.0)
        SynthSimulation.len_results = 0
        #self.step = step
        # self._result_dict = {}
        # self._coarse_simulation = None
        # self.coarse_sim_set = False

        # This attribute is obligatory
        self.need_workspace: bool = False

    @staticmethod
    def sample_fn(x, h):
        """
        Calculates the simulation sample
        :param x: Distribution sample
        :param h: Simluation step
        :return: sample
        """
        # This function can cause many outliers depending on chosen domain of moments function
        return x + h * np.sqrt(1e-4 + np.abs(x))

    @staticmethod
    def sample_fn_no_error(x, h):
        """
        Calculates the simulation sample
        :param x: Distribution sample
        :param h: Simluation step
        :return: sample
        """
        return x

    def level_instance(self, fine_level_params: List[float], coarse_level_params: List[float]):
        config = {}
        config["fine"] = {}
        config["coarse"] = {}
        config["fine"]["step"] = fine_level_params[0]
        config["coarse"]["step"] = coarse_level_params[0]

        config["distr"] = self.config["distr"]

        return LevelSimulation(config_dict=config)

    @staticmethod
    def generate_random_samples(distr):
        SynthSimulation.len_results += 1
        y = distr.rvs(size=1)

        if SynthSimulation.n_nans / (1e-10 + SynthSimulation.len_results) < SynthSimulation.nan_fraction:
            SynthSimulation.n_nans += 1
            y = np.nan

        return y, y

    @staticmethod
    def calculate(config, sample_workspace=None):
        print("config ", config)

        fine_random, coarse_random = SynthSimulation.generate_random_samples(config["distr"])

        fine_step = config["fine"]["step"]
        coarse_step = config["coarse"]["step"]

        fine_result = SynthSimulation.sample_fn(fine_random, fine_step)
        coarse_result = SynthSimulation.sample_fn(coarse_random, coarse_step)



        return fine_result, coarse_result

    def n_ops_estimate(self):
        return (1 / self.step) ** self.config['complexity'] * np.log(max(1 / self.step, 2.0))

    @staticmethod
    def result_format() -> List[QuantitySpec]:
        # When to get this format? Is it mutable?
        spec1 = QuantitySpec(name="length", unit="m", shape=(2, 1), times=[1, 2, 3], locations=[10, 20])
        spec2 = QuantitySpec(name="width", unit="mm", shape=(2, 1), times=[1, 2, 3], locations=[10, 20])
        return [spec1, spec2]

    # @staticmethod
    # def extract_result(sample_id):
    #     # sample time, not implemented in this simulation
    #     time = np.random.random()
    #     return SimulationTest.result_dict[sample_id]

