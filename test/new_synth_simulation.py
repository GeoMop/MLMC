import numpy as np
from typing import List
from synth_simulation import SynthSimulation
from level_simulation import LevelSimulation
from new_simulation import QuantitySpec


class SimulationTest(SynthSimulation):

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
        SimulationTest.n_nans = 0
        SimulationTest.nan_fraction = config.get('nan_fraction', 0.0)
        SimulationTest.len_results = 0
        #self.step = step
        # self._result_dict = {}
        # self._coarse_simulation = None
        # self.coarse_sim_set = False

        # This attribute is obligatory
        self.need_workspace: bool = False

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
        SimulationTest.len_results += 1
        y = distr.rvs(size=1)

        if SimulationTest.n_nans / (1e-10 + SimulationTest.len_results) < SimulationTest.nan_fraction:
            SimulationTest.n_nans += 1
            y = np.nan

        return y, y

    @staticmethod
    def calculate(config, sample_workspace=None):
        print("config ", config)

        fine_random, coarse_random = SimulationTest.generate_random_samples(config["distr"])

        fine_step = config["fine"]["step"]
        coarse_step = config["coarse"]["step"]

        fine_result = SimulationTest.sample_fn(fine_random, fine_step)
        coarse_result = SimulationTest.sample_fn(coarse_random, coarse_step)

        return fine_result, coarse_result

