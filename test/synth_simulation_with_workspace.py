import os
import numpy as np
import yaml
from scipy import stats

from typing import List
from new_simulation import Simulation
from synth_simulation import SynthSimulation
from level_simulation import LevelSimulation
from new_simulation import QuantitySpec


class SimulationTestUseWorkspace(SynthSimulation):

    n_nans = 0
    nan_fraction = 0
    len_results = 0
    result_dict = {}

    CONFIG_FILE = 'synth_sim_config.yaml'

    # Artificial simulation. Just random parameter + numerical error."""
    def __init__(self, config):
        """
        :param config: Dict:
            'distr' scipy.stats.XYZ freezed distribution
            'complexity' number of FLOPS as function of step
        :param step:
        """
        super().__init__(config)
        self.config_yaml = config["config_yaml"]

        SimulationTestUseWorkspace.n_nans = 0
        SimulationTestUseWorkspace.nan_fraction = config.get('nan_fraction', 0.0)
        SimulationTestUseWorkspace.len_results = 0

        # This attribute is obligatory
        self.need_workspace: bool = True

    def level_instance(self, fine_level_params: List[float], coarse_level_params: List[float]):
        config = {}
        config["fine"] = {}
        config["coarse"] = {}

        config["fine"]["step"] = fine_level_params[0]
        config["coarse"]["step"] = coarse_level_params[0]

        return LevelSimulation(config_dict=config, common_files=[self.config_yaml])

    @staticmethod
    def generate_random_samples(distr):
        SimulationTestUseWorkspace.len_results += 1

        if distr == "norm":
            distr = stats.norm()
        elif distr == "lognorm":
            distr = stats.lognorm()
        else:
            raise NotImplementedError("Other distributions are not implemented yet")

        y = distr.rvs(size=SimulationTestUseWorkspace.result_format()[0].shape)

        if SimulationTestUseWorkspace.n_nans / (1e-10 + SimulationTestUseWorkspace.len_results) < SimulationTestUseWorkspace.nan_fraction:
            SimulationTestUseWorkspace.n_nans += 1
            y = np.nan

        return y, y

    @staticmethod
    def calculate(config, sample_workspace):
        config_file = SimulationTestUseWorkspace._read_config(sample_workspace)

        fine_random, coarse_random = SimulationTestUseWorkspace.generate_random_samples(config_file["distr"])
        fine_step = config["fine"]["step"]
        coarse_step = config["coarse"]["step"]

        fine_result = SimulationTestUseWorkspace.sample_fn(fine_random, fine_step)
        coarse_result = SimulationTestUseWorkspace.sample_fn(coarse_random, coarse_step)

        quantity_format = SimulationTestUseWorkspace.result_format()

        results = []
        for result in [fine_result, coarse_result]:
            quantities = []
            for quantity in quantity_format:
                locations = np.array([result + i for i in range(len(quantity.locations))])
                times = np.array([locations for _ in range(len(quantity.times))])
                print("times shape ", times.shape)
                quantities.append(times)

            results.append(quantities)

        return results[0], results[1]

    @staticmethod
    def _read_config(sample_workspace):
        with open(os.path.join(sample_workspace, SimulationTestUseWorkspace.CONFIG_FILE)) as file:
            config = yaml.load(file)

        return config
