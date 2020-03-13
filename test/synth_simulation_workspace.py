import os
import yaml
import numpy as np
from typing import List
from scipy import stats
from simulation import Simulation, QuantitySpec
from level_simulation import LevelSimulation


class SynthSimulationWorkspace(Simulation):

    n_nans = 0
    nan_fraction = 0
    len_results = 0
    result_dict = {}

    CONFIG_FILE = 'synth_sim_config.yaml'

    # Artificial simulation. Just random parameter + numerical error."""
    def __init__(self, config):
        """
        :param config: Dict:
                distr= particular distribution,
                complexity=2,
                nan_fraction=fraction of failed samples
                sim_method=used method for calculating sample result
        """
        super().__init__(config)
        self.config_yaml = config["config_yaml"]

        SynthSimulationWorkspace.n_nans = 0
        SynthSimulationWorkspace.nan_fraction = config.get('nan_fraction', 0.0)
        SynthSimulationWorkspace.len_results = 0

        # This attribute is obligatory
        self.need_workspace: bool = True

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

    def level_instance(self, fine_level_params: List[float], coarse_level_params: List[float]) -> LevelSimulation:
        """

        :param fine_level_params:
        :param coarse_level_params:
        :return:
        """
        config = dict()
        config["fine"] = {}
        config["coarse"] = {}

        config["fine"]["step"] = fine_level_params[0]
        config["coarse"]["step"] = coarse_level_params[0]

        return LevelSimulation(config_dict=config,
                               common_files=[self.config_yaml],
                               task_size=self.n_ops_estimate(fine_level_params[0]),
                               need_sample_workspace=self.need_workspace)

    @staticmethod
    def generate_random_samples(distr, seed):
        """
        Generate random samples from given distribution
        :param distr: scipy distribution
        :param seed: uint32
        :return: fine sample, coarse sample
        """
        SynthSimulationWorkspace.len_results += 1

        if distr == "norm":
            distr = stats.norm(loc=1, scale=2)
        else:
            raise NotImplementedError("Other distributions are not implemented yet")

        distr.random_state = np.random.RandomState(seed)
        y = distr.rvs(size=1)

        if SynthSimulationWorkspace.n_nans / (1e-10 + SynthSimulationWorkspace.len_results) < SynthSimulationWorkspace.nan_fraction:
            SynthSimulationWorkspace.n_nans += 1
            y = [np.nan]

        return y, y

    @staticmethod
    def calculate(config, seed):
        """
        Calculate fine and coarse sample and also extract their results
        :param config: dictionary containing simulation configuration
        :return:
        """
        config_file = SynthSimulationWorkspace._read_config()
        SynthSimulationWorkspace.nan_fraction = config_file["nan_fraction"]

        fine_random, coarse_random = SynthSimulationWorkspace.generate_random_samples(config_file["distr"], seed)

        fine_step = config["fine"]["step"]
        coarse_step = config["coarse"]["step"]

        fine_result = SynthSimulationWorkspace.sample_fn(fine_random, fine_step)

        if coarse_step == 0:
            coarse_result = np.zeros(len(fine_result))
        else:
            coarse_result = SynthSimulationWorkspace.sample_fn(coarse_random, coarse_step)

        if np.isnan(fine_result) or np.isnan(coarse_result):
            raise Exception("result is nan")

        quantity_format = SynthSimulationWorkspace.result_format()

        results = []
        for result in [fine_result, coarse_result]:
            quantities = []

            for quantity in quantity_format:
                if coarse_step == 0:
                    locations = np.array([result for _ in range(len(quantity.locations))])
                else:
                    locations = np.array([result + i for i in range(len(quantity.locations))])
                times = np.array([locations for _ in range(len(quantity.times))])
                quantities.append(times)

            results.append(np.array(quantities))

        return results[0].flatten(), results[1].flatten()

    def n_ops_estimate(self, step):
        # @TODO: how to determine n ops
        return (1 / step) ** 2 * np.log(max(1 / step, 2.0))

    @staticmethod
    def _read_config():
        with open(os.path.join(os.getcwd(), SynthSimulationWorkspace.CONFIG_FILE)) as file:
            config = yaml.load(file)

        return config

    def result_format(self):
        return SynthSimulationWorkspace.result_format()

    @staticmethod
    def result_format() -> List[QuantitySpec]:
        """
        Result format
        :return:
        """
        spec1 = QuantitySpec(name="length", unit="m", shape=(2, 1), times=[1, 2, 3], locations=['10', '20'])
        spec2 = QuantitySpec(name="width", unit="mm", shape=(2, 1), times=[1, 2, 3], locations=['30', '40'])
        # spec1 = QuantitySpec(name="length", unit="m", shape=(2, 1), times=[1, 2, 3], locations=[(1, 2, 3), (4, 5, 6)])
        # spec2 = QuantitySpec(name="width", unit="mm", shape=(2, 1), times=[1, 2, 3], locations=[(7, 8, 9), (10, 11, 12)])
        return [spec1, spec2]