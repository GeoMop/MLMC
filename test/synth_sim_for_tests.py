import numpy as np
from random import random
from time import sleep
from mlmc.sim.synth_simulation import SynthSimulation, SynthSimulationWorkspace


class SynthSimulationForTests(SynthSimulation):

    @staticmethod
    def calculate(config, seed):
        """
        Calculate fine and coarse sample and also extract their results
        :param config: dictionary containing simulation configuration
        :return:
        """
        sleep(random()*2)

        fine_random, coarse_random = SynthSimulation.generate_random_samples(config["distr"], seed)

        fine_step = config["fine"]["step"]
        coarse_step = config["coarse"]["step"]

        fine_result = SynthSimulation.sample_fn(fine_random, fine_step)

        if coarse_step == 0:
            coarse_result = np.zeros(len(fine_result))
        else:
            coarse_result = SynthSimulation.sample_fn(coarse_random, coarse_step)

        if np.isnan(fine_result) or np.isnan(coarse_result):
            raise Exception("result is nan")

        quantity_format = SynthSimulation.result_format()

        results = []
        for result in [fine_result, coarse_result]:
            quantities = []
            for quantity in quantity_format:
                locations = np.array([result + i for i in range(len(quantity.locations))])
                times = np.array([locations for _ in range(len(quantity.times))])
                quantities.append(times)

            results.append(np.array(quantities))

        return results[0].flatten(), results[1].flatten()


class SynthSimulationWorkspaceForTests(SynthSimulationWorkspace):

    @staticmethod
    def calculate(config, seed):
        """
        Calculate fine and coarse sample and also extract their results
        :param config: dictionary containing simulation configuration
        :return:
        """
        sleep(random()*2)

        config_file = SynthSimulationWorkspace._read_config()
        SynthSimulationWorkspace.nan_fraction = config_file["nan_fraction"]

        fine_random, coarse_random = SynthSimulationWorkspace.generate_random_samples(config_file["distr"], seed)

        fine_step = config["fine"]["step"]
        coarse_step = config["coarse"]["step"]

        fine_result = SynthSimulation.sample_fn(fine_random, fine_step)

        if coarse_step == 0:
            coarse_result = np.zeros(len(fine_result))
        else:
            coarse_result = SynthSimulation.sample_fn(coarse_random, coarse_step)

        if np.isnan(fine_result) or np.isnan(coarse_result):
            raise Exception("result is nan")

        quantity_format = SynthSimulation.result_format()

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