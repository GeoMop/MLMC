import numpy as np
from random import random
from time import sleep
from mlmc.sim.synth_simulation import SynthSimulation, SynthSimulationWorkspace


class SynthSimulationForTests(SynthSimulation):

    @staticmethod
    def calculate(config, seed, result_format):
        """
        Calculate fine and coarse sample and also extract their results
        :param config: dictionary containing simulation configuration
        :return:
        """
        sleep(random()*2)
        return SynthSimulation.calculate(config, seed, result_format)


class SynthSimulationWorkspaceForTests(SynthSimulationWorkspace):

    @staticmethod
    def calculate(config, seed, result_format):
        """
        Calculate fine and coarse sample and also extract their results
        :param config: dictionary containing simulation configuration
        :return:
        """
        sleep(random()*2)
        return SynthSimulationWorkspace.calculate(config, seed, result_format)
