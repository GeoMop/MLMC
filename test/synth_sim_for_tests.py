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
        sleep(random()/10)
        return SynthSimulation.calculate(config, seed)


class SynthSimulationWorkspaceForTests(SynthSimulationWorkspace):

    @staticmethod
    def calculate(config, seed):
        """
        Calculate fine and coarse sample and also extract their results
        :param config: dictionary containing simulation configuration
        :return:
        """
        sleep(random()/10)
        return SynthSimulationWorkspace.calculate(config, seed)
