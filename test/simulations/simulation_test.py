from mlmc.sim.simulation import Simulation
import numpy as np
import scipy.stats


class SimulationTest(Simulation):
    """
    Class for 'shooting' simulation 
    Inherits from Simulation
    """
    def __init__(self, config=None, sim_param=0):
        super(Simulation, self).__init__()
        self.sim_param = sim_param
        self._result_dict = {}

    def simulation_sample(self, tag):
        """
        Run simulation
        :param sim_id:    Simulation id
        """
        if self.sim_param == 0:
            x = 0
        else:
            x = self._input_sample + (1 / self.sim_param)

        self.simulation_result = x
        # vrati sim_id, result, metoda extract result
        self._result_dict[tag] = self.simulation_result

        return tag

    def generate_random_sample(self):
        self._input_sample = np.random.lognormal(0, 0.5)
        #self.input_sample = scipy.stats.lognorm.rvs(0.8)
        #self.input_sample = lognorm.rvs(s=0.5, loc=1, scale=1000)

    def get_coarse_sample(self):
        return self._input_sample

    def n_ops_estimate(self):
        return self.sim_param

    @property
    def mesh_step(self):
        return self.sim_param

    @mesh_step.setter
    def mesh_step(self, step):
        self.sim_param = step

    def set_coarse_sim(self, coarse_simulation):
        self._coarse_simulation = coarse_simulation

    def extract_result(self, sim_id):
        return self._result_dict[sim_id]