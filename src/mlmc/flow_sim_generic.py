import numpy as np
import test.simulation_test
import test.simulation_shooting


class FlowSimGeneric:
    """
    Parent class for simulations
    """

    def __init__(self):
        self._sim_param_range = None

    @property
    def sim_param_range(self):
        """
        Simulations param range, range of number of simulation steps
        :return: array of two items 
        """
        return self._sim_param_range

    @sim_param_range.setter
    def sim_param_range(self, sim_param_range):
        if not isinstance(sim_param_range[0], (float, int)) or not isinstance(sim_param_range[1], (float, int)):
            raise ValueError("Simulation parameter range must contains just floats or ints")

        if sim_param_range[0] < 0 or sim_param_range[1] < 0:
            raise ValueError("Simulations parameter ranges must be positive values")

        self._sim_param_range = sim_param_range

    def interpolate_precision(self, t_level=None):
        """
        't' is a parameter from interval [0,1], where 0 corresponds to the lower bound
        and 1 to the upper bound of the simulation parameter range.
        :param t_level: float 0 to 1
        :return: specific simulation 
        """
        if t_level is None:
            sim_param = 0
        else:
            assert 0 <= t_level <= 1
            # logarithmic interpolation
            sim_param = np.round(self.sim_param_range[0] ** (1 - t_level) * self.sim_param_range[1] ** t_level).astype(int)

            #return FlowSim(self.flow_setup, sim_param)
        sim = test.simulation_test.SimulationTest(sim_param)
        #sim = test.simulation_shooting.SimulationShooting(np.array([0, 0]), np.array([10, 0]),
        #                                                 np.array([-100, 200, -300, 400]), 10,
        #                                                  np.array([0, -1]), sim_param)
        sim.sim_param = sim_param
        return sim