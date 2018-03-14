import numpy as np


class Simulation:
    """
    Parent class for simulations
    """

    def __init__(self, config=None, sim_param=0):
        """    
        :param config: Simulation configuration
        :param sim_param: Number of simulation steps
        """
        # Simulation result
        self._simulation_result = None
        self._config = config
        # Fine simulation step
        self._simulation_step = 0
        self.sim_param = sim_param
        self._input_sample = []
        self._coarse_simulation = None

    def cycle(self):
        pass

    def n_ops_estimate(self):
        pass

    def generate_random_sample(self):
        pass

    def get_coarse_sample(self):
        pass

    def set_previous_fine_sim(self, coarse_sim):
        pass

    def get_result(self):
        return self._simulation_result

    @staticmethod
    def log_interpolation(sim_param_range, t_level=None):
        """
        Calculate particular simulation parameter
        :param sim_param_range: Tuple or list of two items, range of simulation parameters
        :param t_level: current level / total number of levels, it means 'precision' of current level fine simulation
        :return: int
        """
        if t_level is None:
            return 0
        else:
            assert 0 <= t_level <= 1
            return np.round(sim_param_range[0] ** (1 - t_level) * sim_param_range[1] ** t_level).astype(int)

    @classmethod
    def make_sim(cls, config, sim_par_range, t_level=None):
        """
        Create specific simulation
        :param config: Simulation configuration
        :param sim_par_range: Tuple or list of two elements, number of  
        :param t_level: Simulation parameter of particular simulation
        :return: Particular simulation object
        """
        sim_par = Simulation.log_interpolation(sim_par_range, t_level)

        return cls(config, sim_par)