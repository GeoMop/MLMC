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
        self.step = sim_param
        self._input_sample = []
        self._coarse_simulation = None

    def set_coarse_sim(self, coarse_sim):
        """
        Must be called, it is part of initialization.
        :param coarse_sim:
        :return:
        """
        pass

    def simulation_sample(self, tag):
        # Forward simulatino for generated input.
        pass

    def n_ops_estimate(self):
        # complexity function
        return self.step

    def generate_random_sample(self):
        # Create new correlated random input for both fine and (related) coarse simulation
        pass

    # def get_coarse_sample(self):
    #     pass


    def extract_result(self):
        return self._simulation_result

    @staticmethod
    def log_interpolation(sim_param_range, t_level):
        """
        Calculate particular simulation parameter
        :param sim_param_range: Tuple or list of two items, range of simulation parameters
        :param t_level: current level / total number of levels, it means 'precision' of current level fine simulation
        :return: int
        """
        assert 0 <= t_level <= 1
        return sim_param_range[0] ** (1 - t_level) * sim_param_range[1] ** t_level

    @classmethod
    def factory(cls, step_range, **kwargs):
        """
        Create specific simulation
        :param config: Simulation configuration
        :param sim_par_range: Tuple or list of two elements, number of
        :param t_level: Simulation parameter of particular simulation
        :return: Particular simulation object
        """


        return lambda t_level, kw=kwargs : cls(Simulation.log_interpolation(step_range, t_level), **kw)
