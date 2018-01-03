import numpy as np


class Level:
    """
    Call Simulation methods
    There are information about random variable - average, dispersion, number of simulation, ...
    """

    def __init__(self, simulation_size, sim, moments_object):
        """
        :param simulation_size: number of simulation steps
        :param sim: instance of object Simulation
        """

        # Number of simulation steps in previous level
        self.n_coarse = simulation_size[0]

        # Number of simulation steps in this level
        self.n_fine = simulation_size[1]

        self._data = []

        # Instance of object Simulation
        self.fine_simulation = sim.make_simulation()
        self.fine_simulation.simulation_step = self.n_fine
        self.coarse_simulation = sim.make_simulation()
        self.coarse_simulation.simulation_step = self.n_coarse

        # Initialization of variables
        self.variance = 0

        # Random variable (fine, coarse)
        self._result = []
        self._variance = 0
        self._moments = []
        self._moments_object = moments_object

        # Default number of simulations is 10
        # that is enough for estimate variance
        self._number_of_simulations = 10
        self.result = self.level()
        self.variance = np.var(self.result)

    @property
    def data(self):
        """
        Simulations data on this level
        """
        return self._data

    @data.setter
    def data(self, values):
        if tuple is not type(values):
            raise TypeError("Item of level data must be tuple")
        self._data.append(values)

    @property
    def result(self):
        """
        Simulations results (fine step simulations result - coarse step simulation result)
        """
        return self._result

    @result.setter
    def result(self, result):
        if not isinstance(result, list):
            raise TypeError("Simulation results must be list")
        self._result = result

    @property
    def variance(self):
        """
        Result variance
        """
        return self._variance

    @variance.setter
    def variance(self, value):
        self._variance = value

    @property
    def number_of_simulations(self):
        """
        Number of simulations
        """
        return self._number_of_simulations

    @number_of_simulations.setter
    def number_of_simulations(self, value):
        self._number_of_simulations = value

    @property
    def moments(self):
        """
        Result moments
        """
        if not self._moments:
            self.get_moments()
        return self._moments

    @moments.setter
    def moments(self, moments):
        if not isinstance(moments, list):
            raise TypeError("Moments must be list")
        self._moments = moments

    @property
    def moments_object(self):
        """
        Moments class instance
        """
        return self._moments_object

    @moments_object.setter
    def moments_object(self, moments_object):
        self._moments_object = moments_object

    def get_moments(self):
        """
        Get moments from results of simulations on this level
        :return: array, moments
        """
        self.moments = self.moments_object.level_moments(self)

    def n_ops_estimate(self):
        """
        :return: fine simulation n
        """
        return self.fine_simulation.simulation_step

    def level(self):
        """
        Implements level of MLMC
        Call Simulation methods
        Set simulation data
        :return: array      self.Y
        """

        for _ in range(self.number_of_simulations):
            self.fine_simulation.random_array()
            fine_step_result = self.fine_simulation.cycle(self.n_fine)
            self.coarse_simulation.set_random_array(self.fine_simulation.get_random_array())

            if self.n_coarse != 0:

                coarse_step_result = self.coarse_simulation.cycle(self.n_fine)
                self.result.append(fine_step_result - coarse_step_result)
            else:
                self.result.append(fine_step_result)

            # Save simulation data
            self.data = (self.fine_simulation.simulation_result, self.coarse_simulation.simulation_result)

        return self.result
