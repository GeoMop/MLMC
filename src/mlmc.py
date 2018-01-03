import numpy as np
from src.mc_level import Level


class MLMC:
    """
    Multi level monte carlo method
    """
    def __init__(self, number_of_levels, n_fine, n_coarse, sim):
        """
        :param number_of_levels:           Number of levels
        :param n_fine:      Maximum steps of one simulation
        :param n_coarse:    Minimum steps of one simulation
        :param sim:         Instance of object Simulation
        """

        self.simulation = sim
        # Initialization of variables
        self._levels = []
        self.target_time = None
        self.target_variance = None
        self.number_of_levels = number_of_levels
        self.n_fine = n_fine
        self.n_coarse = n_coarse
        self._number_of_simulations = None
        self.moments_function = None
        self.moments_object = None
        self.index = 0

    @property
    def levels(self):
        """
        Get monte carlo method levels
        """
        return self._levels

    @levels.setter
    def levels(self, levels):
        if not isinstance(levels, list):
            raise TypeError("Levels must be list")
        self._levels = levels

    @property
    def number_of_simulations(self):
        """
        Number of simulations
        """
        return self._number_of_simulations

    @number_of_simulations.setter
    def number_of_simulations(self, num_of_sim):
        if len(num_of_sim) != self.number_of_levels:
            raise ValueError("Number of simulations must be list and the length of list must be same"
                             " as the number of levels")

        self._number_of_simulations = num_of_sim

    def monte_carlo(self, optimization_type, value):
        """
        Implements complete multilevel Monte Carlo Method
        Calls other necessary methods
        :param optimization_type: int        1 for fix variance and other for fix time
        :param value: float     value of variance or time according to type
        """

        self.index = 0

        if optimization_type == 1:
            self.target_variance = value
        else:
            self.target_time = value

        if self.number_of_levels > 1:
            for _ in range(self.number_of_levels):
                self.create_level()
        else:
            level = Level([0, self.n_fine], self.simulation, self.moments_object)
            self._levels.append(level)

        # Count new number of simulations according to variance of time
        if self.target_variance is not None:
            new_num_of_simulations = self.count_num_of_sim_variance()
        elif self.target_time is not None:
            new_num_of_simulations = self.count_num_of_sim_time()

        self.count_new_num_of_simulations(new_num_of_simulations)

    def count_new_num_of_simulations(self, num_of_simulations):
        """
        For each level counts further number of simulations by appropriate N
        :param num_of_simulations: array      new number of simulations for each level
        """

        for step, level in enumerate(self._levels):
            if self.number_of_simulations is not None:
                num_of_simulations = self.number_of_simulations

            if level.number_of_simulations < num_of_simulations[step]:
                level.number_of_simulations = num_of_simulations[step] - level.number_of_simulations
                # Launch further simulations
                level.level()
                level.number_of_simulations = num_of_simulations[step]

    def create_level(self):
        """
        Create new level add its to the array of levels
        Call method for counting number of simulation steps
        Pass instance of Simulation to Level
        """

        level = Level(self.count_small_n(), self.simulation, self.moments_object)
        self._levels.append(level)

    def count_small_n(self):
        """
        Count number of steps for level
        :return: array  [n_coarse, n_fine]
        """
        fine_steps = np.power(np.power((self.n_fine/self.n_coarse), (1/(self.number_of_levels-1))), self.index)*self.n_coarse

        # Coarse number of simulation steps from previouse level
        if len(self._levels) > 0:
            level = self._levels[self.index - 1]
            coarse_steps = level.n_ops_estimate()

        else:
            coarse_steps = 0
        self.index += 1
        return [np.round(coarse_steps).astype(int), np.round(fine_steps).astype(int)]

    def count_num_of_sim_time(self):
        """
        For each level counts new N according to target_time
        :return: array
        """
        num_of_simulations_time = []
        amount = self.count_sum()

        # Loop through levels
        # Count new number of simulations for each level
        for level in self._levels:
            new_num_of_sim = np.round((self.target_time * np.sqrt(level.variance / level.n_ops_estimate()))
                                      / amount).astype(int)

            num_of_simulations_time.append(new_num_of_sim)
        return num_of_simulations_time

    def count_num_of_sim_variance(self):
        """
        For each level counts new N according to target_variance
        :return: array
        """
        num_of_simulations_var = []
        amount = self.count_sum()

        # Loop through levels
        # Count new number of simulations for each level
        for level in self._levels:
            new_num_of_sim_pom = []

            for moment in level.moments:
                new_num_of_sim_pom.append(np.round((amount * np.sqrt(np.abs(moment) / level.n_ops_estimate()))
                / self.target_variance).astype(int))

            new_num_of_sim = np.max(new_num_of_sim_pom)

            #new_num_of_sim = np.round((amount * np.sqrt(level.variance / level.n_ops_estimate())) / self.target_variance).astype(int)
            num_of_simulations_var.append(new_num_of_sim)

        return num_of_simulations_var

    def count_sum(self):
        """
        Loop through levels and count sum of varinace * simulation n
        :return: float  amount
        """
        amount = 0
        for level in self._levels:
            amount += np.sqrt(level.variance * level.n_ops_estimate())

        return amount
