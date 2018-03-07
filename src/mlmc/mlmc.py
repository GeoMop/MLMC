import numpy as np
from src.mlmc.mc_level import Level


class MLMC:
    """
    Multi level monte carlo method
    """
    def __init__(self, number_of_levels, sim, moments_object):
        """
        :param number_of_levels:    Number of levels
        :param sim:                 Instance of object Simulation
        :param moments_object:      Instance of moments object
        """
        # Object of simulation
        self.simulation = sim
        # Array of level objects
        self._levels = []
        # Time of all mlmc
        self.target_time = None
        # Variance of all mlmc
        self.target_variance = None
        # Number of levels
        self.number_of_levels = number_of_levels
        # The fines simulation step

        self.num_of_simulations = []
        # It is used if want to have fixed number of simulations
        self._number_of_samples = None
        # Instance of selected moments object
        self.moments_object = moments_object
        # Current level
        self.current_level = 0
        # Calculated number of samples
        self._num_of_samples = None

        # Create levels
        for _ in range(self.number_of_levels):
            self._create_level()

    @property
    def levels(self):
        """
        return: list of Level instances
        :return: array of objects (src.mlmc.level.Level())
        """
        return self._levels

    @property
    def number_of_samples(self):
        """
        List of samples in each level
        :return: array 
        """
        return self._number_of_samples

    @number_of_samples.setter
    def number_of_samples(self, num_of_sim):
        if len(num_of_sim) < self.number_of_levels:
            raise ValueError("Number of simulations must be list")

        self._number_of_samples = num_of_sim

    def estimate_n_samples(self):
        # Count new number of simulations according to variance of time
        if self.target_variance is not None:
            self._num_of_samples = self.estimate_n_samples_from_variance()
        elif self.target_time is not None:
            self._num_of_samples = self.estimate_n_samples_from_time()

    def refill_samples(self):
        """
        For each level counts further number of simulations by appropriate N
        """
        for step, level in enumerate(self._levels):
            if self.number_of_samples is not None:
                self.num_of_simulations = self.number_of_samples

            if level.number_of_simulations < self.num_of_simulations[step]:
                level.number_of_simulations = self.num_of_simulations[step] - level.number_of_simulations
                # Launch further simulations
                level.level()
                level.number_of_simulations = self.num_of_simulations[step]

    def _create_level(self):
        """
        Create new level add its to the array of levels
        Call method for counting number of simulation steps
        Pass instance of Simulation to Level
        """
        if self.current_level > 0:
            previous_level_simulation = self._levels[self.current_level-1].fine_simulation
        else:
            previous_level_simulation = self.simulation.interpolate_precision()

        level = Level(self.simulation, previous_level_simulation,
                      self.moments_object, self.current_level/self.number_of_levels)
        self._levels.append(level)
        self.current_level += 1

    def set_target_time(self, target_time):
        """
        For each level counts new N according to target_time
        :return: array
        """
        amount = self._count_sum()
        # Loop through levels
        # Count new number of simulations for each level
        for level in self._levels:
            new_num_of_sim = np.round((target_time * np.sqrt(level.variance / level.n_ops_estimate()))
                                      / amount).astype(int)

            self.num_of_simulations.append(new_num_of_sim)

    def set_target_variance(self, target_variance):
        """
        For each level counts new N according to target_variance
        :return: array
        """
        # Loop through levels
        # Count new number of simulations for each level
        for level in self._levels:
            new_num_of_sim_pom = []
            for index, moment in enumerate(level.moments_estimate[1:]):

                amount = sum([np.sqrt(level.moments_estimate[index+1][1] * level.n_ops_estimate()) for level in self._levels])

                new_num_of_sim_pom.append(np.round((amount * np.sqrt(np.abs(moment[1]) / level.n_ops_estimate()))
                / target_variance[index]).astype(int))
            print("new num of sim pom", new_num_of_sim_pom)

            self.num_of_simulations.append(np.max(new_num_of_sim_pom))

    def _count_sum(self):
        """
        Loop through levels and count sum of variance * simulation step
        :return: float sum
        """
        return sum([np.sqrt(level.variance * level.n_ops_estimate()) for level in self._levels])
