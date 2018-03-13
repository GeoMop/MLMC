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

        JS TODO:

        - Make 'sim_steps_range' part of simulation, an attribute: 'sim_param_range'
          and allow it to be any pair of real positive parameters
          Using logaritmic interpolatin we then compute parameters for individual levels and pass them to the simulation object.
          So the meaning of the simulation parameter is simulation dependent.
        - !! Code is wrong, sim is not instance of Simulation but SimulationSetting.
          Original idea was that sim would be the simulation class however that makes setup of cofigurable simulations
          problematic. Better would be to introduce a method Simulation.interpolate_by_precision(precision)
          with t_level be a number between 0 and 1 given as (l/L) so this will interpolate simulation parameter and return copy
          of the simulation with particular value of the parameter set.
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

    def refill_samples(self):
        """

        JS TODO: Rather let user to call 'set_target_time' or 'set_target_variance'.
        and call refill_samples explicitly.
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
        JS TODO: Rename: 'set_target_time'
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
        JS TODO: rename  'set_target_variance'.
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
