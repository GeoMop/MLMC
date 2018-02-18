import numpy as np
from mc_level import Level


class MLMC:
    """
    Multi level monte carlo method
    """
    def __init__(self, number_of_levels, sim_steps_range, sim, moments_object):
        """
        :param number_of_levels:    Number of levels
        :param sim_steps_range:     Simulations step fine and coarse
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
        self.sim_steps_fine = sim_steps_range[0]
        # The coarsest simulation step
        self.sim_steps_coarse = sim_steps_range[1]
        # Array of number of samples on each level
        # It is used if want to have fixed number of simulations
        self._number_of_samples = None
        # Instance of selected moments object
        self.moments_object = moments_object
        # Current level
        self.index = 0
        # Calculated number of samples
        self._num_of_samples = None

        # Multilevel MC method
        if self.number_of_levels > 1:
            for _ in range(self.number_of_levels):
                self._create_level()
        # One level MC method
        else:
            level = Level([0, self.sim_steps_fine], self.simulation, self.moments_object)
            self._levels.append(level)

    @property
    def levels(self):
        """
        return: list of Level instances
        """
        return self._levels

    @property
    def number_of_samples(self):
        """
        List of samples in each level
        """
        return self._number_of_samples

    @number_of_samples.setter
    def number_of_samples(self, num_of_sim):
        if len(num_of_sim) < self.number_of_levels:
            raise ValueError("Number of simulations must be list")

        self._number_of_samples = num_of_sim

    def monte_carlo(self, optimization_type, values):
        """
        Implements complete multilevel Monte Carlo Method
        Calls other necessary methods
        :param optimization_type: int        1 for fix variance and other for fix time
        :param values: array     value of variance or time according to type

        JS TODO: Rather let user to call 'set_target_time' or 'set_target_variance'.
        and call refill_samples explicitly.
        """
        self.index = 0

        if optimization_type == 1:
            if len(values) < self.moments_object.moments_number:
                raise ValueError("Target variance values length cannot be bigger than moments number")
            self.target_variance = values
        else:
            self.target_time = values


        self.estimate_n_samples()
        self._refill_samples()

    def estimate_n_samples(self):
        # Count new number of simulations according to variance of time
        if self.target_variance is not None:
            self._num_of_samples = self.estimate_n_samples_from_variance()
        elif self.target_time is not None:
            self._num_of_samples = self.estimate_n_samples_from_time()

    def _refill_samples(self):
        """
        For each level counts further number of simulations by appropriate N
        :param num_of_simulations: array      new number of simulations for each level
        """

        for step, level in enumerate(self._levels):
            if self.number_of_samples is not None:
                self._num_of_samples = self.number_of_samples

            if level.number_of_simulations < self._num_of_samples[step]:
                level.number_of_simulations = self._num_of_samples[step] - level.number_of_simulations
                # Launch further simulations
                level.level()
                level.number_of_simulations = self._num_of_samples[step]

    def _create_level(self):
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
        fine_steps = np.power(np.power((self.sim_steps_fine/self.sim_steps_coarse),
                                       (1/(self.number_of_levels-1))), self.index)* self.sim_steps_coarse

        # Coarse number of simulation steps from previous level
        if len(self._levels) > 0:
            level = self._levels[self.index - 1]
            coarse_steps = level.n_fine

        else:
            coarse_steps = 0
        self.index += 1
        return [np.round(coarse_steps).astype(int), np.round(fine_steps).astype(int)]

    def estimate_n_samples_from_time(self):
        """
        For each level counts new N according to target_time
        :return: array
        JS TODO: Rename: 'set_target_time'
        """
        num_of_simulations_time = []
        amount = self._count_sum()

        # Loop through levels
        # Count new number of simulations for each level
        for level in self._levels:
            new_num_of_sim = np.round((self.target_time * np.sqrt(level.variance / level.n_ops_estimate()))
                                      / amount).astype(int)

            num_of_simulations_time.append(new_num_of_sim)
        return num_of_simulations_time

    def estimate_n_samples_from_variance(self):
        """
        For each level counts new N according to target_variance
        :return: array
        JS TODO: rename  'set_target_variance'.
        """
        num_of_simulations_var = []
        amount = self._count_sum()

        # Loop through levels
        # Count new number of simulations for each level
        for level in self._levels:
            new_num_of_sim_pom = []
            level.moments_object.mean = np.mean(level.result)

            for index, moment in enumerate(level.moments_estimate):
                new_num_of_sim_pom.append(np.round((amount * np.sqrt(np.abs(moment[1]) / level.n_ops_estimate()))
                / self.target_variance[index]).astype(int))

            new_num_of_sim = np.max(new_num_of_sim_pom)


            #new_num_of_sim = np.round((amount * np.sqrt(level.variance / level.n_ops_estimate())) / self.target_variance[0]).astype(int)
            num_of_simulations_var.append(new_num_of_sim)

        return num_of_simulations_var

    def _count_sum(self):
        """
        Loop through levels and count sum of varinace * simulation step
        :return: float sum
        """
        return sum([np.sqrt(level.variance * level.n_ops_estimate()) for level in self._levels])
