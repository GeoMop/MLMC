import numpy as np
from mlmc.mc_level import Level


class MLMC:
    """
    Multi level monte carlo method
    """
    def __init__(self, number_of_levels, sim_factory, moments_object, pbs=None):
        """
        :param number_of_levels:    Number of levels
        :param sim:                 Instance of object Simulation
        :param moments_object:      Instance of moments object
        """
        # Object of simulation
        self.simulation_factory = sim_factory
        # Array of level objects
        self.levels = []
        for i_level in range(number_of_levels):
            previous = self.levels[-1].fine_simulation if i_level else None
            level_param = i_level / (number_of_levels - 1)
            level = Level(self.simulation_factory, previous, moments_object, level_param)
            self.levels.append(level)

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
        # Calculated number of samples
        self._num_of_samples = None

        self._pbs = pbs

        # Create levels
        if self._pbs is not None:
            self._pbs.execute()

        self._check_levels() 



    def set_target_time(self, target_time):
        """
        For each level counts new N according to target_time
        :return: array
        """
        amount = self._count_sum()
        # Loop through levels
        # Count new number of simulations for each level
        for level in self.levels:
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

        for level in self.levels:
            new_num_of_sim_pom = []
            for index, moment in enumerate(level.moments[1:]):
                amount = sum([np.sqrt(level.moments[index+1][1] * level.n_ops_estimate()) for level in self.levels])

                new_num_of_sim_pom.append(np.round((amount * np.sqrt(np.abs(moment[1]) / level.n_ops_estimate()))
                / target_variance[index]).astype(int))
            self.num_of_simulations.append(np.max(new_num_of_sim_pom))

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

    #def estimate_n_samples(self):
    #    # Count new number of simulations according to variance of time
    #    if self.target_variance is not None:
    #        self._num_of_samples = self.estimate_n_samples_from_variance()
    #    elif self.target_time is not None:
    #        self._num_of_samples = self.estimate_n_samples_from_time()

    def refill_samples(self):
        """
        For each level counts further number of simulations by appropriate N
        """
        for index, level in enumerate(self.levels):
            if self.number_of_samples is not None:
                self.num_of_simulations = self.number_of_samples

            if level.number_of_simulations < self.num_of_simulations[index]:
                level.number_of_simulations = self.num_of_simulations[index] - level.number_of_simulations
                # Launch further simulations
                level.level()
                level.number_of_simulations = self.num_of_simulations[index]
              
        if self._pbs is not None:
            self._pbs.execute()
        self._check_levels()
        self.save_data()

    def save_data(self):
        """
        Save results for future use
        """
        with open("data", "w") as fout:
            for index, level in enumerate(self.levels):
                fout.write("LEVEL" + "\n")
                fout.write(str(level.number_of_simulations) + "\n")
                fout.write(str(level.n_ops_estimate()) + "\n")
                for tup in level.data:
                    fout.write(str(tup[0])+ " " + str(tup[1]))
                    fout.write("\n")


    def _check_levels(self):
        """
        Check if all simulations are done
        """
        not_done = True
        while not_done is True:
            not_done = False
            for index, level in enumerate(self.levels):
                if level.are_simulations_running():
                    not_done = True



    def _count_sum(self):
        """
        Loop through levels and count sum of variance * simulation step
        :return: float sum
        """
        return sum([np.sqrt(level.variance * level.n_ops_estimate()) for level in self.levels])
