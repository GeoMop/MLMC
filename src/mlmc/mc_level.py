import numpy as np
import scipy as sc
import scipy.stats
import uuid
import copy as cp

class Level:
    """
    Call Simulation methods
    There are information about random variable - average, dispersion, number of simulation, ...
    """

    def __init__(self, sim_factory, previous_level_sim, moments_object, precision):
        """
        :param sim_factory: method that create instance of particular simulation class
        :param previous_level_sim: fine simulation on previous level
        :param moments_object: object for calculating statistical moments
        :param precision: current level number / total number of all levels
        """
        self.data = []

        # Instance of object Simulation
        self.fine_simulation = sim_factory(precision)
        self.coarse_simulation = previous_level_sim
        self.fine_simulation.set_previous_fine_sim(self.coarse_simulation)

        # Initialization of variables
        self._result = []
        self._moments = []
        self.moments_object = moments_object
        self.moments_estimate = []

        # Default number of simulations is 10
        # that is enough for estimate variance
        self.number_of_simulations = 10
        # Run simulations
        self.level()

        self.variance = np.var(self.result)

    @property
    def result(self):
        """
        Simulations results (fine step simulations result - coarse step simulation result)
        :return: array, each item is fine_result - coarse_result
        """
        return self._result

    @result.setter
    def result(self, result):
        if not isinstance(result, list):
            raise TypeError("Simulation results must be list")
        self._result = result

    @property
    def moments(self):
        """
        Result moments
        :return: array of moments
        """
        self.get_moments()
        return self._moments

    @moments.setter
    def moments(self, moments):
        if not isinstance(moments, list):
            raise TypeError("Moments must be list")
        self._moments = moments

    def get_moments(self):
        """
        Get moments from results of simulations on this level
        :return: array, moments
        """
        self.moments = self.level_moments()
        if len(self.moments_estimate) == 0:
            self.moments_estimate = self._moments

    def n_ops_estimate(self):
        """
        :return: fine simulation n
        """
        return self.fine_simulation.n_ops_estimate()

    def _create_simulations(self):
        """
        Generate new random samples for fine and coarse simulation objects
        :return: fine and coarse running simulations
        """
        # Generate random array
        self.fine_simulation.generate_random_sample()

        # Set random array to coarse step simulation
        self.coarse_simulation._input_sample = self.fine_simulation.get_coarse_sample()
        # Run simulations
        return self.fine_simulation.cycle(uuid.uuid1()), self.coarse_simulation.cycle(uuid.uuid1())

    def level(self):
        """
        Implements level of MLMC
        Call Simulation methods
        Set simulation data
        :return: array of results (fine_sim result - coarse_sim result) 
        """
        running_simulations = []
        # Run at the same time maximum of 2000 simulations
        if self.number_of_simulations > 2000:
            num_of_simulations = 2000
        else:
            num_of_simulations = self.number_of_simulations

        # Create pair of fine and coarse simulations and add them to list of all running simulations
        [running_simulations.append(self._create_simulations()) for _ in range(num_of_simulations)]

        while len(running_simulations) > 0:
            for index, (fine_sim, coarse_sim) in enumerate(running_simulations):
                try:
                    # Checks if simulation is already finished
                    if self.fine_simulation.get_result(fine_sim) is not None and self.coarse_simulation.get_result(coarse_sim) is not None:
                        # Save simulations results
                        self.data.append((self.fine_simulation.get_result(fine_sim), self.coarse_simulation.get_result(coarse_sim)))

                        # Create new simulation
                        if num_of_simulations < self.number_of_simulations:
                            running_simulations[index] = self._create_simulations()
                            num_of_simulations += 1
                        else:
                            # Remove simulations pair from running simulations
                            running_simulations.pop(index)
                except ExpWrongResult as e:
                    print(e.message)

        # Level results, fine_sim result - coarse_sim result
        self.result = [data[0] - data[1] for data in self.data]
        return self.result

    def level_moments(self):
        """
        Count moments from level data
        :return: array, moments
        """
        self.moments_object.bounds = sc.stats.mstats.mquantiles(self.result, prob=[self.moments_object.eps, 1 - self.moments_object.eps])
        return self.count_moments(self.data)

    def count_moments(self, level_data):
        """
        Count moments
        :param level_data: array of tuples (fine step result, coarse step result)
        :return: array, moments
        """
        level_results = {}
        level_results["fine"] = []
        level_results["coarse"] = []
        # Initialize array of moments
        moments = []

        # Separate fine step result and coarse step result
        for data in level_data:
            level_results["fine"].append(data[0])
            level_results["coarse"].append(data[1])

        coarse_var = np.var(level_results["coarse"])

        # Count moment for each degree
        for degree in range(self.moments_object.n_moments):
            fine_coarse_diff = []

            for lr_fine, lr_coarse in zip(level_results["fine"], level_results["coarse"]):
                # Moment for fine step result
                fine = self.moments_object.get_moments(lr_fine, degree)

                # For first level use only fine step result
                if coarse_var != 0:
                    # Moment from coarse step result
                    coarse = self.moments_object.get_moments(lr_coarse, degree)
                    fine_coarse_diff.append(fine - coarse)
                else:
                    # Add subtract moment from coarse step result from moment from fine step result
                    fine_coarse_diff.append(fine)

            # Append moment to other moments
            moments.append((np.mean(np.array(fine_coarse_diff)), np.var(np.array(fine_coarse_diff))))

        return moments

class ExpWrongResult(Exception):
    def __init__(self, *args, **kwargs):
        print(*args)
        Exception.__init__(self, *args, **kwargs)
        self.message = "Wrong simulation result"