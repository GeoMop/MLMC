import numpy as np
import scipy as sc
import scipy.stats
import uuid
import os

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
        # Reference to all created simulations.
        #self.simulations = []

        # Collected data samples, pairs (fine, coarse)
        self.data = []

        self.is_zero_level = (precision == 0.0)
        # Instance of object Simulation
        self.fine_simulation = sim_factory(precision)
        # TODO: coarse_simulation can be different to previous_level_sim if they have same mean value
        self.coarse_simulation = previous_level_sim
        self.fine_simulation.set_previous_fine_sim(self.coarse_simulation)

        # Currently running simulations
        self.running_simulations = []

        # Initialization of variables
        self._result = []
        #self._moments = []
        #self.moments_object = moments_object
        #self.moments_estimate = []

        # Default number of simulations is 10
        # that is enough for estimate variance
        self.number_of_simulations=10
        # Run simulations
        self.level()

        self.variance = np.var(self.result)

    @property
    def result(self):
        """
        Simulations results (fine step simulations result - coarse step simulation result)
        :return: array, each item is fine_result - coarse_result
        """
        return [data[0] - data[1] for data in self.data]

    @result.setter
    def result(self, result):
        if not isinstance(result, list):
            raise TypeError("Simulation results must be list")
        self._result = result

    # @property
    # def moments(self):
    #     """
    #     Result moments
    #     :return: array of moments
    #     """
    #     self.get_moments()
    #     return self._moments

    # @moments.setter
    # def moments(self, moments):
    #     if not isinstance(moments, list):
    #         raise TypeError("Moments must be list")
    #     self. = moments

    # def get_moments(self):
    #     """
    #     Get moments from results of simulations on this level
    #     :return: array, moments
    #     """
    #     self.moments = self.level_moments()
    #     if len(self.moments_estimate) == 0:
    #         self.moments_estimate = self._moments

    def n_ops_estimate(self):
        """
        :return: fine simulation n
        """
        return self.fine_simulation.n_ops_estimate()

    def _create_simulations(self):
        """
        Generate new random samples for fine and coarse simulation objects
        :param last_sim: mark last simulation on the level
        :return: fine and coarse running simulations
        """
        # Levels greater than one have fine and coarse simulations
        if self.coarse_simulation is not None:
            # Generate random array
            self.fine_simulation.generate_random_sample()
            # Set random array to coarse step simulation
            self.coarse_simulation._input_sample = self.fine_simulation.get_coarse_sample()
            # Run simulations
            return self.fine_simulation.cycle(uuid.uuid1()), self.coarse_simulation.cycle(uuid.uuid1()) 
        # First level doesn't have coarse simulation
        else:
            # Generate random array
            self.fine_simulation.generate_random_sample()

            # Run simulations
            return self.fine_simulation.cycle(uuid.uuid1()), None

 
    def level(self):
        """
        Implements level of MLMC
        Call Simulation methods
        Set simulation data
        :return: None
        """
        # Create pair of fine and coarse simulations and add them to list of all running simulations
        [self.running_simulations.append(self._create_simulations()) for _ in range(self.number_of_simulations)]

    def are_simulations_running(self):
        # Still running some simulations
        while len(self.running_simulations) > 0:
            # Loop through pair of running simulations
            for index, (fine_sim, coarse_sim) in enumerate(self.running_simulations):
                try:
                    # First level has no coarse simulation
                    if self.coarse_simulation is None:
                        if self.fine_simulation.extract_result(fine_sim) is not None:
                            self.data.append((self.fine_simulation.extract_result(fine_sim), 0))
                            # Remove simulations pair from running simulations
                            self.running_simulations.pop(index)
                    # Other levels have also coarse simulation
                    else:
                        # Checks if simulation is already finished
                        if self.fine_simulation.extract_result(
                                fine_sim) is not None and self.coarse_simulation.extract_result(coarse_sim) is not None:
                            self.data.append((self.fine_simulation.extract_result(fine_sim),
                                              self.coarse_simulation.extract_result(coarse_sim)))
                            # Remove simulations pair from running simulations
                            self.running_simulations.pop(index)
                except ExpWrongResult as e:
                    print(e.message)

        if len(self.running_simulations) > 1:
            return True
        return False


    def estimate_diff_mean(self, moments_obj = None):
        """
        Compute estimate of mead of the level difference:
            E{ M_r(fine_samples) - M_r(coarse_samples) }
        :param moments_obj: Function to compute moment matrix (shape NxR) for given samples vector (shape N).
                            If None just the mean moment is computed
        :return: array, moments
        """

        #moments_obj.bounds = sc.stats.mstats.mquantiles(self.result, prob=[moments_obj.eps,
        #
        if moments_obj is None:
            moments_obj = Mome

        # Separate fine step result and coarse step result
        fine, coarse = zip(*self.data)
        fine_values = np.array(fine)
        coarse_values = np.array(coarse)
        fine_moments = moments_obj.get_moments(fine_values)
        if self.is_zero_level:
            coarse_moments = np.zeros_like(fine_moments)
        else:
            coarse_moments = moments_obj.get_moments(coarse_values)
        diff_mean = np.mean(fine_moments - coarse_moments)
        diff_var = np.var(fine_moments - coarse_moments, ddof = 1)

        return diff_mean, diff_var


class ExpWrongResult(Exception):
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)
        self.message = "Wrong simulation result"
