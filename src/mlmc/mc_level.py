import numpy as np
import scipy as sc
import scipy.stats
import uuid
import copy as cp

class Level:
    """
    Call Simulation methods
    There are information about random variable - average, dispersion, number of simulation, ...

    JS TODO:
    Property and setter methods just boilerplate here, but if done you MUST provide
    documentation of returned data structures otherwise related code is very hard to read.
    """

    def __init__(self, sim, previous_level_sim, moments_object, precision):
        """
        :param simulation_size: number of simulation steps
        :param sim: instance of object Simulation
        """
        self.data = []
        self.simulation = sim

        # Instance of object Simulation
        self.fine_simulation = sim.interpolate_precision(precision)
        self.fine_simulation.n_sim_steps = self.fine_simulation.sim_param
        self.coarse_simulation = previous_level_sim
        self.coarse_simulation.n_sim_steps = self.coarse_simulation.sim_param

        # Initialization of variables
        self._result = []
        self._moments = []
        self.moments_object = moments_object
        self.moments_estimate = []

        # Default number of simulations is 10
        # that is enough for estimate variance
        self.number_of_simulations = 10
        self._estimate_moments()

        self.variance = np.var(self.result)

    @property
    # JS TODO: Very bed, this setter does not have semantics of setter but is rather an append.
    # Extremaly confusing. Expose the level data attribute or make clear MCLevel.append_sample() method.
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

    # JS TODO: No usage of variance in this class. Remove.
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

    def n_ops_estimate(self):
        """
        :return: fine simulation n
        """
        return self.fine_simulation.n_ops_estimate()

    def _create_simulations(self):
        """
        Create fine and coarse simulations and run their simulation
        :return: fine and coarse simulation object
        """
        fine_simulation = cp.deepcopy(self.fine_simulation)
        # Generate random array
        fine_simulation.random_array()
        coarse_simulation = cp.deepcopy(self.coarse_simulation)
        fine_simulation.set_coarse_sim(coarse_simulation)
        # Set random array to coarse step simulation
        coarse_simulation._input_sample = fine_simulation.get_random_array()

        # Run simulations
        fine_simulation.cycle(uuid.uuid1())
        coarse_simulation.cycle(uuid.uuid1())
        return fine_simulation, coarse_simulation

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
                    if fine_sim.simulation_result is not None and coarse_sim.simulation_result is not None:
                        if isinstance(fine_sim.simulation_result, (int, float)) and isinstance(coarse_sim.simulation_result, (int, float)):
                            self.data.append((fine_sim.simulation_result, coarse_sim.simulation_result))
                        else:
                            raise ExpWrongResult()

                        if num_of_simulations < self.number_of_simulations:
                            running_simulations[index] = self._create_simulations()
                            num_of_simulations += 1
                        else:
                            # Remove simulations pair from running simulations
                            running_simulations.pop(index)
                except ExpWrongResult as e:
                    print(e.message)

        self.result = [data[0] - data[1] for data in self.data]
        return self.result

    def _estimate_moments(self):
        """
        Moments estimation
        :return: None
        """
        moments = []
        for k in range(0, 10):
            self._data = []
            self.level()
            if k == 0:
                self.moments_object.mean = np.mean(self.result)

            self.level_moments()
            moments.append(self.moments)

        self.moments_estimate = [(np.mean(m), np.var(m)) for m in zip(*moments)]

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
            moments.append(np.mean(np.array(fine_coarse_diff)))

        return moments

class ExpWrongResult(Exception):
    def __init__(self, *args, **kwargs):
        print(*args)
        Exception.__init__(self, *args, **kwargs)
        self.message = "Wrong simulation result"
