import numpy as np
import scipy as sc


class Moments:
    """
    Class for moments of random distribution
    """

    def __init__(self):
        self._moments_function = None
        self._moments_number = 5
        self.eps = 0

    @property
    def moments_function(self):
        """
        Moment function (monomials or fourier functions)
        """
        return self._moments_function

    @moments_function.setter
    def moments_function(self, function):
        self._moments_function = function

    @property
    def moments_number(self):
        """
        Number of moments
        """
        return self._moments_number

    @moments_number.setter
    def moments_number(self, number):
        if number < 1:
            raise ValueError("Moments number must be greater than 0")
        self._moments_number = number

    def level_moments(self, level):
        """
        Count moments from level data
        :param level: instance of class Level
        :return: array, moments
        """
        self.moments_function.bounds = sc.stats.mstats.mquantiles(level.result, prob=[self.eps, 1 - self.eps])
        moments = self.count_moments(level.data)
        return moments

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

        # Mean of fine step results
        fine_mean = np.mean(level_results["fine"])
        # Mean of coarse step results
        coarse_mean = np.mean(level_results["coarse"])

        # Count moment for each degree
        for degree in range(self.moments_number):
            fine_coarse_diff = []

            for position, value in enumerate(level_results["fine"]):
                # Set mean to moments function
                self.moments_function.mean = fine_mean
                # Moment for fine step result
                fine = self.moments_function.get_moments(value, degree)

                # For first level use only fine step result
                if coarse_mean > 0 or coarse_mean < 0:
                    # Set new mean to moments function
                    self.moments_function.mean = coarse_mean
                    # Moment from coarse step result
                    coarse = self.moments_function.get_moments(level_results["coarse"][position], degree)
                else:
                    coarse = coarse_mean
                # Add subtract moment from coarse step result from moment from fine step result
                fine_coarse_diff.append(fine - coarse)

            # Append moment to other moments
            moments.append(np.mean(fine_coarse_diff))

        return moments
