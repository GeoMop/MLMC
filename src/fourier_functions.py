import numpy as np


class FourierFunctions:
    """
    Fourier functions for distribution approximation
    """
    def __init__(self, mean=0):
        self.mean = mean
        self._bounds = None
        self.fixed_quad_n = None

    @property
    def bounds(self):
        """
        Bounds of random variables
        """
        return self._bounds

    @bounds.setter
    def bounds(self, bounds):
        if len(bounds) != 2:
            raise TypeError("Bounds should be array of two items")
        self._bounds = bounds

    def get_moments(self, value, r):
        """
        Get moment for exact value
        :param value: float, exact value
        :param : int
        :return: moment
        """
        value = self.change_interval(value - self.mean)
        if r == 0:
            return 1
        if r % 2 != 0:
            new_r = int((r + 1) / 2)
            return np.sin(new_r * value)
        if r % 2 == 0:
            new_r = int((r + 1) / 2)
            return np.cos(new_r * value)

    def change_interval(self, value):
        """
        Fitting value from default interval to the bounds interval
        :param value: float
        :return: value remapped to interval from bounds
        """

        return (2 * np.pi * (value - self.bounds[0])) / (self.bounds[1] - self.bounds[0])
