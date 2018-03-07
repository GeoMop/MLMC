import numpy as np


class Moments:
    """
    Class for moments of random distribution
    JS TODO:
    - Put all moments classes into this file. These are small classes so it is better to
     keep them together. This also clarify import, e.g.:
     import mlmc.moments.Monomials
    - encapsulating all attributes through property and setter methods is not good practice in Python
      at least not at initial development stages since it reduce readability of the code. More over number of moments
      should be given in constructor since the set of functions should be fixed during the lifespan of the object.
      On the other hand 'mean' is not known a priori. Therefore:
      - make setter for 'mean', remove it from constructor parameters (but set it to 0.0 and document its meaning.
      - remove moments_number methods
      - rename to n_moments, i.e. public attribute (consistent naming practice for list bounds 'n_*')
      - remove bounds property, keep setter method, add check bounds[1] > bounds[0]

    - Make monomials to return the actual mean as the 0 moment.

    """
    def __init__(self, n_moments=None):
        self.mean = 0.0
        self._bounds = None
        self._moments_function = None
        self.n_moments = n_moments
        self.eps = 0

    @property
    def bounds(self):
        """
        Bounds of random variable
        :return: array 
        """
        return self._bounds

    @bounds.setter
    def bounds(self, bounds):
        if len(bounds) != 2:
            raise TypeError("Bounds should have two items")
        if bounds[1] < bounds[0]:
            raise ValueError("Second bound must be greater than the first one")
        self._bounds = bounds


class Monomials(Moments):
    """
    Monomials for distribution approximation
    """

    def get_moments(self, value, exponent):
        """
        Get moment for exact value
        :param value: float, exact value
        :param exponent: int
        :return: moment
        """
        # Returns the real mean value as the zero moment
        if exponent == 0:
            return value
        return pow(value - self.mean, exponent)


class FourierFunctions(Moments):
    """
    Fourier functions for distribution approximation
    """

    def get_moments(self, value, r):
        """
        Get moment for exact value
        :param value: float, exact value
        :param r: int
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
        if self.bounds[1] - self.bounds[0] == 0:
            return 2 * np.pi * (value - self.bounds[0])
        return (2 * np.pi * (value - self.bounds[0])) / (self.bounds[1] - self.bounds[0])

