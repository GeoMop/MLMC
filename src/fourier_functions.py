import numpy as np


class FourierFunctions:

    def __init__(self, mean = 0):
        self.mean = mean
        self.bounds = None
        self.fixed_quad_n = None

    def set_bounds(self, bounds):
        self.bounds = bounds

    def get_moments(self, x, r):
        x = self.change_interval(x - self.mean)
        if r == 0:
            return 1
        if r % 2 != 0:
            new_r = int((r + 1) / 2)
            return np.sin(new_r * x)
        if r % 2 == 0:
            new_r = int((r + 1) / 2)
            return np.cos(new_r * x)


    def change_interval(self, x):

        """
        Fitting value from first interval to the second interval
        :param x: 
        :return: value x remapped to second interval
        """
        """
        z interval (c, d) na interval (a, b)
        x = (b-a)*(x-c)/(d-c) + a
        (a, b) = (0,2pi)
        """
        c, d = (self.bounds[0], self.bounds[1])

        return (2 * np.pi * (x - c)) / (d - c)

