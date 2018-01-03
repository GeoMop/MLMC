class Monomials:
    """
    Monomials for distribution approximation
    """
    def __init__(self, mean=0):
        self.mean = mean
        self._bounds = None
        self.fixed_quad_n = None

    @property
    def bounds(self):
        """
        Bounds of random variable
        """
        return self._bounds

    @bounds.setter
    def bounds(self, bounds):
        if len(bounds) != 2:
            raise TypeError("Bounds should be array of two items")
        self._bounds = bounds

    def get_moments(self, value, exponent):
        """
        Get moment for exact value
        :param value: float, exact value
        :param exponent: int
        :return: moment
        """
        return pow(value - self.mean, exponent)
