class Moments:
    """
    Class for moments of random distribution
    """
    def __init__(self, mean=0):
        self.mean = mean
        self._bounds = None
        self._moments_function = None
        self._moments_number = 5
        self.eps = 0

    @property
    def bounds(self):
        """
        Bounds of random variable
        """
        return self._bounds

    @bounds.setter
    def bounds(self, bounds):
        if len(bounds) != 2:
            raise TypeError("Bounds should have two items")
        self._bounds = bounds

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
