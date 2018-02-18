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
