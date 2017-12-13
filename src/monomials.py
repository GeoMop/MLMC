
class Monomials:

    def __init__(self, mean=0):
        self.mean = mean
        self.bounds = None
        self.fixed_quad_n = None

    def set_bounds(self, bounds):
        self.bounds = bounds

    def get_moments(self, x, r):
        return pow(x - self.mean, r)

