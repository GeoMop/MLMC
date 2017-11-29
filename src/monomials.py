
class Monomials:

    def __init__(self, mean = 0):
        self.mean = mean

    def get_moments(self, x, r):
        return pow(x - self.mean, r)
