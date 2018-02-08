from mlmc.moments import Moments


class Monomials(Moments):

    def get_moments(self, value, exponent):
        """
        Get moment for exact value
        :param value: float, exact value
        :param exponent: int
        :return: moment
        """
        return pow(value - self.mean, exponent)
