import numpy as np
from scipy import stats
from mlmc.tool import restrict_distribution as distr
from scipy import integrate
from scipy.special import erf, erfinv
import matplotlib.pyplot as plt
from scipy.stats import norm
from statsmodels.distributions.empirical_distribution import ECDF

"""
Some benchmark distributions derived from scipy.stats.rv_continuous.
According to the doc: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.html
it is sufficient to define at least _cdf or _pdf. For decent performance one should define:
_pdf, _cdf, _rvs
"""
def Cauchy():
    return distr.Name(stats.cauchy(), 'Cauchy')

def Gamma():
    return distr.Name(stats.gamma(a=0.5), 'Gamma')

def TwoGaussians():
    submodels=[stats.norm(5, 3),
              stats.norm(0, 0.5)]
    weights=[0.93, .07]
    return distr.Mixture(submodels, weights, name='TwoGaussians')

def FiveFingers():
    submodels=[stats.norm(1 / 10, 1 / 100),
                   stats.norm(3 / 10, 1 / 100),
                   stats.norm(5 / 10, 1 / 100),
                   stats.norm(7 / 10, 1 / 100),
                   stats.norm(9 / 10, 1 / 100)]
    return distr.Mixture(submodels, name='FiveFingers')



def Steps():
    submodels=[stats.uniform(0, 0.3),
                 stats.uniform(0.3, 0.1),
                 stats.uniform(0.4, 0.1),
                 stats.uniform(0.5, 0.3),
                 stats.uniform(0.8, 0.2)]
    weights=[6 / 25, 1 / 8, 1 / 10, 3 / 8, 4 / 25]
    return distr.Mixture(submodels, weights, 'Steps')



class ZeroValue(stats.rv_continuous):
    def __init__(self):
        super().__init__(name='ZeroValue')

    def _pdf(self, x):
        px = np.maximum(x - 0.8, 0)
        return 50 * px

    def _cdf(self, x):
        px = np.maximum(x - 0.8, 0)
        y = 25 * px**2
        return y

    def _ppf(self, q):
        return np.sqrt(q/25) + 0.8

class SmoothZV(stats.rv_continuous):
    def __init__(self):
        super().__init__(name='SmoothZV')

    def _pdf(self, x):
        a = (-1 / x)
        b = (-1 / (1 - x))
        return np.where(x < 0, 0,
                 np.where(x > 1, 0,
                          (a * a + b * b) / (2 + np.exp(a - b) + np.exp(b - a))
                          ))

    def _cdf(self, x):
        a = np.exp(-1 / x)
        b = np.exp(-1 / (1 - x))
        return np.where(x < 0, 0,
                 np.where(x > 1, 0,
                          a / (a + b)
                          ))

    def _ppf(self, q):
        iq = 1/np.log(1/q-1)
        discr = np.sqrt(iq**2 + 0.25)
        base = iq + 0.5
        return np.where(q < 0.5, base - discr,
                 np.where(q > 0.5, base + discr,
                          0
                          ))





class MultivariateNorm(stats.rv_continuous):
    distr = stats.multivariate_normal([0, 0], [[1, 0], [0, 1]])
    domain = [np.array([0, 1]), np.array([0, 1])]

    def pdf(self, values):
        return MultivariateNorm.distr.pdf(values)

    def cdf(self, x):
        return MultivariateNorm.distr.cdf(x)

    # def _cdf(self, x):
    #     def summation():
    #         result = 0
    #         for weight, distr in zip(Discontinuous.weights, Discontinuous.distributions):
    #             result += weight * distr.cdf(x)
    #         return result
    #     return summation()

    def rvs(self, size):
        return MultivariateNorm.distr.rvs(size)


class Abyss(stats.rv_continuous):
    def __init__(self, name="Abyss"):
        super().__init__(name=name)
        self.dist = self
        self.width = 0.05
        self.z = 0.05
        self.renorm = 2 * stats.norm.cdf(-self.width) + self.z * 2 * self.width
        self.renorm = 1 / self.renorm
        self.middle_scale = self.renorm * self.z * 2 * self.width

        self.q0, self.q1 = self._cdf([-self.width, self.width])


    def _pdf(self, x):
        y = np.where(np.logical_and(-self.width < x, x < self.width),
                      self.z,
                      stats.norm.pdf(x))
        return self.renorm * y

    def _cdf(self, x):
        x = np.atleast_1d(x)
        y = np.where(x < -self.width,
                     self.renorm * stats.norm.cdf(x),
                     np.where(x < self.width,
                              0.5 + self.middle_scale * x,
                              1 - self.renorm * stats.norm.cdf(-x)))
        return y


    def _ppf(self, q):
        x = np.where(q < self.q0,
                     stats.norm.ppf(q/self.renorm),
                     np.where(q < self.q1,
                              (q-0.5)/self.middle_scale,
                               -stats.norm.ppf((1-q)/self.renorm)))
        return x






