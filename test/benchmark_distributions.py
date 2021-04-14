import numpy as np
import scipy.stats as st
from scipy import integrate
from scipy.special import erf, erfinv
import matplotlib.pyplot as plt
from scipy.stats import norm
from statsmodels.distributions.empirical_distribution import ECDF


class Gamma(st.rv_continuous):
    domain = [1e-8, 5]

    def _pdf(self, x):
        return 1 / (np.sqrt(np.pi * x)) * np.exp(-x)

    def _cdf(self, x):
        return erf(np.sqrt(x))

    def _ppf(self, x):
        return erfinv(x)**2

    def rvs(self, size):
        x = np.random.uniform(0, 1, size)

        return self._ppf(x)


class TwoGaussians(st.rv_continuous):

    distributions = [st.norm(5, 3),
                     st.norm(0, 0.5)]
    weights = [0.93, .07]

    #domain = [-6.642695234009825, 14.37690544689409]
    domain = [-6.9639446068758595, 18.3417226311775]

    def _pdf(self, x):
        result = 0
        for weight, distr in zip(TwoGaussians.weights, TwoGaussians.distributions):
            result += weight * distr.pdf(x)
        return result

    def _cdf(self, x):
        result = 0
        for weight, distr in zip(TwoGaussians.weights, TwoGaussians.distributions):
            result += weight * distr.cdf(x)
        return result

    def rvs(self, size):
        mixture_idx = np.random.choice(len(TwoGaussians.weights), size=size, replace=True, p=TwoGaussians.weights)
        # y is the mixture sample
        y = np.fromiter((TwoGaussians.distributions[i].rvs() for i in mixture_idx), dtype=np.float64)
        return y


class FiveFingers(st.rv_continuous):
    w = 0.5
    distributions = [st.norm(1/10, 1/100),
                     st.norm(3/10, 1/100),
                     st.norm(5/10, 1/100),
                     st.norm(7/10, 1/100),
                     st.norm(9/10, 1/100)]

    weights = [1/len(distributions)] * len(distributions)
    domain = [0, 1]

    def _pdf(self, x):
        def summation():
            result = 0
            for weight, distr in zip(FiveFingers.weights, FiveFingers.distributions):
                result += weight * distr.pdf(x)
            return result
        return summation()

    def _cdf(self, x):
        def summation():
            result = 0
            for weight, distr in zip(FiveFingers.weights, FiveFingers.distributions):
                result += weight * distr.cdf(x)
            return result
        return summation()

    # def _ppf(self, x):
    #     """
    #     Inverse of cdf
    #     """
    #     def summation():
    #         result = 0
    #         for weight, distr in zip(FiveFingers.weights, FiveFingers.distributions):
    #             result += weight * distr.ppf(x)
    #         return result
    #
    #     #return FiveFingers.w * summation() - (1/FiveFingers.w)*(1 - FiveFingers.w)
    #     return summation()

    def rvs(self, size):
        mixture_idx = np.random.choice(len(FiveFingers.weights), size=size, replace=True, p=FiveFingers.weights)
        # y is the mixture sample
        y = np.fromiter((FiveFingers.distributions[i].rvs() for i in mixture_idx), dtype=np.float64)
        return y


class Cauchy(st.rv_continuous):
    domain = [-20, 20]

    def _pdf(self, x):
        return 1.0 / np.pi / (1.0 + x * x)

    def _cdf(self, x):
        return 0.5 + 1.0 / np.pi * np.arctan(x)

    def _ppf(self, q):
        return np.tan(np.pi * q - np.pi / 2.0)

    def rvs(self, size):
        x = np.random.uniform(0, 1, size)
        return self._ppf(x)


class Discontinuous(st.rv_continuous):
    domain = [0, 1]

    weights = [6/25, 1/8, 1/10, 3/8, 4/25]
    distributions = [st.uniform(0, 0.3-1e-10),
                     st.uniform(0.3, 0.1-1e-10),
                     st.uniform(0.4, 0.1-1e-10),
                     st.uniform(0.5, 0.3-1e-10),
                     st.uniform(0.8, 0.2)]

    def _pdf(self, x):
        def summation():
            result = 0
            for weight, distr in zip(Discontinuous.weights, Discontinuous.distributions):
                result += weight * distr.pdf(x)
            return result

        return summation()

    def _cdf(self, x):
        def summation():
            result = 0
            for weight, distr in zip(Discontinuous.weights, Discontinuous.distributions):
                result += weight * distr.cdf(x)
            return result
        return summation()

    def rvs(self, size):
        mixture_idx = np.random.choice(len(Discontinuous.weights), size=size, replace=True, p=Discontinuous.weights)
        # y is the mixture sample
        y = np.fromiter((Discontinuous.distributions[i].rvs() for i in mixture_idx), dtype=np.float64)
        return y


class Rho4(st.rv_continuous):
    domain = [0, 1]

    def _pdf(self, x):
        if isinstance(x, (list, np.ndarray)):
            y = np.zeros(len(x))
            y[x >= 0.8] = 50 * x[x >= 0.8] - 40
            return y
        else:
            if x <= 0.8:
                return 0
            return 50*x-40

    def _cdf(self, x):
        if isinstance(x, (list, np.ndarray)):
            y = np.zeros(len(x))
            y[x >= 0.8] = 25 * x[x >= 0.8]**2 - 40 * x[x >= 0.8] + 16
            return y
        else:
            if x < 0.8:
                return 0
            return 50 * x ** 2 - 40 * x


    # def rvs(self, size):
    #     mixture_idx = np.random.choice(len(Discontinuous.weights), size=size, replace=True, p=Discontinuous.weights)
    #     # y is the mixture sample
    #     y = np.fromiter((Discontinuous.distributions[i].rvs() for i in mixture_idx), dtype=np.float64)
    #     return y


class MultivariateNorm(st.rv_continuous):
    distr = st.multivariate_normal([0, 0], [[1, 0], [0, 1]])
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


class Abyss(st.rv_continuous):
    def __init__(self, name="Abyss"):
        super().__init__(name=name)
        self.dist = self
        self.width = 0.1
        self.z = 0.1
        self.renorm = 2 * st.norm.cdf(-self.width) + self.z * 2 * self.width
        self.renorm = 1 / self.renorm

    def _pdf(self, x):
        y = np.where(np.logical_and(-self.width < x, x < self.width),
                      self.z,
                      st.norm.pdf(x))
        return self.renorm * y

    def _cdf(self, x):
        y = np.where(x < -self.width,
                     self.renorm * st.norm.cdf(x),
                     np.where(x < self.width,
                              0.5 + self.renorm * self.z * 2 * self.width * x,
                              1 - self.renorm * st.norm.cdf(-x)))
        return y


def test_rho4():
    rho4 = Rho4()

    assert np.isclose(integrate.quad(rho4._pdf, 0, 1)[0], 1)

    domain = rho4.ppf([0.001, 0.999])
    print("domain ", domain)
    a = np.random.uniform(domain[0], domain[1], 1)
    b = np.random.uniform(domain[0], domain[1], 1)

    print("ab.cdf(b) - ab.cdf(a) ", rho4.cdf(b) - rho4.cdf(a))
    print("integrate.quad(ab.pdf, a, b)[0] ", integrate.quad(rho4.pdf, a, b)[0])

    assert np.isclose(rho4.cdf(b) - rho4.cdf(a), integrate.quad(rho4.pdf, a, b)[0], atol=1e-1)

    size = 1000
    values = rho4.rvs(size=size)
    x = np.linspace(0, 1, size)
    plt.plot(x, rho4.pdf(x), 'r-', alpha=0.6, label='rho4 pdf')
    plt.hist(values, bins=1000, density=True, alpha=0.2)
    plt.legend()
    plt.show()

    from statsmodels.distributions.empirical_distribution import ECDF
    ecdf = ECDF(values)
    x = np.linspace(0, 1, size)
    plt.plot(x, ecdf(x), label="ECDF")
    plt.plot(x, rho4.cdf(x), 'r--', alpha=0.6, label='rho4 cdf')

    plt.legend()
    plt.show()


def test_abyss():
    ab = Abyss()

    assert np.isclose(integrate.quad(ab._pdf, -np.inf, np.inf)[0], 1)

    domain = ab.ppf([0.001, 0.999])
    a = np.random.uniform(domain[0], domain[1], 1)
    b = np.random.uniform(domain[0], domain[1], 1)

    assert np.isclose(ab.cdf(b) - ab.cdf(a), integrate.quad(ab.pdf, a, b)[0], atol=1e-1)

    size = 1000
    values = ab.rvs(size=size)
    x = np.linspace(-10, 10, size)
    plt.plot(x, ab.pdf(x), 'r-', alpha=0.6, label='abyss pdf')
    plt.hist(values, bins=1000, density=True, alpha=0.2)
    plt.legend()
    plt.show()

    from statsmodels.distributions.empirical_distribution import ECDF
    ecdf = ECDF(values)
    x = np.linspace(-10, 10, size)
    plt.plot(x, ecdf(x), label="ECDF")
    plt.plot(x, ab.cdf(x), 'r--', alpha=0.6, label='abyss cdf')

    plt.legend()
    plt.show()


def test_two_gaussians():
    tg = TwoGaussians()

    assert np.isclose(integrate.quad(tg._pdf, -np.inf, np.inf)[0], 1)

    domain = tg.ppf([0.001, 0.999])
    a = np.random.uniform(-5, 20, 1)
    b = np.random.uniform(-5, 20, 1)
    assert np.isclose(tg.cdf(b) - tg.cdf(a), integrate.quad(tg.pdf, a, b)[0])

    size = 100000
    values = tg.rvs(size=size)
    x = np.linspace(-10, 20, size)
    plt.plot(x, tg.pdf(x), 'r-', alpha=0.6, label='two gaussians pdf')
    plt.hist(values, bins=1000, density=True, alpha=0.2)
    plt.xlim(-10, 20)
    plt.legend()
    plt.show()

    from statsmodels.distributions.empirical_distribution import ECDF
    ecdf = ECDF(values)
    x = np.linspace(-10, 20, size)
    plt.plot(x, ecdf(x), label="ECDF")
    plt.plot(x, tg.cdf(x), 'r--', alpha=0.6, label='two gaussians cdf')
    plt.xlim(-10, 20)
    plt.legend()
    plt.show()


def test_five_fingers():
    ff = FiveFingers()

    assert np.isclose(integrate.quad(ff.pdf, 0, 1)[0], 1)
    a = 0.1
    b = 0.7
    assert np.isclose(ff.cdf(b) - ff.cdf(a), integrate.quad(ff.pdf, a, b)[0])

    values = ff.rvs(size=10000)
    x = np.linspace(0, 1, 10000)
    plt.plot(x, ff.pdf(x), 'r-', alpha=0.6, label='five fingers pdf')
    plt.hist(values, bins=100, density=True, alpha=0.2)
    plt.xlim(-1, 1)
    plt.legend()
    plt.show()

    from statsmodels.distributions.empirical_distribution import ECDF
    ecdf = ECDF(values)
    plt.plot(x, ecdf(x), label="ECDF")
    plt.plot(x, ff.cdf(x), 'r--', label="cdf")
    plt.legend()
    plt.show()


def test_gamma():
    gamma = Gamma()

    x = np.linspace(0, 3, 100000)
    plt.plot(x, gamma.cdf(x), label="cdf")
    plt.plot(x, gamma.ppf(x), "r-", label="ppf")
    plt.legend()
    plt.ylim(-0.5, 4)
    plt.xlim(-0.1, 3)
    plt.show()

    a = 0.1
    b = 20

    vals = gamma.ppf([0.001, 0.5, 0.999])
    assert np.allclose([0.001, 0.5, 0.999], gamma.cdf(vals))
    assert np.isclose(gamma.cdf(b) - gamma.cdf(a), integrate.quad(gamma.pdf, a, b)[0])

    values = gamma.rvs(size=10000)
    x = np.linspace(gamma.ppf(0.001), gamma.ppf(0.999), 100000)
    plt.plot(x, gamma.pdf(x), 'r-', lw=5, alpha=0.6, label='gamma pdf')

    plt.hist(values, density=True, alpha=0.2)
    plt.xlim(-15, 15)
    plt.legend()
    plt.show()

    x = np.linspace(0, 5, 100000)
    ecdf = ECDF(values)
    plt.plot(x, ecdf(x), label="ECDF")
    plt.plot(x, gamma.cdf(x), label="exact cumulative distr function")
    plt.legend()
    plt.show()


def test_cauchy():
    cauchy = Cauchy()
    x = np.linspace(-2 * np.pi, 2 * np.pi, 10000)

    vals = cauchy.ppf([0.001, 0.5, 0.999])
    assert np.allclose([0.001, 0.5, 0.999], cauchy.cdf(vals))
    assert np.isclose(integrate.quad(cauchy.pdf, -np.inf, np.inf)[0], 1)

    plt.plot(x, cauchy.cdf(x), label="cdf")
    plt.plot(x, cauchy.ppf(x), "r-", label="ppf")
    plt.legend()
    plt.ylim(-5, 5)
    plt.show()

    a = 0.1
    b = 20
    vals = cauchy.ppf([0.001, 0.5, 0.999])
    assert np.allclose([0.001, 0.5, 0.999], cauchy.cdf(vals))
    assert np.isclose(cauchy.cdf(b) - cauchy.cdf(a), integrate.quad(cauchy.pdf, a, b)[0])

    values = cauchy.rvs(size=10000)
    x = np.linspace(cauchy.ppf(0.01), cauchy.ppf(0.99), 100000)
    plt.plot(x, cauchy.pdf(x), 'r-', lw=5, alpha=0.6, label='cauchy pdf')

    plt.hist(values, density=True, alpha=0.2)
    plt.xlim(-15, 15)
    plt.legend()
    plt.show()


def test_discountinuous():
    d = Discontinuous()

    assert np.isclose(integrate.quad(d.pdf, 0, 1)[0], 1)
    a = 0.1
    b = 0.7
    assert np.isclose(d.cdf(b) - d.cdf(a), integrate.quad(d.pdf, a, b)[0])

    values = d.rvs(size=100000)
    x = np.linspace(0, 1, 10000)
    plt.plot(x, d.pdf(x), 'r-', alpha=0.6, label='discontinuous pdf')
    plt.hist(values, bins=200, density=True, alpha=0.2)
    plt.xlim(0, 1)
    plt.legend()
    plt.show()

    from statsmodels.distributions.empirical_distribution import ECDF
    ecdf = ECDF(values)
    plt.plot(x, ecdf(x), label="ECDF")
    plt.plot(x, d.cdf(x), label="exact cumulative distr function")
    plt.show()





if __name__ == "__main__":
    # test_cauchy()
    # test_gamma()
    test_rho4()
    #test_five_fingers()
    #test_two_gaussians()
    #test_discountinuous()

