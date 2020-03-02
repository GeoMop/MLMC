import numpy as np
import scipy.stats as st
from scipy import integrate
from scipy.special import erf, erfinv
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
from mpl_toolkits import mplot3d


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


class BivariateNorm(st.rv_continuous):
    cov_matrix = [[1, 0], [0, 1]]

    theta = np.pi/3
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    # Rotation matrix
    S = np.diag([1, 5])
    # Scale matrix
    cov_matrix = R @ S @ R.T
    print("cov matrix ", cov_matrix)

    distr = st.multivariate_normal([0, 0], cov_matrix)
    #domain = [np.array([-6, 6]), np.array([-6, 6])]

    # without scale and rotation
    domain = [[-2.3314920916072635, 2.3138243437098813], [-2.3314920916072635, 2.3138243437098813]]
    # rotated which converge
    # quantile 0.01
    domain = [[-4.660711459582694, 4.658294931027472], [-3.289568357766129, 3.290929576432855]]
    # quantile 0.001
    domain = [[-6.140244463903041, 6.177752737538267], [-4.319387540342072, 4.346500163341569]]

    #domain = [[-5.140244463903041, 5.577752737538267], [-3.819387540342072, 3.846500163341569]]

    def pdf(self, values):
        # print("PDF input values ", values)
        # print("MultivariateNorm.distr.pdf(values) ", MultivariateNorm.distr.pdf(values))
        return BivariateNorm.distr.pdf(values)

    def cdf(self, x):
        #print("MultivariateNorm.distr.cdf(x) ", MultivariateNorm.distr.cdf([[0, 1], [0,1]]))
        return BivariateNorm.distr.cdf(x)

    # def _cdf(self, x):
    #     def summation():
    #         result = 0
    #         for weight, distr in zip(Discontinuous.weights, Discontinuous.distributions):
    #             result += weight * distr.cdf(x)
    #         return result
    #     return summation()

    def rvs(self, size):
        print("RVS size ", size)
        return BivariateNorm.distr.rvs(size)


class BivariateTwoGaussians(st.rv_continuous):

    cov_matrix = [[1, 0], [0, 1]]

    distributions = [st.multivariate_normal(mean=[5, 3], cov=cov_matrix),
                     st.multivariate_normal(mean=[0, 0.5], cov=cov_matrix)]

    # quantile = 0.01
    # domain_0 = st.norm(5).ppf([quantile, 1 - quantile])
    # domain_1 = st.norm(0).ppf([quantile, 1 - quantile])
    #
    # print("domain 0 ", domain_0)
    # print("domain 1 ", domain_1)
    # exit()


    weights = [0.5,  0.5]
    # quantile 0.01
    #domain = [[-2.0524503142530817, 7.060607769217207], [-1.5591957102507406, 5.063146613002412]]
    # quantile 0.001
    #domain = [[-2.8862275843108507, 7.926870091821058], [-2.390911372521532, 5.8899990982502075]]

    ##############################################################

    weights = [.8, .2]
    # quantile 0.01
    #domain = [[-1.6372417936274075, 7.229829996175572], [-1.1486875841433914, 5.249333502773785]]
    # quantile 0.001
    # domain = [[-2.5346290813531427, 8.047565706537377], [-2.0800682573833424, 6.030247662630675]]


    def pdf(self, values):
        #print("values ", values)
        result = 0
        for weight, distr in zip(BivariateTwoGaussians.weights, BivariateTwoGaussians.distributions):
            #print("distr.pdf(values) ", distr.pdf(values))
            result += weight * distr.pdf(values)
        return result

    def cdf(self, x):
        result = 0
        for weight, distr in zip(BivariateTwoGaussians.weights, BivariateTwoGaussians.distributions):
            result += weight * distr.cdf(x)
        return result

    def rvs(self, size):
        print("size ", size)
        mixture_idx = np.random.choice(len(BivariateTwoGaussians.weights), size=size, replace=True, p=BivariateTwoGaussians.weights)
        # y is the mixture sample
        print("muxture_idx ", mixture_idx)

        data = []
        for i in mixture_idx:

            data.append(BivariateTwoGaussians.distributions[i].rvs())

        return np.array(data)

        #print("data ", data)
        # exit()
        #
        # print("np.dtype([(np.float64), (np.float64)])) ", np.dtype([("f1", np.float64), ("f2", np.float64)]))
        #
        #
        # # print(" ", np.fromiter((BivariateTwoGaussians.distributions[i].rvs() for i in mixture_idx),
        # #                        dtype=np.ndarray))
        # y, x = np.fromiter((BivariateTwoGaussians.distributions[i].rvs() for i in mixture_idx))

        return y


def test_bivariate_two_gaussians():
    distr = BivariateTwoGaussians()

    x, y = np.mgrid[distr.domain[0][0]:distr.domain[0][1]:.01, distr.domain[1][0]:distr.domain[1][1]:.01]
    print("x.shape ", x.shape)
    print("y.shape ", y.shape)

    ax = plt.axes(projection='3d')
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x
    pos[:, :, 1] = y

    #plt.contourf(x, y, distr.pdf(pos))
    #plt.contourf(x, y, distr.pdf(pos), 50, cmap='RdGy')

    #ax.contour3D(x, y, distr.pdf(pos), 50, cmap='binary')

    # ax.plot_surface(x, y, distr.pdf(pos), rstride=1, cstride=1,
    #                 cmap='viridis', edgecolor='none')

    ax.plot_wireframe(x, y, distr.pdf(pos), color='black')

    plt.show()


def test_bivariate_norm():
    x = np.random.uniform(size=(100, 2))
    distr = BivariateNorm()
    #y = distr.pdf(x, mean=mean, cov=cov)
    # print(y)

    #
    # x = np.linspace(0, 5, 100, endpoint=False)
    # y = multivariate_normal.pdf(x, mean=2.5, cov=0.5)
    #
    # print("x ", x)
    # print("y ", y)

    # plt.plot(x, y)

    x, y = np.mgrid[distr.domain[0][0]:distr.domain[0][1]:.01, distr.domain[1][0]:distr.domain[1][1]:.01]
    print("x.shape ", x.shape)
    print("y.shape ", y.shape)
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x
    pos[:, :, 1] = y

    plt.contourf(x, y, distr.pdf(pos))
    plt.contourf(x, y, distr.pdf(pos), 20, cmap='RdGy')

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
    #test_five_fingers()
    #test_two_gaussians()
    #test_discountinuous()
    #test_bivariate_norm()
    test_bivariate_two_gaussians()

