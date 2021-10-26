import numpy as np
from scipy import stats


class RestrictDistribution:
    """
    Renormalization of PDF, CDF for exact distribution
    restricted to given finite domain.
    """
    @classmethod
    def from_quantiles(cls, distr, quantile):
        """
        Factory with domain given by lower quantile or lower and upper quantile.
        :param quantile:
            lower quantile, upper determined as 1-lower
            or (lower, upper) qunatiles
        """
        try:
            lower, upper = quantile
        except TypeError:
            lower = quantile
            upper = 1 - quantile
        return cls(distr, cls.domain_for_quantile(distr, lower, upper))

    @staticmethod
    def domain_for_quantile(distr, lower, upper):
        """
        Determine domain from quantile. Detect force boundaries.
        :param distr: exact distr
        :param quantile: lower bound quantile, 0 = domain from random sampling
        :return: (lower bound, upper bound), (force_left, force_right)
        """
        # if quantile == 0:
        #     # Determine domain by MC sampling.
        #     if hasattr(distr, "domain"):
        #         domain = distr.domain
        #     else:
        #         X = distr.rvs(size=100000)
        #         err = stats.norm.rvs(size=100000)
        #         # X = X * (1 + 0.1 * err)
        #         domain = (np.min(X), np.max(X))
        #     # p_90 = np.percentile(X, 99)
        #     # p_01 = np.percentile(X, 1)
        #     # domain = (p_01, p_90)
        #
        #     # domain = (-20, 20)
        return distr.ppf(np.array([lower, upper]))

    @staticmethod
    def boundary_decay(distr, domain):
        """
        Detect decreasing PDF out of domain bounds.
        :param distr: scipy distribution
        :param domain: (left, right) bounds
        :return: (positive diff on left, negative diff on right)
        """

        eps = 1e-10
        left, right = domain
        l_diff = (distr.pdf(left + eps) - distr.pdf(left)) / eps
        r_diff = (distr.pdf(right) - distr.pdf(right - eps)) / eps

        return l_diff > 0, r_diff < 0

    def __init__(self, distr, domain, force_decay = None):
        """

        :param distr: scipy.stat distribution object.
        :param quantile: float, define lower bound for approx. distr. domain.
        """
        self.distr = distr
        self.a, self.b = domain

        if force_decay is None:
            force_decay = self.boundary_decay(distr, domain)
        self.force_decay = force_decay
        pa, pb = distr.cdf([self.a, self.b])
        self._shift = pa
        self._scale = 1 / (pb - pa)

        if hasattr(self.distr, 'name'):
            self.distr_name = distr.name
        elif hasattr(self.distr, 'dist'):
            self.distr_name = distr.dist.name
        else:
            self.distr_name = 'unknown'

    def pdf(self, x):
        return self.distr.pdf(x) * self._scale

    def cdf(self, x):
        return (self.distr.cdf(x) - self._shift) * self._scale

    def rvs(self, size):
        p_in = self.distr.cdf(self.b) - self.distr.cdf(self.a)
        values = np.empty((size,))
        n_valid = 0
        while n_valid < size:
            n_add = size - n_valid
            _values = self.distr.rvs(size = int(n_add * 1.1 / p_in))
            _values = np.ma.masked_outside(_values, self.a, self.b).compressed()[:n_add]
            values[n_valid: n_valid + len(_values)]  = _values
            n_valid += len(_values)
        return values

        # x = np.random.uniform(0, 1, size)
        # print("self shift ", self.shift)
        # print("self scale ", self.scale)
        #return (self.distr.rvs(size) - self.shift) * self.scale


class Mixture(stats.rv_continuous):
    """
    Mixture of 'submodels' distribution with 'probabilities'.
    """
    def __init__(self, submodels, weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if weights is None:
            weights = len(submodels) * [1/len(submodels)]
        self.weights = weights / np.sum(weights)
        self.submodels = submodels
        try:
            self.name = kwargs['name']
        except:
            pass

    def _pdf(self, x, *args):
        pdf = np.zeros_like(x)
        for submodel, p in zip(self.submodels, self.weights):
            pdf += p * submodel.pdf(x, *args)
        return pdf

    def _cdf(self, x, *args):
        cdf = np.zeros_like(x)
        for submodel, p in zip(self.submodels, self.weights):
            cdf += p * submodel.cdf(x, *args)
        return cdf

    def rvs(self, size):
        """
        TODO: optimize to draw asymptoticaly just 'size' samples in total.
        """
        samples = [distr.rvs(size=size) for distr in self.submodels]
        random_idx = np.random.choice(np.arange(len(self.submodels)), size=size, p=self.weights)
        return np.choose(random_idx, samples)


def Name(distr, name):
    distr.name = name
    return distr
