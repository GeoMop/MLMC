"""


Implementation TODO:
- support for possitive distributions
- compute approximation of more moments then used for approximation, turn problem into
  overdetermined non-linear least square problem
- Make TestMLMC a production class to test validity of MLMC estimatioon on any sampleset
  using subsampling.

Tests:
For given exact distribution with known density.
and given moment functions.

- compute "exact" moments (Gaussian quadrature, moment functions, exact density)
- construct approximation of the density for given exact moments
- compute L2 norm of density and KL divergence for the resulting density approximation

- compute moments approximation using MC and sampling from the dirstirbution
- compute approximation of the density
- compute L2 and KL
- compute sensitivity to the precision of input moments, estimate precision of the result,
  compute requested precision of the moments


"""
import os
import sys
import time
import pytest

import numpy as np
import scipy.stats as stats
from scipy import integrate
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/../src/')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mlmc.plot
import mlmc.estimate
import mlmc.simple_distribution
from mlmc import moments
from test.fixtures.mlmc_test_run import TestMLMC




class GaussTwo(stats.rv_continuous):
    def __init__(self):
        self._pa = stats.norm(loc=5, scale=3)
        self._pb = stats.norm(loc=0, scale=0.5)
        super().__init__(name="GaussTwo")
        self.dist = self
        self._cdf_points = self.rvs(10000)

    def _pdf(self, x):
        return (9 * self._pa.pdf(x) + 1 * self._pb.pdf(x)) / 10

    def _cdf(self, x):
        return (9 * self._pa.cdf(x) + 1 * self._pb.cdf(x)) / 10


class GaussFive(stats.rv_continuous):
    def __init__(self):
        super().__init__(name="GaussFive")
        self.dist = self

    def _pdf(self, x):
        w = 0.5
        val = 0
        for l in np.arange(1, 10, 2):
            val += stats.norm.pdf(x, loc=l/10, scale=1/100)
        return w * val / 5 + (1 - w) * stats.uniform.pdf(x)

    def _cdf(self, x):
        w = 0.5
        val = 0
        for l in np.arange(1, 10, 2):
            val += stats.norm.cdf(x, loc=l/10, scale=1/100)
        return w * val / 5 + (1 - w) * stats.uniform.cdf(x)

class Rampart(stats.rv_continuous):
    def __init__(self):
        super().__init__(name="Rampart")
        self.dist = self

    def _pdf(self, x):
        return 16.0 / 20 * stats.uniform.pdf(x) + \
            4.5 / 20 * stats.uniform.pdf(x, loc=0.3, scale=0.5) - \
            0.5 / 20 * stats.uniform.pdf(x, loc=0.4, scale=0.1)


    def _cdf(self, x):
        return 16.0 / 20 * stats.uniform.cdf(x) + \
            4.5 / 20 * stats.uniform.cdf(x, loc=0.3, scale=0.5) - \
            0.5 / 20 * stats.uniform.cdf(x, loc=0.4, scale=0.1)

class Abyss(stats.rv_continuous):
    def __init__(self):
        super().__init__(name="Abyss")
        self.dist = self
        self.width = 0.1
        self.z = 0.1
        self.renorm = 2 * stats.norm.cdf(-self.width) + self.z * 2 * self.width
        self.renorm = 1 / self.renorm

        # X = np.linspace(-5, 5, 500)
        # plt.plot(X, self.cdf(X))
        # plt.show()


    def _pdf(self, x):
        y = np.where(np.logical_and(-self.width < x, x < self.width),
                      self.z * stats.uniform.pdf( 0.5 * (x / self.width + 1) ),
                      stats.norm.pdf(x))
        return self.renorm * y

    def _cdf(self, x):
        y =  np.where(np.logical_and(-self.width < x, x < self.width),
                      0.5 + self.renorm * self.z * x,
                      stats.norm.cdf(x))
        return y


class CutDistribution:
    """
    Renormalization of PDF, CDF for exact distribution
    restricted to given finite domain.
    """

    def __init__(self, distr_cfg, quantile):
        """

        :param distr_cfg: scipy.stat distribution object.
        :param quantile: float, define lower bound for approx. distr. domain.
        """
        self.idx, distr_cfg = distr_cfg
        self.distr, self.log_flag = distr_cfg
        if type(quantile) is tuple:
            self.domain = quantile
            self.domain_str = "({}, {})".format(*self.domain)
        else:
            self.domain, self.force_decay = self.domain_for_quantile(self.distr, quantile)
            self.domain_str = "Q:{:4.0g}".format(quantile)

        p0, p1 = self.distr.cdf(self.domain)
        self.shift = p0
        self.scale = 1 / (p1 - p0)

    def __str__(self):
        return "{:02}_{} {}".format(self.idx, self.distr.dist.name, self.domain_str)

    @staticmethod
    def domain_for_quantile(distr, quantile):
        """
        Determine domain from quantile. Detect force boundaries.
        :param distr: exact distr
        :param quantile: lower bound quantile, 0 = domain from random sampling
        :return: (lower bound, upper bound), (force_left, force_right)
        """
        if quantile == 0:
            # Determine domain by MC sampling.
            X = distr.rvs(size=1000)
            err = stats.norm.rvs(size=1000)
            X = X * (1 + 0.1 * err)
            domain = (np.min(X), np.max(X))
        else:
            domain = distr.ppf([quantile, 1 - quantile])

        # Detect PDF decay on domain boundaries, test that derivative is positive/negative for left/right side.
        eps = 1e-10
        force_decay = [False, False]
        for side in [0, 1]:
            diff = (distr.pdf(domain[side]) - distr.pdf(domain[side] - eps)) / eps
            if side:
                diff = -diff
            if diff > 0:
                force_decay[side] = True
        return domain, force_decay

    def pdf(self, x):
        return self.distr.pdf(x) * self.scale

    def cdf(self, x):
        return (self.distr.cdf(x) - self.shift) * self.scale

    def var(self):
        N = 10000
        dsize = self.domain[1] - self.domain[0]
        x = np.linspace(self.domain[0], self.domain[1], N)
        px = self.pdf(x)
        avg = np.mean(x * px) * dsize
        return np.mean( (x - avg) ** 2 * px) * dsize

class DistrTestCase:
    """
    Common code for single combination of cut distribution and moments configuration.
    """
    def __init__(self, distr_cfg, quantile, moments_cfg):
        self.distr = CutDistribution(distr_cfg, quantile)

        self.moment_class, self.min_n_moments, self.max_n_moments = moments_cfg
        self.moments_fn = self.moment_class(self.max_n_moments, self.distr.domain, log=self.distr.log_flag, safe_eval=False)

        self.exact_covariance = mlmc.simple_distribution.compute_semiexact_cov(self.moments_fn, self.distr.pdf)
        self.eigenvalues_plot = mlmc.plot.Eigenvalues(title="Eigenvalues, " + self.title)

    @property
    def title(self):
        fn_name = str(self.moment_class.__name__)
        return "distr: {} moment_fn: {}".format(self.distr, fn_name)


    def noise_cov(self, noise):
        noise_mat = np.random.randn(self.moments_fn.size, self.moments_fn.size)
        noise_mat = 0.5 * noise * (noise_mat + noise_mat.T)
        noise_mat[0, 0] = 0
        return self.exact_covariance + noise_mat


    def make_orto_moments(self, noise):
        cov = self.noise_cov(noise)
        orto_moments_fn, info = mlmc.simple_distribution.construct_ortogonal_moments(self.moments_fn, cov, tol=noise)
        evals, threshold, L = info
        print("threshold: ", threshold, " from N: ", self.moments_fn.size)
        self.eigenvalues_plot.add_values(evals, threshold=evals[threshold], label="{:5.2e}".format(noise))
        eye_approx = L @ cov @ L.T
        # test that the decomposition is done well
        assert np.linalg.norm(eye_approx - np.eye(*eye_approx.shape)) < 1e-10
        # TODO: test deviance from exact covariance in some sense
        self.exact_orto_moments = mlmc.simple_distribution.compute_semiexact_moments(orto_moments_fn, self.distr.pdf, tol=1e-13)

        tol_density_approx = 0.01
        moments_data = np.ones((orto_moments_fn.size, 2))
        moments_data[1:, 0] = 0.0
        #moments_data[0,1] = 0.01
        return orto_moments_fn, moments_data


class TestDistr(mlmc.simple_distribution.SimpleDistribution):
    # Auxiliary fixture for plotting convergence of nonlin solve.
    def _calculate_jacobian_matrix(self, multipliers):
        jac = super()._calculate_jacobian_matrix(multipliers)
        self._normalize = jac[0,0]
        self.multipliers = multipliers
        self._plot_distr.add_distribution(self)
        return jac

    def density(self, x):
        vals = super().density(x)
        return vals / self._normalize

distribution_list = [
        # distibution, log_flag
        #(stats.norm(loc=1, scale=2), False),
        (stats.lognorm(scale=np.exp(1), s=1), False),    # Quite hard but peak is not so small comparet to the tail.
    ]
moments_list = [
    (moments.Legendre, 7, 61),
    ]
@pytest.mark.skip
@pytest.mark.parametrize("moments", moments_list)
@pytest.mark.parametrize("distribution", enumerate(distribution_list))
def test_nonlin_solver_convergence(moments, distribution):
    """
    1. compute covariance matrix
    2. for given noise construct orthogonal moments
    3. find distribution, plot density and quad intervals for every Jacobian evaluation
    Plot convergence in terms of densities for every case.
    """
    quantile = 0.001

    case = DistrTestCase(distribution, quantile, moments)
    noise_levels = np.geomspace(1e-5, 1e-3, 6)
    noise_levels = [0.00039810717055349735]
    for noise in noise_levels:
        print("======================")
        print("noise: ", noise)
        orto_moments, moment_data = case.make_orto_moments(noise)
        distr_obj = TestDistr(orto_moments, moment_data,
                                domain=case.distr.domain, force_decay=case.distr.force_decay)
        distr_obj._plot_distr = mlmc.plot.Distribution(exact_distr=case.distr, error_plot=None)
        min_result = distr_obj.estimate_density_minimize(tol=3*noise)
        # print("lambda: ", distr_obj.multipliers)
        distr_obj._plot_distr.show(file="NonlinSolve_{}_{}".format(case.title, noise))







# 24m without update; trust-ncg
# 24m 21s with update; trust-ncg
# 8m 15 s with updates, dogleg, diff2
# 7m 43s with updates, diff, - better error

distribution_list = [
        # distibution, log_flag
        (stats.norm(loc=1, scale=2), False),            # 9-13, 9-27, 7-9
        # (stats.norm(loc=1, scale=10), False),           # 8-18, 8-14, 7-10
        (stats.lognorm(scale=np.exp(1), s=1), False),   # >50 it, , 3-11,  some errors as non-SPD
        # (stats.lognorm(scale=np.exp(-3), s=2), False),  # > 50 it, , errors non-SPD
        # (stats.lognorm(scale=np.exp(-3), s=2), True),   # > 50 it  , errors non-SPD
        # (stats.chi2(df=10), False),                     # 13-39, 10-21, 4-14
        # (stats.chi2(df=5), True),                       # 12-48, 14 -50, 9-11
        # (stats.weibull_min(c=0.5), False),              # >50, , non-SPD
        # (stats.weibull_min(c=1), False),                # 21->50, 19-50, non-SPD, Exponential
        # (stats.weibull_min(c=2), False),                # 11-22, 11-35, 5-8 1x nonSPD, Rayleigh distribution
        # (stats.weibull_min(c=5, scale=4), False),       # 9-16, 10-18, 6-7, (1x>50) close to normal
        # (stats.weibull_min(c=1.5), True),               # 14-39, 14-37, 6-7, Infinite derivative at zero
        (stats.gamma(a=1), False),
        (GaussTwo(), False),
        (GaussFive(), False),
        (stats.cauchy(scale=0.5), False),
        (Rampart(), False)
    ]

moments_list = [
    # moments_class, min and max number of moments, use_covariance flag
    #(moments.Monomial, 3, 10),
    #(moments.Fourier, 5, 61),
    #(moments.Legendre, 7, 61, False),
    (moments.Legendre, 3, 7),
    ]
@pytest.mark.skip
@pytest.mark.parametrize("moments", moments_list)
@pytest.mark.parametrize("distribution", enumerate(distribution_list))
def test_nonlin_solver_robustness(moments, distribution):
    quantile = 0.01

    case = DistrTestCase(distribution, quantile, moments)
    plot_distr = mlmc.plot.Distribution(exact_distr=case.distr, error_plot="kl", legend_title="noise",
                                        log_x=case.distr.log_flag)
    n_samples = np.geomspace(2**8, 2**20, 4)
    noise_levels = np.sqrt( case.distr.var() / n_samples)
    for noise in noise_levels:
        print("======================")
        print("noise: ", noise)
        orto_moments, moment_data = case.make_orto_moments(noise)
        distr_obj = mlmc.simple_distribution.SimpleDistribution(orto_moments, moment_data,
                                domain=case.distr.domain, force_decay=case.distr.force_decay)

        min_result = distr_obj.estimate_density_minimize(tol=noise)
        plot_distr.add_distribution(distr_obj, label=str(noise))
    plot_distr.show(file="NSolve_rob_{}".format(case.title))




def plot_scatter(ax, X, Y, title, xy_scale, **kw):
    ax.set_title(title)
    ax.set_xscale(xy_scale[0])
    ax.set_yscale(xy_scale[1])
    if xy_scale[0] == 'log':
        ax.set_xlim((1e-5, 1e1))
        lx = np.geomspace(1e-5, 0.1, 100)
    else:
        #ax.set_xlim((0, 1))
        pass
    if xy_scale[1] == 'log':
        ax.set_ylim((1e-2, 1e2))
    else:
        ax.set_ylim((0, 1.2))
    ax.scatter(X, Y, marker='.', edgecolors = 'none', **kw)
    return ax, lx



def kl_estimates(distribution, moments, ax):
    quantile = 0.01

    case = DistrTestCase(distribution, quantile, moments)
    orto_moments, moment_data = case.make_orto_moments(0)
    exact_distr = mlmc.simple_distribution.SimpleDistribution(orto_moments, moment_data,
                                                            domain=case.distr.domain,
                                                            force_decay=case.distr.force_decay)

    # true_pdf = case.distr.pdf
    # a, b = case.distr.domain
    tolerance = 1e-10
    min_result = exact_distr.estimate_density_minimize(tol=tolerance)
    # exact_tol = max(min_result.res_norm, tolerance)
    exact_mu = case.exact_orto_moments
    exact_eval_0, exact_eval_max = exact_distr.jacobian_spectrum()[[0, -1]]
    mu_diffs, l_diffs, eigs = [], [], []
    #ratio_distribution = stats.lognorm(s=0.1)
    ratio_distribution = stats.norm(scale=0.01*np.linalg.norm(exact_distr.multipliers[1:]))
    raw_distr = mlmc.simple_distribution.SimpleDistribution(orto_moments, moment_data,
                                                            domain=case.distr.domain,
                                                            force_decay=case.distr.force_decay)

    size = len(exact_distr.multipliers)
    linf_log_approx_error = np.max(np.log(case.distr.pdf(exact_distr._quad_points))
                                   - np.log(exact_distr.density(exact_distr._quad_points)))
    b_factor_estimate = np.exp(linf_log_approx_error)
    linf_inv_distr = np.max(1/case.distr.pdf(exact_distr._quad_points))
    Am_factor_estimate = (orto_moments.size + 1) * np.sqrt(linf_inv_distr)


    # max_norm_momemnts = np.max(np.linalg.norm(exact_distr._quad_moments, axis=1))
    # print("max norm of momments: ", max_norm_momemnts)

    for _ in range(1000):
        s = 3 * stats.uniform.rvs(size=1)[0]
        lambda_inex = exact_distr.multipliers + s*ratio_distribution.rvs(size)
        raw_distr._initialize_params(size)
        raw_distr.multipliers = lambda_inex
        raw_distr.set_quadrature(exact_distr)
        raw_distr.moments = raw_distr.moments_by_quadrature()
        #m0 = density_integral(raw_distr)

        # if not np.isclose(moments[0], m0):
        #     continue

        #raw_distr.multipliers[0] += np.log(moments[0])
        #raw_distr.moments = raw_distr.moments_by_quadrature()

        #assert np.isclose(raw_distr.moments[0],  1.0, atol=1e-5)
        raw_eval_0, raw_eval_max  = raw_distr.jacobian_spectrum()[[0, -1]]
        #print("evals: ", raw_eval_0, raw_eval_max)
        lambda_diff = -(exact_distr.multipliers - raw_distr.multipliers)
        l_diff_norm = np.linalg.norm(lambda_diff[:])
        mu_diff = exact_mu - raw_distr.moments
        mu_diff_norm = np.linalg.norm(mu_diff[:])
        l_diffs.append(l_diff_norm)
        mu_diffs.append(mu_diff_norm)
        eigs.append((raw_eval_0, raw_eval_max))

        # kl_true_exact = kl(true_pdf, exact_distr.density, exact_distr)
        # kl_exact_raw = kl(exact_distr.density, raw_distr.density, exact_distr)
        # kl_true_raw = kl(true_pdf, raw_distr.density, exact_distr)
        # err = kl_true_exact + kl_exact_raw - kl_true_raw
        # print("KL(true, exact): {} + KL(exact, raw): {} - KL(true, raw) : {}  =  {}".format(
        #     kl_true_exact, kl_exact_raw, kl_true_raw, err))
        # assert np.abs(err) < 10 * exact_tol + exact_tol * kl_true_raw

        #print("lamda diff: {} < mu diff / L0: {} |  {}".format(
        #    l_diff_norm, mu_diff_norm, l_diff_norm - mu_diff_norm))

        # lambda_dif_mu = lambda_diff @ exact_mu
        # lambda_dif_mu_dif = np.dot(lambda_diff, mu_diff)
        # theory_bound = mu_diff @ mu_diff / raw_eval_0
        # kl_modif = KL_div_modif(exact_distr, raw_distr, exact_distr)
        # print("KL(rho_e, rho_n):   {} = kl_modif: {} | {}".format(kl_exact_raw, kl_modif, kl_exact_raw - kl_modif))
        # print("KL(rho_e, rho_n):   {} = (l_e - l_n) . mu_e: {} | {}".format(kl_exact_raw, lambda_dif_mu, kl_exact_raw - lambda_dif_mu))
        # print("(l_e - l_n) . mu_e: {} < dl . dmu            {} | {}".format(lambda_dif_mu, lambda_dif_mu_dif , lambda_dif_mu - lambda_dif_mu_dif))
        # print("dl . dmu:           {} < dmu / L0            {} | {}".format(lambda_dif_mu_dif, theory_bound, lambda_dif_mu_dif - theory_bound))


    Y = np.array(l_diffs) * np.array(np.array(eigs)[:, 0]) / np.array(mu_diffs)
    #Y = np.array(eigs)
    ax, lx = plot_scatter(ax, mu_diffs, Y, case.title, ('log', 'linear'), color='red')
    #ax, lx = plot_scatter(ax, mu_diffs, Y[:,0], case.title, ('log', 'log'), color='red')
    #plot_scatter(ax, mu_diffs, Y[:, 1], case.title, ('log', 'log'), color='blue')
    ax.set_ylabel("$\\alpha_0|\lambda_0 - \lambda_r| / |\mu_0 - \mu_r|$")
    ax.set_xlabel("$|\mu_0 - \mu_r|$")
    ax.axhline(y=1.0, color='red', alpha=0.3)
    #ax.plot(lx, lx , color='red', label="raw $1/\\alpha_0$", alpha=0.3)

    # plot_scatter(ax, mu_diffs, l_diffs, case.title, ('log', 'log'), color='blue')
    # shaw_coef = 2 * b_factor_estimate * np.exp(1)
    # shaw_mu_lim = 1 / (4 * np.exp(1) * b_factor_estimate * Am_factor_estimate)
    # ax.plot(lx, lx * shaw_coef, color='blue', label="shaw", alpha=0.3)
    # ax.axvline(x=shaw_mu_lim, color='blue', alpha=0.3)
    case.eigenvalues_plot.show("")

    def plot_mineig_by_lambda():
        plt.suptitle(case.title)
        lx = np.geomspace(1e-10, 0.1, 100)
        Y = exact_eval_0 * np.ones_like(lx)
        plt.plot(lx, Y, color='red')

        plt.scatter(l_diffs, eigs, marker='.')
        #plt.ylim((1e-5, 0.1))
        plt.xlim((1e-5, 0.1))
        # #lx = np.linspace(1e-10, 0.1, 100)
        # plt.plot(lx, lx / raw_eval_0, color='orange')
        # #plt.plot(lx, lx / raw_eval_max, color='green')
        plt.xscale('log')
        # plt.yscale('log')
        plt.show()



@pytest.mark.parametrize("moments", moments_list)
def test_kl_estimates(moments):
    """
    Test estimates for KL divergence
    1. find exact moments, find exact lambda
    2. generate varied lambda
    3. compute corresponding moments and KL
    4. plot individual terms

    """
    distribution_list = [
        # distibution, log_flag
        (stats.lognorm(scale=np.exp(1), s=1), False),  # >50 it, , 3-11,  some errors as non-SPD
        (stats.gamma(a=1), False),
        (GaussTwo(), False),
        (GaussFive(), False),
        (stats.cauchy(scale=0.5), False),
        (Abyss(), False)
    ]
    shape = (2, 3)
    fig, axes = plt.subplots(*shape, sharex=True, sharey=True,
                             figsize=(15, 10))
    #fig.suptitle("Mu -> Lambda")
    axes = axes.flatten()
    for distr, ax in zip(enumerate(distribution_list), axes[:len(distribution_list)]):
       kl_estimates(distr, moments, ax)
    plt.tight_layout()
    #mlmc.plot._show_and_save(fig, "", "mu_to_lambda_lim")
    mlmc.plot._show_and_save(fig, "", "mu_to_alpha")






def density_integral(distr):
    q_density = distr.density(distr._quad_points)
    return q_density @ distr._quad_weights

def kl(prior, post, quad_distr):
    p = prior(quad_distr._quad_points)
    q = post(quad_distr._quad_points)
    integrand = p * np.log(p / q) - p + q
    return integrand @ quad_distr._quad_weights

def KL_div_modif(prior_distr, post_distr, quad_distr):
    p = prior_distr.density(quad_distr._quad_points)
    q = post_distr.density(quad_distr._quad_points)
    moms = quad_distr._quad_moments.T
    integrand = - p * np.dot( (prior_distr.multipliers - post_distr.multipliers), moms) - p + q
    return integrand @ quad_distr._quad_weights



class ConvResult:
    """
    Results of a convergency calculation.
    TODO: make some of these part of Distribution.
    """
    def __init__(self):
        self.size = 0
        # Moment sizes used
        self.noise = 0.0
        # Noise level used in Covariance and moments.
        self.kl = np.nan
        # KL divergence KL(exact, approx) for individual sizes
        self.l2 = np.nan
        # L2 distance of densities
        self.time = 0
        # times of calculations of density approx.
        self.nit = 0
        # number of iterations in minimization problems
        self.residual_norm = 0
        # norm of residual of non-lin solver
        self.success = False

    def __str__(self):
        return "#{} it:{} err:{:6.2g} kl:{:6.2g} l2:{:6.2g} lambda_0:{:4.2} bound:{:6.2g}".format(self.size, self.nit, self.residual_norm,
                                                               self.kl, self.l2, self.lambda_0, self.theory_bound)

class DistributionDomainCase:
    """
    Class to perform tests for fully specified exact distribution (distr. + domain) and moment functions.
    Exact moments and other objects are created once. Then various tests are performed.
    """

    def __init__(self, moments, distribution, quantile):
        # Setup distribution.
        i_distr, distribution = distribution
        distr, log_flag = distribution
        self.log_flag = log_flag
        self.quantile = quantile
        self.distr_name = "{:02}_{}".format(i_distr, distr.dist.name)
        self.cut_distr = CutDistribution(distr, quantile)
        self.moments_data = moments
        moment_class, min_n_moments, max_n_moments, self.use_covariance = moments
        self.fn_name = str(moment_class.__name__)
        # domain_str = "({:5.2}, {:5.2})".format(*self.domain)
        self.eigenvalues_plot = None


    @property
    def title(self):
        cov = "_cov" if self.use_covariance else ""
        return "distr: {} quantile: {} moment_fn: {}{}".format(self.distr_name, self.quantile, self.fn_name, cov)

    def pdfname(self, subtitle):
        return "{}_{}.pdf".format(self.title, subtitle)

    @property
    def domain(self):
        return self.cut_distr.domain

    def pdf(self, x):
        return self.cut_distr.pdf(x)

    def setup_moments(self, moments_data, noise_level):
        """
        Setup moments without transformation.
        :param moments_data: config tuple
        :param noise_level: magnitude of added Gauss noise.
        :return:
        """
        tol_exact_moments = 1e-13
        moment_class, min_n_moments, max_n_moments, self.use_covariance = moments_data
        log = self.log_flag
        self.moment_sizes = np.round(np.geomspace(min_n_moments, max_n_moments, 8)).astype(int)
        #self.moment_sizes = [3,4,5,6,7]
        self.moments_fn = moment_class(max_n_moments, self.domain, log=log, safe_eval=False)


        if self.use_covariance:
            size = self.moments_fn.size
            base_moments = self.moments_fn
            exact_cov = mlmc.simple_distribution.compute_semiexact_cov(base_moments, self.pdf)
            noise = np.random.randn(size**2).reshape((size, size))
            noise += noise.T
            noise *= 0.5 * noise_level
            noise[0, 0] = 0
            cov = exact_cov + noise
            self.moments_fn, info = mlmc.simple_distribution.construct_ortogonal_moments(base_moments, cov, noise_level)
            evals, threshold, L = info

            eye_approx = L @ cov @ L.T
            # test that the decomposition is done well
            assert np.linalg.norm(eye_approx - np.eye(*eye_approx.shape)) < 1e-10

            print("threshold: ", threshold, " from N: ", size)
            if self.eigenvalues_plot:
                threshold = evals[threshold]
                noise_label = "{:5.2e}".format(noise_level)
                self.eigenvalues_plot.add_values(evals, threshold=threshold, label=noise_label)
            self.tol_density_approx = 0.01
        else:
            self.exact_moments += noise_level * np.random.randn(self.moments_fn.size)
            self.tol_density_approx = 1e-4

        self.exact_moments = mlmc.simple_distribution.compute_semiexact_moments(self.moments_fn,
                                self.pdf, tol=tol_exact_moments)



    def check_convergence(self, results):
        # summary values
        sizes = np.log([r.size for r in results])
        kl = np.log([r.kl for r in results])
        sizes = sizes[~np.isnan(kl)]
        kl = kl[~np.isnan(kl)]
        n_failed = sum([not r.success for r in results])
        total_its = sum([r.nit for r in results])
        total_time = sum([r.time for r in results])
        if len(kl) > 2:
            s1, s0 = np.polyfit(sizes, kl, 1)
        max_err = np.max(kl)
        min_err = np.min(kl)

        # print
        print("CASE {} | failed: {} kl_decay: {} nit: {} time: {:3.1}".format(
            self.title, n_failed, s1, total_its, total_time))

        # check
        #assert n_failed == 0
        #assert s1 < -1
        # # Check convergence
        # if domain_quantile > 0.01:
        #     continue
        # if not (n_failed[-1] == 0 and (max_err < tol_density_approx * 8 or s1 < -1)):
        #     warn_log.append((i_q, n_failed[-1], s1, s0, max_err))
        #     fail = "FF"
        # else:
        #     fail = ' q'
        # print(
        #     fail + ": ({:5.3g}, {:5.3g});  failed: {} cumit: {} tavg: {:5.3g};  s1: {:5.3g} s0: {:5.3g} kl: ({:5.3g}, {:5.3g})".format(
        #         domain[0], domain[1], n_failed[-1], tot_nit, cumtime / tot_nit, s1, s0, min_err, max_err))


    def make_approx(self, distr_class, noise, moments_data, tol):
        result = ConvResult()
        distr_obj = distr_class(self.moments_fn, moments_data,
                                domain=self.domain, force_decay=self.cut_distr.force_decay)
        t0 = time.time()
        min_result = distr_obj.estimate_density_minimize(tol=tol)
        moments = mlmc.simple_distribution.compute_semiexact_moments(self.moments_fn, distr_obj.density)
        print("moments approx error: ", np.linalg.norm(moments - self.exact_moments), "m0: ", moments[0])

        # result = profile(lambda : distr_obj.estimate_density_minimize(tol_exact_moments))
        t1 = time.time()
        result.size = moments_data.shape[0]
        result.noise = noise
        result.time = t1 - t0
        result.residual_norm = min_result.fun_norm
        result.success = min_result.success
        result.moments = moments
        result.lambda_0 = min_result.eigvals[0]

        if result.success:
            result.nit = min_result.nit
        result.distribution = distr_obj
        return result, distr_obj



    def exact_conv(self):
        """
        Test density approximation for varying number of exact moments.
        :return:
        """
        # Setup moments.
        self.setup_moments(self.moments_data, noise_level=0)

        mlmc.plot.moments(self.moments_fn, size=21, title=self.title+"_moments")

        results = []
        distr_plot = mlmc.plot.Distribution(exact_distr=self.cut_distr, title=self.title+"_exact",
                                            log_x=self.log_flag, error_plot='kl')
        for i_m, n_moments in enumerate(self.moment_sizes):
            if n_moments > self.moments_fn.size:
                continue
            print("======================")
            print("EXACT CONV - ", self.title)

            # moments_fn = moment_fn(n_moments, domain, log=log_flag, safe_eval=False )
            # print(i_m, n_moments, domain, force_decay)
            moments_data = np.empty((n_moments, 2))
            moments_data[:, 0] = self.exact_moments[:n_moments]
            moments_data[:, 1] = 1.0

            if self.use_covariance:
                modif_cov = mlmc.simple_distribution.compute_semiexact_cov(self.moments_fn, self.pdf)
                diff_norm = np.linalg.norm(modif_cov - np.eye(*modif_cov.shape))
                print("#{} cov mat norm: {}".format(n_moments, diff_norm))

                result, distr_obj = self.make_approx(mlmc.simple_distribution.SimpleDistribution, 0.0, moments_data,
                                                     tol=1e-5)
            else:
                # TODO:
                # Use SimpleDistribution only as soon as it use regularization that improve convergency even without
                # cov matrix. preconditioning.
                result, distr_obj = self.make_approx(mlmc._distribution.Distribution, 0.0, moments_data)
            distr_plot.add_distribution(distr_obj, label="#{}".format(n_moments))
            results.append(result)

        #self.check_convergence(results)
        distr_plot.show(file=self.pdfname("_pdf_exact"))
        distr_plot.reset()
        return results

    def inexact_conv(self):
        """
        Test density approximation for maximal number of moments
        and varying amount of noise added to covariance matrix.
        :return:
        """
        min_noise = 1e-5
        max_noise = 1e-2
        results = []
        distr_plot = mlmc.plot.Distribution(exact_distr=self.cut_distr, legend_title="noise level", title="Density, " + self.title,
                                            log_x=self.log_flag, error_plot='kl')
        self.eigenvalues_plot = mlmc.plot.Eigenvalues(title = "Eigenvalues, " + self.title)

        geom_seq = np.geomspace(min_noise, max_noise, 4)
        noise_levels = np.concatenate(([0.0], geom_seq))
        for noise in noise_levels:
            print("======================")
            print("INEXACT CONV - ", self.title)
            self.setup_moments(self.moments_data, noise_level=noise)
            n_moments = len(self.exact_moments)

            moments_data = np.zeros((n_moments, 2))
            moments_data[0, 0] = 1.0
            moments_data[:, 1] = 1.0

            if self.use_covariance:
                modif_cov = mlmc.simple_distribution.compute_semiexact_cov(self.moments_fn, self.pdf)
                diff_norm = np.linalg.norm(modif_cov - np.eye(*modif_cov.shape)) / n_moments
                ref_moments = np.zeros(n_moments)
                ref_moments[0] = 1.0
                mom_err = np.linalg.norm(self.exact_moments - ref_moments) / np.sqrt(n_moments)
                print("noise: {:6.2g} error of natural cov: {:6.2g} natural moments: {:6.2g}".format(
                    noise, diff_norm, mom_err))
                # ASSERT
                if not mom_err/(noise + 1e-10) < 50:
                    print("Violated ASSERT: {} < {}".format(mom_err/(noise + 1e-10), 50))

                result, distr_obj = self.make_approx(mlmc.simple_distribution.SimpleDistribution, noise, moments_data,
                                                     tol=max(0.1*noise, 1e-10))


                self.compute_diffs(result)
            else:
                # TODO:
                # Use SimpleDistribution only as soon as it use regularization that improve convergency even without
                # cov matrix. preconditioning.
                result, distr_obj = self.make_approx(mlmc._distribution.Distribution, noise, moments_data)
            distr_plot.add_distribution(distr_obj, label="{:5.1e}".format(noise))
            results.append(result)

        #self.check_convergence(results)
        self.eigenvalues_plot.show(file = self.pdfname("_eigenvalues"))
        distr_plot.show(file=self.pdfname("_pdf_iexact"))
        distr_plot.reset()
        return results











def plot_convergence(quantiles, conv_val, title):
    """
    Plot convergence with moment size for various quantiles.
    :param quantiles: iterable with quantiles
    :param conv_val: matrix of ConvResult, n_quantiles x n_moments
    :param title: plot title and filename used to save
    :return:
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    for iq, q in enumerate(quantiles):
        results = conv_val[iq]
        #X = [r.size for r in results]
        X = np.arange(len(results))
        kl = [r.kl for r in results]
        #l2 = [r.l2 for r in results]
        bound = [r.theory_bound for r in results]
        lambda_dif_mu = [r.lambda_dif_mu for r in results]
        lambda_dif_mu_dif = [r.lambda_dif_mu for r in results]
        col = plt.cm.tab10(plt.Normalize(0,10)(iq))
        ax.plot(X, kl, ls='dotted', c='black', label="kl_q="+str(q))
        #ax.plot(X, l2, ls='dashed', c=col, label="l2_q=" + str(q), marker='d')
        ax.plot(X, bound, ls='dotted', c='red', label="mu_bound_q=" + str(q), marker='d')
        ax.plot(X, lambda_dif_mu, ls='dotted', c='blue', label="lambda_diff_mu q=" + str(q), marker='d')
        ax.plot(X, lambda_dif_mu_dif, ls='dotted', c='orange', label="lambda dif mu dif_q=" + str(q), marker='d')

    ax.set_yscale('log')
    ax.set_xscale('log')
    fig.legend()
    fig.suptitle(title)
    fname = title + ".pdf"
    fig.savefig(fname)


distribution_list = [
        # distibution, log_flag
        (stats.norm(loc=1, scale=2), False),
        #(stats.norm(loc=1, scale=10), False),
        (stats.lognorm(scale=np.exp(1), s=1), False),    # Quite hard but peak is not so small comparet to the tail.
        #(stats.lognorm(scale=np.exp(-3), s=2), False),  # Extremely difficult to fit due to very narrow peak and long tail.
        # (stats.lognorm(scale=np.exp(-3), s=2), True),    # Still difficult for Lagrange with many moments.
        # (stats.chi2(df=10), False), # Monomial: s1=nan, Fourier: s1= -1.6, Legendre: s1=nan
        # (stats.chi2(df=5), True), # Monomial: s1=-10, Fourier: s1=-1.6, Legendre: OK
        # (stats.weibull_min(c=0.5), False),  # Exponential # Monomial stuck, Fourier stuck
        # (stats.weibull_min(c=1), False),  # Exponential
        # (stats.weibull_min(c=2), False),  # Rayleigh distribution
        # (stats.weibull_min(c=5, scale=4), False),   # close to normal
        # (stats.weibull_min(c=1.5), True),  # Infinite derivative at zero
    ]

moments_list = [
    # moments_class, min and max number of moments, use_covariance flag
    #(moments.Monomial, 3, 10),
    #(moments.Fourier, 5, 61),
    #(moments.Legendre, 7, 61, False),
    (moments.Legendre, 7, 61),
    ]




@pytest.mark.skip
@pytest.mark.parametrize("moments", [
    # moments_class, min and max number of moments, use_covariance flag
    #(moments.Monomial, 3, 10),
    #(moments.Fourier, 5, 61),
    #(moments.Legendre, 7, 61, False),
    (moments.Legendre, 7, 61, True),
    ])
@pytest.mark.parametrize("distribution", enumerate(distribution_list))
def test_pdf_approx_exact_moments(moments, distribution):
    """
    Test reconstruction of the density function from exact moments.
    - various distributions
    - various moments functions
    - test convergency with increasing number of moments
    :return:
    """
    quantiles = np.array([0.01])
    #quantiles = np.array([0.01])
    conv = {}
    # Dict of result matricies (n_quantiles, n_moments) for every performed kind of test.
    for i_q, quantile in enumerate(quantiles):
        np.random.seed(1234)
        case = DistributionDomainCase(moments, distribution, quantile)
        # tests = [case.exact_conv, case.inexact_conv]
        tests = [case.inexact_conv]
        for test_fn in tests:
            name = test_fn.__name__
            test_results = test_fn()
            values = conv.setdefault(name, (case.title, []))
            values[1].append(test_results)

    for key, values in conv.items():
        title, results = values
        title = "{}_conv_{}".format(title, key)
        if results[0] is not None:
            plot_convergence(quantiles, results, title=title)

    # kl_collected = np.empty( (len(quantiles), len(moment_sizes)) )
    # l2_collected = np.empty_like(kl_collected)
    # n_failed = []
    # warn_log = []
    #
    #     kl_collected[i_q, :], l2_collected[i_q, :] = exact_conv(cut_distr, moments_fn, tol_exact_moments, title)
    #
    #
    # plot_convergence(moment_sizes, quantiles, kl_collected, l2_collected, title)
    #
    # #assert not warn_log
    # if warn_log:
    #     for warn in warn_log:
    #         print(warn)




# distr_sublist = [
#         (stats.norm(loc=1, scale=10), False),
#         (stats.lognorm(scale=np.exp(1), s=1), False),    # # No D_KL convergence under 1e-3.
#         #(stats.lognorm(scale=np.exp(-3), s=2), False),  # Extremely difficult to fit due to very narrow peak and long tail.
#         (stats.lognorm(scale=np.exp(-3), s=2), True),    # Still difficult for Lagrange with many moments.
#         (stats.chi2(df=5), True), # Monomial: s1=-10, Fourier: s1=-1.6, Legendre: OK
#         # (stats.weibull_min(c=0.5), False),  # No D_KL convergence under 1e-1.
#         (stats.weibull_min(c=1), False),  # Exponential
#         (stats.weibull_min(c=2), False),  # D_KL steady under 1e-6.
#         (stats.weibull_min(c=1.5), True),  # Infinite derivative at zero
#     ]
#
# @pytest.mark.skip
# @pytest.mark.parametrize("distribution", distribution_list)
# def test_pdf_approx_iexact_moments(distribution):
#     np.random.seed(67)
#
#     distr, log_flag = distribution
#     distr_name = distr.dist.name
#     print("Inexact PDF approx for distribution: {}".format(distr_name))
#
#     domain_quantile = 0
#     domain, force_decay = domain_for_quantile(distr, domain_quantile)
#     moments_fn = moments.Legendre(21, domain, log=log_flag, safe_eval=False)
#
#
#
#     # Approximation for exact moments
#     tol_exact_moments = 1e-6
#     density = lambda x: distr.pdf(x)
#     exact_moments = mlmc.distribution.compute_exact_moments(moments_fn, density, tol=tol_exact_moments)
#     # Estimate variances
#     X = distr.rvs(size=1000)
#     est_moment_vars = np.var(moments_fn(X), axis=0, ddof=1)
#     #print(est_moment_vars)
#
#     # fail: 1, 6
#     # fine: 4
#     distr_plot = DistrPlot(distr, distr_name+", for inexact moments")
#     moment_errors = np.exp(np.linspace(np.log(0.01), np.log(0.000001), 20))
#     kl_collected = np.empty( len(moment_errors) )
#     l2_collected = np.empty_like(kl_collected)
#     warn_log = []
#     n_failed = 0
#     cum_time = 0
#     cum_it = 0
#     for i_m, err in enumerate(moment_errors):
#         perturb = 0* stats.norm.rvs(size = moments_fn.size) * err * np.sqrt(est_moment_vars)
#         moments_data = np.empty((moments_fn.size, 2))
#         moments_data[:, 0] = exact_moments + perturb
#         moments_data[:, 1] = est_moment_vars
#         distr_obj = mlmc.distribution.Distribution(moments_fn, moments_data,
#                                                    domain=domain, force_decay=force_decay)
#         t0 = time.time()
#         # result = distr_obj.estimate_density(tol_exact_moments)
#         result = distr_obj.estimate_density_minimize(err*4)
#         #result = profile(lambda : distr_obj.estimate_density_minimize(tol_exact_moments))
#         t1 = time.time()
#         cum_time += t1 - t0
#         nit = getattr(result, 'nit', result.njev)
#         cum_it += nit
#         fn_norm = result.fun_norm
#         if result.success:
#             kl_div = mlmc.distribution.KL_divergence(density, distr_obj.density, domain[0], domain[1])
#             l2_dist = mlmc.distribution.L2_distance(distr_obj.density, density, domain[0], domain[1])
#             kl_collected[i_m] = kl_div
#             l2_collected[i_m] = l2_dist
#             #print("q: {}, err: {:7.3g} :: nit: {} fn: {} ; kl: {} l2: {}".format(
#             #    domain_quantile, err, nit, fn_norm, kl_div, l2_dist))
#             distr_plot.plot_approximation(distr_obj, str(err))
#         else:
#             n_failed+=1
#             print("q: {}, err {} :: nit: {} fn:{} ; msg: {}".format(
#                 domain_quantile, err, nit, fn_norm, result.message))
#
#             kl_collected[i_m] = np.nan
#             l2_collected[i_m] = np.nan
#
#     # Check convergence
#     #print(kl_collected)
#     s1, s0 = np.polyfit(np.log(moment_errors), np.log(kl_collected), 1)
#     max_err = np.max(kl_collected)
#     min_err = np.min(kl_collected)
#     if not (n_failed == 0 and (max_err < 1e-6 or s1 > 0)):
#         warn_log.append((domain_quantile, n_failed,  s1, s0, max_err))
#         fail = 'NQ'
#     else:
#         fail = ' q'
#     fail = ' q'
#     print(fail + ": ({:5.3g}, {:5.3g});  failed: {} tavg: {:5.3g};  s1: {:5.3g} s0: {:5.3g} kl: ({:5.3g}, {:5.3g})".format(
#         domain[0], domain[1], n_failed, cum_time/cum_it, s1, s0, min_err, max_err))
#
#     #distr_plot.show()
#     distr_plot.clean()
#
#
#
#     def plot_convergence():
#         plt.plot(moment_errors, kl_collected, ls='solid', c='red')
#         plt.plot(moment_errors, l2_collected, ls='dashed', c='blue')
#         plt.yscale('log')
#         plt.xscale('log')
#         plt.legend()
#         plt.show()
#
#     #plot_convergence()
#
#     # if warn_log:
#     #     for warn in warn_log:
#     #         print(warn)
#
#
#
#
#
#




# shape = 0.1
# values = np.random.lognormal(0, shape, 100000)
#
# moments_number = 10
# bounds = [0, 2]
# toleration = 0.05
# eps = 1e-6
#
# bounds = sc.stats.mstats.mquantiles(values, prob=[eps, 1 - eps])
# print(bounds)
#
#
# mean = np.mean(values)
# print(mean)0
#
# basis_function = FourierFunctions(mean)
# basis_function.set_bounds(bounds)
# basis_function.fixed_quad_n = moments_number * 2
# """
# basis_function = Monomials(mean)
# basis_function.set_bounds(bounds)
# basis_function.fixed_quad_n = moments_number + 1
# """
# #print(np.mean(np.sin(values)))
# moments = []
# for k in range(moments_number):
#     val = []
#
#     for value in values:
#         val.append(basis_function.get_moments(value, k))
#
#     moments.append(np.mean(val))
#
# print("momenty", moments)
#
#
# zacatek = t.time()
# # Run distribution
# distribution = DistributionFixedQuad(basis_function, moments_number, moments, toleration)
# #d.set_values(values)
# lagrangian_parameters = distribution.estimate_density()
#
# konec = t.time()
#
# print("celkovy cas", konec- zacatek)
# print(lagrangian_parameters)
#
#
# ## Difference between approximate and exact density
# sum = 0
# X = np.linspace(bounds[0], bounds[1], 100)
# for x in X:
#    sum += abs(distribution.density(x) - sc.stats.lognorm.pdf(x, shape))**2
# print(sum)
#
#
# ## Set approximate density values
# approximate_density = []
# X = np.linspace(bounds[0], bounds[1], 100)
# for x in X:
#     approximate_density.append(distribution.density(x))
#
#
# ## Show approximate and exact density
# plt.plot(X, approximate_density, 'r')
# plt.plot(X, [sc.stats.lognorm.pdf(x, shape) for x in X])
# plt.ylim((0, 10))
# plt.show()
#
# """
# ## Show approximate and exact density in logaritmic scale
# X = np.linspace(bounds[0], bounds[1], 100)
# plt.plot(X, -np.log(approximate_density), 'r')
# plt.plot(X, -np.log([sc.stats.lognorm.pdf(x, shape) for x in X]))
# plt.ylim((-10, 10))
# plt.show()
# """
#
#
#
# import numpy as np
# import scipy as sc
# import matplotlib.pyplot as plt
# import sys
# sys.path.insert(0, '/home/martin/Documents/MLMC/src')
# from distribution import Distribution
# from distribution_fixed_quad import DistributionFixedQuad
# from monomials import Monomials
# from fourier_functions import FourierFunctions
#
# shape = 0.1
# values = np.random.normal(0, shape, 100000)
#
# moments_number = 15
# bounds = [0, 2]
# toleration = 0.05
# eps = 1e-6
#
# bounds = sc.stats.mstats.mquantiles(values, prob=[eps, 1 - eps])
#
# mean = np.mean(values)
#
# basis_function = FourierFunctions(mean)
# basis_function.set_bounds(bounds)
# basis_function.fixed_quad_n = moments_number * 2
# """
# basis_function = Monomials(mean)
# basis_function.set_bounds(bounds)
# basis_function.fixed_quad_n = moments_number + 1
# """
#
# moments = []
# for k in range(moments_number):
#     val = []
#
#     for value in values:
#         val.append(basis_function.get_moments(value, k))
#
#     moments.append(np.mean(val))
#
#
# # Run distribution
# distribution = DistributionFixedQuad(basis_function, moments_number, moments, toleration)
# #d.set_values(values)
# lagrangian_parameters = distribution.estimate_density()
#
# print(moments)
# print(lagrangian_parameters)
#
#
# ## Difference between approximate and exact density
# sum = 0
# X = np.linspace(bounds[0], bounds[1], 100)
# for x in X:
#    sum += abs(distribution.density(x) - sc.stats.norm.pdf(x))
# print(sum)
#
#
# ## Set approximate density values
# approximate_density = []
# X = np.linspace(bounds[0], bounds[1], 100)
# for x in X:
#     approximate_density.append(distribution.density(x))
#
#
# ## Show approximate and exact density
# plt.plot(X, approximate_density, 'r')
# plt.plot(X, [sc.stats.norm.pdf(x, 0, shape) for x in X])
# plt.ylim((0, 10))
# plt.show()
#
# """
# ## Show approximate and exact density in logaritmic scale
# X = np.linspace(bounds[0], bounds[1], 100)
# plt.plot(X, -np.log(approximate_density), 'r')
# plt.plot(X, -np.log([sc.stats.norm.pdf(x) for x in X]))
# plt.ylim((-10, 10))
# plt.show()
# """

def compute_mlmc_distribution(nl, distr, nm):
    """
    Test approximation moments from first estimate and from final number of samples
    :param nl: int. Number of levels
    :param distr: Distributions as [distr obj, log (bool), simulation function]
    :return: TestMLMC instance
    """
    n_moments = nm
    repet_number = 1
    start_moments_n = nm
    all_variances = []
    all_means = []
    d = distr[0]
    for i in range(repet_number):
        mc_test = TestMLMC(nl, n_moments, d, distr[1], distr[2])
        # number of samples on each level
        mc_test.mc.set_initial_n_samples()
        mc_test.mc.refill_samples()
        mc_test.mc.wait_for_simulations()
        mc_test.mc.set_target_variance(1e-5, mc_test.moments_fn)
        mc_test.mc.refill_samples()
        mc_test.mc.wait_for_simulations()

        # Moments as tuple (means, vars)
        moments = mc_test.mc.estimate_moments(mc_test.moments_fn)
        # Variances
        variances = np.sqrt(moments[1]) * 3

        all_variances.append(variances)
        all_means.append(moments[0])

    # Exact moments from distribution
    exact_moments = mlmc._distribution.compute_exact_moments(mc_test.moments_fn, d.pdf, 1e-10)

    means = (np.mean(all_means, axis=0))
    vars = np.mean(all_variances, axis=0)
    moments_data = np.empty((len(exact_moments[0:start_moments_n]), 2))

    rnd = [np.random.normal(0, 0.01) for v in vars]
    exact_moments = exact_moments + rnd

    moments_data[:, 0] = exact_moments[0:start_moments_n]
    moments_data[:, 1] = np.zeros(len(exact_moments[0:start_moments_n]))

    moments_data[:, 0] = means[:start_moments_n]
    moments_data[:, 1] = vars[:start_moments_n]

    mc_test.moments_fn.size = start_moments_n

    distr_obj = mlmc._distribution.Distribution(mc_test.moments_fn, moments_data)
    # distr_obj.choose_parameters_from_samples()
    distr_obj.domain = mc_test.moments_fn.domain
    # result = distr_obj.estimate_density(tol=0.0001)
    result = distr_obj.estimate_density_minimize(tol=1)

    mc_test.distr_obj = distr_obj
    # density = density_from_prior_estimate(distr_obj, mc_test, exact_moments, d, moments_data)

    return mc_test


def density_from_prior_estimate(distr_obj, mc_test, exact_moments, exact_density_object, moments_data):
    """
    Estimate current density from prior one
    :param distr_obj: Distribution object
    :param mc_test: TestMLMC instance
    :param exact_moments: list, exact moments
    :param exact_density_object: Exact density object for artificial distributions
    :param moments_data: Moments data from MLMC
    :return: Density values
    """
    size = 1e5
    x = np.linspace(distr_obj.domain[0], distr_obj.domain[1], size)
    density = distr_obj.density(x)
    # exact_density = exact_density_object.pdf(x)
    # tol = 1e-5
    # last_density = density
    # last_distr_obj = distr_obj

    while True: # Now there is no termination criterion !!
        # Last multipliers and density
        multipliers = distr_obj.multipliers
        multipliers = np.append(multipliers, 0)

        # Add new moment, default zero
        moments = np.empty((len(moments_data) + 1, 2))
        moments[:, 0] = exact_moments[0:len(moments_data) + 1]
        moments[:, 1] = np.zeros(len(moments_data) + 1)
        moments_data = moments

        # Set new moments size
        mc_test.moments_fn.size = len(moments_data)

        # Compute new density
        distr_obj = mlmc._distribution.Distribution(mc_test.moments_fn, moments_data)
        distr_obj.multipliers = multipliers
        distr_obj.domain = mc_test.moments_fn.domain
        distr_obj.estimate_density_minimize(tol=1e-15)
        density = distr_obj.density(x)

        kl_div = mlmc._distribution.KL_divergence(exact_density_object.pdf, distr_obj.density,
                                                  mc_test.moments_fn.domain[0],
                                                  mc_test.moments_fn.domain[1])
        L2_norm = mlmc._distribution.L2_distance(exact_density_object.pdf, distr_obj.density,
                                                 mc_test.moments_fn.domain[0],
                                                 mc_test.moments_fn.domain[1])

        plt.plot(x, density, label="entropy density")
        plt.plot(x, exact_density_object.pdf(x), label="pdf")
        plt.legend()
        plt.show()

    return density


@pytest.mark.skip
def test_distributions():
    """
    Plot densities and histogram for chosen distributions
    :return: None
    """
    mlmc_list = []
    # List of distributions
    distributions = [
        (stats.norm(loc=1, scale=2), False, '_sample_fn')
        #(stats.lognorm(scale=np.exp(5), s=1), True, '_sample_fn'),  # worse conv of higher moments
        # (stats.lognorm(scale=np.exp(-5), s=1), True, '_sample_fn_basic'),
        #(stats.chi2(df=10), True, '_sample_fn')#,
        # (stats.weibull_min(c=20), True, '_sample_fn'),   # Exponential
        # (stats.weibull_min(c=1.5), True, '_sample_fn_basic'),  # Infinite derivative at zero
        # (stats.weibull_min(c=3), True, '_sample_fn_basic')  # Close to normal
         ]
    levels = [1]#, 2, 3, 5, 7, 9]
    n_moments = 10
    # Loop through distributions and levels
    for distr in distributions:
        for level in levels:
            mlmc_list.append(compute_mlmc_distribution(level, distr, n_moments))

    fig = plt.figure(figsize=(30, 10))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    n_moments = 5
    # One level MC samples
    mc0_samples = mlmc_list[0].mc.levels[0].sample_values[:, 0]
    mlmc_list[0].ref_domain = (np.min(mc0_samples), np.max(mc0_samples))

    # Plot densities according to TestMLMC instances data
    for test_mc in mlmc_list:
        test_mc.mc.clean_subsamples()
        test_mc.mc.update_moments(test_mc.moments_fn)
        domain, est_domain, mc_test = mlmc.estimate.compute_results(mlmc_list[0], n_moments, test_mc)
        mlmc.estimate.plot_pdf_approx(ax1, ax2, mc0_samples, mc_test, domain, est_domain)
    ax1.legend()
    ax2.legend()
    fig.savefig('compare_distributions.pdf')
    plt.show()


#test_distributions()
