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
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/../src/')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mlmc.estimate
import mlmc.distribution
from mlmc import moments
from test.fixtures.mlmc_test_run import TestMLMC


class CutDistribution:
    """
    Renormalization of PDF, CDF for exact distribution
    restricted to given finite domain.
    """

    def __init__(self, distr, quantile):
        """

        :param distr: scipy.stat distribution object.
        :param quantile: float, define lower bound for approx. distr. domain.
        """
        self.distr = distr
        self.quantile = quantile
        self.domain, self.force_decay = self.domain_for_quantile(distr, quantile)
        p0, p1 = distr.cdf(self.domain)
        self.shift = p0
        self.scale = 1 / (p1 - p0)

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

class ConvResult:
    """
    Results of a convergency calculation.
    """
    def __init__(self):
        self.size = 0
        # Moment sizes used
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
        return "#{} it:{} err:{} kl:{:6.2g} l2:{:6.2g}".format(self.size, self.nit, self.residual_norm,
                                                               self.kl, self.l2)

class DistributionDomainCase:
    """
    Class to perform tests for fully specified exact distribution (distr. + domain) and moment functions.
    Exact moments and other objects are created once. Then various tests are performed.
    """
    tol_exact_moments = 1e-6
    tol_density_approx = tol_exact_moments * 8

    def __init__(self, moments, distribution, quantile):
        # Setup distribution.
        i_distr, distribution = distribution
        distr, log_flag = distribution
        self.quantile = quantile
        self.distr_name = "{:02}_{}".format(i_distr, distr.dist.name)
        self.cut_distr = CutDistribution(distr, quantile)
        # domain_str = "({:5.2}, {:5.2})".format(*self.domain)

        # Setup moments.
        moment_class, min_n_moments, max_n_moments = moments
        self.moment_sizes = np.round(np.exp(np.linspace(np.log(min_n_moments), np.log(max_n_moments), 8))).astype(int)
        #self.moment_sizes = [3,4,5,6,7]

        # TODO: modify for CovDistribution
        self.moments_fn = moment_class(self.moment_sizes[-1], self.domain, log=log_flag, safe_eval=False)
        self.fn_name = str(moment_class.__name__)
        self.exact_moments = mlmc.distribution.compute_exact_moments(self.moments_fn,
                                self.pdf, tol=self.tol_exact_moments)

        self.title = "distr: {} quantile: {} moment_fn: {}".format(self.distr_name, quantile, self.fn_name)

    @property
    def domain(self):
        return self.cut_distr.domain

    def pdf(self, x):
        return self.cut_distr.pdf(x)


    def check_convergence(self, results):
        # summary values
        sizes = np.log([r.size for r in results])
        kl = np.log([r.kl for r in results])
        sizes = sizes[~np.isnan(kl)]
        kl = kl[~np.isnan(kl)]
        n_failed = sum([not r.success for r in results])
        total_its = sum([r.nit for r in results])
        total_time = sum([r.time for r in results])
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


    def make_approx(self, distr_class, moments_data):
        result = ConvResult()

        distr_obj = distr_class(self.moments_fn, moments_data,
                                domain=self.domain, force_decay=self.cut_distr.force_decay)
        t0 = time.time()
        min_result = distr_obj.estimate_density_minimize(self.tol_density_approx)
        # result = profile(lambda : distr_obj.estimate_density_minimize(tol_exact_moments))
        t1 = time.time()
        result.size = moments_data.shape[0]
        result.time = t1 - t0
        result.residual_norm = min_result.fun_norm
        result.success = min_result.success

        if result.success:
            result.nit = min_result.nit
            a, b = self.domain
            result.kl = mlmc.distribution.KL_divergence(self.pdf, distr_obj.density, a, b)
            result.l2 = mlmc.distribution.L2_distance(self.pdf, distr_obj.density, a, b)
        print(result)
        return result, distr_obj



    def exact_conv(self):
        results = []
        distr_plot = mlmc.estimate.DistributionPlot(exact_distr=self.cut_distr, title=self.title)
        for i_m, n_moments in enumerate(self.moment_sizes):
            # moments_fn = moment_fn(n_moments, domain, log=log_flag, safe_eval=False )
            # print(i_m, n_moments, domain, force_decay)
            moments_data = np.empty((n_moments, 2))
            moments_data[:, 0] = self.exact_moments[:n_moments]
            moments_data[:, 1] = 1.0

            # TODO: modify for CovDistribution
            result, distr_obj = self.make_approx(mlmc.distribution.Distribution, moments_data)
            distr_plot.add_distribution(distr_obj, label="#{}".format(n_moments))
            results.append(result)

        self.check_convergence(results)
        distr_plot.show(save=self.title + "_exact_conv.pdf")
        distr_plot.reset()
        return results

    def inexact_conv(self):
        return None

    def covariance_exact_conv(self):
        return None

    def covariance_inexact_conv(self):
        return None

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
        sizes = [r.size for r in results]
        kl = [r.kl for r in results]
        l2 = [r.l2 for r in results]
        col = plt.cm.tab10(plt.Normalize(0,10)(iq))
        ax.plot(sizes, kl, ls='solid', c=col, label="kl_q="+str(q), marker='o')
        ax.plot(sizes, l2, ls='dashed', c=col, label="l2_q=" + str(q), marker='d')
    ax.set_yscale('log')
    ax.set_xscale('log')
    fig.legend()
    fig.suptitle(title)
    fname = title + ".pdf"
    fig.savefig(fname)


distribution_list = [
        # distibution, log_flag
        (stats.norm(loc=1, scale=2), False),
        # (stats.norm(loc=1, scale=10), False),
        # (stats.lognorm(scale=np.exp(1), s=1), False),    # Quite hard but peak is not so small comparet to the tail.
        # #(stats.lognorm(scale=np.exp(-3), s=2), False),  # Extremely difficult to fit due to very narrow peak and long tail.
        # (stats.lognorm(scale=np.exp(-3), s=2), True),    # Still difficult for Lagrange with many moments.
        # (stats.chi2(df=10), False), # Monomial: s1=nan, Fourier: s1= -1.6, Legendre: s1=nan
        # (stats.chi2(df=5), True), # Monomial: s1=-10, Fourier: s1=-1.6, Legendre: OK
        # (stats.weibull_min(c=0.5), False),  # Exponential # Monomial stuck, Fourier stuck
        # (stats.weibull_min(c=1), False),  # Exponential
        # (stats.weibull_min(c=2), False),  # Rayleigh distribution
        # (stats.weibull_min(c=5, scale=4), False),   # close to normal
        # (stats.weibull_min(c=1.5), True),  # Infinite derivative at zero
    ]



#@pytest.mark.skip
@pytest.mark.parametrize("moments", [
    #(moments.Monomial, 3, 10),
    #(moments.Fourier, 5, 61),
    (moments.Legendre, 7, 61),
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
    quantiles = np.array([0.001, 0.01])
    conv = {}
    # Dict of result matricies (n_quantiles, n_moments) for every performed kind of test.
    for i_q, quantile in enumerate(quantiles):
        np.random.seed(1234)
        case = DistributionDomainCase(moments, distribution, quantile)
        tests = [case.exact_conv, case.inexact_conv, case.covariance_exact_conv, case.covariance_inexact_conv]
        for test_fn in tests:
            name = test_fn.__name__
            print("test_fn: ", name)
            values = conv.setdefault(name, (case.title, []))
            values[1].append(test_fn())

    for key, values in conv.items():
        title, results = values
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




distr_sublist = [
        (stats.norm(loc=1, scale=10), False),
        (stats.lognorm(scale=np.exp(1), s=1), False),    # # No D_KL convergence under 1e-3.
        #(stats.lognorm(scale=np.exp(-3), s=2), False),  # Extremely difficult to fit due to very narrow peak and long tail.
        (stats.lognorm(scale=np.exp(-3), s=2), True),    # Still difficult for Lagrange with many moments.
        (stats.chi2(df=5), True), # Monomial: s1=-10, Fourier: s1=-1.6, Legendre: OK
        # (stats.weibull_min(c=0.5), False),  # No D_KL convergence under 1e-1.
        (stats.weibull_min(c=1), False),  # Exponential
        (stats.weibull_min(c=2), False),  # D_KL steady under 1e-6.
        (stats.weibull_min(c=1.5), True),  # Infinite derivative at zero
    ]

@pytest.mark.skip
@pytest.mark.parametrize("distribution", distribution_list)
def test_pdf_approx_iexact_moments(distribution):
    np.random.seed(67)

    distr, log_flag = distribution
    distr_name = distr.dist.name
    print("Inexact PDF approx for distribution: {}".format(distr_name))

    domain_quantile = 0
    domain, force_decay = domain_for_quantile(distr, domain_quantile)
    moments_fn = moments.Legendre(21, domain, log=log_flag, safe_eval=False)



    # Approximation for exact moments
    tol_exact_moments = 1e-6
    density = lambda x: distr.pdf(x)
    exact_moments = mlmc.distribution.compute_exact_moments(moments_fn, density, tol=tol_exact_moments)
    # Estimate variances
    X = distr.rvs(size=1000)
    est_moment_vars = np.var(moments_fn(X), axis=0, ddof=1)
    #print(est_moment_vars)

    # fail: 1, 6
    # fine: 4
    distr_plot = DistrPlot(distr, distr_name+", for inexact moments")
    moment_errors = np.exp(np.linspace(np.log(0.01), np.log(0.000001), 20))
    kl_collected = np.empty( len(moment_errors) )
    l2_collected = np.empty_like(kl_collected)
    warn_log = []
    n_failed = 0
    cum_time = 0
    cum_it = 0
    for i_m, err in enumerate(moment_errors):
        perturb = 0* stats.norm.rvs(size = moments_fn.size) * err * np.sqrt(est_moment_vars)
        moments_data = np.empty((moments_fn.size, 2))
        moments_data[:, 0] = exact_moments + perturb
        moments_data[:, 1] = est_moment_vars
        distr_obj = mlmc.distribution.Distribution(moments_fn, moments_data,
                                                   domain=domain, force_decay=force_decay)
        t0 = time.time()
        # result = distr_obj.estimate_density(tol_exact_moments)
        result = distr_obj.estimate_density_minimize(err*4)
        #result = profile(lambda : distr_obj.estimate_density_minimize(tol_exact_moments))
        t1 = time.time()
        cum_time += t1 - t0
        nit = getattr(result, 'nit', result.njev)
        cum_it += nit
        fn_norm = result.fun_norm
        if result.success:
            kl_div = mlmc.distribution.KL_divergence(density, distr_obj.density, domain[0], domain[1])
            l2_dist = mlmc.distribution.L2_distance(distr_obj.density, density, domain[0], domain[1])
            kl_collected[i_m] = kl_div
            l2_collected[i_m] = l2_dist
            #print("q: {}, err: {:7.3g} :: nit: {} fn: {} ; kl: {} l2: {}".format(
            #    domain_quantile, err, nit, fn_norm, kl_div, l2_dist))
            distr_plot.plot_approximation(distr_obj, str(err))
        else:
            n_failed+=1
            print("q: {}, err {} :: nit: {} fn:{} ; msg: {}".format(
                domain_quantile, err, nit, fn_norm, result.message))

            kl_collected[i_m] = np.nan
            l2_collected[i_m] = np.nan

    # Check convergence
    #print(kl_collected)
    s1, s0 = np.polyfit(np.log(moment_errors), np.log(kl_collected), 1)
    max_err = np.max(kl_collected)
    min_err = np.min(kl_collected)
    if not (n_failed == 0 and (max_err < 1e-6 or s1 > 0)):
        warn_log.append((domain_quantile, n_failed,  s1, s0, max_err))
        fail = 'NQ'
    else:
        fail = ' q'
    fail = ' q'
    print(fail + ": ({:5.3g}, {:5.3g});  failed: {} tavg: {:5.3g};  s1: {:5.3g} s0: {:5.3g} kl: ({:5.3g}, {:5.3g})".format(
        domain[0], domain[1], n_failed, cum_time/cum_it, s1, s0, min_err, max_err))

    #distr_plot.show()
    distr_plot.clean()



    def plot_convergence():
        plt.plot(moment_errors, kl_collected, ls='solid', c='red')
        plt.plot(moment_errors, l2_collected, ls='dashed', c='blue')
        plt.yscale('log')
        plt.xscale('log')
        plt.legend()
        plt.show()

    #plot_convergence()

    # if warn_log:
    #     for warn in warn_log:
    #         print(warn)










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
    exact_moments = mlmc.distribution.compute_exact_moments(mc_test.moments_fn, d.pdf, 1e-10)

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

    distr_obj = mlmc.distribution.Distribution(mc_test.moments_fn, moments_data)
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
        distr_obj = mlmc.distribution.Distribution(mc_test.moments_fn, moments_data)
        distr_obj.multipliers = multipliers
        distr_obj.domain = mc_test.moments_fn.domain
        distr_obj.estimate_density_minimize(tol=1e-15)
        density = distr_obj.density(x)

        kl_div = mlmc.distribution.KL_divergence(exact_density_object.pdf, distr_obj.density,
                                                 mc_test.moments_fn.domain[0],
                                                 mc_test.moments_fn.domain[1])
        L2_norm = mlmc.distribution.L2_distance(exact_density_object.pdf, distr_obj.density,
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
