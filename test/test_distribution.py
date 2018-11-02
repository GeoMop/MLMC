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

import mlmc.postprocess
import mlmc.distribution
from mlmc.distribution import Distribution
from mlmc import moments
from test.fixtures.mlmc_test_run import TestMLMC


class DistrPlot:
    def __init__(self, distr, title):
        self._exact_distr = distr
        self.plot_matrix = []
        self.fig, self.axes = plt.subplots(1, 3, figsize=(15,5))
        self.fig.suptitle(title)

    def plot_borders(self, ax, domain):
        l1 = ax.axvline(x=domain[0], ymin=0, ymax=0.1)
        l2 = ax.axvline(x=domain[1], ymin=0, ymax=0.1)
        return [l1, l2]

    def plot_approximation(self, approx_obj, label):
        plots = []

        domain = approx_obj.domain
        d_size = domain[1] - domain[0]
        slack = 0 #0.05
        extended_domain = (domain[0] - slack*d_size, domain[1] + slack*d_size)
        X = np.linspace(extended_domain[0], extended_domain[1], 1000)
        Y = approx_obj.density(X)
        Y0 = self._exact_distr.pdf(X)

        ax = self.axes[0]
        ax.set_title("PDF")
        np.set_printoptions(precision=2)
        lab = str(np.array(domain))
        line, = ax.plot(X, Y, label=label)
        plots.append(line)
        ax.plot(X, Y0, c='black', label="exact PDF")
        plots += self.plot_borders(ax, domain)


        ax = self.axes[1]
        ax.set_title("log(PDF)")
        line, = ax.plot(X, -np.log(Y))
        plots.append(line)
        ax.plot(X, -np.log(Y0), c='black', label="exact PDF")
        plots += self.plot_borders(ax, domain)

        ax = self.axes[2]
        ax.set_title("PDF diff")
        line, = ax.plot(X, Y - Y0)
        plots.append(line)
        plots += self.plot_borders(ax, domain)

        self.plot_matrix.append(plots)

    def show(self):
        for i, plots in enumerate(self.plot_matrix):
            col = plt.cm.jet(plt.Normalize(0, len(self.plot_matrix))(i))
            for line in plots:
                line.set_color(col)
        self.fig.legend()
        plt.show()

    def clean(self):
        plt.close()

def profile(fun, skip=False):
    import statprof
    if skip:
        return fun()
    statprof.start()
    try:
        result = fun()
    finally:
        statprof.stop()
    statprof.display()
    return result


def domain_for_quantile(distr, quantile):
    if quantile == 0:
        # Determine domain by MC sampling.
        X = distr.rvs(size=1000)
        err = stats.norm.rvs(size=1000)
        X = X * (1 + 0.1 * err)
        domain = (np.min(X), np.max(X))
    else:
        domain = distr.ppf([quantile, 1 - quantile])

    # Try to detect PDF decay on domain boundaries
    eps = 1e-10
    force_decay = [False, False]
    for side in [0, 1]:
        diff = (distr.pdf(domain[side]) - distr.pdf(domain[side] - eps)) / eps
        if side:
            diff = -diff
        if diff > 0:
            force_decay[side] = True
    return domain, force_decay

#@pytest.mark.skip
@pytest.mark.parametrize("moment_fn, max_n_moments", [
    (moments.Monomial, 10),
    #(moments.Fourier, 61),
    (moments.Legendre, 61)])
@pytest.mark.parametrize("distribution",[
        (stats.norm(loc=1, scale=2), False),
        (stats.norm(loc=1, scale=10), False),
        (stats.lognorm(scale=np.exp(1), s=1), False),    # Quite hard but peak is not so small comparet to the tail.
        #(stats.lognorm(scale=np.exp(-3), s=2), False),  # Extremely difficult to fit due to very narrow peak and long tail.
        (stats.lognorm(scale=np.exp(-3), s=2), True),    # Still difficult for Lagrange with many moments.
        (stats.chi2(df=10), False), # Monomial: s1=nan, Fourier: s1= -1.6, Legendre: s1=nan
        (stats.chi2(df=5), True), # Monomial: s1=-10, Fourier: s1=-1.6, Legendre: OK
        (stats.weibull_min(c=0.5), False),  # Exponential # Monomial stuck, Fourier stuck
        (stats.weibull_min(c=1), False),  # Exponential
        (stats.weibull_min(c=2), False),  # Rayleigh distribution
        (stats.weibull_min(c=5, scale=4), False),   # close to normal
        (stats.weibull_min(c=1.5), True),  # Infinite derivative at zero
    ])
def test_pdf_approx_exact_moments(moment_fn, max_n_moments, distribution):
    """
    Test reconstruction of the density function from exact moments.
    - various distributions
    - various moments functions
    - test convergency with increasing number of moments
    :return:
    """
    np.random.seed(1234)
    distr, log_flag = distribution
    fn_name = moment_fn.__name__
    distr_name = distr.dist.name
    density = lambda x: distr.pdf(x)
    print("Testing moments: {} for distribution: {}".format(fn_name, distr_name))

    tol_exact_moments = 1e-6
    tol_density_approx = tol_exact_moments * 8
    quantiles = np.array([0.0001, 0])
    quantiles = np.array([0.0001])


    moment_sizes = np.round(np.exp(np.linspace(np.log(3), np.log(max_n_moments), 5))).astype(int)
    moment_sizes = [61]
    kl_collected = np.empty( (len(quantiles), len(moment_sizes)) )
    l2_collected = np.empty_like(kl_collected)

    n_failed = []
    warn_log = []
    distr_plot = DistrPlot(distr, distr_name + " " + fn_name)
    for i_q, domain_quantile in enumerate(quantiles):
        domain, force_decay = domain_for_quantile(distr, domain_quantile)

        # Approximation for exact moments
        moments_fn = moment_fn(moment_sizes[-1], domain, log=log_flag, safe_eval=False )
        exact_moments = mlmc.distribution.compute_exact_moments(moments_fn, density, tol=tol_exact_moments)

        n_failed.append(0)
        cumtime = 0
        tot_nit = 0
        for i_m, n_moments in enumerate(moment_sizes):
            moments_fn = moment_fn(n_moments, domain, log=log_flag, safe_eval=False )
            #print(i_m, n_moments, domain, force_decay)
            moments_data = np.empty((n_moments, 2))
            moments_data[:, 0] = exact_moments[:n_moments]
            moments_data[:, 1] = 1.0
            is_positive = log_flag
            distr_obj = mlmc.distribution.Distribution(moments_fn, moments_data,
                                                       domain=domain, force_decay=force_decay)
            t0 = time.time()
            # result = distr_obj.estimate_density(tol_exact_moments)
            result = distr_obj.estimate_density_minimize(tol_density_approx)
            #result = profile(lambda : distr_obj.estimate_density_minimize(tol_exact_moments))
            t1 = time.time()
            cumtime += (t1 - t0)
            nit = getattr(result, 'nit', result.njev)
            tot_nit += nit
            fn_norm = result.fun_norm
            if result.success:

                kl_div = mlmc.distribution.KL_divergence(density, distr_obj.density, domain[0], domain[1])
                l2_dist = mlmc.distribution.L2_distance(distr_obj.density, density, domain[0], domain[1])
                kl_collected[i_q, i_m] = kl_div
                l2_collected[i_q, i_m] = l2_dist
                print("q: {}, m: {} :: nit: {} fn: {} ; kl: {} l2: {}".format(
                    domain_quantile, n_moments, nit, fn_norm, kl_div, l2_dist))
                if i_m + 1 == len(moment_sizes):
                    # plot for last case
                    distr_plot.plot_approximation(distr_obj, label=str(domain))
            else:
                n_failed[-1]+=1
                #print("q: {}, m: {} :: nit: {} fn:{} ; msg: {}".format(
                #    domain_quantile, n_moments, nit, fn_norm, result.message))

                kl_collected[i_q, i_m] = np.nan
                l2_collected[i_q, i_m] = np.nan

        # Check convergence
        s1, s0 = np.polyfit(np.log(moment_sizes), np.log(kl_collected[i_q,:]), 1)
        max_err = np.max(kl_collected[i_q,:])
        min_err = np.min(kl_collected[i_q,:])
        if domain_quantile > 0.01:
            continue
        if not (n_failed[-1] == 0 and (max_err < tol_density_approx * 8 or s1 < -1)):
            warn_log.append((i_q, n_failed[-1],  s1, s0, max_err))
            fail = "FF"
        else:
            fail = ' q'
        print(fail + ": ({:5.3g}, {:5.3g});  failed: {} cumit: {} tavg: {:5.3g};  s1: {:5.3g} s0: {:5.3g} kl: ({:5.3g}, {:5.3g})".format(
            domain[0], domain[1], n_failed[-1], tot_nit, cumtime/tot_nit, s1, s0, min_err, max_err))

    #distr_plot.show()
    distr_plot.clean()



    def plot_convergence():
        for iq, q in enumerate(quantiles):
            col = plt.cm.tab10(plt.Normalize(0,10)(iq))
            plt.plot(moment_sizes, kl_collected[iq,:], ls='solid', c=col, label="kl_q="+str(q), marker='o')
            plt.plot(moment_sizes, l2_collected[iq,:], ls='dashed', c=col, label="l2_q=" + str(q), marker='d')
        plt.yscale('log')
        plt.xscale('log')
        plt.legend()
        plt.show()

    #plot_convergence()

    if warn_log:
        for warn in warn_log:
            print(warn)



#@pytest.mark.skip
@pytest.mark.parametrize("distribution",[

        (stats.norm(loc=1, scale=10), False),
        (stats.lognorm(scale=np.exp(1), s=1), False),    # # No D_KL convergence under 1e-3.
        #(stats.lognorm(scale=np.exp(-3), s=2), False),  # Extremely difficult to fit due to very narrow peak and long tail.
        (stats.lognorm(scale=np.exp(-3), s=2), True),    # Still difficult for Lagrange with many moments.
        (stats.chi2(df=5), True), # Monomial: s1=-10, Fourier: s1=-1.6, Legendre: OK
        # (stats.weibull_min(c=0.5), False),  # No D_KL convergence under 1e-1.
        (stats.weibull_min(c=1), False),  # Exponential
        (stats.weibull_min(c=2), False),  # D_KL steady under 1e-6.
        (stats.weibull_min(c=1.5), True),  # Infinite derivative at zero
    ])
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
    n_moments = 5
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
        domain, est_domain, mc_test = mlmc.postprocess.compute_results(mlmc_list[0], n_moments, test_mc)
        mlmc.postprocess.plot_pdf_approx(ax1, ax2, mc0_samples, mc_test, domain, est_domain)
    ax1.legend()
    ax2.legend()
    fig.savefig('compare_distributions.pdf')
    plt.show()


test_distributions()
