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

import numpy as np
import scipy.stats as stats
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/../src/')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mlmc.postprocess
import mlmc.distribution
from mlmc.distribution import Distribution
import mlmc.moments
#import test_mlmc
import time


import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.integrate as integrate


def check_distr_approx(moment_class, distribution, distr_args):
    """
    :param moment_class:
    :param distribution:
    :return:
    """
    fn_name = moment_class.__name__
    distr_name = distribution.__class__.__name__
    print("Testing moments: {} for distribution: {}".format(fn_name, distr_name))

    # Approximation for exact moments

    density = lambda x: distribution.pdf(x, **distr_args)
    domain = distribution.ppf([0.01, 0.99], **distr_args)
    print("domain: ", domain)

    n_moments = 10
    tol = 1e-4

    mean = distribution.mean(**distr_args)
    variance = distribution.var(**distr_args)
    moments_fn = moment_class(n_moments, domain)
    exact_moments = mlmc.distribution.compute_exact_moments(moments_fn, density, tol)
    moments_data = np.empty((n_moments, 2))
    moments_data[:, 0] = exact_moments
    moments_data[:, 1] = tol
    is_positive = (domain[0] > 0.0)
    distr_obj = mlmc.distribution.Distribution(moments_fn, moments_data, is_positive)
    distr_obj.choose_parameters_from_moments(mean, variance)

    # Just for test plotting
    distr_obj.fn_name = fn_name
    t1 = time.clock()
    result = distr_obj.estimate_density(tol)
    t2 = time.clock()
    t = t2 - t1
    nit = getattr(result, 'nit', result.njev)
    kl_div = mlmc.distribution.KL_divergence(distr_obj.density, density, domain[0], domain[1])
    l2_dist = mlmc.distribution.L2_distance(distr_obj.density, density, domain[0], domain[1])
    print("Conv: {} Nit: {} Time: {} KL: {} L2: {}".format(
        result.success, nit, t, kl_div, l2_dist
    ))
    return distr_obj


def plot_approximations(dist, args, approx_objs):
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    domain = approx_objs[0].domain
    X = np.linspace(domain[0], domain[1], 1000)
    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(1, 3, 1)
    ax_log = fig.add_subplot(1, 3, 2)
    ax_diff = fig.add_subplot(1, 3, 3)
    cmap = plt.cm.jet
    norm = mpl.colors.Normalize(vmin=0, vmax=2)

    Y0 = dist.pdf(X, **args)
    ax.plot(X, Y0, c='red')
    for i, approx in enumerate(approx_objs):
        Y = approx.density(X)
        ax.plot(X, Y, c=cmap(norm(i)), label=approx.fn_name)
        ax_log.plot(X, -np.log(Y), c=cmap(norm(i)), label=approx.fn_name)
        ax_diff.plot(X, Y - Y0, c=cmap(norm(i)), label=approx.fn_name)
    ax.set_ylim(0, 2)
    # ax_diff.set_ylim(0, 1)
    fig.legend()
    plt.show()


def test_distribution():
    moment_fns = [mlmc.moments.Monomial, mlmc.moments.Fourier, mlmc.moments.Legendre]
    # moment_fns = [mlmc.moments.monomial_moments]
    distrs = [
        (stats.norm, dict(loc=1.0, scale=2.0)),
        (stats.lognorm, dict(s=0.5, scale=np.exp(2.0)))
    ]

    for distr, args in distrs:
        approx_objs = []
        for fn in moment_fns:
            approx_objs.append(check_distr_approx(fn, distr, args))
        plot_approximations(distr, args, approx_objs)


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

@pytest.mark.skip
def test_mlmc_distribution(nl, distr):
    """
    Test approximation moments from first estimate and from final number of samples
    :param nl: int. Number of levels
    :param distr: Distributions as [distr obj, log (bool), simulation function]
    :return: TestMLMC instance
    """
    n_moments = 20
    repet_number = 1
    start_moments_n = 5
    all_variances = []
    all_means = []
    d = distr[0]
    for i in range(repet_number):
        mc_test = test_mlmc.TestMLMC(nl, n_moments, d, distr[1], distr[2])
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
    result = distr_obj.estimate_density_minimize(tol=1e-15)

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
    levels = [1, 5]#, 2, 3, 5, 7, 9]
    # Loop through distributions and levels
    for distr in distributions:
        for level in levels:
            mlmc_list.append(test_mlmc_distribution(level, distr))

    fig = plt.figure(figsize=(30, 10))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    n_moments = 5
    # One level MC samples
    mc0_samples = mlmc_list[0].mc.levels[0].sample_values[:, 0]
    mlmc_list[0].ref_domain = (np.min(mc0_samples), np.max(mc0_samples))

    # Plot densities according to TestMLMC instances data
    for test_mc in mlmc_list:
        test_mc.mc.update_moments(test_mc.moments_fn)
        domain, est_domain, mc_test = mlmc.postprocess.compute_results(mlmc_list[0], n_moments, test_mc)
        mlmc.postprocess.plot_pdf_approx(ax1, ax2, mc0_samples, mc_test, domain, est_domain)
    ax1.legend()
    ax2.legend()
    fig.savefig('compare_distributions.pdf')
    plt.show()


