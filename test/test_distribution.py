"""


Implementation TODO:
- weight inexact moments by their variance; change in nonlinearity W*F(x) = 0
- support for possitive distributions
- compute approximation of more moments then used for approximation, turn problem into
  overdetermined non-linear least square problem

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

import mlmc.distribution
import mlmc.moments
import time





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

    density = lambda x : distribution.pdf(x, **distr_args)
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
    fig = plt.figure(figsize=(15,5))
    ax = fig.add_subplot(1,3,1)
    ax_log = fig.add_subplot(1,3,2)
    ax_diff = fig.add_subplot(1, 3, 3)
    cmap = plt.cm.jet
    norm = mpl.colors.Normalize(vmin=0, vmax=2)

    Y0 = dist.pdf(X, **args)
    ax.plot(X, Y0, c='red')
    for i, approx in enumerate(approx_objs):
        Y = approx.density(X)
        ax.plot(X, Y, c = cmap(norm(i)), label=approx.fn_name)
        ax_log.plot(X, -np.log(Y), c = cmap(norm(i)), label=approx.fn_name)
        ax_diff.plot(X, Y-Y0, c = cmap(norm(i)), label=approx.fn_name)
    ax.set_ylim(0, 2)
    #ax_diff.set_ylim(0, 1)
    fig.legend()
    plt.show()




def test_distribution():
    moment_fns = [mlmc.moments.Monomial, mlmc.moments.Fourier, mlmc.moments.Legendre]
    #moment_fns = [mlmc.moments.monomial_moments]
    distrs = [
        (stats.norm, dict(loc=1.0, scale=2.0)),
        (stats.lognorm, dict(s=0.5, scale=np.exp(2.0)))
        ]

    for distr, args in distrs:
        approx_objs = []
        for fn in moment_fns:
            approx_objs.append(check_distr_approx(fn, distr, args))
        #plot_approximations(distr, args, approx_objs)


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
#
#
#
#
#
#
#
#
