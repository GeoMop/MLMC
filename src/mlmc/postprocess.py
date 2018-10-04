import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/../src/')
from mlmc.distribution import Distribution

# This file will contain postprocessing methods


def plot_pdf_approx(ax1, ax2, mc0_samples, mlmc_wrapper, domain, est_domain):
    """
    Plot density and distribution, plot contains density estimation from MLMC and histogram created from One level MC
    :param ax1: First figure subplot
    :param ax2: Second figure subplot
    :param mc0_samples: One level MC samples
    :param mlmc_wrapper: Object with mlmc instance, must contains distribution object
    :param domain: Domain from one level MC
    :param est_domain: Domain from MLMC
    :return: None
    """
    # X = np.exp(np.linspace(np.log(domain[0]), np.log(domain[1]), 1000))
    # bins = np.exp(np.linspace(np.log(domain[0]), np.log(10), 60))
    X = np.linspace(domain[0], domain[1], 1000)
    bins = np.linspace(domain[0], domain[1], len(mc0_samples)/15)

    distr_obj = mlmc_wrapper.distr_obj

    n_levels = mlmc_wrapper.mc.n_levels
    color = "C{}".format(n_levels)
    label = "l {}".format(n_levels)
    Y = distr_obj.density(X)
    ax1.plot(X, Y, c=color, label=label)

    Y = distr_obj.cdf(X)
    ax2.plot(X, Y, c=color, label=label)

    if n_levels == 1:
        ax1.hist(mc0_samples, normed=True, bins=bins, alpha=0.3, label='full MC', color=color)
        X, Y = ecdf(mc0_samples)
        ax2.plot(X, Y, 'red')

    ax1.axvline(x=domain[0], c=color)
    ax1.axvline(x=domain[1], c=color)


def compute_results(mlmc_l0, n_moments, mlmc_wrapper):
    """
    Compute density and moments domains
    :param mlmc_l0: One level Monte-Carlo method
    :param n_moments: int, Number of moments
    :param mc_wrapper: Object with mlmc instance, must contains also moments function object
    :return: domain - tuple, domain from 1LMC
             est_domain - tuple, domain estimated by mlmc instance
             mc_wrapper - with current distr_obj (object estimating distribution)
    """
    mlmc = mlmc_wrapper.mc
    moments_fn = mlmc_wrapper.moments_fn
    domain = mlmc_l0.ref_domain
    est_domain = mlmc.estimate_domain()

    t_var = 1e-5
    ref_diff_vars, _ = mlmc.estimate_diff_vars(moments_fn)
    # ref_moments, ref_vars = mc.estimate_moments(moments_fn)
    # ref_std = np.sqrt(ref_vars)
    # ref_diff_vars_max = np.max(ref_diff_vars, axis=1)
    # ref_n_samples = mc.set_target_variance(t_var, prescribe_vars=ref_diff_vars)
    # ref_n_samples = np.max(ref_n_samples, axis=1)
    # ref_cost = mc.estimate_cost(n_samples=ref_n_samples)
    # ref_total_std = np.sqrt(np.sum(ref_diff_vars / ref_n_samples[:, None]) / n_moments)
    # ref_total_std_x = np.sqrt(np.mean(ref_vars))

    est_moments, est_vars = mlmc.estimate_moments(moments_fn)

    # def describe(arr):
    #     print("arr ", arr)
    #     q1, q3 = np.percentile(arr, [25, 75])
    #     print("q1 ", q1)
    #     print("q2 ", q3)
    #     return "{:f8.2} < {:f8.2} | {:f8.2} | {:f8.2} < {:f8.2}".format(
    #         np.min(arr), q1, np.mean(arr), q3, np.max(arr))

    moments_data = np.stack((est_moments, est_vars), axis=1)
    distr_obj = Distribution(moments_fn, moments_data)
    distr_obj.domain = domain
    distr_obj.estimate_density_minimize(1)
    mlmc_wrapper.distr_obj = distr_obj

    return domain, est_domain, mlmc_wrapper

def ecdf(x):
    xs = np.sort(x)
    ys = np.arange(1, len(xs) + 1) / float(len(xs))
    return xs, ys
