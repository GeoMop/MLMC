import os
import sys
import time
import pytest

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.ticker
from scipy.interpolate import interp1d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/../src/')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mlmc.estimate
import mlmc.distribution
import mlmc.simple_distribution
import mlmc.simple_distribution_total_var
from mlmc import moments
import test.benchmark_distributions as bd
import mlmc.tool.plot as plot
from test.fixtures.mlmc_test_run import MLMCTest
import mlmc.spline_approx as spline_approx
from mlmc.moments import Legendre
from textwrap import wrap

import pandas as pd
import pickle


distr_names = {'_norm': "norm", '_lognorm': "lognorm", '_two_gaussians': "two_gaussians", "_five_fingers": "five_fingers",
               "_cauchy": "cauchy", "_discontinuous": "discontinuous"}


def plot_KL_div_inexact():
    """
    Plot KL divergence for different noise level of exact moments
    """
    dir_name = "KL_div_inexact_numpy"
    if not os.path.exists(dir_name):
        raise FileNotFoundError

    for key, name in distr_names.items():

        work_dir = os.path.join(dir_name, name)
        if os.path.exists(work_dir):
            noise_levels = np.load(os.path.join(work_dir, "noise_levels.npy"))
            n_moments = np.load(os.path.join(work_dir, "n_moments.npy"))

            kl_plot = plot.KL_divergence(iter_plot=True, log_y=True, log_x=True,
                                         title=name + "_n_mom_{}".format(n_moments), xlabel="noise std",
                                         ylabel="KL divergence",
                                         truncation_err_label="trunc. err, m: {}".format(n_moments))

            distr_plot = plot.SimpleDistribution(title="{}_inexact".format(name), cdf_plot=True, error_plot=False)

            for noise_level in noise_levels:

                kl_plot.truncation_err = np.load(os.path.join(work_dir, "truncation_err.npy"))

                _, kl_div = np.load('{}/{}_{}.npy'.format(work_dir, noise_level, "add-value"))
                _, nit, success = np.load('{}/{}_{}.npy'.format(work_dir, noise_level, "add-iteration"))
                _, diff_linalg_norm = np.load('{}/{}_{}.npy'.format(work_dir, noise_level, "add-moments"))

                kl_plot.add_value((noise_level, kl_div))
                kl_plot.add_iteration(x=noise_level, n_iter=nit, failed=success)
                kl_plot.add_moments_l2_norm((noise_level, diff_linalg_norm))

                domain = np.load('{}/{}_{}.npy'.format(work_dir, noise_level, "domain"))
                X = np.load('{}/{}_{}.npy'.format(work_dir, noise_level, "X"))

                Y_pdf = np.load('{}/{}_{}.npy'.format(work_dir, noise_level, "Y_pdf"))
                Y_cdf = np.load('{}/{}_{}.npy'.format(work_dir, noise_level, "Y_cdf"))

                print("Y pdf ", Y_pdf[10])

                distr_plot.add_distribution(X, Y_pdf, Y_cdf, domain, label="{}_{}".format(name, noise_level))

            Y_exact_pdf = np.load('{}/{}_{}.npy'.format(work_dir, noise_level, "Y_pdf_exact"))
            Y_exact_cdf = np.load('{}/{}_{}.npy'.format(work_dir, noise_level, "Y_cdf_exact"))
            distr_plot._add_exact_distr(X, Y_exact_pdf, Y_exact_cdf)

            kl_plot.show(None)
            distr_plot.show(None)

def plot_KL_div_reg_inexact():
    """
    Plot KL divergence for different noise level of exact moments
    """
    dir_name = "KL_div_inexact_reg_numpy"
    if not os.path.exists(dir_name):
        raise FileNotFoundError

    for key, name in distr_names.items():

        work_dir = os.path.join(dir_name, name)
        if os.path.exists(work_dir):
            noise_levels = np.load(os.path.join(work_dir, "noise_levels.npy"))
            n_moments = np.load(os.path.join(work_dir, "n_moments.npy"))

            kl_plot = plot.KL_divergence(iter_plot=True, log_y=True, log_x=True,
                                         title=name + "_n_mom_{}".format(n_moments), xlabel="noise std",
                                         ylabel="KL divergence",
                                         truncation_err_label="trunc. err, m: {}".format(n_moments))

            distr_plot = plot.SimpleDistribution(title="{}_inexact".format(name), cdf_plot=True, error_plot=False)

            for noise_level in noise_levels:

                kl_plot.truncation_err = np.load(os.path.join(work_dir, "truncation_err.npy"))

                _, kl_div = np.load('{}/{}_{}.npy'.format(work_dir, noise_level, "add-value"))
                _, nit, success = np.load('{}/{}_{}.npy'.format(work_dir, noise_level, "add-iteration"))
                _, diff_linalg_norm = np.load('{}/{}_{}.npy'.format(work_dir, noise_level, "add-moments"))

                kl_plot.add_value((noise_level, kl_div))
                kl_plot.add_iteration(x=noise_level, n_iter=nit, failed=success)
                kl_plot.add_moments_l2_norm((noise_level, diff_linalg_norm))

                domain = np.load('{}/{}_{}.npy'.format(work_dir, noise_level, "domain"))
                X = np.load('{}/{}_{}.npy'.format(work_dir, noise_level, "X"))

                Y_pdf = np.load('{}/{}_{}.npy'.format(work_dir, noise_level, "Y_pdf"))
                Y_cdf = np.load('{}/{}_{}.npy'.format(work_dir, noise_level, "Y_cdf"))

                print("Y pdf ", Y_pdf[10])

                distr_plot.add_distribution(X, Y_pdf, Y_cdf, domain, label="{}_{}".format(name, noise_level))

                reg_params = np.load('{}/{}_{}.npy'.format(work_dir, noise_level, "reg-params"))
                min_results = np.load('{}/{}_{}.npy'.format(work_dir, noise_level, "min-results"))
                plot_reg_params(reg_params, min_results)

            Y_exact_pdf = np.load('{}/{}_{}.npy'.format(work_dir, noise_level, "Y_pdf_exact"))
            Y_exact_cdf = np.load('{}/{}_{}.npy'.format(work_dir, noise_level, "Y_cdf_exact"))
            distr_plot._add_exact_distr(X, Y_exact_pdf, Y_exact_cdf)

            kl_plot.show(None)
            distr_plot.show(None)


def plot_reg_params(reg_params, min_results):
    zipped = zip(reg_params, min_results)

    for reg_param, min_result in zip(reg_params, min_results):
        print("reg_param: {}, min_result: {}".format(reg_param, min_result))

    sorted_zip = sorted(zipped, key=lambda x: x[1])

    best_params = []
    # best_params.append(0)
    min_best = None
    for s_tuple in sorted_zip:
        if min_best is None:
            min_best = s_tuple
        print(s_tuple)
        if len(best_params) < 10:
            best_params.append(s_tuple[0])

    fig, ax = plt.subplots()
    ax.plot(reg_params, min_results)
    ax.plot(min_best[0], min_best[1], 'x', color='red')
    ax.set_ylabel("MSE")
    ax.set_xlabel(r"$\log(\alpha)$")
    ax.set_xscale('log')
    ax.legend(loc='best')
    logfmt = matplotlib.ticker.LogFormatterExponent(base=10.0, labelOnlyBase=True)
    ax.xaxis.set_major_formatter(logfmt)

    plt.show()


def plot_find_reg_param():
    dir_name = "find_reg_param"
    noise_level = "0.01"
    if not os.path.exists(dir_name):
        raise FileNotFoundError
    for key, name in distr_names.items():
        work_dir = os.path.join(dir_name, name)
        if os.path.exists(work_dir):
            reg_params = np.load('{}/{}_{}.npy'.format(work_dir, noise_level, "reg-params"))
            min_results = np.load('{}/{}_{}.npy'.format(work_dir, noise_level, "min-results"))
            plot_reg_params(reg_params, min_results)


if __name__ == "__main__":
    #plot_KL_div_inexact()
    #plot_KL_div_reg_inexact()
    plot_find_reg_param()
