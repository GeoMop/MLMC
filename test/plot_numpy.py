import os
import sys
import time
import pytest

import numpy as np
import scipy.stats as stats
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


def plot_KL_div_exact():
    """
    Plot KL divergence for different noise level of exact moments
    """
    distr_names = {'normální rozdělení': "norm",
                   'lognormální rozdělení': "lognorm",
                   'rozdělení two_gaussians': "two_gaussians",
                   "rozdělení five_fingers": "five_fingers",
                    "Cauchy rozdělení": "cauchy",
                   "nespojité rozdělení": "discontinuous"}

    dir_name = "/home/martin/Documents/MLMC_exact_plot/test/KL_div_exact_numpy_2"
    #dir_name = "/home/martin/Documents/MLMC_exact_plot/test/KL_div_exact_numpy_4"

    if not os.path.exists(dir_name):
        raise FileNotFoundError

    kl_div_mom_err_plot = plot.KL_div_mom_err(title="KL_div_R_exact", x_label="R",
                                              y_label=r'$D(\rho \Vert \rho_R)$', x_log=True)

    all_constants = []
    for distr_title, name in distr_names.items():

        work_dir = os.path.join(dir_name, name)
        if os.path.exists(work_dir):
            #noise_levels = np.load(os.path.join(work_dir, "noise_levels.npy"))
            moment_sizes = np.load(os.path.join(work_dir, "moment_sizes.npy"))

            kl_plot = plot.KL_divergence(iter_plot=False,
                                         log_y=True,
                                         log_x=True,
                                         kl_mom_err=False,
                                         title=name + "_exact_mom", xlabel="noise std",
                                         ylabel="KL divergence",
                                         truncation_err_label="trunc. err, m: {}")

            distr_plot = plot.SimpleDistribution(title="{}_exact".format(name), cdf_plot=False, error_plot=False)

            #moment_sizes = [2, 8, 15, 30, 45, 60, 76, 87]

            constraint_values = []
            for n_mom in moment_sizes:

                #kl_plot.truncation_err = np.load(os.path.join(work_dir, "truncation_err.npy"))
                try:
                    _, kl_div = np.load('{}/{}_{}.npy'.format(work_dir, n_mom, "add-value"))
                except FileNotFoundError:
                    kl_div = -1
                    kl_plot.add_value((n_mom, kl_div))
                    constraint_values.append(np.exp(-0.25 * n_mom))
                    continue

                _, nit, success = np.load('{}/{}_{}.npy'.format(work_dir, n_mom, "add-iteration"))
                #_, diff_linalg_norm = np.load('{}/{}_{}.npy'.format(work_dir, noise_level, "add-moments"))

                #constraint_values.append(1/np.power(n_mom, 2))
                #constraint_values.append(np.exp(-0.25 * n_mom))
                kl_plot.add_value((n_mom, kl_div))
                kl_plot.add_iteration(x=n_mom, n_iter=nit, failed=success)
                #kl_plot.add_moments_l2_norm((noise_level, diff_linalg_norm))

                domain = np.load('{}/{}_{}.npy'.format(work_dir, n_mom, "domain"))
                X = np.load('{}/{}_{}.npy'.format(work_dir, n_mom, "X"))

                Y_pdf = np.load('{}/{}_{}.npy'.format(work_dir, n_mom, "Y_pdf"))
                Y_cdf = np.load('{}/{}_{}.npy'.format(work_dir, n_mom, "Y_cdf"))
                threshold = np.load('{}/{}_{}.npy'.format(work_dir, n_mom, "threshold"))

                print("Y pdf ", Y_pdf[10])

                distr_plot.add_distribution(X, Y_pdf, Y_cdf, domain, label="R={}, ".format(n_mom) +  r'$D(\rho \Vert \rho_{R})$' + ":{:0.4g}".format(kl_div))



            #kl_div_mom_err_plot.add_ininity_norm(constraint_values)

            kl_div_mom_err_plot.add_values(kl_div=kl_plot._y, mom_err=moment_sizes, density=distr_title)
            kl_div_mom_err_plot.add_iters(kl_plot._iter_x, kl_plot._iterations, kl_plot._failed_iter_x,
                                          kl_plot._failed_iterations)

            try:
                Y_exact_pdf = np.load('{}/{}_{}.npy'.format(work_dir, n_mom, "Y_pdf_exact"))
                Y_exact_cdf = np.load('{}/{}_{}.npy'.format(work_dir, n_mom, "Y_cdf_exact"))
                distr_plot._add_exact_distr(X, Y_exact_pdf, Y_exact_cdf)
            except:
                pass

            # kl_plot.show(None)
            # distr_plot.show(None)
    print("all konstants ", all_constants)
    print("len all konstants ", len(all_constants))

    #all_constants.append(constraint_values)
    kl_div_mom_err_plot.show()



def plot_KL_div_inexact():
    """
    Plot KL divergence for different noise level of exact moments
    """
    distr_names = {'normální rozdělení': "norm",
                   'lognormální rozdělení': "lognorm",
                   'rozdělení two_gaussians': "two_gaussians",
                   #"rozdělení five_fingers": "five_fingers",
                   "Cauchy rozdělení": "cauchy",
                   "nespojité rozdělení": "discontinuous"
        }

    #dir_name = "/home/martin/Documents/MLMC/test/KL_div_inexact_numpy_4_final"
    dir_name = "/home/martin/Documents/MLMC/test/KL_div_inexact_numpy_2_err"
    #dir_name = "/home/martin/Documents/MLMC/test/KL_div_inexact_numpy_4_err"
    #dir_name = "/home/martin/Documents/MLMC/test/KL_div_inexact_numpy_4_err_35_e16"
    dir_name = "/home/martin/Documents/MLMC/test/KL_div_inexact_numpy_2_err_35_e10"

    dir_name = "/home/martin/Documents/MLMC/test/KL_div_inexact_for_reg_1"
    if not os.path.exists(dir_name):
        raise FileNotFoundError

    kl_div_mom_err_plot = plot.KL_div_mom_err(title="densities",  x_label=r'$|\mu - \hat{\mu}|^2$',
                                              y_label=r'$D(\rho_{35} \Vert \hat{\rho}_{35})$')

    max_values = []
    for distr_title, name in distr_names.items():

        work_dir = os.path.join(dir_name, name)
        if os.path.exists(work_dir):
            noise_levels = np.load(os.path.join(work_dir, "noise_levels.npy"))
            n_moments = np.load(os.path.join(work_dir, "n_moments.npy"))

            noise_levels = noise_levels[:50]

            print("noise levels ", noise_levels)
            print("len noise levels ", len(noise_levels))

            # noise_levels = [noise_levels[0], noise_levels[6], noise_levels[12], noise_levels[22], noise_levels[32],
            #                 noise_levels[40], noise_levels[-1]]

            kl_plot = plot.KL_divergence(iter_plot=False,
                                         log_y=True,
                                         log_x=True,
                                         kl_mom_err=False,
                                         title=name + "_n_mom_{}".format(n_moments), xlabel="noise std",
                                         ylabel="KL divergence",
                                         truncation_err_label="trunc. err, m: {}".format(n_moments))

            distr_plot = plot.SimpleDistribution(title="{}_inexact".format(name), cdf_plot=True, error_plot=False)

            print("noise levels ", noise_levels)

            for noise_level in noise_levels:

                kl_plot.truncation_err = trunc_err =np.load(os.path.join(work_dir, "truncation_err.npy"))

                kl_div_mom_err_plot.add_truncation_error(trunc_err)

                _, kl_div = np.load('{}/{}_{}.npy'.format(work_dir, noise_level, "add-value"))
                _, nit, success = np.load('{}/{}_{}.npy'.format(work_dir, noise_level, "add-iteration"))
                _, diff_linalg_norm = np.load('{}/{}_{}.npy'.format(work_dir, noise_level, "add-moments"))

                print("kl div ", kl_div)

                kl_plot.add_value((noise_level, kl_div))
                kl_plot.add_iteration(x=noise_level, n_iter=nit, failed=success)
                kl_plot.add_moments_l2_norm((noise_level, diff_linalg_norm))

                domain = np.load('{}/{}_{}.npy'.format(work_dir, noise_level, "domain"))
                threshold = np.load('{}/{}_{}.npy'.format(work_dir, noise_level, "threshold"))
                X = np.load('{}/{}_{}.npy'.format(work_dir, noise_level, "X"))

                Y_pdf = np.load('{}/{}_{}.npy'.format(work_dir, noise_level, "Y_pdf"))
                Y_cdf = np.load('{}/{}_{}.npy'.format(work_dir, noise_level, "Y_cdf"))

                y_pdf_log = np.load('{}/{}_{}.npy'.format(work_dir, noise_level, "Y_pdf_log"))
                print("y_pdf log ", y_pdf_log)
                print("np.max(np.abs(y_pdf log)) ", np.max(np.abs(y_pdf_log)))
                max_values.append(np.max(np.abs(y_pdf_log)))

                print("Y pdf ", Y_pdf[10])

                # print("len X ", X)
                # print("len kl div ")

                distr_plot.add_distribution(X, Y_pdf, Y_cdf, domain, label=r'$\sigma=$'  + "{:0.3g}, th:{}, ".format(noise_level, threshold)
                                                                           + r'$D(\rho_{35} \Vert \hat{\rho}_{35})$' + ":{:0.4g}".format(kl_div))


            #kl_div_mom_err_plot.add_ininity_norm(max_values)

            kl_div_mom_err_plot.add_values(kl_div=kl_plot._y, mom_err=kl_plot._mom_err_y, density=distr_title)
            kl_div_mom_err_plot.add_iters(kl_plot._iter_x, kl_plot._iterations, kl_plot._failed_iter_x,
                                          kl_plot._failed_iterations)

            Y_exact_pdf = np.load('{}/{}_{}.npy'.format(work_dir, noise_level, "Y_pdf_exact"))
            Y_exact_cdf = np.load('{}/{}_{}.npy'.format(work_dir, noise_level, "Y_cdf_exact"))
            distr_plot._add_exact_distr(X, Y_exact_pdf, Y_exact_cdf)

            print("max values ", max_values)

            #kl_plot.show(None)
            distr_plot.show(None)
    kl_div_mom_err_plot.show()


def plot_kl_div_mom_err():
    orth_method = 2
    distr_names = {'normální rozdělení': "norm",
                   'lognormální rozdělení': "lognorm",
                   'rozdělení two_gaussians': "two_gaussians",
                   "rozdělení five_fingers": "five_fingers",
                   "Cauchy rozdělení": "cauchy",
                   "nespojité rozdělení": "discontinuous"
                   }

    dir_name = "/home/martin/Documents/MLMC/test/KL_div_inexact_numpy_{}_err".format(orth_method)
    #dir_name = "/home/martin/Documents/MLMC/test/KL_div_inexact_numpy_4_err_35_e16"

    dir_name = "/home/martin/Documents/MLMC/test/KL_div_inexact_numpy_2_err_35_e10"
    if not os.path.exists(dir_name):
        raise FileNotFoundError

    kl_div_mom_err_plot = plot.KL_div_mom_err(title="densities", x_label=r'$|\mu - \hat{\mu}|^2$',
                                              y_label=r'$D(\rho_{35} \Vert \hat{\rho}_{35})$')


    for distr_title, name in distr_names.items():

        work_dir = os.path.join(dir_name, name)
        if os.path.exists(work_dir):
            noise_levels = np.load(os.path.join(work_dir, "noise_levels.npy"))
            n_moments = np.load(os.path.join(work_dir, "n_moments.npy"))

            kl_plot = plot.KL_divergence(iter_plot=False,
                                         log_y=True,
                                         log_x=True,
                                         kl_mom_err=False,
                                         title=name + "_KL_div_{}".format(n_moments), xlabel="noise std",
                                         ylabel="KL divergence",
                                         truncation_err_label="trunc. err, m: {}".format(n_moments))

            max_values = []
            print("noise levels ", noise_levels)
            kl_plot.truncation_err = trunc_err = np.load(os.path.join(work_dir, "truncation_err.npy"))

            kl_div_mom_err_plot.add_truncation_error(trunc_err)


            for noise_level in noise_levels:


                _, kl_div = np.load('{}/{}_{}.npy'.format(work_dir, noise_level, "add-value"))
                _, nit, success = np.load('{}/{}_{}.npy'.format(work_dir, noise_level, "add-iteration"))
                _, diff_linalg_norm = np.load('{}/{}_{}.npy'.format(work_dir, noise_level, "add-moments"))

                y_pdf_log = np.load('{}/{}_{}.npy'.format(work_dir, noise_level, "Y_pdf_log"))
                print("y_pdf log ", y_pdf_log)
                print("np.max(np.abs(y_pdf log)) ", np.max(np.abs(y_pdf_log)))
                max_values.append(np.max(np.abs(y_pdf_log)))

                kl_plot.add_value((noise_level, kl_div))
                kl_plot.add_iteration(x=noise_level, n_iter=nit, failed=success)
                kl_plot.add_moments_l2_norm((noise_level, diff_linalg_norm))

            kl_div_mom_err_plot.add_values(kl_div=kl_plot._y, mom_err=kl_plot._mom_err_y, density=distr_title)
            kl_div_mom_err_plot.add_iters(kl_plot._iter_x, kl_plot._iterations, kl_plot._failed_iter_x,
                                          kl_plot._failed_iterations)

            #kl_div_mom_err_plot.add_inexact_constr(max_values)



    #print("max values ", max_values)
    kl_div_mom_err_plot.show()


def plot_KL_div_reg_inexact_noises():
    """
    Plot KL divergence for different noise level of exact moments
    """
    distr_names = {#'normální rozdělení': "norm",
        'lognormální rozdělení': "lognorm",
        # 'rozdělení two_gaussians': "two_gaussians",
        # "rozdělení five_fingers": "five_fingers",
        # "Cauchy rozdělení": "cauchy",
        # "nespojité rozdělení": "discontinuous"
    }
    orth_method = 4

    dir_name = "/home/martin/Documents/MLMC/test/reg_KL_div_inexact_35_{}".format(orth_method)
    dir_name_inexact = "/home/martin/Documents/MLMC/test/KL_div_inexact_for_reg_{}".format(orth_method)

    dir_name = "/home/martin/Documents/orth_methods/reg_KL_div_inexact_35_{}".format(orth_method)
    #dir_name_inexact = "/home/martin/Documents/MLMC/test/KL_div_inexact_for_reg_2"
    #dir_name = "/home/martin/Documents/orth_methods/KL_div_inexact_for_reg_1"

    if not os.path.exists(dir_name):
        raise FileNotFoundError

    for key, name in distr_names.items():

        work_dir = os.path.join(dir_name, name)
        work_dir_inexact = os.path.join(dir_name_inexact, name)

        if os.path.exists(work_dir):
            noise_levels = np.load(os.path.join(work_dir, "noise_levels.npy"))
            n_moments = np.load(os.path.join(work_dir, "n_moments.npy"))

            kl_plot = plot.KL_divergence(iter_plot=True, log_y=True, log_x=True,
                                         title=name + "_n_mom_{}".format(n_moments), xlabel="noise std",
                                         ylabel="KL divergence",
                                         truncation_err_label="trunc. err, m: {}".format(n_moments))

            #noise_levels = [noise_levels[0]]
            print("noise levels ", noise_levels)

            distr_plot = plot.SimpleDistribution(title="{}_inexact_reg_{}".format(name, orth_method), cdf_plot=False, error_plot=False)
            reg_params_plot = plot.RegParametersPlot(title="{}_reg_params_orth_{}".format(name, orth_method), reg_info=True)

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
                threshold = np.load('{}/{}_{}.npy'.format(work_dir, noise_level, "threshold"))

                Y_pdf_inexact = np.load('{}/{}_{}.npy'.format(work_dir_inexact, noise_level, "Y_pdf"))
                Y_cdf_inexact = np.load('{}/{}_{}.npy'.format(work_dir_inexact, noise_level, "Y_cdf"))
                _, kl_div_inexact = np.load('{}/{}_{}.npy'.format(work_dir_inexact, noise_level, "add-value"))
                threshold_inexact = np.load('{}/{}_{}.npy'.format(work_dir_inexact, noise_level, "threshold"))

                print("Y pdf ", Y_pdf[10])

                distr_plot.add_original_distribution(X, Y_pdf_inexact, Y_cdf_inexact, domain,
                                                     label=r'$\sigma=$' + "{:0.3g}, th:{}, ".format(noise_level,
                                                                                                     threshold_inexact)
                                                           + r'$D(\rho \Vert \hat{\rho}_{35})$' +
                                                           ":{:0.4g}".format(kl_div_inexact))

                distr_plot.add_distribution(X, Y_pdf, Y_cdf, domain, label=r'$\sigma=$' + "{:0.3g}, th:{}, ".format(
                                                                          noise_level,threshold)
                                                                           + r'$D(\rho \Vert \hat{\rho}_{35})$' +
                                                                           ":{:0.4g}".format(kl_div))


                reg_params = np.load('{}/{}_{}.npy'.format(work_dir, noise_level, "reg-params"))
                min_results = np.load('{}/{}_{}.npy'.format(work_dir, noise_level, "min-results"))
                info = np.load('{}/{}_{}.npy'.format(work_dir, noise_level, "info"))

                print("info ", info)

                reg_params_plot.add_values(reg_params, min_results, label=r'$\sigma=$' + "{:0.3g}".format(noise_level))
                reg_params_plot.add_info(info=info)

            Y_exact_pdf = np.load('{}/{}_{}.npy'.format(work_dir, noise_level, "Y_pdf_exact"))
            Y_exact_cdf = np.load('{}/{}_{}.npy'.format(work_dir, noise_level, "Y_cdf_exact"))

            distr_plot._add_exact_distr(X, Y_exact_pdf, Y_exact_cdf)


            distr_plot.show(None)

            reg_params_plot.show()



    #plot_reg_params(reg_params, min_results)

                #kl_plot.show(None)


def plot_KL_div_reg_inexact():
    """
    Plot KL divergence for different noise level of exact moments
    """
    distr_names = {  'normální rozdělení': "norm",
        'lognormální rozdělení': "lognorm",
        'rozdělení two_gaussians': "two_gaussians",
        "rozdělení five_fingers": "five_fingers",
        "Cauchy rozdělení": "cauchy",
        "nespojité rozdělení": "discontinuous"
    }

    orth_method = 4

    dir_name = "/home/martin/Documents/MLMC/test/reg_KL_div_inexact_35_{}".format(orth_method)
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


            noise_levels = [noise_levels[0]]

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

    import matplotlib
    import matplotlib.pyplot as plt
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

def plot_legendre():
    import matplotlib.pyplot as plt
    size = 10
    label_fontsize = 16
    x = np.linspace(-1, 1, 1000)

    leg_poly = np.polynomial.legendre.legvander(x, deg=size - 1)

    fig, ax = plt.subplots(1, 1, figsize=(22, 10))

    print("leg poly shape ", leg_poly.shape)

    ax.set_ylabel(r'$P_{r}(x)$', size=label_fontsize)
    ax.set_xlabel(r'$x$', size=label_fontsize)

    ax.set_ylim([-1.1, 1.1])
    ax.set_xlim([-1, 1])

    for index in range(len(leg_poly[0])):
        ax.plot(x, leg_poly[:, index], label=r'$P_{}(x)$'.format(index))
        print("m shape ", leg_poly[:, index].shape)
        print("m ", leg_poly[:, index])

    ax.legend(fontsize=label_fontsize)
    fig.show()
    file = "legendre_poly.pdf"
    fig.savefig(file)


if __name__ == "__main__":
    #plot_legendre()

    #plot_KL_div_exact()
    #plot_KL_div_inexact()
    #plot_kl_div_mom_err()
    #plot_KL_div_reg_inexact()
    plot_KL_div_reg_inexact_noises()
    #plot_find_reg_param()
