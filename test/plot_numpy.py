import os
import sys
import time
import pytest

import numpy as np
import ruamel.yaml as yaml
import scipy.stats as stats
from scipy.interpolate import interp1d

import mlmc.tool.plot as plot
import pickle


distr_names = {'_norm': "norm", '_lognorm': "lognorm", '_two_gaussians': "two_gaussians", "_five_fingers": "five_fingers",
               "_cauchy": "cauchy", "_discontinuous": "discontinuous"}


def plot_KL_div_exact_iter(data_dir=None):
    """
    Plot KL divergence for different noise level of exact moments
    """
    distr_names = {'rozdělení two_gaussians': "two-gaussians",
                    "rozdělení five_fingers": "five-fingers",
                    'lognormální rozdělení': "lognorm",
                   "Cauchy rozdělení": "cauchy",
                    "Abyss rozdělení": "abyss",
                    'rozdělení zero-value': "zero-value",
                   }

    if data_dir is None:
        data_dir = "/home/martin/Documents/MLMC_exact_plot/test/KL_div_exact_numpy_2"
        #data_dir = "/home/martin/Documents/MLMC_article/test/KL_div_exact_numpy_2_charon_quad_no_precond"
        #dir_name = "/home/martin/Documents/MLMC_exact_plot/test/KL_div_exact_numpy_4"

    if not os.path.exists(data_dir):
        raise FileNotFoundError

    kl_div_mom_err_plot = plot.KL_div_mom_err(title="KL_div_R_exact", x_label="R",
                                              y_label=r'$D(\rho \Vert \rho_R)$', x_log=True)

    iter_plot = plot.Iterations(title="mu_err_iterations", x_label="iteration step m",
                                y_label=r'$\sum_{k=1}^{M}(\mu_k - \mu_k^{m})^2$', x_log=False)

    all_constants = []
    for distr_title, name in distr_names.items():
        work_dir = os.path.join(data_dir, name)
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
            mom_err = []

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

                iter_res_mom = np.load('{}/{}_{}.npy'.format(work_dir, n_mom, "res_moments"))
                mom_err.append(iter_res_mom[-1])

                constraint_values.append(1/np.power(n_mom, 2))
                #constraint_values.append(np.exp(-0.25 * n_mom))
                kl_plot.add_value((n_mom, kl_div))
                kl_plot.add_iteration(x=n_mom, n_iter=nit, failed=success)
                #kl_plot.add_moments_l2_norm((noise_level, diff_linalg_norm))

                domain = np.load('{}/{}_{}.npy'.format(work_dir, n_mom, "domain"))
                X = np.load('{}/{}_{}.npy'.format(work_dir, n_mom, "X"))

                Y_pdf = np.load('{}/{}_{}.npy'.format(work_dir, n_mom, "Y_pdf"))
                Y_cdf = np.load('{}/{}_{}.npy'.format(work_dir, n_mom, "Y_cdf"))
                threshold = np.load('{}/{}_{}.npy'.format(work_dir, n_mom, "threshold"))
                distr_plot.add_distribution(X, Y_pdf, Y_cdf, domain, label="M={}, ".format(n_mom) +  r'$D(\rho \Vert \rho_{M})$' + ":{:0.2e}".format(kl_div))
                #iter_plot.add_ininity_norm(constraint_values)
                iter_plot.add_values(iter_res_mom, label=name)

            kl_div_mom_err_plot.add_ininity_norm(constraint_values)

            kl_div_mom_err_plot.add_values(kl_div=kl_plot._y, mom_err=moment_sizes, density=distr_title)
            kl_div_mom_err_plot.add_iters(kl_plot._iter_x, kl_plot._iterations, kl_plot._failed_iter_x,
                                          kl_plot._failed_iterations)

            try:
                Y_exact_pdf = np.load('{}/{}_{}.npy'.format(work_dir, n_mom, "Y_pdf_exact"))
                Y_exact_cdf = np.load('{}/{}_{}.npy'.format(work_dir, n_mom, "Y_cdf_exact"))
                distr_plot._add_exact_distr(X, Y_exact_pdf, Y_exact_cdf)
            except:
                pass

            #kl_plot.show(None)
            #distr_plot.show(None)

    #all_constants.append(constraint_values)
    #kl_div_mom_err_plot.show()
    iter_plot.show()


def plot_KL_div_exact(dir_name=None):
    """
    Plot KL divergence for different noise level of exact moments
    """
    distr_names = {'normální rozdělení': "norm",
                   'lognormální rozdělení': "lognorm",
                   'rozdělení two_gaussians': "two_gaussians",
                   'rozdělení rho4': "Rho4",
                   "rozdělení five_fingers": "five_fingers",
                   "Cauchy rozdělení": "cauchy",
                   # "nespojité rozdělení": "discontinuous"
                   }
    if dir_name is None:
        dir_name = "/home/martin/Documents/MLMC_exact_plot/test/KL_div_exact_numpy_2"
        dir_name = "/home/martin/Documents/MLMC_article/test/KL_div_exact_numpy_2"
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


                constraint_values.append(1/np.power(n_mom, 2))
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



            kl_div_mom_err_plot.add_ininity_norm(constraint_values)

            kl_div_mom_err_plot.add_values(kl_div=kl_plot._y, mom_err=moment_sizes, density=distr_title)
            kl_div_mom_err_plot.add_iters(kl_plot._iter_x, kl_plot._iterations, kl_plot._failed_iter_x,
                                          kl_plot._failed_iterations)

            try:
                Y_exact_pdf = np.load('{}/{}_{}.npy'.format(work_dir, n_mom, "Y_pdf_exact"))
                Y_exact_cdf = np.load('{}/{}_{}.npy'.format(work_dir, n_mom, "Y_cdf_exact"))
                distr_plot._add_exact_distr(X, Y_exact_pdf, Y_exact_cdf)
            except:
                pass

            kl_plot.show(None)
            distr_plot.show(None)
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
                   "rozdělení five_fingers": "five_fingers",
                   "Cauchy rozdělení": "cauchy",
                   "nespojité rozdělení": "discontinuous"
        }

    #dir_name = "/home/martin/Documents/MLMC/test/KL_div_inexact_numpy_4_final"
    dir_name = "/home/martin/Documents/MLMC/test/KL_div_inexact_numpy_2_err"
    #dir_name = "/home/martin/Documents/MLMC/test/KL_div_inexact_numpy_4_err"
    #dir_name = "/home/martin/Documents/MLMC/test/KL_div_inexact_numpy_4_err_35_e16"
    #dir_name = "/home/martin/Documents/MLMC/test/KL_div_inexact_numpy_2_err_35_e10"

    #dir_name = "/home/martin/Documents/MLMC/test/KL_div_inexact_for_reg_1"

    orth_method = 2

    dir_name = "/home/martin/Documents/mlmc_data_final/orth_{}/KL_div_inexact_for_reg_{}_all".\
        format(orth_method, orth_method)

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

            noise_levels = [noise_levels[0], noise_levels[6], noise_levels[12], noise_levels[22], noise_levels[32],
                            noise_levels[40], noise_levels[-1]]

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


def plot_KL_div_inexact_iter(dir_name=None):
    """
    Plot KL divergence for different noise level of exact moments
    """
    distr_names = {  # 'normální rozdělení': "norm",
        'lognormální rozdělení': "lognorm",
        'rozdělení two_gaussians': "two-gaussians",
        'rozdělení zero-value': "zero-value",
        "rozdělení five_fingers": "five-fingers",
        "Cauchy rozdělení": "cauchy",
        "Abyss rozdělení": "abyss",
        # "nespojité rozdělení": "discontinuous"
    }

    if dir_name is None:
        # dir_name = "/home/martin/Documents/MLMC/test/KL_div_inexact_numpy_4_final"
        dir_name = "/home/martin/Documents/MLMC/test/KL_div_inexact_numpy_2_err"
        dir_name = "/home/martin/Documents/MLMC_article/test/inexact_precond_data/KL_div_inexact_2"

    if not os.path.exists(dir_name):
        raise FileNotFoundError

    kl_div_mom_err_plot = plot.KL_div_mom_err(title="densities",  x_label=r'$|\mu - \hat{\mu}|^2$',
                                              y_label=r'$D(\rho_{35} \Vert \hat{\rho}_{35})$')

    iter_plot = plot.Iterations(title="mu_err_iterations", x_label="iteration step m",
                                y_label=r'$\sum_{k=1}^{M}(\mu_k - \mu_k^{m})^2$', x_log=False)

    max_values = []
    for distr_title, name in distr_names.items():

        work_dir = os.path.join(dir_name, name)

        print("work dir ", work_dir)
        if os.path.exists(work_dir):
            noise_levels = np.load(os.path.join(work_dir, "noise_levels.npy"))
            n_moments = np.load(os.path.join(work_dir, "n_moments.npy"))
            noise_levels = noise_levels[:50]

            print("noise levels ", noise_levels)
            print("len noise levels ", len(noise_levels))

            # noise_levels = [noise_levels[0], noise_levels[6], noise_levels[12], noise_levels[22], noise_levels[32],
            #                 noise_levels[40], noise_levels[-1]]

            noise_levels = [noise_levels[0]]

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

                iter_res_mom = np.load('{}/{}.npy'.format(work_dir, "res_moments"))
                kl_plot.add_value((noise_level, kl_div))
                kl_plot.add_iteration(x=noise_level, n_iter=nit, failed=success)
                kl_plot.add_moments_l2_norm((noise_level, diff_linalg_norm))

                domain = np.load('{}/{}_{}.npy'.format(work_dir, noise_level, "domain"))
                threshold = np.load('{}/{}_{}.npy'.format(work_dir, noise_level, "threshold"))
                X = np.load('{}/{}_{}.npy'.format(work_dir, noise_level, "X"))

                Y_pdf = np.load('{}/{}_{}.npy'.format(work_dir, noise_level, "Y_pdf"))
                Y_cdf = np.load('{}/{}_{}.npy'.format(work_dir, noise_level, "Y_cdf"))

                y_pdf_log = np.load('{}/{}_{}.npy'.format(work_dir, noise_level, "Y_pdf_log"))
                max_values.append(np.max(np.abs(y_pdf_log)))
                distr_plot.add_distribution(X, Y_pdf, Y_cdf, domain, label=r'$\sigma=$'  + "{:0.3g}, th:{}, ".format(noise_level, threshold)
                                                                           + r'$D(\rho_{35} \Vert \hat{\rho}_{35})$' + ":{:0.4g}".format(kl_div))
                iter_plot.add_values(iter_res_mom, label=name)

                print("iter res mom ", iter_res_mom)

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
    iter_plot.show()


def plot_kl_div_mom_err():
    orth_method = 4
    distr_names = {'normální rozdělení': "norm",
                   'lognormální rozdělení': "lognorm",
                   'rozdělení two_gaussians': "two_gaussians",
                   "rozdělení five_fingers": "five_fingers",
                   "Cauchy rozdělení": "cauchy",
                   "nespojité rozdělení": "discontinuous"
                   }

    dir_name = "/home/martin/Documents/MLMC/test/KL_div_inexact_numpy_{}_err".format(orth_method)

    dir_name = "/home/martin/Documents/MLMC/test/KL_div_inexact_numpy_2_err_35_e10"

    dir_name = "/home/martin/Documents/mlmc_data_final/orth_{}/KL_div_inexact_for_reg_{}_all".format(orth_method,
                                                                                                         orth_method)

    if not os.path.exists(dir_name):
        raise FileNotFoundError

    kl_div_mom_err_plot = plot.KL_div_mom_err(title="densities_orth_{}".format(orth_method), x_label=r'$|\mu - \hat{\mu}|^2$',
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


def plot_overall_times(sampling_info_path, estimated_times, scheduled_times, finished_times):
    sampling_plot = plot.SamplingPlots(title="Sampling algo", single_fig=True)

    estimated_label = r'$C^t$'
    scheduled_label = r'$C^s$'
    collected_label = r'$C^c$'

    sampling_plot.add_estimated_n(estimated_times, label=estimated_label)
    sampling_plot.add_scheduled_n(scheduled_times, label=scheduled_label)
    sampling_plot.add_collected_n(finished_times, label=collected_label)

    sampling_plot.show(None)
    sampling_plot.show(file=os.path.join(sampling_info_path, "sampling_algo_times_overall"))
    sampling_plot.reset()


def plot_overall_times_sim_times(sampling_info_path, estimated_times, scheduled_times, finished_times, sim_estimated,
                                 sim_scheduled, sim_collected):
        estimated_label = r'$C^t$ total'
        scheduled_label = r'$C^s$ total'
        collected_label = r'$C^c$ total'

        estimated_sub_label = r'$C^t$ sim'
        scheduled_sub_label = r'$C^s$ sim'
        collected_sub_label = r'$C^c$ sim'

        sampling_plot = plot.SamplingPlots(title="Sampling algo", single_fig=True)

        sampling_plot.add_estimated_n(estimated_times, estimated_sub_time=sim_estimated, label=estimated_label, sub_label=estimated_sub_label)
        sampling_plot.add_scheduled_n(scheduled_times, scheduled_sub_time=sim_scheduled, label=scheduled_label, sub_label=scheduled_sub_label)
        sampling_plot.add_collected_n(finished_times, collected_sub_time=sim_collected, label=collected_label, sub_label=collected_sub_label)

        sampling_plot.show(None)
        sampling_plot.show(file=os.path.join(sampling_info_path, "sampling_algo_times_overall_sim_times"))
        sampling_plot.reset()


def plot_overall_times_flow_times(sampling_info_path, estimated_times, scheduled_times, finished_times, flow_estimated, flow_scheduled, flow_collected):
    estimated_label = r'$C^t$ total'
    scheduled_label = r'$C^s$ total'
    collected_label = r'$C^c$ total'

    estimated_sub_label = r'$C^t$ flow'
    scheduled_sub_label = r'$C^s$ flow'
    collected_sub_label = r'$C^c$ flow'

    sampling_plot = plot.SamplingPlots(title="Sampling algo", single_fig=True)

    sampling_plot.add_estimated_n(estimated_times, estimated_sub_time=flow_estimated, label=estimated_label, sub_label=estimated_sub_label)
    sampling_plot.add_scheduled_n(scheduled_times, scheduled_sub_time=flow_scheduled, label=scheduled_label, sub_label=scheduled_sub_label)
    sampling_plot.add_collected_n(finished_times, collected_sub_time=flow_collected, label=collected_label, sub_label=collected_sub_label)

    sampling_plot.show(None)
    sampling_plot.show(file=os.path.join(sampling_info_path, "sampling_algo_times_overall_flow_times"))
    sampling_plot.reset()


def plot_sampling_data():
    n_levels = [5]
    for nl in n_levels:
        sampling_info_path = "/home/martin/Documents/MLMC_article/data/sampling_info"
        #sampling_info_path = "/home/martin/Documents/metamodels/data/1000_ele/cl_0_3_s_4/sampling_info"

        n_target_samples = []
        n_scheduled_samples = []
        n_collected_samples = []
        n_finished_samples = [] # n finished samples is same as n collected samples
        n_failed_samples = []
        n_estimated = []
        variances = []
        n_ops = []
        times = []

        times_scheduled_samples = []
        running_times = []
        flow_running_times = []

        for i in range(0, 100):
            sampling_info_path_iter = os.path.join(sampling_info_path, str(i))
            if os.path.isdir(sampling_info_path_iter):
                n_target_samples.append(np.load(os.path.join(sampling_info_path_iter, "n_target_samples.npy")))
                n_scheduled_samples.append(np.load(os.path.join(sampling_info_path_iter, "n_scheduled_samples.npy")))
                n_collected_samples.append(np.load(os.path.join(sampling_info_path_iter, "n_collected_samples.npy")))
                n_finished_samples.append(np.load(os.path.join(sampling_info_path_iter, "n_finished_samples.npy")))
                n_failed_samples.append(np.load(os.path.join(sampling_info_path_iter, "n_failed_samples.npy"), allow_pickle=True))
                n_estimated.append(np.load(os.path.join(sampling_info_path_iter, "n_estimated.npy")))
                variances.append(np.load(os.path.join(sampling_info_path_iter, "variances.npy")))
                n_ops.append(np.load(os.path.join(sampling_info_path_iter, "n_ops.npy")))
                times.append(np.load(os.path.join(sampling_info_path_iter, "time.npy")))

                running_times.append(np.load(os.path.join(sampling_info_path_iter, "running_times.npy")))
                flow_running_times.append(np.load(os.path.join(sampling_info_path_iter, "flow_running_times.npy")))
                if os.path.exists(os.path.join(sampling_info_path_iter, "scheduled_samples_time.npy")):
                    times_scheduled_samples.append(np.load(os.path.join(sampling_info_path_iter, "scheduled_samples_time.npy")))
            else:
                break

        def time_for_sample_func(data):
            new_n_ops = []
            for nop in data:
                nop = np.squeeze(nop)
                if len(nop) > 0:
                    new_n_ops.append(nop[:, 0]/nop[:, 1])
            return new_n_ops

        n_ops = time_for_sample_func(n_ops)
        running_times = time_for_sample_func(running_times)
        flow_running_times = time_for_sample_func(flow_running_times)
        time_for_sample = np.array(n_ops) + np.array(running_times)

        sim_times = np.array(running_times)
        field_times = np.array(running_times) - np.array(flow_running_times)
        overheads = np.array(n_ops) - field_times

        time_for_sample = sim_times + overheads

        # for nc in n_collected_samples:
        #     print(nc)

        for ne in n_estimated:
            print(ne)


        print("n collected samples ", n_collected_samples)
        print("n ops ", n_ops)
        print("running times ", running_times)
        print("flow running times ", flow_running_times)
        print("np.array(running_times) - np.array(flow_running_times) ", np.array(running_times) - np.array(flow_running_times))

        n_scheduled_times = time_for_sample * np.array(n_scheduled_samples) # Scheduled is the same as Target
        n_estimated_times = time_for_sample * np.array(n_estimated)
        n_finished_times = time_for_sample * np.array(n_collected_samples)


        estimated_times = np.add(np.sum(n_estimated_times, axis=1), np.array(times))
        scheduled_times = np.sum(n_scheduled_times, axis=1) + np.squeeze(times)
        finished_times = np.sum(n_finished_times, axis=1) + np.squeeze(times)

        # Flow execution times from profiler
        flow_n_scheduled_times = flow_running_times * np.array(n_scheduled_samples)  # Scheduled is the same as Target
        flow_n_estimated_times = flow_running_times * np.array(n_estimated)
        flow_n_finished_times = flow_running_times * np.array(n_collected_samples)

        flow_scheduled = np.sum(flow_n_scheduled_times, axis=1)
        flow_collected = np.sum(flow_n_finished_times, axis=1)
        flow_estimated = np.sum(flow_n_estimated_times, axis=1)

        # Simulation CPU times
        sim_n_scheduled_times = sim_times * np.array(n_scheduled_samples)  # Scheduled is the same as Target
        sim_n_estimated_times = sim_times * np.array(n_estimated)
        sim_n_collected_times = sim_times *np.array(n_collected_samples)

        sim_scheduled_times = np.sum(sim_n_scheduled_times, axis=1)
        sim_collected_times = np.sum(sim_n_collected_times, axis=1)
        sim_estimated_times = np.sum(sim_n_estimated_times, axis=1)

        # Plots
        #plot_overall_times(sampling_info_path, estimated_times, scheduled_times, finished_times)
        plot_overall_times_sim_times(sampling_info_path, estimated_times, scheduled_times, finished_times,
                                     sim_estimated_times, sim_scheduled_times, sim_collected_times)
        #plot_overall_times_flow_times(sampling_info_path, estimated_times, scheduled_times, finished_times, flow_estimated, flow_scheduled, flow_collected)


        #sampling_plot.add_total_time(np.squeeze(times) + np.sum(np.squeeze(n_ops), axis=1))

        #print("n_target_samples ", n_target_samples)
        # print("n estimated ", n_estimated)
        # print("n_scheduled ", n_scheduled_samples)
        # print("n_collected ", n_collected_samples)
        #
        # print("sum n_scheduled ", np.sum(n_scheduled_samples, axis=1))
        # print("sum n_collected ", np.sum(n_collected_samples, axis=1))

        # print("n_ops ", n_ops)
        # print("n collected times ", n_collected_times)
        # print("time ", times[:len(times_scheduled_samples)])
        # print("np.sum(np.squeeze(n_ops), axis=1) ", np.sum(np.squeeze(n_ops), axis=1))
        # print("np.sum(np.squeeze(running_times), axis=1) ", np.sum(np.squeeze(running_times), axis=1))
        # print("np.sum(np.squeeze(flow_running_times), axis=1) ", np.sum(np.squeeze(flow_running_times), axis=1))
        # field_generating_time = np.sum(np.squeeze(running_times), axis=1) - 2*np.sum(np.squeeze(flow_running_times), axis=1)


def plot_run_times(n_ops, running_times, extract_mesh_times, make_field_times, generate_rnd_times,
                       fine_flow_times, coarse_flow_times, time_for_sample):

    import matplotlib
    import matplotlib.pyplot as plt

    print("running times ", running_times)

    mean_running_times = np.mean(running_times, axis=0)
    mean_extract_mesh_times = np.mean(extract_mesh_times, axis=0)
    mean_make_field_times = np.mean(make_field_times, axis=0)
    mean_generate_rnd_times = np.mean(generate_rnd_times, axis=0)
    mean_fine_flow_times = np.mean(fine_flow_times, axis=0)
    mean_coarse_flow_times = np.mean(coarse_flow_times, axis=0)


    print("mean running times ", mean_running_times)
    print("mean extract mesh times ", mean_extract_mesh_times)
    print("mean make field times ", mean_make_field_times)
    print("mean generate rnd times ", mean_generate_rnd_times)
    print("mean fine flow times ", mean_fine_flow_times)
    print("mean coarse flow times ", mean_coarse_flow_times)

    preprocess_times = mean_extract_mesh_times + mean_make_field_times + mean_generate_rnd_times

    flow_times = mean_coarse_flow_times + mean_fine_flow_times

    print("presprocess times ", preprocess_times)
    print("flwo times ", flow_times)

    sample_times = preprocess_times + flow_times

    print("sample times ", sample_times)

    preprocess_sample_ratio = preprocess_times / sample_times
    flow_sample_ratio = flow_times / sample_times

    x = range(0, len(sample_times))
    matplotlib.rcParams.update({'lines.markersize': 14})
    matplotlib.rcParams.update({'font.size': 32})

    #####
    ## Preprocess / sample time and flow / sample time
    #####
    fig_ratio, ax_ratio = plt.subplots(1, 1, figsize=(22, 10))
    ax_ratio.scatter(x, preprocess_sample_ratio, label='preprocess / sample time')
    ax_ratio.scatter(x, flow_sample_ratio, label='flow / sample time')
    ax_ratio.legend(loc='upper right')
    # plt.xlim(-0.5, 1000)
    #plt.yscale('log')
    ax_ratio.set_xticks(x)
    ax_ratio.set_xlabel("levels")
    fig_ratio.show()
    file = "flow_preprocess_ratio_times.pdf"
    fig_ratio.savefig(file)

    #####
    ## Preprocess and flow times
    #####
    fig_times, ax_times = plt.subplots(1, 1, figsize=(22, 10))
    print("preprocess times ", preprocess_times)
    print("flow_times ", flow_times)
    ax_times.scatter(x, flow_times, label='sample flow time')
    ax_times.scatter(x, preprocess_times, label='sample preprocess time')
    ax_times.legend(loc='upper left')
    # plt.xlim(-0.5, 1000)
    ax_times.set_yscale('log')
    ax_times.set_xticks(x)
    ax_times.set_xlabel("level")
    fig_times.show()
    file = "flow_preprocess_times.pdf"
    fig_times.savefig(file)


    #####
    ## Parts of preproces time
    #####
    fig_pre_times, ax_pre_times = plt.subplots(1, 1, figsize=(22, 10))
    ax_pre_times.set_yscale('log')
    ax_pre_times.scatter(x, mean_generate_rnd_times, label='generate rnd time')
    ax_pre_times.scatter(x, mean_extract_mesh_times, label='extract mesh time')
    ax_pre_times.scatter(x, mean_make_field_times, label='make field time')
    ax_pre_times.legend(loc='upper left')
    # plt.xlim(-0.5, 1000)

    ax_pre_times.set_xticks(x)
    ax_pre_times.set_xlabel("level")
    fig_pre_times.show()
    file = "preprocess_times.pdf"
    fig_pre_times.savefig(file)
    exit()


def compare_overall_times():
    n_levels = [1, 2, 3, 5]
    #n_levels = [1]
    for nl in n_levels:
        sampling_info_path = "/home/martin/Documents/MLMC_article/data/times/L{}/sampling_info".format(nl)
        # sampling_info_path = "/home/martin/Documents/metamodels/data/1000_ele/cl_0_3_s_4/sampling_info"

        n_target_samples = []
        n_scheduled_samples = []
        n_collected_samples = []
        n_finished_samples = []  # n finished samples is same as n collected samples
        n_failed_samples = []
        n_estimated = []
        variances = []
        n_ops = []
        times = []

        times_scheduled_samples = []
        running_times = []
        extract_mesh_times = []
        make_field_times = []
        generate_rnd_times = []
        fine_flow_times = []
        coarse_flow_times = []

        for i in range(0, 100):
            sampling_info_path_iter = os.path.join(sampling_info_path, str(i))
            if os.path.isdir(sampling_info_path_iter):
                n_target_samples.append(np.load(os.path.join(sampling_info_path_iter, "n_target_samples.npy")))
                n_scheduled_samples.append(np.load(os.path.join(sampling_info_path_iter, "n_scheduled_samples.npy")))
                n_collected_samples.append(np.load(os.path.join(sampling_info_path_iter, "n_collected_samples.npy")))
                n_finished_samples.append(np.load(os.path.join(sampling_info_path_iter, "n_finished_samples.npy")))
                n_failed_samples.append(
                    np.load(os.path.join(sampling_info_path_iter, "n_failed_samples.npy"), allow_pickle=True))
                n_estimated.append(np.load(os.path.join(sampling_info_path_iter, "n_estimated.npy")))
                variances.append(np.load(os.path.join(sampling_info_path_iter, "variances.npy")))
                n_ops.append(np.load(os.path.join(sampling_info_path_iter, "n_ops.npy")))
                times.append(np.load(os.path.join(sampling_info_path_iter, "time.npy")))

                running_times.append(np.load(os.path.join(sampling_info_path_iter, "running_times.npy")))
                extract_mesh_times.append(np.load(os.path.join(sampling_info_path_iter, "extract_mesh_times.npy")))
                make_field_times.append(np.load(os.path.join(sampling_info_path_iter, "make_field_times.npy")))
                generate_rnd_times.append(np.load(os.path.join(sampling_info_path_iter, "generate_rnd_times.npy")))
                fine_flow_times.append(np.load(os.path.join(sampling_info_path_iter, "fine_flow_times.npy")))
                coarse_flow_times.append(np.load(os.path.join(sampling_info_path_iter, "coarse_flow_times.npy")))

                if os.path.exists(os.path.join(sampling_info_path_iter, "scheduled_samples_time.npy")):
                    times_scheduled_samples.append(
                        np.load(os.path.join(sampling_info_path_iter, "scheduled_samples_time.npy")))
            else:
                break

        # print("extract mesh times ", extract_mesh_times[-1])
        # exit()

        def time_for_sample_func(data):
            new_n_ops = []
            for nop in data:
                nop = np.squeeze(nop)
                if len(nop) > 0:
                    #print("nop ", nop)
                    new_n_ops.append(nop[..., 0]/nop[..., 1])
            return new_n_ops

        n_ops = time_for_sample_func(n_ops)
        running_times = time_for_sample_func(running_times) # level_sim._calculate time.time()
        extract_mesh_times = time_for_sample_func(extract_mesh_times)
        make_field_times = time_for_sample_func(make_field_times)
        generate_rnd_times = time_for_sample_func(generate_rnd_times)
        fine_flow_times = time_for_sample_func(fine_flow_times)
        coarse_flow_times = time_for_sample_func(coarse_flow_times)
        time_for_sample = np.array(n_ops) + np.array(running_times)

        mean_running_times = np.mean(running_times, axis=0)
        mean_extract_mesh_times = np.mean(extract_mesh_times, axis=0)
        mean_make_field_times = np.mean(make_field_times, axis=0)
        mean_generate_rnd_times = np.mean(generate_rnd_times, axis=0)
        mean_fine_flow_times = np.mean(fine_flow_times, axis=0)
        mean_coarse_flow_times = np.mean(coarse_flow_times, axis=0)
        mean_n_ops = np.mean(n_ops, axis=0)

        # print("mean running times ", mean_running_times)
        # print("mean extract mesh times ", mean_extract_mesh_times)
        # print("mean make field times ", mean_make_field_times)
        # print("mean generate rnd times ", mean_generate_rnd_times)
        # print("mean fine flow times ", mean_fine_flow_times)
        # print("mean coarse flow times ", mean_coarse_flow_times)

        print("n ops ", mean_n_ops)


        preprocess_times = mean_extract_mesh_times + mean_make_field_times + mean_generate_rnd_times
        flow_times = mean_coarse_flow_times + mean_fine_flow_times
        sample_times = preprocess_times + flow_times


        n_estimated = n_estimated[-1]  # best prediction
        # print("sample times ", sample_times)
        # print("n estimated ", n_estimated)
        #


        print("L{} time ".format(nl), np.sum(mean_n_ops * n_estimated))


def new_plot_sampling_data():
    n_levels = [5]
    for nl in n_levels:
        sampling_info_path = "/home/martin/Documents/MLMC_article/data/times/L5/sampling_info"
        #sampling_info_path = "/home/martin/Documents/metamodels/data/1000_ele/cl_0_3_s_4/sampling_info"

        n_target_samples = []
        n_scheduled_samples = []
        n_collected_samples = []
        n_finished_samples = [] # n finished samples is same as n collected samples
        n_failed_samples = []
        n_estimated = []
        variances = []
        n_ops = []
        times = []

        times_scheduled_samples = []
        running_times = []
        extract_mesh_times = []
        make_field_times = []
        generate_rnd_times = []
        fine_flow_times = []
        coarse_flow_times = []

        for i in range(0, 100):
            sampling_info_path_iter = os.path.join(sampling_info_path, str(i))
            if os.path.isdir(sampling_info_path_iter):
                n_target_samples.append(np.load(os.path.join(sampling_info_path_iter, "n_target_samples.npy")))
                n_scheduled_samples.append(np.load(os.path.join(sampling_info_path_iter, "n_scheduled_samples.npy")))
                n_collected_samples.append(np.load(os.path.join(sampling_info_path_iter, "n_collected_samples.npy")))
                n_finished_samples.append(np.load(os.path.join(sampling_info_path_iter, "n_finished_samples.npy")))
                n_failed_samples.append(np.load(os.path.join(sampling_info_path_iter, "n_failed_samples.npy"), allow_pickle=True))
                n_estimated.append(np.load(os.path.join(sampling_info_path_iter, "n_estimated.npy")))
                variances.append(np.load(os.path.join(sampling_info_path_iter, "variances.npy")))
                n_ops.append(np.load(os.path.join(sampling_info_path_iter, "n_ops.npy")))
                times.append(np.load(os.path.join(sampling_info_path_iter, "time.npy")))

                running_times.append(np.load(os.path.join(sampling_info_path_iter, "running_times.npy")))
                extract_mesh_times.append(np.load(os.path.join(sampling_info_path_iter, "extract_mesh_times.npy")))
                make_field_times.append(np.load(os.path.join(sampling_info_path_iter, "make_field_times.npy")))
                generate_rnd_times.append(np.load(os.path.join(sampling_info_path_iter, "generate_rnd_times.npy")))
                fine_flow_times.append(np.load(os.path.join(sampling_info_path_iter, "fine_flow_times.npy")))
                coarse_flow_times.append(np.load(os.path.join(sampling_info_path_iter, "coarse_flow_times.npy")))

                if os.path.exists(os.path.join(sampling_info_path_iter, "scheduled_samples_time.npy")):
                    times_scheduled_samples.append(np.load(os.path.join(sampling_info_path_iter, "scheduled_samples_time.npy")))
            else:
                break

        def time_for_sample_func(data):
            new_n_ops = []
            for nop in data:
                nop = np.squeeze(nop)
                if len(nop) > 0:
                    new_n_ops.append(nop[:, 0]/nop[:, 1])
            return new_n_ops


        n_ops = time_for_sample_func(n_ops)
        running_times = time_for_sample_func(running_times) # level_sim._calculate time.time()
        extract_mesh_times = time_for_sample_func(extract_mesh_times)
        make_field_times = time_for_sample_func(make_field_times)
        generate_rnd_times = time_for_sample_func(generate_rnd_times)
        fine_flow_times = time_for_sample_func(fine_flow_times)
        coarse_flow_times = time_for_sample_func(coarse_flow_times)
        time_for_sample = np.array(n_ops) + np.array(running_times)

        plot_run_times(n_ops, running_times, extract_mesh_times, make_field_times, generate_rnd_times,
                       fine_flow_times, coarse_flow_times, time_for_sample)

        sim_times = np.array(running_times)
        field_times = np.array(extract_mesh_times) + np.array(make_field_times) + np.array(generate_rnd_times)
        flow_running_times = np.array(fine_flow_times) + np.array(coarse_flow_times)
        overheads = np.array(n_ops) - field_times



        time_for_sample = sim_times + overheads

        print("n collected samples ", n_collected_samples)

        print("n ops ", n_ops)
        print("running times ", running_times)
        print("flow running times ", flow_running_times)
        print("np.array(running_times) - np.array(flow_running_times) ", np.array(running_times) - np.array(flow_running_times))


        n_scheduled_times = time_for_sample * np.array(n_scheduled_samples) # Scheduled is the same as Target
        n_estimated_times = time_for_sample * np.array(n_estimated)
        n_finished_times = time_for_sample * np.array(n_collected_samples)


        estimated_times = np.add(np.sum(n_estimated_times, axis=1), np.array(times))
        scheduled_times = np.sum(n_scheduled_times, axis=1) + np.squeeze(times)
        finished_times = np.sum(n_finished_times, axis=1) + np.squeeze(times)

        # Flow execution times from profiler
        flow_n_scheduled_times = flow_running_times * np.array(n_scheduled_samples)  # Scheduled is the same as Target
        flow_n_estimated_times = flow_running_times * np.array(n_estimated)
        flow_n_finished_times = flow_running_times * np.array(n_collected_samples)

        flow_scheduled = np.sum(flow_n_scheduled_times, axis=1)
        flow_collected = np.sum(flow_n_finished_times, axis=1)
        flow_estimated = np.sum(flow_n_estimated_times, axis=1)

        # Simulation CPU times
        sim_n_scheduled_times = sim_times * np.array(n_scheduled_samples)  # Scheduled is the same as Target
        sim_n_estimated_times = sim_times * np.array(n_estimated)
        sim_n_collected_times = sim_times *np.array(n_collected_samples)

        sim_scheduled_times = np.sum(sim_n_scheduled_times, axis=1)
        sim_collected_times = np.sum(sim_n_collected_times, axis=1)
        sim_estimated_times = np.sum(sim_n_estimated_times, axis=1)

        # Plots
        plot_overall_times(sampling_info_path, estimated_times, scheduled_times, finished_times)
        plot_overall_times_sim_times(sampling_info_path, estimated_times, scheduled_times, finished_times,
                                     sim_estimated_times, sim_scheduled_times, sim_collected_times)
        plot_overall_times_flow_times(sampling_info_path, estimated_times, scheduled_times, finished_times, flow_estimated, flow_scheduled, flow_collected)




def analyze_n_ops():
    cl = 1
    sigma = 1
    levels = 1
    sampling_info_path = "/home/martin/Sync/Documents/flow123d_results/flow_experiments/Exponential/" \
                         "corr_length_0_{}/sigma_{}/L{}/jobs".format(cl, sigma, levels)

    sampling_info_path = "/home/martin/Sync/Documents/flow123d_results/flow_experiments/Exponential/" \
                         "corr_length_0_1_n_ops/sigma_1/L1/times"

    directory = os.fsencode(sampling_info_path)

    print("os.listdir(directory) ", os.listdir(directory))


    total_times = [0, 0]
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".yaml"):
            print("file name ", filename)
            time = {}
            with open(os.path.join(sampling_info_path, filename)) as file:
                times = yaml.load(file, yaml.Loader)

                for level_id, t, n_samples in times:
                    time.setdefault(level_id, []).append((t, n_samples))

            for l_id in range(levels):
                if l_id != 0:
                    continue
                if l_id in time:
                    print("level id ", l_id)
                    print("time ", time[l_id])

                print("time[l_id][-1][0] ", time[l_id][-1][0])
                total_times[0] += time[l_id][-1][0]
                total_times[1] += time[l_id][-1][1]

                #print(time[l_id][-1][0]/time[l_id][-1][1])

    print("times ", total_times)


if __name__ == "__main__":


    # analyze_n_ops()
    # exit()

    #plot_sampling_data()
    #new_plot_sampling_data()
    #compare_overall_times()
    #exit()
    #plot_legendre()

    plot_KL_div_exact()
    #plot_KL_div_inexact()
    #plot_kl_div_mom_err()
    #plot_KL_div_reg_inexact()
    #plot_KL_div_reg_inexact_noises()
    #plot_MEM_spline_vars()
    #plot_KL_div_reg_noises()
    #plot_KL_div_inexact_seeds_all()
    #plot_KL_div_reg_inexact_seeds()
    #plot_find_reg_param()
