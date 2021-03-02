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


def plot_MEM_spline_vars():
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
    orth_method = 4
    n_levels = 1
    n_moments = 35

    #dir_name_inexact = "/home/martin/Documents/MLMC/test/KL_div_inexact_for_reg_{}".format(orth_method)

    dir_name = "/home/martin/Documents/MLMC/test/MEM_spline_orth:{}_L:{}_M:{}".format(orth_method, n_levels, n_moments)

    dir_name = "/home/martin/Documents/mlmc_data_dp/spline/orth_{}/MEM_spline_orth:{}_L:{}_M:{}".format(orth_method, orth_method,
                                                                                         n_levels, n_moments)

    #dir_name = "/home/martin/Documents/MLMC/test/reg_KL_div_inexact_35_{}_five_fingers_1e-2".format(orth_method)

    if not os.path.exists(dir_name):
        raise FileNotFoundError

    kl_div_mom_err_plot = plot.KL_div_mom_err(title="spline_densities_reg_orth_{}".format(orth_method), x_label=r'$|\mu - \hat{\mu}|^2$',
                                              y_label=r'$D(\rho_{35} \Vert \hat{\rho}_{35})$')

    for key, name in distr_names.items():

        work_dir = os.path.join(dir_name, name)
        #work_dir_inexact = os.path.join(dir_name_inexact, name)

        if os.path.exists(work_dir):
            target_vars = np.load(os.path.join(work_dir, "target_vars.npy"))
            n_moments = np.load(os.path.join(work_dir, "n_moments.npy"))

            kl_plot = plot.KL_divergence(iter_plot=True, log_y=True, log_x=True,
                                         title=name + "_n_mom_{}".format(n_moments), xlabel="noise std",
                                         ylabel="KL divergence",
                                         truncation_err_label="trunc. err, m: {}".format(n_moments))

            #target_vars =target_vars[-1:]

            target_vars = target_vars[:1]

            print("target_vars ", target_vars)

            #noise_levels = [1e-2]

            target_vars = np.flip(target_vars)

            distr_plot = plot.SimpleDistribution(title="{}_inexact_reg_{}".format(name, orth_method), cdf_plot=False, error_plot=False)
            reg_params_plot = plot.RegParametersPlot(title="{}_reg_params_orth_{}".format(name, orth_method),
                                                     reg_kl=True, reg_info=True)

            spline_inter_points_plot = plot.SplineInterpolationPointsPlot(title="{}_reg_params_orth_{}".format(name, orth_method),
                                                                          x_log=False)

            #kl_div_mom_err_plot.add_truncation_error(trunc_err)

            for target_var in target_vars:

                #kl_plot.truncation_err = np.load(os.path.join(work_dir, "truncation_err.npy"))

                _, kl_div = np.load('{}/{}_{}.npy'.format(work_dir, target_var, "add-value"))
                _, nit, success = np.load('{}/{}_{}.npy'.format(work_dir, target_var, "add-iteration"))
                _, diff_linalg_norm = np.load('{}/{}_{}.npy'.format(work_dir, target_var, "add-moments"))

                kl_plot.add_value((target_var, kl_div))
                kl_plot.add_iteration(x=target_var, n_iter=nit, failed=success)
                kl_plot.add_moments_l2_norm((target_var, diff_linalg_norm))

                ###################
                ### Without REG ###
                ###################
                domain = np.load('{}/{}_{}.npy'.format(work_dir, target_var, "domain"))
                X = np.load('{}/{}_{}.npy'.format(work_dir, target_var, "X"))

                Y_pdf = np.load('{}/{}_{}.npy'.format(work_dir, target_var, "Y_pdf"))
                Y_cdf = np.load('{}/{}_{}.npy'.format(work_dir, target_var, "Y_cdf"))
                _, kl_div = np.load('{}/{}_{}.npy'.format(work_dir, target_var, "add-value"))
                threshold = np.load('{}/{}_{}.npy'.format(work_dir, target_var, "threshold"))

                ###################
                ### With REG    ###
                ###################
                name = "_reg"
                domain_reg = np.load('{}/{}_{}.npy'.format(work_dir, target_var, "domain" + name))
                X_reg = np.load('{}/{}_{}.npy'.format(work_dir, target_var, "X" + name))
                Y_pdf_reg = np.load('{}/{}_{}.npy'.format(work_dir, target_var, "Y_pdf" + name))
                Y_cdf_reg = np.load('{}/{}_{}.npy'.format(work_dir, target_var, "Y_cdf" + name))
                threshold_reg = np.load('{}/{}_{}.npy'.format(work_dir, target_var, "threshold" + name))
                _, kl_div_reg = np.load('{}/{}_{}.npy'.format(work_dir, target_var, "add-value" + name))

                if orth_method == 4:
                    threshold = 34 - threshold
                    threshold_reg = 34 - threshold_reg


                distr_plot.add_original_distribution(X, Y_pdf, Y_cdf, domain,
                                                     label=r'$\sigma=$' + "{:0.3g}, th:{}, ".format(target_var,
                                                                                                    threshold)
                                                           + r'$D(\rho \Vert \hat{\rho}_{35})$' +
                                                           ":{:0.4g}".format(kl_div))


                distr_plot.add_distribution(X_reg, Y_pdf_reg, Y_cdf_reg, domain_reg, label=r'$\sigma=$' + "{:0.3g}, th:{}, ".format(
                                                                          target_var, threshold_reg)
                                                                           + r'$D(\rho \Vert \hat{\rho}_{35})$' +
                                                                           ":{:0.4g}".format(kl_div_reg))

                ###################
                ### Spline      ###
                ###################
                name = "_bspline"
                domain_bspline = np.load('{}/{}_{}.npy'.format(work_dir, target_var, "domain" + name))
                X_bspline = np.load('{}/{}_{}.npy'.format(work_dir, target_var, "X" + name))
                Y_pdf_bspline = np.load('{}/{}_{}.npy'.format(work_dir, target_var, "Y_pdf" + name))
                Y_cdf_bspline = np.load('{}/{}_{}.npy'.format(work_dir, target_var, "Y_cdf" + name))
                kl_div_bspline = np.load('{}/{}_{}.npy'.format(work_dir, target_var, "result" + name))

                distr_plot.add_distribution(X_bspline, Y_pdf_bspline, Y_cdf_bspline, domain_bspline,
                                            label="Bspline " + r'$\sigma=$' + "{:0.3g}, ".format(
                                                target_var)
                                                  + r'$D(\rho \Vert \hat{\rho}_{35})$' +
                                                  ":{:0.4g}, L2: {}".format(kl_div_bspline[0], kl_div_bspline[1]))

                Y_pdf_bspline_l2 = np.load('{}/{}_{}.npy'.format(work_dir, target_var, "Y_pdf_l2" + name))
                Y_cdf_bspline_l2 = np.load('{}/{}_{}.npy'.format(work_dir, target_var, "Y_cdf_l2" + name))
                kl_l2_best = np.load('{}/{}_{}.npy'.format(work_dir, target_var, "l2_dist" + name))

                distr_plot.add_distribution(X_bspline, Y_pdf_bspline_l2, Y_cdf_bspline_l2, domain_bspline,
                                            label="L2 best Bspline " + r'$\sigma=$' + "{:0.3g}, ".format(
                                                target_var)
                                                  + r'$D(\rho \Vert \hat{\rho}_{35})$' +
                                                  ":{:0.4g} L2: {}".format(kl_l2_best[0], kl_l2_best[1]))

                all_n_int_points = np.load('{}/{}_{}.npy'.format(work_dir, target_var, "spline_int_points"))
                kl_div_l2_dist = np.load('{}/{}_{}.npy'.format(work_dir, target_var, "spline_kl_divs_l2_dist"))

                spline_inter_points_plot.add_values(all_n_int_points, kl_div_l2_dist, label=r'$\sigma=$' + "{:0.3g}".format(target_var))

                reg_params = np.load('{}/{}_{}.npy'.format(work_dir, target_var, "reg-params"))
                min_results = np.load('{}/{}_{}.npy'.format(work_dir, target_var, "min-results"))
                cond_numbers = np.load('{}/{}_{}.npy'.format(work_dir, target_var, "cond-numbers"))
                info = np.load('{}/{}_{}.npy'.format(work_dir, target_var, "info"))

                print("reg params ", reg_params)
                print("min results ", min_results)

                reg_params_plot.add_values(reg_params, min_results, label=r'$\sigma=$' + "{:0.3g}".format(target_var))
                reg_params_plot.add_cond_numbers(cond_numbers)
                reg_params_plot.add_info(info=info)


            Y_exact_pdf = np.load('{}/{}_{}.npy'.format(work_dir, target_var, "Y_pdf_exact"))
            Y_exact_cdf = np.load('{}/{}_{}.npy'.format(work_dir, target_var, "Y_cdf_exact"))

            distr_plot._add_exact_distr(X, Y_exact_pdf, Y_exact_cdf)

            distr_plot.show(None)

            reg_params_plot.show()
            spline_inter_points_plot.show()

            kl_div_mom_err_plot.add_values(kl_div=kl_plot._y, mom_err=kl_plot._mom_err_y, density=key)
            kl_div_mom_err_plot.add_iters(kl_plot._iter_x, kl_plot._iterations, kl_plot._failed_iter_x,
                                          kl_plot._failed_iterations)

    kl_div_mom_err_plot.show()



def plot_KL_div_reg_inexact_noises():
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
    orth_method = 1

    #dir_name = "/home/martin/Documents/MLMC/test/reg_KL_div_inexact_35_{}".format(orth_method)
    dir_name_inexact = "/home/martin/Documents/MLMC/test/KL_div_inexact_for_reg_{}".format(orth_method)

    dir_name = "/home/martin/Documents/orth_methods/reg_KL_div_inexact_35_{}".format(orth_method)
    #dir_name_inexact = "/home/martin/Documents/MLMC/test/KL_div_inexact_for_reg_2"
    #dir_name = "/home/martin/Documents/orth_methods/KL_div_inexact_for_reg_1"

    #dir_name = "/home/martin/Documents/new_mlmc_seeds_original/mlmc_seed_1/orth_{}/reg_KL_div_inexact_35_{}".format(orth_method,
    #                                                                                                   orth_method)

    dir_name_inexact = "/home/martin/Documents/mlmc_data_dp/orth_{}/KL_div_inexact_for_reg_{}".format(orth_method,
                                                                                           orth_method)
    dir_name = "/home/martin/Documents/mlmc_data_dp/orth_{}/reg_KL_div_inexact_35_{}".format(orth_method,
                                                                                            orth_method)

    #dir_name = "/home/martin/Documents/MLMC/test/reg_KL_div_inexact_35_{}_five_fingers_1e-2".format(orth_method)

    if not os.path.exists(dir_name):
        raise FileNotFoundError

    kl_div_mom_err_plot = plot.KL_div_mom_err(title="densities_reg_orth_{}".format(orth_method), x_label=r'$|\mu - \hat{\mu}|^2$',
                                              y_label=r'$D(\rho_{35} \Vert \hat{\rho}_{35})$')

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

            noise_levels = noise_levels[:-1]
            print("noise levels ", noise_levels)

            #noise_levels = [1e-2]

            noise_levels = np.flip(noise_levels)

            distr_plot = plot.SimpleDistribution(title="{}_inexact_reg_{}".format(name, orth_method), cdf_plot=False, error_plot=False)
            reg_params_plot = plot.RegParametersPlot(title="{}_reg_params_orth_{}".format(name, orth_method),
                                                     reg_kl=True, reg_info=True)

            #kl_div_mom_err_plot.add_truncation_error(trunc_err)

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

                if orth_method == 4:
                    threshold = 34 - threshold
                    threshold_inexact = 34 - threshold_inexact

                print("Y pdf ", Y_pdf[10])

                print("KL div ", kl_div)

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

            kl_div_mom_err_plot.add_values(kl_div=kl_plot._y, mom_err=kl_plot._mom_err_y, density=key)
            kl_div_mom_err_plot.add_iters(kl_plot._iter_x, kl_plot._iterations, kl_plot._failed_iter_x,
                                          kl_plot._failed_iterations)

    kl_div_mom_err_plot.show()



    #plot_reg_params(reg_params, min_results)

                #kl_plot.show(None)

def load_mlmc(path):
    with open(path, "rb") as writer:
        mlmc = pickle.load(writer)
    return mlmc


def plot_MEM_spline():
    """
    Plot KL divergence for different noise level of exact moments
    """
    distr_names = {'normální rozdělení': "norm",
        # 'lognormální rozdělení': "lognorm",
        # 'rozdělení two_gaussians': "two_gaussians",
        # "rozdělení five_fingers": "five_fingers",
        # "Cauchy rozdělení": "cauchy",
        # "nespojité rozdělení": "discontinuous"
    }
    orth_method = 2

    #dir_name = "/home/martin/Documents/MLMC/test/reg_KL_div_inexact_35_{}".format(orth_method)
    dir_name_inexact = "/home/martin/Documents/MLMC/test/KL_div_inexact_for_reg_{}".format(orth_method)

    dir_name = "/home/martin/Documents/orth_methods/reg_KL_div_inexact_35_{}".format(orth_method)
    #dir_name_inexact = "/home/martin/Documents/MLMC/test/KL_div_inexact_for_reg_2"
    #dir_name = "/home/martin/Documents/orth_methods/KL_div_inexact_for_reg_1"

    #dir_name = "/home/martin/Documents/new_mlmc_seeds_original/mlmc_seed_1/orth_{}/reg_KL_div_inexact_35_{}".format(orth_method,
    #                                                                                                   orth_method)

    dir_name_inexact = "/home/martin/Documents/mlmc_data_dp/orth_{}/KL_div_inexact_for_reg_{}".format(orth_method,
                                                                                           orth_method)
    dir_name = "/home/martin/Documents/mlmc_data_dp/orth_{}/reg_KL_div_inexact_35_{}".format(orth_method,
                                                                                            orth_method)

    dir_name = "/home/martin/Documents/MLMC/test/reg_KL_div_inexact_35_{}_cauchy_1e-2".format(orth_method)


    n_levels = 1
    n_moments = 5
    target_var = 1e-6
    quantile = 0.001
    interpolation_points = 20

    dir_name = "MEM_spline_L:{}_M:{}_TV:{}_q:{}_:int_point".format(n_levels, n_moments, target_var, quantile, interpolation_points)

    if not os.path.exists(dir_name):
        raise FileNotFoundError

    kl_div_mom_err_plot = plot.KL_div_mom_err(title="densities_reg_orth_{}".format(orth_method), x_label=r'$|\mu - \hat{\mu}|^2$',
                                              y_label=r'$D(\rho_{35} \Vert \hat{\rho}_{35})$')

    for key, name in distr_names.items():

        work_dir = os.path.join(dir_name, name)
        work_dir_inexact = os.path.join(dir_name_inexact, name)

        if os.path.exists(work_dir):
            noise_levels = np.load(os.path.join(work_dir, "noise_levels.npy"))
            n_moments = np.load(os.path.join(work_dir, "n_moments.npy"))

            mlmc = load_mlmc(os.path.join(work_dir, "saved_mlmc"))
            print("mlmc levels ", mlmc.levels)

            n_samples = []
            # for level in mlmc.levels:
            #     n_samples.append(level._n_collected_samples)

            int_points_domain = np.load(os.path.join(work_dir, "int_points_domain"))
            density = np.load(os.path.join(work_dir, "density"))

            # int_points_domain = [0, 0]
            # int_points_domain[0] = cut_distr.domain[0] - 1000
            # int_points_domain[1] = cut_distr.domain[1] + 1000

            density = True
            spline_plot = plot.Spline_plot(bspline=True,
                                           title="levels: {}, int_points_domain: {}".format(n_levels,
                                                                                            int_points_domain),
                                           density=density)

            #interpolation_points = 5
            polynomial_degree = np.load(os.path.join(work_dir, "polynomial_degree"))
            accuracy = np.load(os.path.join(work_dir, "accuracy"))

            # X = np.linspace(cut_distr.domain[0]-10, cut_distr.domain[1]+10, 1000)
            X = np.load(os.path.join(work_dir, "X"))
            interpolation_points = np.load(os.path.join(work_dir, "interpolation_points"))


            # distr_obj = make_spline_approx(int_points_domain, mlmc, polynomial_degree, accuracy)

            # interpolation_points = [300, 500, 750, 1000, 1250]


            spline_plot.interpolation_points = interpolation_points

            interpolation_points = [interpolation_points]
            for n_int_points in interpolation_points:

                if density:
                    pdf = np.load(os.path.join(work_dir, "indicator_pdf"))
                    pdf_X = np.load(os.path.join(work_dir, "indicator_pdf_X"))
                    spline_plot.add_indicator_density((pdf_X, pdf))
                else:
                    cdf = np.load(os.path.join(work_dir, "indicator_cdf"))
                    cdf_X = np.load(os.path.join(work_dir, "indicator_cdf_X"))
                    spline_plot.add_indicator((cdf_X, cdf))

                if density:
                    pdf = np.load(os.path.join(work_dir, "smooth_pdf"))
                    pdf_X = np.load(os.path.join(work_dir, "smooth_pdf_X"))
                    spline_plot.add_smooth_density((pdf_X, pdf))
                else:
                    cdf = np.load(os.path.join(work_dir, "smooth_cdf"))
                    cdf_X = np.load(os.path.join(work_dir, "smooth_cdf_X"))
                    spline_plot.add_smooth((cdf_X, cdf))

                if density:
                    pdf = np.load(os.path.join(work_dir, "spline_pdf"))
                    pdf_X= np.load(os.path.join(work_dir, "spline_pdf_X"))
                    spline_plot.add_bspline_density((pdf_X, pdf))
                else:
                    cdf = np.load(os.path.join(work_dir, "spline_cdf"))
                    cdf_X = np.load(os.path.join(work_dir, "spline_cdf_X"))
                    spline_plot.add_bspline((cdf_X, cdf))

            exact_cdf = np.load(os.path.join(work_dir, "exact_cdf"))
            spline_plot.add_exact_values(X, exact_cdf)
            if density:
                exact_pdf = np.load(os.path.join(work_dir, "exact_pdf"))
                spline_plot.add_density_exact_values(X, exact_pdf)

            ecdf = np.load(os.path.join(work_dir, "ecdf"))
            ecdf_X = np.load(os.path.join(work_dir, "ecdf_X"))
            spline_plot.add_ecdf(ecdf_X, ecdf)
            spline_plot.show()


def plot_KL_div_reg_noises():
    """
    Plot KL divergence for different noise level of exact moments
    """
    sdistr_names = {#'normální rozdělení': "norm",
        # 'lognormální rozdělení': "lognorm",
        # 'rozdělení two_gaussians': "two_gaussians",
        # "rozdělení five_fingers": "five_fingers",
        "Cauchy rozdělení": "cauchy",
        # "nespojité rozdělení": "discontinuous"
    }
    orth_method = 2

    dir_name = "/home/martin/Documents/MLMC/test/reg_KL_div_inexact_35_{}".format(orth_method)
    dir_name_inexact = "/home/martin/Documents/MLMC/test/KL_div_inexact_for_reg_{}".format(orth_method)

    dir_name = "/home/martin/Documents/orth_methods/reg_KL_div_inexact_35_{}".format(orth_method)
    #dir_name_inexact = "/home/martin/Documents/MLMC/test/KL_div_inexact_for_reg_2"
    #dir_name = "/home/martin/Documents/orth_methods/KL_div_inexact_for_reg_1"

    # dir_name = "/home/martin/Documents/mlmc_seeds/mlmc_seed_5/orth_{}/reg_KL_div_inexact_35_{}".format(orth_method,
    #                                                                                                    orth_method)

    dir_name = "/home/martin/Documents/mlmc_data_final/orth_{}/reg_KL_div_inexact_35_{}".format(orth_method,
                                                                                            orth_method)

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

                # Y_pdf_inexact = np.load('{}/{}_{}.npy'.format(work_dir_inexact, noise_level, "Y_pdf"))
                # Y_cdf_inexact = np.load('{}/{}_{}.npy'.format(work_dir_inexact, noise_level, "Y_cdf"))
                # _, kl_div_inexact = np.load('{}/{}_{}.npy'.format(work_dir_inexact, noise_level, "add-value"))
                # threshold_inexact = np.load('{}/{}_{}.npy'.format(work_dir_inexact, noise_level, "threshold"))

                if orth_method == 4:
                    threshold = 34 - threshold
                    #threshold_inexact = 34 - threshold_inexact



                print("Y pdf ", Y_pdf[10])

                # distr_plot.add_original_distribution(X, Y_pdf_inexact, Y_cdf_inexact, domain,
                #                                      label=r'$\sigma=$' + "{:0.3g}, th:{}, ".format(noise_level,
                #                                                                                      threshold_inexact)
                #                                            + r'$D(\rho \Vert \hat{\rho}_{35})$' +
                #                                            ":{:0.4g}".format(kl_div_inexact))

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

def plot_KL_div_inexact_seeds_all():
    """
    Plot KL divergence for different noise level of exact moments
    """
    distr_names = {'normální rozdělení': "norm",
        # 'lognormální rozdělení': "lognorm",
        # 'rozdělení two_gaussians': "two_gaussians",
        # "rozdělení five_fingers": "five_fingers",
        # "Cauchy rozdělení": "cauchy",
        # "nespojité rozdělení": "discontinuous"
    }
    orth_method = 2
    dir_name = "/home/martin/Documents/new_mlmc_seeds/mlmc_seed_1/orth_{}/KL_div_inexact_for_reg_all".format(orth_method)
    #dir_name = "/home/martin/Documents/new_mlmc_seeds/mlmc_seed_1/orth_{}/KL_div_inexact_for_reg_{}".format(orth_method, orth_method)

    exact_dir_name = "/home/martin/Documents/MLMC_exact_plot/test/KL_div_exact_numpy_2".format(orth_method)

    n_seeds = 30
    n_reg_params = 150

    if not os.path.exists(dir_name):
        raise FileNotFoundError
    
    #all_noise_levels = [0.1]

    for key, name in distr_names.items():
        work_dir = os.path.join(dir_name, name)

        if os.path.exists(work_dir):
            noise_levels = np.load(os.path.join(work_dir, "noise_levels.npy"))
            n_moments = np.load(os.path.join(work_dir, "n_moments.npy"))

            noise_levels = noise_levels[:50]

            print("noise levels ", noise_levels)
            print("len noise levels ", len(noise_levels))

            noise_levels = [noise_levels[0], noise_levels[6], noise_levels[12], noise_levels[22], noise_levels[32],
                            noise_levels[40], noise_levels[-1]]
            #
            #
            noise_levels = np.flip(noise_levels)

            #noise_levels = [noise_levels[-1]]

            # if all_noise_levels:
            #     noise_levels = all_noise_levels

            print("noise levels ", noise_levels)

            kl_plot = plot.KL_divergence(iter_plot=False,
                                         log_y=True,
                                         log_x=True,
                                         kl_mom_err=False,
                                         title=name + "_n_mom_{}".format(n_moments), xlabel="noise std",
                                         ylabel="KL divergence",
                                         truncation_err_label="trunc. err, m: {}".format(n_moments))

            distr_plot = plot.SimpleDistribution(title="{}_inexact_all_{}".format(name, orth_method), cdf_plot=False, error_plot=False)
            reg_params_plot = plot.RegParametersPlot(title="{}_reg_params_orth_{}".format(name, orth_method), reg_info=True)

            for noise_level in noise_levels:

                all_kl_div = []
                all_nit = []
                all_success = []
                all_diff_linalg_norm = []

                all_X = []
                all_pdf_Y = []
                all_cdf_Y = []

                for seed in range(1, n_seeds):
                    print("seed ", seed)
                    seed_dir = "/home/martin/Documents/new_mlmc_seeds/mlmc_seed_{}/orth_{}/KL_div_inexact_for_reg_all".\
                        format(seed, orth_method, orth_method)

                    # seed_dir = dir_name = "/home/martin/Documents/new_mlmc_seeds/mlmc_seed_{}/orth_{}/KL_div_inexact_for_reg_{}".\
                    #       format(seed, orth_method, orth_method)

                    print("seed dir ", seed_dir)
                    work_dir = os.path.join(seed_dir, name)
                    exact_work_dir = os.path.join(exact_dir_name, name)
                    print("work_dir ", work_dir)

                    # if not os.path.exists(work_dir):
                    #     continue

                    try:
                        _, kl_div = np.load('{}/{}_{}.npy'.format(work_dir, noise_level, "add-value"))
                        _, nit, success = np.load('{}/{}_{}.npy'.format(work_dir, noise_level, "add-iteration"))
                        _, diff_linalg_norm = np.load('{}/{}_{}.npy'.format(work_dir, noise_level, "add-moments"))

                        if success:
                            continue
                        print("success ", success)

                        all_kl_div.append(kl_div)
                        all_nit.append(nit)
                        all_success.append(success)
                        all_diff_linalg_norm.append(diff_linalg_norm)

                        domain = np.load('{}/{}_{}.npy'.format(work_dir, noise_level, "domain"))
                        X = np.load('{}/{}_{}.npy'.format(work_dir, noise_level, "X"))

                        print("X ", X)

                        all_X.append(X)

                        print("domain ", domain)

                        Y_pdf = np.load('{}/{}_{}.npy'.format(work_dir, noise_level, "Y_pdf"))
                        Y_cdf = np.load('{}/{}_{}.npy'.format(work_dir, noise_level, "Y_cdf"))
                        threshold = np.load('{}/{}_{}.npy'.format(work_dir, noise_level, "threshold"))

                        if orth_method == 4:
                            threshold = 34 - threshold

                        all_pdf_Y.append(Y_pdf)
                        all_cdf_Y.append(Y_cdf)

                        # distr_plot.add_distribution(X, Y_pdf, Y_cdf, domain,
                        #                             label=r'$\sigma=$' + "{:0.3g}, th:{}, ".format(noise_level,
                        #                                                                            threshold)
                        #                                   + r'$D(\rho_{35} \Vert \hat{\rho}_{35})$' + ":{:0.4g}".format(
                        #                                 kl_div))
                        kl_plot.truncation_err = np.load(os.path.join(work_dir, "truncation_err.npy"))

                    except FileNotFoundError as e:
                        print("ERR MSG ", e)
                        continue

                kl_div = np.mean(all_kl_div)
                nit = np.mean(all_nit)
                success = np.mean(all_success)
                diff_linalg_norm = np.mean(all_diff_linalg_norm)

                kl_plot.add_value((noise_level, kl_div))
                kl_plot.add_iteration(x=noise_level, n_iter=nit, failed=success)
                kl_plot.add_moments_l2_norm((noise_level, diff_linalg_norm))

                print("ALL X ", all_X)

                print("ALL PDF Y ", all_pdf_Y)

                print("ALL CDF Y ", all_cdf_Y)

                distr_plot.add_distribution(np.mean(np.array(all_X), axis=0),
                                            np.mean(np.array(all_pdf_Y), axis=0),
                                            np.mean(np.array(all_cdf_Y), axis=0), domain,
                                            label="avg from {} ".format(len(all_pdf_Y)) + r'$\sigma=$' + "{:0.3g} ".format(noise_level)
                                              + r'$D(\rho_{35} \Vert \hat{\rho}_{35})$' + ":{:0.4g}".format(kl_div))


            Y_exact_pdf = np.load('{}/{}_{}.npy'.format(exact_work_dir, 39, "Y_pdf_exact"))
            Y_exact_cdf = np.load('{}/{}_{}.npy'.format(exact_work_dir, 39, "Y_cdf_exact"))
            distr_plot._add_exact_distr(X, Y_exact_pdf, Y_exact_cdf)

            distr_plot.show(None)

            reg_params_plot.show()

            print("valid seeds ", len(all_pdf_Y))


def plot_KL_div_reg_inexact_seeds():
    """
    Plot KL divergence for different noise level of exact moments
    """
    distr_names = {#'normální rozdělení': "norm",
         #'lognormální rozdělení': "lognorm",
         'rozdělení two_gaussians': "two_gaussians",
          #"rozdělení five_fingers": "five_fingers",
         #"Cauchy rozdělení": "cauchy",
        #"nespojité rozdělení": "discontinuous"
    }
    orth_method = 2

    dir_name_inexact = "/home/martin/Documents/new_mlmc_seeds/mlmc_seed_1/orth_{}/KL_div_inexact_for_reg_{}".\
        format(orth_method, orth_method)
    dir_name = "/home/martin/Documents/new_mlmc_seeds/mlmc_seed_1/orth_{}/reg_KL_div_inexact_35_{}".format(orth_method, orth_method)
    exact_dir_name = "/home/martin/Documents/MLMC_exact_plot/test/KL_div_exact_numpy_2".format(orth_method)

    # dir_name = "/home/martin/Documents/new_mlmc_seeds/mlmc_seed_1/orth_4_2/reg_KL_div_inexact_35_4"
    # dir_name_inexact = "/home/martin/Documents/new_mlmc_seeds/mlmc_seed_1/orth_4_2/KL_div_inexact_for_reg_4"

    min_seed = 1
    n_seeds = 2
    n_reg_params = 150

    if not os.path.exists(dir_name):
        raise FileNotFoundError

    for key, name in distr_names.items():

        work_dir = os.path.join(dir_name, name)
        work_dir_inexact = os.path.join(dir_name_inexact, name)

        print("work_dir ", work_dir)

        if os.path.exists(work_dir):
            noise_levels = np.load(os.path.join(work_dir, "noise_levels.npy"))
            n_moments = np.load(os.path.join(work_dir, "n_moments.npy"))

            kl_plot = plot.KL_divergence(iter_plot=True, log_y=True, log_x=True,
                                         title=name + "_n_mom_{}".format(n_moments), xlabel="noise std",
                                         ylabel="KL divergence",
                                         truncation_err_label="trunc. err, m: {}".format(n_moments))

            noise_levels = [noise_levels[-2]]
            print("noise levels ", noise_levels)

            distr_plot = plot.SimpleDistribution(title="{}_inexact_reg_{}".format(name, orth_method), cdf_plot=False, error_plot=False)
            reg_params_plot = plot.RegParametersPlot(title="{}_reg_params_orth_{}".format(name, orth_method), reg_info=True)

            for noise_level in noise_levels:

                all_kl_div = []
                all_nit = []
                all_success = []
                all_diff_linalg_norm = []
                all_reg_params = []
                all_min_results = []
                all_info = []

                all_X = []
                all_pdf_Y = []
                all_cdf_Y = []
                all_pdf_Y_inexact = []
                all_cdf_Y_inexact = []
                all_kl_div_inexact = []

                all_reg_min_res = []
                for seed in range(min_seed, n_seeds):
                    seed_dir = "/home/martin/Documents/new_mlmc_seeds/mlmc_seed_{}/orth_{}/reg_KL_div_inexact_35_{}".\
                        format(seed, orth_method, orth_method)
                    seed_dir_inexact = "/home/martin/Documents/new_mlmc_seeds/mlmc_seed_{}/orth_{}/KL_div_inexact_for_reg_{}".\
                        format(seed, orth_method, orth_method)
                    # seed_dir = "/home/martin/Documents/new_mlmc_seeds/mlmc_seed_{}/orth_4_2/reg_KL_div_inexact_35_4". \
                    #     format(seed)
                    # seed_dir_inexact = "/home/martin/Documents/new_mlmc_seeds/mlmc_seed_{}/orth_4_2/KL_div_inexact_for_reg_4". \
                    #     format(seed)


                    work_dir = os.path.join(seed_dir, name)
                    work_dir_inexact = os.path.join(seed_dir_inexact, name)
                    exact_work_dir = os.path.join(exact_dir_name, name)
                    print("work_dir ", work_dir)
                    print("work dir inexact ", work_dir_inexact)

                    try:
                        _, kl_div = np.load('{}/{}_{}.npy'.format(work_dir, noise_level, "add-value"))
                        _, nit, success = np.load('{}/{}_{}.npy'.format(work_dir, noise_level, "add-iteration"))
                        _, diff_linalg_norm = np.load('{}/{}_{}.npy'.format(work_dir, noise_level, "add-moments"))

                        if success:
                            continue

                        threshold = np.load('{}/{}_{}.npy'.format(work_dir, noise_level, "threshold"))

                        # if kl_div > 0.03:
                        #     continue

                        if orth_method == 4:
                            threshold = 34 - threshold

                        # if threshold == 0:
                        #     continue
                        # else:
                        #     print("SEED ", seed)

                        reg_par = np.load('{}/{}_{}.npy'.format(work_dir, noise_level, "reg-params"))

                        print("reg params shape ", reg_par.shape)

                        if reg_par.shape[0] < n_reg_params:
                            reg_par = np.append(reg_par, np.ones(n_reg_params-reg_par.shape[0])*reg_par[-1])
                        all_reg_params.append(reg_par)

                        min_res = np.load('{}/{}_{}.npy'.format(work_dir, noise_level, "min-results"))
                        if min_res.shape[0] < n_reg_params:
                            min_res = np.append(min_res, np.ones(n_reg_params-min_res.shape[0])*min_res[-1])

                        all_min_results.append(min_res)

                        info = np.load('{}/{}_{}.npy'.format(work_dir, noise_level, "info"))

                        while info.shape[0] < n_reg_params:
                            # print("info shape ", info.shape)
                            # print("info[-1, :] ", info[-1, :])

                            info = list(info)
                            info.append(info[-1][:])

                            info = np.array(info)

                            # info = np.append(info, info[-1, :], axis=1)
                            #print("info shape ", info.shape)

                        # if info.shape[0] < n_reg_params:
                        #     print("info shape ", info.shape)
                        #     print("info[-1, :] ", info[-1, :])
                        #
                        #     info = list(info)
                        #     info.append(info[-1][:])
                        #
                        #     info = np.array(info)
                        #
                        #     # info = np.append(info, info[-1, :], axis=1)
                        #     print("info shape ", info.shape)
                        #
                        # if info.shape[0] < n_reg_params:
                        #     print("info shape ", info.shape)
                        #     print("info[-1, :] ", info[-1, :])
                        #
                        #     info = list(info)
                        #     info.append(info[-1][:])
                        #
                        #     info = np.array(info)
                        #
                        #     # info = np.append(info, info[-1, :], axis=1)
                        #     print("info shape ", info.shape)


                        zipped = zip(reg_par, min_res, info[:, 0])
                        sorted_zip = sorted(zipped, key=lambda x: x[1])

                        min_kl_div = None
                        for s_tuple in sorted_zip:
                            if min_kl_div is None:
                                min_kl_div = s_tuple

                        # if threshold == 0:
                        #     continue
                        # else:
                        #     print("SEED: {}, min_kl_div: {}".format(seed, min_kl_div))
                        #

                        all_reg_min_res.append(min_kl_div)

                        all_info.append(info)

                        all_kl_div.append(kl_div)
                        all_nit.append(nit)
                        all_success.append(success)
                        all_diff_linalg_norm.append(diff_linalg_norm)

                        domain = np.load('{}/{}_{}.npy'.format(work_dir, noise_level, "domain"))
                        X = np.load('{}/{}_{}.npy'.format(work_dir, noise_level, "X"))

                        #print("X ", X)

                        all_X.append(X)

                        #print("domain ", domain)

                        Y_pdf = np.load('{}/{}_{}.npy'.format(work_dir, noise_level, "Y_pdf"))
                        Y_cdf = np.load('{}/{}_{}.npy'.format(work_dir, noise_level, "Y_cdf"))


                        all_pdf_Y.append(Y_pdf)
                        all_cdf_Y.append(Y_cdf)

                        distr_plot.add_distribution(X, Y_pdf, Y_cdf, domain,
                                                    label=r'$\sigma=$' + "{:0.3g}, th:{}, ".format(
                                                        noise_level, threshold)
                                                          + r'$D(\rho \Vert \hat{\rho}_{35})$' +
                                                          ":{:0.4g}".format(kl_div))

                        _, kl_div = np.load('{}/{}_{}.npy'.format(work_dir_inexact, noise_level, "add-value"))
                        _, nit, success = np.load('{}/{}_{}.npy'.format(work_dir, noise_level, "add-iteration"))

                        #print("KL div inexact ", kl_div)

                        if not success:
                            print("NIT ", nit)
                            all_kl_div_inexact.append(kl_div)

                        Y_pdf_inexact = np.load('{}/{}_{}.npy'.format(work_dir_inexact, noise_level, "Y_pdf"))
                        Y_cdf_inexact = np.load('{}/{}_{}.npy'.format(work_dir_inexact, noise_level, "Y_cdf"))
                        _, kl_div_inexact = np.load('{}/{}_{}.npy'.format(work_dir_inexact, noise_level, "add-value"))
                        threshold_inexact = np.load('{}/{}_{}.npy'.format(work_dir_inexact, noise_level, "threshold"))

                        if orth_method == 4:
                            threshold_inexact = 34 - threshold_inexact

                        reg_params_plot.add_values(reg_par, min_res,
                                                   label=r'$\sigma=$' + "{:0.3g}".format(noise_level))
                        reg_params_plot.add_info(info=info)

                        # distr_plot.add_distribution(X, Y_pdf_inexact, Y_cdf_inexact, domain,
                        #                             label="INEXACT " + r'$\sigma=$' + "{:0.3g}, th:{}, ".format(
                        #                                 noise_level, threshold)
                        #                                   + r'$D(\rho \Vert \hat{\rho}_{35})$' +
                        #                                   ":{:0.4g}".format(kl_div))


                        all_pdf_Y_inexact.append(Y_pdf_inexact)
                        all_cdf_Y_inexact.append(Y_cdf_inexact)

                        kl_plot.truncation_err = np.load(os.path.join(work_dir, "truncation_err.npy"))

                        # Y_exact_pdf = np.load('{}/{}_{}.npy'.format(exact_work_dir, 39, "Y_pdf_exact"))
                        # Y_exact_cdf = np.load('{}/{}_{}.npy'.format(exact_work_dir, 39, "Y_cdf_exact"))
                        #
                        # distr_plot._add_exact_distr(X, Y_exact_pdf, Y_exact_cdf)
                        # distr_plot.show()

                    except FileNotFoundError as e:
                        print("ERR MSG ", e)
                        continue

                #print("all KL div ", all_kl_div)
                kl_div = np.mean(all_kl_div)
                nit = np.mean(all_nit)
                success = np.mean(all_success)
                diff_linalg_norm = np.mean(all_diff_linalg_norm)

                kl_plot.add_value((noise_level, kl_div))
                kl_plot.add_iteration(x=noise_level, n_iter=nit, failed=success)
                kl_plot.add_moments_l2_norm((noise_level, diff_linalg_norm))

                # print("all reg params ", all_reg_params)
                # print("all min results ", all_min_results)
                # print("all info ", all_info)
                #
                #print("{}, all min kl div:{} ".format(name), all_reg_min_res)
                print("{}, ALL MIN KL DIV MEAN: {} ".format(name, np.mean(all_reg_min_res, axis=0)))

                reg_params = np.mean(all_reg_params, axis=0)
                min_results = np.mean(all_min_results, axis=0)
                info = np.mean(all_info, axis=0)

                # print("reg params ", reg_params)
                # print("min results ", min_results)
                # print("info ", info)

                reg_params_plot.add_values(reg_params, min_results, label=r'$\sigma=$' + "{:0.3g}".format(noise_level))
                reg_params_plot.add_info(info=info)



                # distr_plot.add_original_distribution(X, Y_pdf_inexact, Y_cdf_inexact, domain,
                #                                      label=r'$\sigma=$' + "{:0.3g}, th:{}, ".format(noise_level,
                #                                                                                     threshold_inexact)
                #                                            + r'$D(\rho \Vert \hat{\rho}_{35})$' +
                #                                            ":{:0.4g}".format(kl_div_inexact))

            Y_exact_pdf = np.load('{}/{}_{}.npy'.format(exact_work_dir, 39, "Y_pdf_exact"))
            Y_exact_cdf = np.load('{}/{}_{}.npy'.format(exact_work_dir, 39, "Y_cdf_exact"))


            # print("ALL X ", all_X)
            #
            # print("ALL PDF Y ", all_pdf_Y)
            #
            # print("ALL CDF Y ", all_cdf_Y)

            #print("all KL div inexact ", all_kl_div_inexact)

            distr_plot.add_original_distribution(np.mean(np.array(all_X)), np.mean(np.array(all_pdf_Y_inexact)),
                                                 np.mean(np.array(all_cdf_Y_inexact)),
                                                 domain, label="original, KL: " + ":{:0.4g}".format(
                    np.mean(all_kl_div_inexact)))



            distr_plot.add_distribution(np.mean(np.array(all_X), axis=0),
                                        np.mean(np.array(all_pdf_Y), axis=0),
                                        np.mean(np.array(all_cdf_Y), axis=0), domain, label="average, KL: " + ":{:0.4g}".format(np.mean(all_kl_div)) + "rep: {}".format(len(all_kl_div)))


            distr_plot._add_exact_distr(X, Y_exact_pdf, Y_exact_cdf)

            # print("ALL PDF Y INEXACT", all_pdf_Y_inexact)
            #
            # print("ALL CDF Y INEXACT", all_cdf_Y_inexact)
            #
            # print("ALL KL DIV INEXACT ", all_kl_div_inexact)


            distr_plot.show(None)

            reg_params_plot.show()

            print("valid seeds ", len(all_pdf_Y))


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


def plot_overall_times(sampling_info_path, estimated_times, scheduled_times, finished_times):
    sampling_plot = plot.SamplingPlots(title="Sampling algo", single_fig=True)

    sampling_plot.add_estimated_n(estimated_times)
    sampling_plot.add_scheduled_n(scheduled_times)
    sampling_plot.add_collected_n(finished_times)

    sampling_plot.show(None)
    sampling_plot.show(file=os.path.join(sampling_info_path, "sampling_algo_times_overall"))
    sampling_plot.reset()


def plot_overall_times_sim_times(sampling_info_path, estimated_times, scheduled_times, finished_times, sim_estimated,
                                 sim_scheduled, sim_collected):
        estimated_sub_label = "E sim"
        scheduled_sub_label = "S sim"
        collected_sub_label = "C sim"

        sampling_plot = plot.SamplingPlots(title="Sampling algo", single_fig=True)

        sampling_plot.add_estimated_n(estimated_times, estimated_sub_time=sim_estimated, sub_label=estimated_sub_label)
        sampling_plot.add_scheduled_n(scheduled_times, scheduled_sub_time=sim_scheduled, sub_label=scheduled_sub_label)
        sampling_plot.add_collected_n(finished_times, collected_sub_time=sim_collected, sub_label=collected_sub_label)

        sampling_plot.show(None)
        sampling_plot.show(file=os.path.join(sampling_info_path, "sampling_algo_times_overall_sim_times"))
        sampling_plot.reset()


def plot_overall_times_flow_times(sampling_info_path, estimated_times, scheduled_times, finished_times, flow_estimated, flow_scheduled, flow_collected):
    estimated_sub_label = "E flow"
    scheduled_sub_label = "S flow"
    collected_sub_label = "C flow"

    sampling_plot = plot.SamplingPlots(title="Sampling algo", single_fig=True)

    sampling_plot.add_estimated_n(estimated_times, estimated_sub_time=flow_estimated, sub_label=estimated_sub_label)
    sampling_plot.add_scheduled_n(scheduled_times, scheduled_sub_time=flow_scheduled, sub_label=scheduled_sub_label)
    sampling_plot.add_collected_n(finished_times, collected_sub_time=flow_collected, sub_label=collected_sub_label)

    sampling_plot.show(None)
    sampling_plot.show(file=os.path.join(sampling_info_path, "sampling_algo_times_overall_flow_times"))
    sampling_plot.reset()


def plot_sampling_data():
    n_levels = [5]
    for nl in n_levels:
        sampling_info_path = "/home/martin/Documents/MLMC_article/data/sampling_info"

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
        time_for_sample = n_ops


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
        sim_n_scheduled_times = (np.array(running_times) - np.array(flow_running_times)) * np.array(
            n_scheduled_samples)  # Scheduled is the same as Target
        sim_n_estimated_times = (np.array(running_times) - np.array(flow_running_times)) * np.array(n_estimated)
        sim_n_collected_times = (np.array(running_times) - np.array(flow_running_times)) * np.array(n_collected_samples)

        sim_scheduled_times = np.sum(sim_n_scheduled_times, axis=1)
        sim_collected_times = np.sum(sim_n_collected_times, axis=1)
        sim_estimated_times = np.sum(sim_n_estimated_times, axis=1)

        # Plots
        plot_overall_times(sampling_info_path, estimated_times, scheduled_times, finished_times)
        plot_overall_times_sim_times(sampling_info_path, estimated_times, scheduled_times, finished_times,
                                     sim_estimated_times, sim_scheduled_times, sim_collected_times)
        plot_overall_times_flow_times(sampling_info_path, estimated_times, scheduled_times, finished_times, flow_estimated, flow_scheduled, flow_collected)


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


# def analyze_n_ops():
#     cl = 1
#     sigma = 1
#     levels = 3
#     sampling_info_path = "/home/martin/Sync/Documents/flow123d_results/flow_experiments/Exponential/" \
#                          "corr_length_0_{}/sigma_{}/L{}/jobs".format(cl, sigma, levels)
#
#     directory = os.fsencode(sampling_info_path)
#
#     print("os.listdir(directory) ", os.listdir(directory))
#
#     for file in os.listdir(directory):
#         filename = os.fsdecode(file)
#         if filename.endswith(".yaml"):
#             print("file name ", filename)
#             time = {}
#             with open(os.path.join(sampling_info_path, filename)) as file:
#                 times = yaml.load(file, yaml.Loader)
#
#                 for level_id, t, n_samples in times:
#                     time.setdefault(level_id, []).append((t, n_samples))
#
#             for l_id in range(levels):
#                 if l_id != 0:
#                     continue
#                 if l_id in time:
#                     print("level id ", l_id)
#                     print("time ", time[l_id])
#                 #print(time[l_id][-1][0]/time[l_id][-1][1])


if __name__ == "__main__":


    #analyze_n_ops()

    plot_sampling_data()
    exit()
    #plot_legendre()

    #plot_KL_div_exact()
    #plot_KL_div_inexact()
    #plot_kl_div_mom_err()
    #plot_KL_div_reg_inexact()
    #plot_KL_div_reg_inexact_noises()
    plot_MEM_spline_vars()
    #plot_KL_div_reg_noises()
    #plot_KL_div_inexact_seeds_all()
    #plot_KL_div_reg_inexact_seeds()
    #plot_find_reg_param()
