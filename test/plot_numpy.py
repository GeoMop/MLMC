import os
import numpy as np
import mlmc.tool.plot as plot



distr_names = {'_norm': "norm", '_lognorm': "lognorm", '_two_gaussians': "two_gaussians", "_five_fingers": "five_fingers",
               "_cauchy": "cauchy", "_discontinuous": "discontinuous"}


def plot_KL_div_exact_iter(data_dir=None):
    """
    Plot KL divergence for different noise level of exact moments
    """
    distr_names = {'two-gaussians': "TwoGaussians",
                    "five-fingers": "FiveFingers",
                    'lognorm': "lognorm",
                   "cauchy": "Cauchy",
                    "abyss": "Abyss",
                    'zero-value': "ZeroValue",
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
                iter_plot.add_values(iter_res_mom, label=distr_title)

            # kl_div_mom_err_plot.add_ininity_norm(constraint_values)
            #
            # kl_div_mom_err_plot.add_values(kl_div=kl_plot._y, mom_err=moment_sizes, density=distr_title)
            # kl_div_mom_err_plot.add_iters(kl_plot._iter_x, kl_plot._iterations, kl_plot._failed_iter_x,
            #                               kl_plot._failed_iterations)

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


def plot_KL_div_inexact_iter(dir_name=None):
    """
    Plot KL divergence for different noise level of exact moments
    """
    distr_names = {'two-gaussians': "TwoGaussians",
                   "five-fingers": "FiveFingers",
                   'lognorm': "lognorm",
                   "cauchy": "Cauchy",
                   "abyss": "Abyss",
                   'zero-value': "ZeroValue",
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

        if os.path.exists(work_dir):
            noise_levels = np.load(os.path.join(work_dir, "noise_levels.npy"))
            n_moments = np.load(os.path.join(work_dir, "n_moments.npy"))
            noise_levels = noise_levels[:50]


            noise_levels = [noise_levels[0]]

            kl_plot = plot.KL_divergence(iter_plot=False,
                                         log_y=True,
                                         log_x=True,
                                         kl_mom_err=False,
                                         title=name + "_n_mom_{}".format(n_moments), xlabel="noise std",
                                         ylabel="KL divergence",
                                         truncation_err_label="trunc. err, m: {}".format(n_moments))

            distr_plot = plot.SimpleDistribution(title="{}_inexact".format(name), cdf_plot=True, error_plot=False)

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
                iter_plot.add_values(iter_res_mom, label=distr_title)


            #kl_div_mom_err_plot.add_ininity_norm(max_values)
            kl_div_mom_err_plot.add_values(kl_div=kl_plot._y, mom_err=kl_plot._mom_err_y, density=distr_title)
            kl_div_mom_err_plot.add_iters(kl_plot._iter_x, kl_plot._iterations, kl_plot._failed_iter_x,
                                          kl_plot._failed_iterations)

            Y_exact_pdf = np.load('{}/{}_{}.npy'.format(work_dir, noise_level, "Y_pdf_exact"))
            Y_exact_cdf = np.load('{}/{}_{}.npy'.format(work_dir, noise_level, "Y_cdf_exact"))
            distr_plot._add_exact_distr(X, Y_exact_pdf, Y_exact_cdf)

            #kl_plot.show(None)
            distr_plot.show(None)
    kl_div_mom_err_plot.show()
    iter_plot.show()


def plot_KL_div_inexact_precond_no_precond(precond_dir_name=None, no_precond_dir_name=None):
    """
    Plot
    :param precond_dir_name: dir with precondition approach data
    :param no_precond_dir_name: dir with no precondition approach data
    """

    distr_names = {'two-gaussians': "TwoGaussians",
                   "five-fingers": "FiveFingers",
                   'lognorm': "lognorm",
                   "cauchy": "Cauchy",
                   "abyss": "Abyss",
                   'zero-value': "ZeroValue",
                   }
    noise = 1e-5
    dir_name = "/home/martin/Documents/MLMC_article/test/inexact_precond_data_L_{}/KL_div_inexact_2".format(noise)

    if no_precond_dir_name is None:
        no_precond_dir_name = "/home/martin/Documents/MLMC_article/test/inexact_precond_data_L_{}/KL_div_inexact_6".format(noise)

    if precond_dir_name is not None:
        dir_name = precond_dir_name

    if not os.path.exists(dir_name) or not os.path.exists(no_precond_dir_name):
        raise FileNotFoundError

    kl_div_mom_err_plot = plot.KL_div_mom_err(title="densities", x_label=r'$|\mu - \hat{\mu}|^2$',
                                              y_label=r'$D(\rho_{35} \Vert \hat{\rho}_{35})$')

    iter_plot = plot.IterationsComparison(title="mu_err_iterations", x_label="iteration step m",
                                y_label=r'$\sum_{k=1}^{M}(\mu_k - \mu_k^{m})^2$', x_log=False)

    max_values = []
    for distr_title, name in distr_names.items():

        work_dir = os.path.join(dir_name, name)
        no_precond_work_dir = os.path.join(no_precond_dir_name, name)
        if os.path.exists(work_dir):
            noise_levels = np.load(os.path.join(work_dir, "noise_levels.npy"))
            n_moments = np.load(os.path.join(work_dir, "n_moments.npy"))

            noise_levels = noise_levels[:50]


            noise_levels = [noise_levels[0]]

            kl_plot = plot.KL_divergence(iter_plot=False,
                                         log_y=True,
                                         log_x=True,
                                         kl_mom_err=False,
                                         title=name + "_n_mom_{}".format(n_moments), xlabel="noise std",
                                         ylabel="KL divergence",
                                         truncation_err_label="trunc. err, m: {}".format(n_moments))

            distr_plot = plot.SimpleDistribution(title="{}_inexact".format(name), cdf_plot=True, error_plot=False)

            for noise_level in noise_levels:
                kl_plot.truncation_err = trunc_err = np.load(os.path.join(work_dir, "truncation_err.npy"))

                kl_div_mom_err_plot.add_truncation_error(trunc_err)

                _, kl_div = np.load('{}/{}_{}.npy'.format(work_dir, noise_level, "add-value"))
                _, nit, success = np.load('{}/{}_{}.npy'.format(work_dir, noise_level, "add-iteration"))
                _, diff_linalg_norm = np.load('{}/{}_{}.npy'.format(work_dir, noise_level, "add-moments"))

                iter_res_mom = np.load('{}/{}.npy'.format(work_dir, "res_moments"))
                try:
                    no_precond_res_mom = np.load('{}/{}.npy'.format(no_precond_work_dir, "res_moments"))
                except:
                    no_precond_res_mom = None

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

                distr_plot.add_distribution(X, Y_pdf, Y_cdf, domain,
                                            label=r'$\sigma=$' + "{:0.3g}, th:{}, ".format(noise_level, threshold)
                                                  + r'$D(\rho_{35} \Vert \hat{\rho}_{35})$' + ":{:0.4g}".format(kl_div))

                iter_plot.add_values(iter_res_mom, no_precond_res_mom, label=distr_title)

            # kl_div_mom_err_plot.add_ininity_norm(max_values)

            kl_div_mom_err_plot.add_values(kl_div=kl_plot._y, mom_err=kl_plot._mom_err_y, density=distr_title)
            kl_div_mom_err_plot.add_iters(kl_plot._iter_x, kl_plot._iterations, kl_plot._failed_iter_x,
                                          kl_plot._failed_iterations)

            Y_exact_pdf = np.load('{}/{}_{}.npy'.format(work_dir, noise_level, "Y_pdf_exact"))
            Y_exact_cdf = np.load('{}/{}_{}.npy'.format(work_dir, noise_level, "Y_cdf_exact"))
            distr_plot._add_exact_distr(X, Y_exact_pdf, Y_exact_cdf)

            print("max values ", max_values)

            # kl_plot.show(None)
            #distr_plot.show(None)
    #kl_div_mom_err_plot.show()
    iter_plot.show()



if __name__ == "__main__":
    # plot_KL_div_exact_iter()
    plot_KL_div_inexact_iter()
    plot_KL_div_inexact_precond_no_precond()
