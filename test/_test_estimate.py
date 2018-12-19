import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
from scipy import stats
import time

from pandas import DataFrame

import pytest
src_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(src_path, '..', 'src'))
import mlmc.simple_distribution
import mlmc.moments as moments
import mlmc.mc_level
import mlmc.estimate

import fixtures.mlmc_test_run
from test_distribution import DistrPlot


from mlmc.distribution import Distribution
import test.test_distribution as test_distribution
from mlmc.simple_distribution import SimpleDistribution


class TestEstimate(mlmc.estimate.Estimate):

    def __init__(self):
        self._eigen_values = {}
        self._estimate_cov_matrix = None
        self._fig_file = None

    def construct_orthogonal_moments(self, moments_fn, mlmc_obj=None, exact_cov=None, threshold=0, plot=True, rm_below_threshold=False):
        """
        Construct orthogonal moments object
        :param moments_fn: mlmc.Moments instance
        :param mlmc_obj: mlmc.MLMC instance
        :param exact_cov: Exact covariance matrix
        :param threshold: Threshold value for eigenvalues
        :param plot: bool, Plot eigenvalues
        :param rm_below_threshold: bool, Remove eigenvalues below threshold
        :return: mlmc.Moments instance
        """
        cov = None
        # Estimate covariance
        if mlmc_obj is not None and exact_cov is None:
            cov = self.estimate_covariance(moments_fn, mlmc_obj.levels)

        # Use exact covariance
        if exact_cov is not None:
            cov = exact_cov

        # if cov is None:
        #     return

        # centered covariance
        M = np.eye(moments_fn.size)
        M[:, 0] = -cov[:, 0]
        cov_center = M @ cov @ M.T
        eval, evec = np.linalg.eigh(cov_center)
        evec = np.flip(evec, axis=1)
        i_first_positive = np.argmax(eval > 0)

        # Remove values below threshold
        if rm_below_threshold is True:
            eval = eval[i_first_positive:]

            if i_first_positive != 0:
                evec = evec[:, :-i_first_positive]
        # Replace values below threshold by first value above threshold
        else:
            i_first_positive = np.argmax(eval > threshold)
            eval[:i_first_positive] = eval[i_first_positive]

        L = evec.T @ M

        if plot:
            self.plot_values(eval, log=True, fig_file=self._fig_file)#, treshold=treshold, errors = err_pos_diag)

        eval = np.flip(eval, axis=0)
        L = -(1/np.sqrt(eval))[:, None] * L

        ortogonal_moments_obj = mlmc.moments.TransformedMoments(moments_fn, L)
        return ortogonal_moments_obj, np.flip(eval, axis=0), L

    def _plot_eigen_moment_var(self, eigenvalues, moments_data):
        """
        Plot moments var with corresponding eigenvalue
        """
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(30, 10))
        ax = fig.add_subplot(1, 1, 1)

        values = list(eigenvalues.values())[0]

        X = np.arange(len(values))
        a, b = np.min(values), np.max(values)

        ax.set_yscale('log', nonposy="clip")
        ax.set_ylim(a / ((b / a) ** 0.05), b * (b / a) ** 0.05)

        errors = np.flip(moments_data[:, 1], axis=0)
        ax.errorbar(X[1:], values[1:], yerr=errors[1:], fmt='o', capthick=10)

        fig.legend()
        fig.show()

    def plot_all_eigen_values(self, eigenvalues):
        """
        Plot all eigenvalues, currently not used
        :param eigenvalues: dictionary
        """
        log = True
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(30, 10))
        ax = fig.add_subplot(1, 1, 1)

        all_values = []
        for l, v in eigenvalues.items():
            all_values.extend(v)

        a, b = np.min(all_values), np.max(all_values)
        if log:
            ax.set_yscale('log')
            #ax.set_ylim(a / ((b / a) ** 0.05), b * (b / a) ** 0.05)
        else:
            ax.set_ylim(a - 0.05 * (b - a), b + 0.05 * (b - a))

        for label, eigen in eigenvalues.items():
            self.plot_eigen_values(eigen, ax, label)

        x_original = []
        y_original = []
        for index, value in enumerate(eigenvalues['estimate cov']):
            y_original.append(value)
            x_original.append(int(index))

        fig.legend()
        fig.show()
        plt.show()

    def plot_eigen_values(self, values, ax, label="", errors=None, log=False, treshold=None):
        """
        Auxiliary method for self.plot_all_eigen_values()
        """
        X = np.arange(len(values))
        if errors is None:
            ax.scatter(X, values, label=label)
        else:
            ax.errorbar(X, values, label=label, yerr=errors, fmt='o', ecolor='g', capthick=2)
        for i, x in enumerate(X):
            ax.annotate(str(x), (x + 0.1, values[i]), label=label)
        if treshold is not None:
            ax.axvline(x=treshold - 0.1)

    def plot_values(self, values, errors=None, log=False, treshold=None, fig_file=None):
        """
        Plot eigenvalues

        """
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(1, 1, 1)
        X = np.arange(len(values))

        a, b = np.min(values), np.max(values)

        if log:
            ax.set_yscale('log')
            ax.set_ylim(a / ((b/a)**0.05), b * (b/a)**0.05)
        else:
            ax.set_ylim(a - 0.05 * (b - a), b + 0.05 * (b - a))
        if errors is None:
            ax.scatter(X, values)
        else:
            ax.errorbar(X, values, yerr=errors, fmt='o', ecolor='g', capthick=2)
        for i, x in enumerate(X):
            ax.annotate(str(x), (x+0.1, values[i]))
        if treshold is not None:
            ax.axvline(x=treshold-0.1)
        if fig_file is not None:
            fig.savefig(fig_file)
        print("Backend: ", plt.get_backend())
        fig.show()
        print("Continue")
        plt.show()


    # @pytest.mark.parametrize("moment_fn, max_n_moments", [
    #     (moments.Monomial, 10),
    #     #(moments.Fourier, 61),
    #     (moments.Legendre, 61)])
    # @pytest.mark.parametrize("distribution",[
    #         (stats.norm(loc=1, scale=2), False)
    #         # (stats.norm(loc=1, scale=10), False),
    #         # (stats.lognorm(scale=np.exp(1), s=1), False),    # Quite hard but peak is not so small comparet to the tail.
    #         # #(stats.lognorm(scale=np.exp(-3), s=2), False),  # Extremely difficult to fit due to very narrow peak and long tail.
    #         # (stats.lognorm(scale=np.exp(-3), s=2), True),    # Still difficult for Lagrange with many moments.
    #         # (stats.chi2(df=10), False), # Monomial: s1=nan, Fourier: s1= -1.6, Legendre: s1=nan
    #         # (stats.chi2(df=5), True), # Monomial: s1=-10, Fourier: s1=-1.6, Legendre: OK
    #         # (stats.weibull_min(c=0.5), False),  # Exponential # Monomial stuck, Fourier stuck
    #         # (stats.weibull_min(c=1), False),  # Exponential
    #         # (stats.weibull_min(c=2), False),  # Rayleigh distribution
    #         # (stats.weibull_min(c=5, scale=4), False),   # close to normal
    #         # (stats.weibull_min(c=1.5), True),  # Infinite derivative at zero
    #     ])
    def _test_pdf_approx_exact_moments(self, moment_fn, max_n_moments, distribution):
        """
        Test reconstruction of the density function from exact moments.
        - various distributions
        - various moments functions
        - test convergence with increasing number of moments
        :return:
        """
        #np.random.seed(1234)
        distr, log_flag = distribution
        fn_name = moment_fn.__name__
        distr_name = distr.dist.name
        density = lambda x: distr.pdf(x)
        print("Testing moments: {} for distribution: {}".format(fn_name, distr_name))

        tol_exact_moments = 1e-6
        tol_density_approx = 1e-6
        quantiles = np.array([0.0001])

        #moment_sizes = np.round(np.exp(np.linspace(np.log(3), np.log(max_n_moments), 5))).astype(int)
        moment_sizes = [max_n_moments]
        kl_collected = np.empty((len(quantiles), len(moment_sizes)))
        l2_collected = np.empty_like(kl_collected)

        n_failed = []
        warn_log = []
        distr_plot = DistrPlot(distr, distr_name + " " + fn_name)
        for i_q, domain_quantile in enumerate(quantiles):
            domain, force_decay = self.domain_for_quantile(distr, domain_quantile)

            # Approximation for exact moments
            moments_fn = moment_fn(moment_sizes[-1], domain, log=log_flag, safe_eval=False)

            # Exact moments and covariance
            exact_moments = mlmc.simple_distribution.compute_exact_moments(moments_fn, density, tol=tol_exact_moments)
            exact_cov = mlmc.simple_distribution.compute_exact_cov(moments_fn, density)

            moments_fn, eval, _ = self.construct_orthogonal_moments(moments_fn, exact_cov=exact_cov)

            # Transform exact moments
            exact_moments = mlmc.simple_distribution.compute_exact_moments(moments_fn, density, tol=tol_exact_moments)

            n_failed.append(0)
            cumtime = 0
            tot_nit = 0
            for i_m, n_moments in enumerate(moment_sizes):
                #moments_fn = orthogonal_moments_obj(n_moments, domain, log=log_flag, safe_eval=False)

                moments_data = np.empty((len(exact_moments), 2))
                moments_data[:, 0] = exact_moments[:]
                moments_data[:, 1] = 1e-6

                print("moments data ", moments_data)
                is_positive = log_flag
                distr_obj = SimpleDistribution(moments_fn, moments_data,
                                                           domain=moments_fn.domain, force_decay=force_decay)
                t0 = time.time()
                # result = distr_obj.estimate_density(tol_exact_moments)

                result = distr_obj.estimate_density_minimize(tol_density_approx)

                #result = profile(lambda : distr_obj.estimate_density_minimize(tol_exact_moments))
                t1 = time.time()
                cumtime += (t1 - t0)
                nit = getattr(result, 'nit', result.njev)
                tot_nit += nit
                fn_norm = result.fun_norm

                kl_div = mlmc.simple_distribution.KL_divergence(density, distr_obj.density, domain[0], domain[1])
                l2_dist = mlmc.simple_distribution.L2_distance(distr_obj.density, density, domain[0], domain[1])
                kl_collected[i_q, i_m] = kl_div
                l2_collected[i_q, i_m] = l2_dist

                #assert kl_div < 1, "kl_div: " + str(kl_div)
                print("q: {}, m: {} :: nit: {} fn: {} ; kl: {} l2: {}".format(
                    domain_quantile, n_moments, nit, fn_norm, kl_div, l2_dist))
                if i_m + 1 == len(moment_sizes):
                    # plot for last case
                    pass

                distr_plot.plot_approximation(distr_obj, label=str(domain))

                n_failed[-1] += 1
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

        distr_plot.show()
        #distr_plot.clean()

    # @TODO duplicated in test_distribution
    def domain_for_quantile(self, distr, quantile):
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

    def _test_pdf_mlmc(self):
        """
        Test reconstruction of the density function from exact moments.
        - various distributions
        - various moments functions
        - test convergency with increasing number of moments
        :return:
        """
        #np.random.seed(3)  # To be reproducible
        n_levels = [1]  # [1, 2, 3, 5, 7]
        #n_moments = [30]#, 30, 100]#, 20, 40, 60, 80]

        moment_size = 20
        n_moments = [moment_size]
        threshold = 0
        plots = None

        kl_div_all = {}
        l2_all = {}

        distr = [
            (stats.norm(loc=1, scale=2), False, '_sample_fn', "(loc=1, scale=2)"),
            # (stats.norm(loc=1, scale=10), False, '_sample_fn'),
            #(stats.lognorm(scale=np.exp(5), s=1), True, '_sample_fn', "(scale=np.exp(5), s=1)"),# not from exact zero
            # (stats.lognorm(scale=np.exp(-3), s=2), True, '_sample_fn'),  # worse conv of higher moments
            #(stats.chi2(df=10), True, '_sample_fn', "(df=10)"),
             #(stats.chi2(df=5), True, '_sample_fn'),
            #WORSE (stats.weibull_min(c=0.5), False, '_sample_fn'),  # Exponential
            #(stats.weibull_min(c=1), False, '_sample_fn', "(c=1)"),  # Exponential
            # (stats.weibull_min(c=2), False, '_sample_fn'),  # Rayleigh distribution
            #(stats.weibull_min(c=5, scale=4), False, '_sample_fn', "(c=5, scale=4)"),  # close to normal
            # (stats.weibull_min(c=1.5), True, '_sample_fn'),  # Infinite derivative at zero
            # (stats.lognorm(scale=np.exp(-5), s=1), True, '_sample_fn_no_error'),
            #(stats.weibull_min(c=20), True, '_sample_fn_no_error', "(c=20)"),   # Exponential
            # (stats.weibull_min(c=20), True, '_sample_fn_no_error'),   # Exponential
            #(stats.weibull_min(c=3), True, '_sample_fn_no_error', "(c=3)")    # Close to normal
        ]

        # Save to csv file object
        csv = None#SaveCsv("/home/martin/Desktop/Distributions")
        quantile_list = [0.01]#, 0.001, 0.0001, 0.000001]

        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(1, 1, 1)
        color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b', 'g', 'r']
        colors = iter(color_list)
        max_moments = np.max(n_moments)

        for nl in n_levels:
            for d, log_flag, sim, params in distr:
                kl_div_distr = {}
                l2_distr = {}

                distr_name = d.dist.name + params
                # Add distribution to csv file object
                # if csv is not None:
                #     csv.add_distribution(distr_name)

                # distr_fig_dir = os.path.join(csv.data_dir, distr_name)
                # if not os.path.exists(distr_fig_dir):
                #     os.makedirs(distr_fig_dir)

                config = {"threshold": threshold, "distr_name": distr_name, "moment_size": moment_size,
                          "color_list": color_list, "quantile_list": quantile_list}

                #for moment_size in n_moments:
                for quantile in quantile_list:
                    mc_test = fixtures.mlmc_test_run.TestMLMC(nl, moment_size, d, log_flag, sim, quantile=quantile)
                    # number of samples on each level
                    #total_samples = mc_test.mc.sample_range(100000, 100)
                    mc_test.mc.set_initial_n_samples()
                    mc_test.mc.target_var_adding_samples(1e-6, mc_test.moments_fn)
                    #mc_test.generate_samples(total_samples)
                    #self._fig_file = os.path.join(distr_fig_dir, "{}_moments.png".format(moment_size))

                    #config['total_samples'] = total_samples

                    fn_name = "Legendre"
                    distr_name = d.dist.name
                    density = lambda x: d.pdf(x)
                    print("Testing moments: {} for distribution: {}".format(fn_name, distr_name))

                    tol_exact_moments = 1e-6
                    tol_density_approx = 1e-6
                    quantiles = np.array([quantile])
                    n_failed = []
                    warn_log = []

                    for i_q, domain_quantile in enumerate(quantiles):
                        est_mlmc = mlmc.estimate.Estimate(mc_test.mc)
                        # domain, force_decay = self.domain_for_quantile(d, domain_quantile)
                        est_moments_obj, est_eigen_values, est_moments, L_estimate = self._estimated_eval_moments(mc_test.moments_fn,
                                                                                                                  mc_test.mc, threshold)

                        exact_moments_obj, exact_eigen_values, exact_moments, exact_cov, L_exact = self._exact_eval_moments(mc_test,
                                                                                                                      threshold,
                                                                                                                      density)

                        # moments_obj_L, eigen_values_L, est_moments, _ = self._estimated_eval_moments(exact_moments_obj,
                        #                                                                              mc_test.mc,
                        #                                                                              threshold,
                        #                                                                               #L_estimate
                        #                                                                              )

                        transform_moments_obj, transform_eigen_values, transform_moments = self._transform_eval_moments(
                                                                                                                        exact_moments_obj,
                                                                                                                        threshold,
                                                                                                                        density,
                            _mlmc=mc_test.mc

                        )

                        plots = self._plot_eval(est_eigen_values, exact_eigen_values, max_moments, colors, ax,
                                                                              transform_eval=transform_eigen_values,
                                                                              #L_eval=eigen_values_L,
                                                log=True)


                #         exact_moments_data = np.empty((len(exact_moments), 2))
                #         exact_moments_data[:, 0] = exact_moments[:]
                #         exact_moments_data[:, 1] = 1e-5
                #
                #         distr_obj_exact = SimpleDistribution(exact_moments_obj, exact_moments_data, domain=exact_moments_obj.domain)
                #         distr_obj_exact.estimate_density_minimize(tol_density_approx)  # 0.95 two side quantile
                #
                #         # self.plot_all_eigen_values(self._eigen_values)
                #         #
                #         # self._plot_eigen_moment_var(self._eigen_values, moments_data)
                #         #
                #         distr_plot = DistrPlot(d, distr_name + " " + fn_name)
                #         #distr_obj, result = self._estimate_distr(est_moments_obj, est_moments, tol_density_approx)
                #         distr_obj, result = self._estimate_distr(exact_moments_obj, exact_moments, tol_density_approx)
                #         domain = distr_obj.domain
                #
                #         n_failed.append(0)
                #         cumtime = 0
                #         tot_nit = 0
                #
                #         kl_div = 0
                #         min_err = 0
                #         # distr_obj = mlmc.distribution.Distribution(moments_obj, moments_data,
                #         #                                            domain=domain, force_decay=force_decay)
                #         t0 = time.time()
                #         #result = distr_obj.estimate_density(tol_exact_moments)
                #         #result = distr_obj.estimate_density_minimize(tol_density_approx)
                #         #result = profile(lambda : distr_obj.estimate_density_minimize(tol_exact_moments))
                #         t1 = time.time()
                #         cumtime += (t1 - t0)
                #         # Number of iterations
                #         nit = getattr(result, 'nit', result.njev)
                #         tot_nit += nit
                #         fn_norm = result.fun_norm
                #         if result.success:
                #
                #             kl_div = mlmc.simple_distribution.KL_divergence(d.pdf, distr_obj.density, distr_obj.domain[0],
                #                                                             distr_obj.domain[1])
                #             l2_dist = mlmc.simple_distribution.L2_distance(distr_obj.density, d.pdf, distr_obj.domain[0],
                #                                                             distr_obj.domain[1])
                #
                #             print("q: {}, m: {} :: nit: {} fn: {} ; kl: {} l2: {}".format(
                #                 domain_quantile, n_moments, nit, fn_norm, kl_div, l2_dist))
                #
                #             # plot for last case
                #
                #         else:
                #             n_failed[-1] += 1
                #             #print("q: {}, m: {} :: nit: {} fn:{} ; msg: {}".format(
                #             #    domain_quantile, n_moments, nit, fn_norm, result.message))
                #
                #             kl_div = mlmc.simple_distribution.KL_divergence(d.pdf, distr_obj.density,
                #                                                             distr_obj.domain[0],
                #                                                             distr_obj.domain[1])
                #             l2_dist = mlmc.simple_distribution.L2_distance(distr_obj.density, d.pdf,
                #                                                            distr_obj.domain[0],
                #                                                            distr_obj.domain[1])
                #
                #             # Check convergence
                #             #s1, s0 = np.polyfit(np.log(moment_size), np.log(kl_collected[i_q,:]), 1)
                #             max_err = kl_div#np.max(kl_collected[i_q,:])
                #             min_err = kl_div#np.min(kl_collected[i_q,:])
                #             if domain_quantile > 0.01:
                #                 continue
                #             if not (n_failed[-1] == 0 and (max_err < tol_density_approx * 8)):
                #                 warn_log.append((i_q, n_failed[-1], max_err))
                #                 fail = "FF"
                #             else:
                #                 fail = ' q'
                #             print(fail + ": ({:5.3g}, {:5.3g});  failed: {} cumit: {} tavg: {:5.3g}; kl: ({:5.3g}, {:5.3g})".format(
                #                 domain[0], domain[1], n_failed[-1], tot_nit, cumtime/tot_nit, min_err, max_err))
                #
                #         distr_values = {"moments": moment_size, "KL divergence": kl_div, "L2 dist": l2_dist,
                #                         "threshold": threshold}
                #
                #         # Add distribution values to csv file
                #         if csv is not None:
                #             csv.add_distribution_values(distr_values)
                #
                #         distr_plot.plot_approximation(distr_obj, label=str(domain))
                #
                #     kl_div_distr[moment_size] = kl_div
                #     l2_distr[moment_size] = l2_dist
                #
                # kl_div_all[distr_name] = kl_div_distr
                # l2_all[distr_name] = l2_distr

        # self._plot_kl_div(kl_div_all)
        # self._plot_l2_div(l2_all)
        #distr_plot.show()
        if plots is not None:
            self._display_eval_plot(plots, ax, config)

    def _moments_vars(self):
        """
        Test reconstruction of the density function from exact moments.
        - various distributions
        - various moments functions
        - test convergency with increasing number of moments
        :return:
        """
        #np.random.seed(3)  # To be reproducible
        n_levels = [1]  # [1, 2, 3, 5, 7]
        n_moments = [30]#, 30, 100]#, 20, 40, 60, 80]

        moment_size = 30
        threshold = 0
        plots = None

        distr = [
            #(stats.norm(loc=1, scale=2), False, '_sample_fn_no_error', "(loc=1, scale=2)"),
            # (stats.norm(loc=1, scale=10), False, '_sample_fn'),
            (stats.lognorm(scale=np.exp(5), s=1), True, '_sample_fn', "(scale=np.exp(5), s=1)"),# not from exact zero
            # (stats.lognorm(scale=np.exp(-3), s=2), True, '_sample_fn'),  # worse conv of higher moments
            #(stats.chi2(df=10), True, '_sample_fn', "(df=10)"),
             #(stats.chi2(df=5), True, '_sample_fn'),
            #WORSE (stats.weibull_min(c=0.5), False, '_sample_fn'),  # Exponential
            #(stats.weibull_min(c=1), False, '_sample_fn', "(c=1)"),  # Exponential
            # (stats.weibull_min(c=2), False, '_sample_fn'),  # Rayleigh distribution
            #(stats.weibull_min(c=5, scale=4), False, '_sample_fn', "(c=5, scale=4)"),  # close to normal
            # (stats.weibull_min(c=1.5), True, '_sample_fn'),  # Infinite derivative at zero
            # (stats.lognorm(scale=np.exp(-5), s=1), True, '_sample_fn_no_error'),
            #(stats.weibull_min(c=20), True, '_sample_fn_no_error', "(c=20)"),   # Exponential
            # (stats.weibull_min(c=20), True, '_sample_fn_no_error'),   # Exponential
            #(stats.weibull_min(c=3), True, '_sample_fn_no_error', "(c=3)")    # Close to normal
        ]

        # Save to csv file object
        csv = SaveCsv("/home/martin/Desktop/Distributions")
        quantile_list = [0.1, 0.01, 0.001, 0.0001]#, 0.0000001]

        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(1, 1, 1)
        color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b', 'g', 'r']
        colors = iter(color_list)
        max_moments = np.max(n_moments)

        for nl in n_levels:
            for d, log_flag, sim, params in distr:

                distr_name = d.dist.name + params
                # Add distribution to csv file object
                if csv is not None:
                    csv.add_distribution(distr_name)

                distr_fig_dir = os.path.join(csv.data_dir, distr_name)
                if not os.path.exists(distr_fig_dir):
                    os.makedirs(distr_fig_dir)

                config = {"threshold": threshold, "distr_name": distr_name, "moment_size": moment_size,
                          "color_list": color_list, "quantile_list": quantile_list}

                #for moment_size in n_moments:
                for quantile in quantile_list:
                    mc_test = fixtures.mlmc_test_run.TestMLMC(nl, moment_size, d, log_flag, sim, quantile=quantile)
                    # number of samples on each level
                    total_samples = mc_test.mc.sample_range(100000, 100)
                    #mc_test.mc.set_initial_n_samples()
                    mc_test.mc.target_var_adding_samples(1e-6, mc_test.moments_fn)
                    mc_test.generate_samples(total_samples)
                    #self._fig_file = os.path.join(distr_fig_dir, "{}_moments.png".format(moment_size))

                    fn_name = "Legendre"
                    distr_name = d.dist.name
                    density = lambda x: d.pdf(x)
                    print("Testing moments: {} for distribution: {}".format(fn_name, distr_name))
                    quantiles = np.array([quantile])

                    for i_q, domain_quantile in enumerate(quantiles):
                        est_mlmc = mlmc.estimate.Estimate(mc_test.mc)
                        # domain, force_decay = self.domain_for_quantile(d, domain_quantile)
                        estimated_moments = mc_test.mc.estimate_moments(mc_test.moments_fn)
                        exact_moments = mlmc.simple_distribution.compute_exact_moments(mc_test.moments_fn, density,
                                                                                       tol=1e-10)

                        moments_mean = []
                        moments_var = []
                        moments_mean.append(estimated_moments[0])
                        moments_var.append(estimated_moments[1])

                        x = np.arange(0, len(estimated_moments[0]))
                        x = x - 0.3
                        default_x = x
                        c = next(colors)

                        plt.plot(default_x, exact_moments, marker="^",  markersize=10, markerfacecolor='None', color=c,  linestyle='None')
                        plt.errorbar(x, estimated_moments[0], yerr=estimated_moments[1], fmt='o', capsize=3, color=c)

        import matplotlib.patches as mpatches
        import matplotlib.lines as mlines

        # # Axes title
        ax.set_title("Momenty, rozptyl = 1e-6, 1LMC")

        ax.set_yscale('symlog')
        ax.set_xscale('symlog')
        plt.xlabel("moment")
        triangle = mlines.Line2D([], [], color='black', marker='^', linestyle='None',
                                        markersize=10, markerfacecolor='None')
        circle = mlines.Line2D([], [], color='black', marker='o', linestyle='None')
        handles = []
        handles.append(triangle)
        handles.append(circle)

        labels = ['Přesné momenty', 'Odhadnuté momenty']
        for c, quantile in zip(color_list[0:len(quantile_list)], quantile_list):
            handles.append(mpatches.Patch(color=c))
            labels.append("{} kvantil".format(quantile))

        # Set some handlers legend color to black
        leg = plt.legend(handles, labels)
        #plt.legend()
        # plt.savefig("{}_moments_{}".format(moment_size, distr_name))
        plt.show()

    def _estimate_distr(self, moments_obj, moments, tol_density_approx):
        """
        Estimate distribution
        :param moments_obj: Moments object
        :param moments: Estimated moments, list
        :param tol_density_approx: Density approximation tolerance
        :return: tuple (distribution object, estimate density result)
        """
        if len(moments) > 2:
            moments_data = np.empty((len(moments), 2))
            moments_data[:, 0] = moments
            moments_data[:, 1] = 1e-7
        else:
            moments_data = np.empty((len(moments[0]), 2))
            moments_data[:, 0] = moments[0]
            moments_data[:, 1] = moments[1]

        distr_obj = SimpleDistribution(moments_obj, moments_data, domain=moments_obj.domain)
        result = distr_obj.estimate_density_minimize(tol_density_approx)  # 0.95 two side quantile
        return distr_obj, result

    def _display_eval_plot(self, plots, ax, config):
        """
        Display eigenvalues plots and change legend handlers and
        :param plots: Tuple of plot objects
        :param ax: Fig axes
        :param config: Dictionary with some plot params
        :return: None
        """
        import matplotlib.patches as mpatches
        import matplotlib.lines as mlines
        est_plot, exact_plot, transform_plot, L_plot = plots
        color_list = config.get('color_list')
        quantile_list = config.get('quantile_list')
        threshold = config.get('threshold')
        total_samples = config.get('total_samples')
        moment_size = config.get('moment_size')
        distr_name = config.get('distr_name')

        # Axes title
        ax.set_title("Vlastní čísla, prahová hodnota {}, N = {}, počet momentů = {}, rozdělení: {}".
                     format(threshold, total_samples, moment_size, distr_name))

        triangle = mlines.Line2D([], [], color='black', marker='^', linestyle='None',
                                 markersize=10, markerfacecolor='None')
        circle = mlines.Line2D([], [], color='black', marker='o', linestyle='None', markerfacecolor='None')

        handles = [est_plot, exact_plot]
        labels = ['Odhad cov matice', 'Přesná cov matice']
        if transform_plot is not None:
            handles.append(circle)
            labels.append("Transformovana matice")

        if L_plot is not None:
            handles.append(triangle)
            labels.append("Přesná matice L")

        for c, quantile in zip(color_list[0:len(quantile_list)], quantile_list):
            handles.append(mpatches.Patch(color=c))
            labels.append("{} kvantil".format(quantile))

        # Set some handlers legend color to black
        leg = plt.legend(handles, labels)
        leg.legendHandles[0].set_color('black')
        leg.legendHandles[1].set_color('black')
        leg.legendHandles[2].set_color('black')
        leg.legendHandles[3].set_color('black')

        #plt.savefig("{}_moments_{}".format(moment_size, distr_name))
        plt.show()

    def _estimated_eval_moments(self, moments_fn, mlmc, threshold):
        """
        Estimate eigenvalues and moments, values are estimated by MLMC
        :param mc_test: MLMC wrapper object
        :param threshold: Values below threshold are replaced by first value above threshold
        :return:
        """
        moments_obj, eval, L = self.construct_orthogonal_moments(moments_fn, mlmc,
                                                             threshold=threshold, plot=False)
        moments = mlmc.estimate_moments(moments_obj)
        return moments_obj, eval, moments, L

    def _exact_eval_moments(self, mc_test, threshold, density):
        """
        Compute exact eigenvalues and moments
        :param mc_test: MLMC wrapper object
        :param threshold: Values below threshold are replaced by first value above threshold
        :param density: Pdf of given distribution
        :return: Transform moments object, eigenvalues, exact moments, exact covariance matrix, matrix L
        """
        exact_cov = mlmc.simple_distribution.compute_exact_cov(mc_test.moments_fn, density)

        moments_object, eval, L = self.construct_orthogonal_moments(mc_test.moments_fn, exact_cov=exact_cov,
                                                                    threshold=threshold, plot=False)

        exact_moments = mlmc.simple_distribution.compute_exact_moments(moments_object, density, tol=1e-4)
        return moments_object, eval, exact_moments, exact_cov, L

    def _transform_eval_moments(self, moments_obj, threshold, density, _mlmc=None):
        """
        Eigenvalues from transform matrix
        :param moments_obj: Moments object, moments are estimated by mlmc
        :param threshold: Values below threshold are replaced by first value above threshold
        :return: Transform moments object, eigenvalues, estimated moments
        """
        #exact_cov = mlmc.simple_distribution.compute_exact_cov(moments_obj, density)

        moments_obj, eval, _ = self.construct_orthogonal_moments(moments_obj, mlmc_obj=_mlmc,
                                                                 threshold=threshold, plot=False)
        moments = 0#_mlmc.estimate_moments(est_moments_obj)
        return moments_obj, eval, moments

    def _plot_eval(self, est_eval, exact_eval, max_moments, colors, ax, transform_eval=None, L_eval=None, log=True):
        """
        Plot eigenvalues ...
        :param est_eval: Estimated eigenvalues
        :param exact_eval: Exact eigenvalues
        :param max_moments: Maximal number of moments
        :param colors: Iterator, gives color
        :param ax: Plot axis
        :param transform_eval: Eigenvalues from transform basis
        :param L_eval: Estimated eigenvalues with exact L matrix
        :param log: bool, logarithmic scale
        :return: None
        """

        X = np.arange(len(est_eval))
        X += (max_moments - len(est_eval))

        color = next(colors)
        a, b = np.min(est_eval), np.max(est_eval)

        if log:
            ax.set_yscale('log')
            ax.set_ylim(a / ((b / a) ** 0.05), 10**1)#b * (b / a) ** 0.05)
        else:
            ax.set_ylim(a - 0.05 * (b - a), b + 0.05 * (b - a))

        transform_plot = None
        L_plot = None

        est_plot, = ax.plot(X, est_eval, linestyle='-', color=color)
        exact_plot, = ax.plot(X, exact_eval, linestyle='--', color=color)
        if transform_eval is not None:
            transform_plot, = ax.plot(X, transform_eval, marker="o", markersize=10, markerfacecolor='None', color=color,
                                      linestyle='None')
        if L_eval is not None:
            L_plot, = ax.plot(X, L_eval, marker="^", markersize=10, markerfacecolor='None', color=color, linestyle='None')

        return est_plot, exact_plot, transform_plot, L_plot

    def _plot_kl_div(self, kl_div_all):
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(30, 10))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_yscale('log')
        ax.set_xlabel("moments")
        ax.set_ylabel("KL div")

        for lab, distr_kl in kl_div_all.items():
            _x = []
            _y = []
            for x, y in distr_kl.items():
                _x.append(x)
                _y.append(y)

            ax.scatter(_x, _y, label=lab)
            fig.legend()
        fig.show()
        plt.show()

    def _plot_l2_div(self, l2_all):
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(30, 10))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_yscale('log')
        ax.set_xlabel("moments")
        ax.set_ylabel("L2")

        for lab, l2_distr in l2_all.items():
            _x = []
            _y = []
            for x, y in l2_distr.items():
                _x.append(x)
                _y.append(y)

            ax.scatter(_x, _y, label=lab)
        fig.legend()
        fig.show()
        plt.show()


if __name__ == "__main__":
    test_estimate = TestEstimate()
    # test_estimate._moments_vars()
    # exit()
    test_estimate._test_pdf_mlmc()
    exit()
    moments_functions = [(moments.Legendre, 20)]#, (moments.Legendre, 10)]

    distributions = [
                (stats.norm(loc=1, scale=2), False),
                #(stats.norm(loc=1, scale=10), False),
                #(stats.lognorm(scale=np.exp(5), s=1), True)
                #(stats.lognorm(scale=np.exp(1), s=1), False),    # Quite hard but peak is not so small comparet to the tail.
                # #(stats.lognorm(scale=np.exp(-3), s=2), False),  # Extremely difficult to fit due to very narrow peak and long tail.
                #(stats.lognorm(scale=np.exp(-3), s=2), True),    # Still difficult for Lagrange with many moments.
                #(stats.chi2(df=10), False), #Monomial: s1=nan, Fourier: s1= -1.6, Legendre: s1=nan
                #(stats.chi2(df=5), True), #Monomial: s1=-10, Fourier: s1=-1.6, Legendre: OK
                # (stats.weibull_min(c=0.5), False),  # Exponential # Monomial stuck, Fourier stuck
                # (stats.weibull_min(c=1), False),  # Exponential
                # (stats.weibull_min(c=2), False),  # Rayleigh distribution
                # (stats.weibull_min(c=5, scale=4), False),   # close to normal
                # (stats.weibull_min(c=1.5), True),  # Infinite derivative at zero
            ]
    for distribution in distributions:
        for moments_fn, max_n_moments in moments_functions:
            test_estimate._test_pdf_approx_exact_moments(moments_fn, max_n_moments, distribution)