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
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/../src/')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mlmc.estimate
import mlmc.distribution
import mlmc.simple_distribution
from mlmc import moments
import test.benchmark_distributions as bd
import mlmc.tool.plot as plot
from test.fixtures.mlmc_test_run import MLMCTest
import mlmc.spline_approx as spline_approx
from mlmc.moments import Legendre
from textwrap import wrap

import pandas as pd


class CutDistribution:
    """
    Renormalization of PDF, CDF for exact distribution
    restricted to given finite inter_points_domain.
    """

    def __init__(self, distr, quantile):
        """

        :param distr: scipy.stat distribution object.
        :param quantile: float, define lower bound for approx. distr. inter_points_domain.
        """
        self.distr = distr
        self.quantile = quantile
        self.domain, self.force_decay = self.domain_for_quantile(distr, quantile)
        p0, p1 = distr.cdf(self.domain)
        self.shift = p0
        self.scale = 1 / (p1 - p0)

    @staticmethod
    def domain_for_quantile(distr, quantile):
        """
        Determine inter_points_domain from quantile. Detect force boundaries.
        :param distr: exact distr
        :param quantile: lower bound quantile, 0 = inter_points_domain from random sampling
        :return: (lower bound, upper bound), (force_left, force_right)
        """
        if quantile == 0:
            # Determine inter_points_domain by MC sampling.
            if hasattr(distr, "domain"):
                domain = distr.domain
            else:
                X = distr.rvs(size=1000)
                err = stats.norm.rvs(size=1000)
                #X = X * (1 + 0.1 * err)
                domain = (np.min(X), np.max(X))
            # p_90 = np.percentile(X, 99)
            # p_01 = np.percentile(X, 1)
            # inter_points_domain = (p_01, p_90)

            #inter_points_domain = (-20, 20)
        else:
            domain = distr.ppf([quantile, 1 - quantile])

        # Detect PDF decay on inter_points_domain boundaries, test that derivative is positive/negative for left/right side.
        eps = 1e-10
        force_decay = [False, False]
        for side in [0, 1]:
            diff = (distr.pdf(domain[side]) - distr.pdf(domain[side] - eps)) / eps
            if side:
                diff = -diff
            if diff > 0:
                force_decay[side] = True
        return domain, force_decay

    def pdf(self, x):
        return self.distr.pdf(x) * self.scale

    def cdf(self, x):
        return (self.distr.cdf(x) - self.shift) * self.scale

    def rvs(self, size=10):
        return self.distr.rvs(size)
        # x = np.random.uniform(0, 1, size)
        # print("self shift ", self.shift)
        # print("self scale ", self.scale)
        #return (self.distr.rvs(size) - self.shift) * self.scale


class ConvResult:
    """
    Results of a convergence calculation.
    """
    def __init__(self):
        self.size = 0
        # Moment sizes used
        self.noise = 0.0
        # Noise level used in Covariance and moments.
        self.kl = np.nan
        self.kl_2 = np.nan
        # KL divergence KL(exact, approx) for individual sizes
        self.l2 = np.nan
        # L2 distance of densities
        self.tv = np.nan
        # Total variation
        self.time = 0
        # times of calculations of density approx.
        self.nit = 0
        # number of iterations in minimization problems
        self.residual_norm = 0
        # norm of residual of non-lin solver
        self.success = False

    def __str__(self):
        return "#{} it:{} err:{} kl:{:6.2g} l2:{:6.2g} tv:{:6.2g}".format(self.size, self.nit, self.residual_norm,
                                                               self.kl, self.l2, self.tv)


class DistributionDomainCase:
    """
    Class to perform tests for fully specified exact distribution (distr. + inter_points_domain) and moment functions.
    Exact moments and other objects are created once. Then various tests are performed.
    """

    def __init__(self, moments, distribution, quantile):
        # Setup distribution.
        i_distr, distribution = distribution
        distr, log_flag = distribution
        self.log_flag = log_flag
        self.quantile = quantile

        if 'dist' in distr.__dict__:
            self.distr_name = "{:02}_{}".format(i_distr, distr.dist.name)
            self.cut_distr = CutDistribution(distr, quantile)
        else:
            self.distr_name = "{:02}_{}".format(i_distr, distr.name)
            self.cut_distr = CutDistribution(distr, 0)

        self.moments_data = moments
        moment_class, min_n_moments, max_n_moments, self.use_covariance = moments
        self.fn_name = str(moment_class.__name__)

        # domain_str = "({:5.2}, {:5.2})".format(*self.inter_points_domain)
        self.eigenvalues_plot = None

    @property
    def title(self):
        cov = "_cov" if self.use_covariance else ""
        return "distr: {} quantile: {} moment_fn: {}{}".format(self.distr_name, self.quantile, self.fn_name, cov)

    def pdfname(self, subtitle):
        return "{}_{}.pdf".format(self.title, subtitle)

    @property
    def domain(self):
        return self.cut_distr.domain

    def pdf(self, x):
        return self.cut_distr.pdf(x)

    # def create_correlation(self, cov):
    #     cov_diag = np.diag(cov)
    #     variable_std = np.sqrt(cov_diag)
    #     corr = np.eye(cov.shape[0])
    #
    #     for i in range(len(cov_diag)):
    #         for j in range(i+1):
    #             corr[j, i] = corr[i, j] = cov[i, j] / (variable_std[i]*variable_std[j])
    #
    #     return corr

    def setup_moments(self, moments_data, noise_level, reg_param=0, reg_param_beta=0):
        """
        Setup moments without transformation.
        :param moments_data: config tuple
        :param noise_level: magnitude of added Gauss noise.
        :return:
        """
        tol_exact_moments = 1e-6
        moment_class, min_n_moments, max_n_moments, self.use_covariance = moments_data
        log = self.log_flag
        if min_n_moments == max_n_moments:
            self.moment_sizes = np.array([max_n_moments])#[36, 38, 40, 42, 44, 46, 48, 50, 52, 54])+1#[max_n_moments])#10, 18, 32, 64])
        else:
            self.moment_sizes = np.round(np.exp(np.linspace(np.log(min_n_moments), np.log(max_n_moments), 8))).astype(int)
        #self.moment_sizes = [3,4,5,6,7]

        self.moments_fn = moment_class(max_n_moments, self.domain, log=log, safe_eval=False)

        if self.use_covariance:
            size = self.moments_fn.size
            base_moments = self.moments_fn

            # @TODO: remove regularization
            exact_cov, reg_matrix = mlmc.simple_distribution.compute_semiexact_cov_2(base_moments, self.pdf, reg_param=reg_param,
                                                                         reg_param_beta=reg_param_beta)

            self.moments_without_noise = exact_cov[:, 0]

            # Add regularization
            exact_cov += reg_matrix

            np.random.seed(1234)
            noise = np.random.randn(size**2).reshape((size, size))
            noise += noise.T
            noise *= 0.5 * noise_level
            noise[0, 0] = 0
            cov = exact_cov + noise
            moments = cov[:, 0]

            self.moments_fn, info, cov_centered = mlmc.simple_distribution.construct_orthogonal_moments(base_moments,
                                                                                                        cov,
                                                                                                        noise_level,
                                                                                                        reg_param=reg_param)
            self._cov_with_noise = cov
            self._cov_centered = cov_centered
            original_evals, evals, threshold, L = info
            self.L = L

            #eye_approx = L @ cov @ L.T
            # print("eye approx ")
            # print(pd.DataFrame(eye_approx))

            # print("np.linalg.norm(eye_approx - np.eye(*eye_approx.shape)) ", np.linalg.norm(eye_approx - np.eye(*eye_approx.shape)))
            # test that the decomposition is done well
            #assert np.linalg.norm(eye_approx - np.eye(*eye_approx.shape)) < 1e-9 # 1e-10 failed with Cauchy for more moments

            print("threshold: ", threshold, " from N: ", size)
            if self.eigenvalues_plot:
                threshold = evals[threshold]
                noise_label = "{:5.2e}".format(noise_level)
                self.eigenvalues_plot.add_values(evals, threshold=threshold, label=noise_label)

                # noise_label = "original evals, {:5.2e}".format(noise_level)
                # self.eigenvalues_plot.add_values(original_evals, threshold=threshold, label=noise_label)

            self.tol_density_approx = 0.01

            self.exact_moments = mlmc.simple_distribution.compute_semiexact_moments(self.moments_fn,
                                                                                      self.pdf, tol=tol_exact_moments)

        else:
            self.exact_moments = mlmc.simple_distribution.compute_semiexact_moments(self.moments_fn,
                                                                                    self.pdf, tol=tol_exact_moments)
            self.exact_moments += noise_level * np.random.randn(self.moments_fn.size)
            self.tol_density_approx = 1e-8

        return info, moments

    def check_convergence(self, results):
        # summary values
        sizes = np.log([r.size for r in results])
        kl = np.log([r.kl for r in results])
        sizes = sizes[~np.isnan(kl)]
        kl = kl[~np.isnan(kl)]
        n_failed = sum([not r.success for r in results])
        total_its = sum([r.nit for r in results])
        total_time = sum([r.time for r in results])
        if len(kl) > 2:
            s1, s0 = np.polyfit(sizes, kl, 1)
        max_err = np.max(kl)
        min_err = np.min(kl)

        # print
        print("CASE {} | failed: {} kl_decay: {} nit: {} time: {:3.1}".format(
            self.title, n_failed, s1, total_its, total_time))

        # check
        #assert n_failed == 0
        #assert s1 < -1
        # # Check convergence
        # if domain_quantile > 0.01:
        #     continue
        # if not (n_failed[-1] == 0 and (max_err < tol_density_approx * 8 or s1 < -1)):
        #     warn_log.append((i_q, n_failed[-1], s1, s0, max_err))
        #     fail = "FF"
        # else:
        #     fail = ' q'
        # print(
        #     fail + ": ({:5.3g}, {:5.3g});  failed: {} cumit: {} tavg: {:5.3g};  s1: {:5.3g} s0: {:5.3g} kl: ({:5.3g}, {:5.3g})".format(
        #         inter_points_domain[0], inter_points_domain[1], n_failed[-1], tot_nit, cumtime / tot_nit, s1, s0, min_err, max_err))

    def make_approx(self, distr_class, noise, moments_data, tol, reg_param=0, prior_distr_obj=None, reg_param_beta=0):
        result = ConvResult()

        distr_obj = distr_class(self.moments_fn, moments_data,
                                domain=self.domain, force_decay=self.cut_distr.force_decay, reg_param=reg_param, reg_param_beta=reg_param_beta)

        # multipliers = None
        # if prior_distr_obj is not None:
        #     multipliers = prior_distr_obj.multipliers
        #     distr_obj.reg_domain = [distr_obj.moments_fn.domain[0], 0]

        t0 = time.time()
        min_result = distr_obj.estimate_density_minimize(tol=tol, multipliers=None)

        moments = mlmc.simple_distribution.compute_semiexact_moments(self.moments_fn, distr_obj.density)

        print("moments approx error: ", np.linalg.norm(moments - self.exact_moments), "m0: ", moments[0])

        # result = profile(lambda : distr_obj.estimate_density_minimize(tol_exact_moments))
        t1 = time.time()
        result.size = moments_data.shape[0]
        result.noise = noise
        result.time = t1 - t0
        result.residual_norm = min_result.fun_norm
        result.success = min_result.success

        if result.success:
            result.nit = min_result.nit
        a, b = self.domain
        result.kl = mlmc.simple_distribution.KL_divergence(self.pdf, distr_obj.density, a, b)
        result.kl_2 = mlmc.simple_distribution.KL_divergence_2(self.pdf, distr_obj.density, a, b)
        result.l2 = mlmc.simple_distribution.L2_distance(self.pdf, distr_obj.density, a, b)
        result.tv = 0#mlmc.simple_distribution.total_variation_int(distr_obj.density_derivation, a, b)
        print(result)
        X = np.linspace(self.cut_distr.domain[0], self.cut_distr.domain[1], 10)
        density_vals = distr_obj.density(X)
        exact_vals = self.pdf(X)
        #print("vals: ", density_vals)
        #print("exact: ", exact_vals)
        return result, distr_obj

    # def mc_conv(self):
    #
    #     moments_size = 20
    #     n_samples = 100000
    #     reg_param = 0
    #     distr_tol = 1e-6
    #     moment_class, min_n_moments, max_n_moments, use_covariance = self.moments_data
    #     log = self.log_flag
    #
    #     distr_plot = plot.Distribution(exact_distr=self.cut_distr,
    #                                    title="Preconditioning reg, {},  n_moments: {}".format(self.title,moments_size),
    #                                    log_x=self.log_flag, error_plot=None, reg_plot=True, cdf_plot=False,
    #                                    log_density=True)
    #     eigenvalues_plot = plot.Eigenvalues(title="Eigenvalues, " + self.title)
    #
    #     ##########################################################################################
    #
    #     print("self domain ", self.domain)
    #     X = self.cut_distr.rvs(size=10000)
    #     true_domain = (np.min(X), np.max(X))
    #
    #     moments_fn = moment_class(moments_size, true_domain, log=log, safe_eval=False)
    #
    #     print("moments function ", moments_fn)
    #
    #     samples = self.cut_distr.rvs(size=n_samples)
    #
    #     print("samples ", samples)
    #
    #
    #     # Calculate moments and covariance with 'noise'
    #     moments = moments_fn(samples)
    #
    #     print("moments ", moments)
    #     var_moments = np.var(moments, axis=0, ddof=1)
    #     mean_moments = np.mean(moments, axis=0)
    #     print("moments mean: {}".format(mean_moments))
    #     print("moments vars: {}".format(var_moments))
    #
    #     exact_moments = mlmc.simple_distribution.compute_semiexact_moments(moments_fn, self.pdf)
    #
    #     print("exact moments: {}".format(exact_moments))
    #
    #     exit()
    #
    #     cov = np.matmul(moments.T, moments) / n_samples
    #
    #     print("cov")
    #     print(pd.DataFrame(cov))
    #     print("cov[:, 0] ", cov[:, 0])
    #
    #     ###########################################################################################
    #
    #     # Orthogonalize moments
    #     new_moments_fn, info = mlmc.simple_distribution.construct_orthogonal_moments(moments_fn, cov,
    #                                                                                  np.sqrt(var_moments),
    #                                                                                   reg_param=reg_param)
    #     original_evals, evals, threshold, L = info
    #
    #     eigenvalues_plot.add_values(evals, threshold=threshold)
    #
    #     #############################################################################################
    #
    #     # Transform moments
    #     trans_moments = new_moments_fn(samples)
    #     print("transformed moments ", trans_moments)
    #
    #
    #     trans_exact_moments = mlmc.simple_distribution.compute_semiexact_moments(new_moments_fn, self.pdf)
    #
    #     var_moments_trans = np.var(trans_moments, axis=0, ddof=1)
    #     mean_moments_trans = np.mean(trans_moments, axis=0)
    #     print("Transformed moments mean: {}, var: {}".format(mean_moments_trans, var_moments_trans))
    #
    #     print("trans exact moments: {}".format(trans_exact_moments))
    #
    #     cov_trans = np.matmul(trans_moments.T, trans_moments) / n_samples
    #     print("Trans cov")
    #
    #     print(cov_trans)
    #
    #     print(pd.DataFrame(cov_trans))
    #
    #     print("Trans cov[:, 0] ", cov_trans[:, 0])
    #
    #     moments_data = np.empty((new_moments_fn.size, 2))
    #     moments_data[:, 0] = trans_exact_moments#mean_moments_trans
    #     moments_data[:, 1] = 1#var_moments_trans
    #     moments_data[0, 1] = 1.0
    #
    #     ##################################################################################################
    #
    #     result = ConvResult()
    #
    #     distr_obj = mlmc.simple_distribution.SimpleDistribution(new_moments_fn, moments_data,
    #                             domain=self.domain, force_decay=self.cut_distr.force_decay, reg_param=reg_param)
    #
    #     min_result = distr_obj.estimate_density_minimize(tol=distr_tol)
    #
    #     distr_plot.add_distribution(distr_obj, label="trans mom var: {}, threshold: {}".format(var_moments_trans, threshold),
    #                                 size=moments_size, reg_param=reg_param)
    #
    #
    #
    #     # # result = profile(lambda : distr_obj.estimate_density_minimize(tol_exact_moments))
    #     # t1 = time.time()
    #     # result.size = transformed_moments.shape[0]
    #     # result.noise = np.sqrt(var_moments)
    #     # result.residual_norm = min_result.fun_norm
    #     # result.success = min_result.success
    #     #
    #     # if result.success:
    #     #     result.nit = min_result.nit
    #     # a, b = self.domain
    #     # result.kl = mlmc.simple_distribution.KL_divergence(self.pdf, distr_obj.density, a, b)
    #     # result.kl_2 = mlmc.simple_distribution.KL_divergence_2(self.pdf, distr_obj.density, a, b)
    #     # result.l2 = mlmc.simple_distribution.L2_distance(self.pdf, distr_obj.density, a, b)
    #     # result.tv = 0  # mlmc.simple_distribution.total_variation_int(distr_obj.density_derivation, a, b)
    #     # print(result)
    #     # X = np.linspace(self.cut_distr.domain[0], self.cut_distr.domain[1], 10)
    #     # density_vals = distr_obj.density(X)
    #     # exact_vals = self.pdf(X)
    #     # # print("vals: ", density_vals)
    #     # # print("exact: ", exact_vals)
    #
    #     eigenvalues_plot.show(None)
    #     distr_plot.show(None)

    def mlmc_conv(self, mc=None, distr_plot=None, moments_fn=None):
        #self.setup_moments(self.moments_data, noise_level=0)
        results = []
        kl_divergences = []
        n_levels = 1
        target_vars = [1e-6, 1e-5, 1e-4]
        target_vars = [1e-4]
        distr_accuracy = 1e-12
        mom_class, min_mom, max_mom, _ = self.moments_data

        log_flag = self.log_flag
        a, b = self.domain

        for target_var in target_vars:
            if distr_plot is None:
                distr_plot = plot.Distribution(exact_distr=self.cut_distr,
                                               title="Density, {},  n_moments: {}, target_var: {}".format(self.title,
                                                                                                          max_mom,
                                                                                                          target_var),
                                               log_x=self.log_flag, error_plot=None, reg_plot=True, cdf_plot=False,
                                               log_density=True)

            print("self moments data ", self.moments_data)

            mc_test = MLMCTest(n_levels, max_mom, self.cut_distr.distr, log_flag, "_sample_fn", moments_class=mom_class)
            # number of samples on each level
            mc_test.mc.set_initial_n_samples()
            mc_test.mc.refill_samples()
            mc_test.mc.wait_for_simulations()
            mc_test.mc.select_values({"quantity": (b"quantity_1", "="), "time": (0, "=")})
            estimator = mlmc.estimate.Estimate(mc_test.mc, mc_test.moments_fn)

            estimator.target_var_adding_samples(target_var, mc_test.moments_fn)
            mc = mc_test.mc

            mc_test.mc.update_moments(mc_test.moments_fn)
            means, vars = estimator.estimate_moments(mc_test.moments_fn)

            print("means ", means)
            print("vars ", vars)

            exact_moments = mlmc.simple_distribution.compute_exact_moments(mc_test.moments_fn, self.pdf)

            print("exact moments: {}".format(exact_moments))


            #reg_params = [1e-3, 1e-3*2, 1e-3*5, 1e-3*7]#[0, 1e-5, 1e-6, 1e-7]#[0, 1e-1, 1e-3]
            reg_params = [1e-5]

            for reg_param in reg_params:

                #expected_value = np.mean(means[:, 1])
                info = estimator.construct_density(tol=distr_accuracy, reg_param=reg_param, orth_moments_tol=target_var, exact_pdf=self.pdf)

                original_evals, evals, threshold, L = info

                mc0_samples = np.concatenate(mc.levels[0].sample_values[:, 0])

                distr_plot.add_distribution(estimator.distribution, label="reg param: {}, threshold: {}".format(reg_param, threshold),
                                            size=max_mom, reg_param=reg_param)

                distr_plot.add_raw_samples(mc0_samples)

                #plot_mom_indices = np.arange(1, max_mom, 1)
                #distr_plot.add_distribution(, size=, reg_param=reg_param)

                kl =mlmc.simple_distribution.KL_divergence(self.cut_distr.pdf, estimator.distribution.density, a, b)
                kl_divergences.append(kl)
                #l2 = mlmc.simple_distribution.L2_distance(self.cut_distr.pdf, estimator.distribution, a, b)

        print("kl divergences ", kl_divergences)

        distr_plot.show(None)
        #self._plot_kl_div(target_vars, kl_divergences)

        # fig = plt.figure()
        # ax = fig.add_subplot(1, 1, 1)
        # # ax.plot(noise_levels, tv, label="total variation")
        # ax.plot(target_vars, kl_divergences, 'o', c='r')
        # ax.set_xlabel("noise level")
        # ax.set_ylabel("KL divergence")
        # # ax.plot(noise_levels, l2, label="l2 norm")
        # # ax.plot(reg_parameters, int_density, label="abs(density-1)")
        # # ax.set_yscale('log')
        # ax.set_xscale('log')
        # ax.legend()
        # plt.show()

        # #self.check_convergence(results)
        # print("show pdf mlmc")
        # distr_plot.show(None)#=self.pdfname("_pdf_mlmc"))
        # distr_plot.reset()
        # #plt.show()
        return results

    def exact_conv(self):
        """
        Test density approximation for varying number of exact moments.
        :return:
        """
        results = []
        distr_plot = plot.Distribution(exact_distr=self.cut_distr, title=self.title+"_exact", cdf_plot=False,
                                            log_x=self.log_flag, error_plot='kl', multipliers_plot=True)

        mom_class, min_mom, max_mom, log_flag = self.moments_data
        moments_num = [max_mom]

        for i_m, n_moments in enumerate(moments_num):
            self.moments_data = (mom_class, n_moments, n_moments, log_flag)
            # Setup moments.
            self.setup_moments(self.moments_data, noise_level=0)

            if n_moments > self.moments_fn.size:
                continue
            # moments_fn = moment_fn(n_moments, inter_points_domain, log=log_flag, safe_eval=False )
            # print(i_m, n_moments, inter_points_domain, force_decay)
            moments_data = np.empty((n_moments, 2))
            moments_data[:, 0] = self.exact_moments[:n_moments]
            moments_data[:, 1] = 1.0

            if self.use_covariance:
                modif_cov = mlmc.simple_distribution.compute_exact_cov(self.moments_fn, self.pdf)
                diff_norm = np.linalg.norm(modif_cov - np.eye(*modif_cov.shape))
                print("#{} cov mat norm: {}".format(n_moments, diff_norm))

                result, distr_obj = self.make_approx(mlmc.simple_distribution.SimpleDistribution, 0.0, moments_data,
                                                     tol=1e-10)
            else:
                # TODO:
                # Use SimpleDistribution only as soon as it use regularization that improve convergency even without
                # cov matrix. preconditioning.
                result, distr_obj = self.make_approx(mlmc.distribution.Distribution, 0.0, moments_data)
            distr_plot.add_distribution(distr_obj, label="#{}".format(n_moments) +
                                                         "\n total variation {:6.2g}".format(result.tv))
            results.append(result)

        # mult_tranform_back = distr_obj.multipliers  # @ np.linalg.inv(self.L)
        # final_jac = distr_obj._calculate_jacobian_matrix(mult_tranform_back)

        final_jac = distr_obj.final_jac

        distr_obj_exact_conv_int = mlmc.simple_distribution.compute_exact_cov(distr_obj.moments_fn, distr_obj.density)
        M = np.eye(len(self._cov_with_noise[0]))
        M[:, 0] = -self._cov_with_noise[:, 0]

        # print("M @ L-1 @ H @ L.T-1 @ M.T")
        # print(pd.DataFrame(
        #     M @ (np.linalg.inv(self.L) @ final_jac @ np.linalg.inv(self.L.T)) @ M.T))
        #
        # print("orig cov centered")
        # print(pd.DataFrame(self._cov_centered))

        #self.check_convergence(results)
        #plt.show()
        distr_plot.show(None)#file=self.pdfname("_pdf_exact"))
        distr_plot.reset()

        #self._plot_kl_div(moments_num, [r.kl for r in results])
        #self._plot_kl_div(moments_num, [r.kl_2 for r in results])

        return results

    def _plot_kl_div(self, x, kl):

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        # ax.plot(noise_levels, tv, label="total variation")
        ax.plot(x, kl, 'o', c='r')
        #ax.set_xlabel("noise level")
        ax.set_xlabel("noise level")
        ax.set_ylabel("KL divergence")
        # ax.plot(noise_levels, l2, label="l2 norm")
        # ax.plot(reg_parameters, int_density, label="abs(density-1)")
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.legend()

        plt.show()

    def find_regularization_param(self):
        np.random.seed(1234)
        noise_level = 1e-2
        reg_params = np.linspace(1e-9, 1e-5, num=50)

        min_results = []
        print("reg params ", reg_params)

        #reg_params = np.logspace(-7, -5, num=40)
        print("log params ", reg_params)

        #reg_params = [1e-6]
        # print("geospace ", np.geomspace(1e-8, 1e-6, num=20))
        #
        # plt.plot(reg_params, np.zeros(25), 'o')
        # plt.show()

        moment_class, min_n_moments, max_n_moments, self.use_covariance = self.moments_data
        log = self.log_flag
        if min_n_moments == max_n_moments:
            self.moment_sizes = np.array(
                [max_n_moments])  # [36, 38, 40, 42, 44, 46, 48, 50, 52, 54])+1#[max_n_moments])#10, 18, 32, 64])
        else:
            self.moment_sizes = np.round(
                np.exp(np.linspace(np.log(min_n_moments), np.log(max_n_moments), 8))).astype(int)
        self.moments_fn = moment_class(max_n_moments, self.domain, log=log, safe_eval=False)

        _, _, n_moments, _ = self.moments_data
        # distr_plot = plot.Distribution(exact_distr=self.cut_distr,
        #                                title="Preconditioning reg, {},  n_moments: {}, noise: {}".format(self.title,
        #                                                                                                  n_moments,
        #                                                                                                  noise_level),
        #                                log_x=self.log_flag, error_plot=None, reg_plot=True, cdf_plot=False,
        #                                log_density=True)

        size = 100
        noises = []

        noise = np.random.randn(self.moments_fn.size ** 2).reshape((self.moments_fn.size, self.moments_fn.size))
        noise += noise.T
        noise *= 0.5 * noise_level
        noise[0, 0] = 0
        fine_noise = noise

        for i in range(size):
            noise = np.random.randn(self.moments_fn.size ** 2).reshape((self.moments_fn.size, self.moments_fn.size))
            noise += noise.T
            noise *= 0.5 * noise_level * 1.2
            noise[0, 0] = 0
            noises.append(noise)

        for reg_param in reg_params:
            exact_cov, reg_matrix = mlmc.simple_distribution.compute_semiexact_cov_2(self.moments_fn, self.pdf, reg_param=reg_param)

            self.exact_moments = exact_cov[:, 0]
            # Add noise and regularization
            cov = exact_cov + fine_noise[:len(exact_cov), :len(exact_cov)]
            cov += reg_matrix

            moments_with_noise = cov[:, 0]
            self.moments_fn, info, cov_centered = mlmc.simple_distribution.construct_orthogonal_moments(self.moments_fn,
                                                                                                        cov,
                                                                                                        noise_level,
                                                                                                       reg_param=reg_param)
            original_evals, evals, threshold, L = info
            fine_moments = np.matmul(moments_with_noise, L.T)

            moments_data = np.empty((len(fine_moments), 2))
            moments_data[:, 0] = fine_moments  # self.exact_moments
            moments_data[:, 1] = 1  # noise ** 2
            moments_data[0, 1] = 1.0

            _, distr_obj = self.make_approx(mlmc.simple_distribution.SimpleDistribution, noise, moments_data,
                                            tol=1e-8, reg_param=reg_param)

            estimated_density_covariance, reg_matrix = mlmc.simple_distribution.compute_semiexact_cov_2(self.moments_fn,
                                                                                                distr_obj.density)

            M = np.eye(len(cov[0]))
            M[:, 0] = -cov[:, 0]

            print("cov centered")
            print(pd.DataFrame(cov_centered))

            print("M-1 @ L-1 @ H @ L.T-1 @ M.T-1")
            print(pd.DataFrame(
                M @ (np.linalg.inv(L) @ distr_obj.final_jac @ np.linalg.inv(L.T)) @ M.T))

            print("cov")
            print(pd.DataFrame(cov))

            print("L-1 @ H @ L.T-1")
            print(pd.DataFrame(
                (np.linalg.inv(L) @ distr_obj.final_jac @ np.linalg.inv(L.T))))

            # print(pd.DataFrame(
            #     M @ (np.linalg.inv(L) @ estimated_density_covariance @ np.linalg.inv(L.T)) @ M.T))


            moments_from_density = np.linalg.inv(L) @ distr_obj.final_jac @ np.linalg.inv(L.T)[:, 0]

            print("moments from density ", moments_from_density)
            print("moments_with_noise ", moments_with_noise)

            estimated_density_exact_moments = estimated_density_covariance[:, 0]

            result = []
            for i in range(size):
                cov = exact_cov + noises[i][:len(exact_cov), :len(exact_cov)]
                coarse_moments = cov[:, 0]
                coarse_moments[0] = 1
                coarse_moments = np.matmul(coarse_moments, L.T)

                # _, distr_obj = self.make_approx(mlmc.simple_distribution.SimpleDistribution, noise, moments_data,
                #                                      tol=1e-7, reg_param=reg_param)
                #
                # estimate_density_exact_moments = mlmc.simple_distribution.compute_semiexact_moments(self.moments_fn,
                #                                                                                     distr_obj.density)

                res = (moments_from_density - coarse_moments)**2

                print("res to result ", res)
                result.append(res)

            # distr_plot.add_distribution(distr_obj,
            #                             label="noise: {}, threshold: {}, reg param: {}".format(noise_level, threshold,
            #                                                                                    reg_param),
            #                             size=len(coarse_moments), reg_param=reg_param)

            min_results.append(np.sum(result))

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        # ax.plot(noise_levels, tv, label="total variation")

        print("reg params ", reg_params)
        print("min results ", min_results)

        zipped = zip(reg_params, min_results)

        for reg_param, min_result in zip(reg_params, min_results):
            print("reg_param: {}, min_result: {}".format(reg_param, min_result))

        sorted_zip = sorted(zipped, key=lambda x: x[1])

        best_params = []
        #best_params.append(0)
        for s_tuple in sorted_zip:
            print(s_tuple)
            if len(best_params) < 5:
                best_params.append(s_tuple[0])

        ax.plot(reg_params, min_results, 'o', c='r')
        # ax.set_xlabel("noise level")
        ax.set_xlabel("regularization param (alpha)")
        ax.set_ylabel("min")
        # ax.plot(noise_levels, l2, label="l2 norm")
        # ax.plot(reg_parameters, int_density, label="abs(density-1)")
        #ax.set_yscale('log')
        #ax.set_xscale('log')
        ax.legend()

        plt.show()
        print("best params ", best_params)

        self.determine_regularization_param(best_params)

    def determine_regularization_param(self, reg_params=None):
        """
        Test density approximation for maximal number of moments
        and varying amount of noise added to covariance matrix.
        :return:
        """
        min_noise = 1e-6
        max_noise = 1e-2
        results = []

        _, _, n_moments, _ = self.moments_data
        distr_plot = plot.Distribution(exact_distr=self.cut_distr, title="Preconditioning reg, {},  n_moments: {}, noise: {}".format(self.title, n_moments, max_noise),
                                            log_x=self.log_flag, error_plot=None, reg_plot=False, cdf_plot=False, log_density=True)
        self.eigenvalues_plot = plot.Eigenvalues(title="Eigenvalues, " + self.title)

        geom_seq = np.exp(np.linspace(np.log(min_noise), np.log(max_noise), 20))
        noise_levels = np.flip(np.concatenate(([0.0], geom_seq)), axis=0)
        #noise_levels = geom_seq
        #print("noise levels ", noise_levels)

        noise_levels = noise_levels[:1]
        #noise_levels = [5.99484250e-02, 3.59381366e-01, 2.15443469e+00]
        #noise_levels = [3.59381366e-01]#, 1e-1]

        #noise_levels = [1e-3, 1e-2, 1e-1, 1e1, 1e2, 1e3]
        print("noise levels ", noise_levels)
        #exit()

        #noise_levels = [max_noise]
        #plot_mom_indices = np.array([0, 1, 2])
        plot_mom_indices = None

        kl_total = []
        kl_2_total = []
        l2 = []

        moments = []
        all_exact_moments = []

        for noise in noise_levels:
            kl_2 = []
            kl = []
            for _ in range(1):

                print("======================")
                print("noise ", noise)

                reg_parameters = [1e3*2] # functional reg param = 0
                reg_param_beta = 0

                reg_parameters = [0]

                # reg_parameters = [1.623776739188721e-07, 2.06913808111479e-07,
                #                   3.359818286283781e-07, 1.8329807108324375e-06,
                #                   2.6366508987303554e-07]

                # reg inv
                # reg_parameters = np.array([0, 0.0006, 0.0007])

                dir = self.title + "noise: ".format(noise)
                if not os.path.exists(dir):
                    os.makedirs(dir)

                tv = []
                # kl = []
                # l2 = []
                int_density = []

                if reg_params is not None:
                    reg_parameters = reg_params

                for reg_param in reg_parameters:
                    print("reg parameter ", reg_param)
                    info, moments_with_noise = self.setup_moments(self.moments_data, noise_level=noise, reg_param=reg_param,
                                                             reg_param_beta=reg_param_beta)
                    n_moments = len(self.exact_moments)

                    original_evals, evals, threshold, L = info
                    new_moments = np.matmul(moments_with_noise, L.T)

                    moments_data = np.empty((n_moments, 2))
                    moments_data[:, 0] = new_moments
                    moments_data[:, 1] = noise ** 2
                    moments_data[0, 1] = 1.0

                    print("moments data ", moments_data)

                    # modif_cov, reg_matrix = mlmc.simple_distribution.compute_semiexact_cov_2(self.moments_fn, self.pdf, reg_param=reg_param,
                    #                                                              reg_param_beta=reg_param_beta)
                    #
                    # modif_cov += reg_matrix
                    # # print("modif cov")
                    # # print(pd.DataFrame(modif_cov))
                    # # print("modif cov inv")
                    # # print(np.linalg.inv(pd.DataFrame(modif_cov)))
                    #
                    # diff_norm = np.linalg.norm(modif_cov - np.eye(*modif_cov.shape)) / n_moments
                    # ref_moments = np.zeros(n_moments)
                    # ref_moments[0] = 1.0

                    # print("ref moments ", ref_moments)
                    # mom_err = np.linalg.norm(self.exact_moments - ref_moments) / np.sqrt(n_moments)
                    # print("noise: {:6.2g} error of natural cov: {:6.2g} natural moments: {:6.2g}".format(
                    #     noise, diff_norm, mom_err))

                    # distr_plot = plot.Distribution(exact_distr=self.cut_distr, title="Density, " + self.title,
                    #                                      log_x=self.log_flag, error_plot='kl')
                    result, distr_obj = self.make_approx(mlmc.simple_distribution.SimpleDistribution, noise, moments_data,
                                                         tol=1e-8, reg_param=reg_param, reg_param_beta=reg_param_beta)

                    m = mlmc.simple_distribution.compute_exact_moments(self.moments_fn, distr_obj.density)
                    e_m = mlmc.simple_distribution.compute_exact_moments(self.moments_fn, self.pdf)
                    moments.append(m)
                    all_exact_moments.append(e_m)

                    # if reg_param > 0:
                    #     distr_obj._analyze_reg_term_jacobian([reg_param])

                    # result, distr_obj = self.make_approx(mlmc.simple_distribution.SimpleDistribution, noise, moments_data,
                    #                                      tol=1e-10, reg_param=reg_param, prior_distr_obj=distr_obj)

                    print("DISTR OBJ reg param {}, MULTIPLIERS {}".format(reg_param, distr_obj.multipliers))

                    distr_plot.add_distribution(distr_obj, label="noise: {}, threshold: {}, reg param: {}".format(noise, threshold, reg_param),
                                               size=n_moments, mom_indices=plot_mom_indices, reg_param=reg_param)

                    results.append(result)

                    final_jac = distr_obj.final_jac

                    print("ORIGINAL COV CENTERED")
                    print(pd.DataFrame(self._cov_centered))

                    M = np.eye(len(self._cov_with_noise[0]))
                    M[:, 0] = -self._cov_with_noise[:, 0]

                    print("M-1 @ L-1 @ H @ L.T-1 @ M.T-1")
                    print(pd.DataFrame(
                        np.linalg.inv(M) @ (
                                    np.linalg.inv(L) @ final_jac @ np.linalg.inv(L.T)) @ np.linalg.inv(M.T)))


                    tv.append(result.tv)
                    l2.append(result.l2)
                    kl.append(result.kl)
                    kl_2.append(result.kl_2)

                    distr_obj._update_quadrature(distr_obj.multipliers)
                    q_density = distr_obj._density_in_quads(distr_obj.multipliers)
                    q_gradient = distr_obj._quad_moments.T * q_density
                    integral = np.dot(q_gradient, distr_obj._quad_weights) / distr_obj._moment_errs

                    int_density.append(abs(sum(integral)-1))

            kl_total.append(np.mean(kl))
            kl_2_total.append(np.mean(kl_2))

                #distr_plot.show(file=os.path.join(dir, self.pdfname("reg_param_{}_pdf_iexact".format(reg_param))))
            #distr_plot.reset()

            # print("kl ", kl)
            # print("tv ", tv)
            # print("l2 ", l2)
            # print("density ", int_density)

        print("FINAL moments ", moments)
        print("exact moments ", all_exact_moments)

        for exact, estimated in zip(moments, all_exact_moments):
            print("(exact-estimated)**2", (exact-estimated)**2)
            print("sum(exact-estimated)**2", np.sum((exact - estimated) ** 2))

        distr_plot.show(file="determine_param {}".format(self.title))#file=os.path.join(dir, self.pdfname("_pdf_iexact")))
        distr_plot.reset()

        print("kl divergence", kl)

        self._plot_kl_div(noise_levels, kl_total)

        self.plot_gradients(distr_obj.gradients)
        #self._plot_kl_div(noise_levels, kl_2_total)
        plt.show()

        #self.check_convergence(results)
        #self.eigenvalues_plot.show(file=None)#self.pdfname("_eigenvalues"))

        return results

    def plot_gradients(self, gradients):
        print("gradients ", gradients)
        print("gradients LEN ", len(gradients))
        gradients = [np.linalg.norm(gradient) for gradient in gradients]

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(gradients)
        plt.show()

    def inexact_conv(self):
        """
        Test density approximation for maximal number of moments
        and varying amount of noise added to covariance matrix.
        :return:
        """
        min_noise = 1e-6
        max_noise = 0.01
        results = []

        distr_plot = plot.Distribution(exact_distr=self.cut_distr, title="", cdf_plot=False,
                                            log_x=self.log_flag, error_plot='kl')

        self.eigenvalues_plot = plot.Eigenvalues(title="Eigenvalues, " + self.title)

        geom_seq = np.exp(np.linspace(np.log(min_noise), np.log(max_noise), 5))
        noise_levels = np.flip(np.concatenate(([0.0], geom_seq)), axis=0)

        noise_levels = noise_levels[:1]

        print("noise levels ", noise_levels)
        # exit()
        # print("self moments data ", self.moments_data)
        # exit()

        mom_class, min_mom, max_mom, log_flag = self.moments_data

        moments_num = [5]#, 10, 20, 30]

        for noise in noise_levels:
            for m in moments_num:#np.arange(min_mom, max_mom, 5):

                for self.use_covariance in [True]:
                    print("self use covariance ", self.use_covariance)

                    self.moments_data = (mom_class, m, m, log_flag)
                    info, moments_with_noise = self.setup_moments(self.moments_data, noise_level=noise)

                    n_moments = len(self.exact_moments)

                    original_evals, evals, threshold, L = info

                    n_moments = len(self.exact_moments)
                    moments_data = np.empty((n_moments, 2))
                    moments_data[:, 0] = self.exact_moments[:n_moments]
                    moments_data[:, 1] = 1.0

                    print("moments data ", moments_data)

                    if self.use_covariance:
                        print("if use covariance ", self.use_covariance)

                        modif_cov = mlmc.simple_distribution.compute_semiexact_cov(self.moments_fn, self.pdf)

                        print("modif_cov ", modif_cov)

                        diff_norm = np.linalg.norm(modif_cov - np.eye(*modif_cov.shape)) / n_moments
                        ref_moments = np.zeros(n_moments)
                        ref_moments[0] = 1.0
                        mom_err = np.linalg.norm(self.exact_moments[:n_moments] - ref_moments) / np.sqrt(n_moments)
                        print("noise: {:6.2g} error of natural cov: {:6.2g} natural moments: {:6.2g}".format(
                            noise, diff_norm, mom_err))

                        #assert mom_err/(noise + 1e-10) < 50  - 59 for five fingers dist

                        result, distr_obj = self.make_approx(mlmc.simple_distribution.SimpleDistribution, noise, moments_data,
                                                             tol=1e-5)

                        distr_plot.add_distribution(distr_obj,
                                                    label="{} moments, {} threshold".format(n_moments, threshold))
                        results.append(result)

                    else:
                        print("without covariance")

                        print("moments data ", moments_data)

                        # TODO:
                        # Use SimpleDistribution only as soon as it use regularization that improve convergency even without
                        # cov matrix. preconditioning.
                        result, distr_obj = self.make_approx(mlmc.simple_distribution.SimpleDistribution, noise, moments_data, tol=1e-5)
                        distr_plot.add_distribution(distr_obj, label="{} moments".format(n_moments))
                        results.append(result)

        #self.check_convergence(results)
        self.eigenvalues_plot.show(None)#file=self.pdfname("_eigenvalues"))
        distr_plot.show(None)#"PDF aprox")#file=self.pdfname("_pdf_iexact"))
        distr_plot.reset()
        plt.show()
        return results


distribution_list = [
        # distibution, log_flag
        (stats.norm(loc=1, scale=2), False),
        (stats.norm(loc=1, scale=10), False),
        # (stats.lognorm(scale=np.exp(1), s=1), False),    # Quite hard but peak is not so small comparet to the tail.
        # #(stats.lognorm(scale=np.exp(-3), s=2), False),  # Extremely difficult to fit due to very narrow peak and long tail.
        # (stats.lognorm(scale=np.exp(-3), s=2), True),    # Still difficult for Lagrange with many moments.
        # (stats.chi2(df=10), False), # Monomial: s1=nan, Fourier: s1= -1.6, Legendre: s1=nan
        # (stats.chi2(df=5), True), # Monomial: s1=-10, Fourier: s1=-1.6, Legendre: OK
        # (stats.weibull_min(c=0.5), False),  # Exponential # Monomial stuck, Fourier stuck
        # (stats.weibull_min(c=1), False),  # Exponential
        # (stats.weibull_min(c=2), False),  # Rayleigh distribution
        # (stats.weibull_min(c=5, scale=4), False),   # close to normal
        # (stats.weibull_min(c=1.5), True),  # Infinite derivative at zero
    ]


#@pytest.mark.skip
@pytest.mark.parametrize("moments", [
    # moments_class, min and max number of moments, use_covariance flag
    #(moments.Monomial, 3, 10),
    #(moments.Fourier, 5, 61),
    #(moments.Legendre, 7, 61, False),
    (moments.Legendre, 7, 61, True),
    ])
@pytest.mark.parametrize("distribution", enumerate(distribution_list))
def test_pdf_approx_exact_moments(moments, distribution):
    """
    Test reconstruction of the density function from exact moments.
    - various distributions
    - various moments functions
    - test convergency with increasing number of moments
    :return:
    """
    quantiles = np.array([0.001])
    #quantiles = np.array([0.01])
    conv = {}
    # Dict of result matricies (n_quantiles, n_moments) for every performed kind of test.
    for i_q, quantile in enumerate(quantiles):
        np.random.seed(1234)

        case = DistributionDomainCase(moments, distribution, quantile)
        #tests = [case.exact_conv, case.inexact_conv]
        #tests = [case.mlmc_conv]
        #tests = [case.mc_conv]
        #tests = [case.exact_conv]
        #tests = [case.determine_regularization_param]
        tests = [case.find_regularization_param]
        for test_fn in tests:
            name = test_fn.__name__
            test_results = test_fn()
            values = conv.setdefault(name, (case.title, []))
            values[1].append(test_results)

    for key, values in conv.items():
        title, results = values
        title = "{}_conv_{}".format(title, key)
        if results[0] is not None:
            plot.plot_convergence(quantiles, results, title=title)

    # kl_collected = np.empty( (len(quantiles), len(moment_sizes)) )
    # l2_collected = np.empty_like(kl_collected)
    # n_failed = []
    # warn_log = []
    #
    #     kl_collected[i_q, :], l2_collected[i_q, :] = exact_conv(cut_distr, moments_fn, tol_exact_moments, title)
    #
    #
    # plot_convergence(moment_sizes, quantiles, kl_collected, l2_collected, title)
    #
    # #assert not warn_log
    # if warn_log:
    #     for warn in warn_log:
    #         print(warn)


# @pytest.mark.skip
# def test_distributions():
#     """
#     Plot densities and histogram for chosen distributions
#     :return: None
#     """
#     mlmc_list = []
#     # List of distributions
#     distributions = [
#         (stats.norm(loc=1, scale=2), False, '_sample_fn')
#         #(stats.lognorm(scale=np.exp(5), s=1), True, '_sample_fn'),  # worse conv of higher moments
#         # (stats.lognorm(scale=np.exp(-5), s=1), True, '_sample_fn_basic'),
#         #(stats.chi2(df=10), True, '_sample_fn')#,
#         # (stats.weibull_min(c=20), True, '_sample_fn'),   # Exponential
#         # (stats.weibull_min(c=1.5), True, '_sample_fn_basic'),  # Infinite derivative at zero
#         # (stats.weibull_min(c=3), True, '_sample_fn_basic')  # Close to normal
#          ]
#     levels = [1]#, 2, 3, 5, 7, 9]
#     n_moments = 10
#     # Loop through distributions and levels
#     for distr in distributions:
#         for level in levels:
#             mlmc_list.append(compute_mlmc_distribution(level, distr, n_moments))
#
#     fig = plt.figure(figsize=(30, 10))
#     ax1 = fig.add_subplot(1, 2, 1)
#     ax2 = fig.add_subplot(1, 2, 2)
#
#     n_moments = 5
#     # One level MC samples
#     mc0_samples = mlmc_list[0].mc.levels[0].sample_values[:, 0]
#     mlmc_list[0].ref_domain = (np.min(mc0_samples), np.max(mc0_samples))
#
#     # Plot densities according to TestMLMC instances data
#     for test_mc in mlmc_list:
#         test_mc.mc.clean_subsamples()
#         test_mc.mc.update_moments(test_mc.moments_fn)
#         inter_points_domain, est_domain, mc_test = mlmc.estimate.compute_results(mlmc_list[0], n_moments, test_mc)
#         mlmc.estimate.plot_pdf_approx(ax1, ax2, mc0_samples, mc_test, inter_points_domain, est_domain)
#     ax1.legend()
#     ax2.legend()
#     fig.savefig('compare_distributions.pdf')
#     plt.show()


def test_total_variation():
    function = lambda x: np.sin(x)
    lower_bound, higher_bound = 0, 2 * np.pi
    total_variation = mlmc.simple_distribution.total_variation_vec(function, lower_bound, higher_bound)
    tv = mlmc.simple_distribution.total_variation_int(function, lower_bound, higher_bound)

    assert np.isclose(total_variation, 4, rtol=1e-2, atol=0)
    assert np.isclose(tv, 4, rtol=1e-1, atol=0)

    function = lambda x: x**2
    lower_bound, higher_bound = -5, 5
    total_variation = mlmc.simple_distribution.total_variation_vec(function, lower_bound, higher_bound)
    tv = mlmc.simple_distribution.total_variation_int(function, lower_bound, higher_bound)

    assert np.isclose(total_variation, lower_bound**2 + higher_bound**2, rtol=1e-2, atol=0)
    assert np.isclose(tv, lower_bound ** 2 + higher_bound ** 2, rtol=1e-2, atol=0)

    function = lambda x: x
    lower_bound, higher_bound = -5, 5
    total_variation = mlmc.simple_distribution.total_variation_vec(function, lower_bound, higher_bound)
    tv = mlmc.simple_distribution.total_variation_int(function, lower_bound, higher_bound)
    assert np.isclose(total_variation, abs(lower_bound) + abs(higher_bound), rtol=1e-2, atol=0)
    assert np.isclose(tv, abs(lower_bound) + abs(higher_bound), rtol=1e-2, atol=0)


def plot_derivatives():
    function = lambda x: x
    lower_bound, higher_bound = -5, 5
    x = np.linspace(lower_bound, higher_bound, 1000)
    y = mlmc.simple_distribution.l1_norm(function, x)
    hubert_y = mlmc.simple_distribution.hubert_norm(function, x)

    plt.plot(x, y, '--')
    plt.plot(x, hubert_y, linestyle=':')
    plt.show()


def run_distr():
    distribution_list = [
        # distibution, log_flag
        # (stats.dgamma(1,1), False) # not good
        # (stats.beta(0.5, 0.5), False) # Looks great
        (bd.TwoGaussians(name='two_gaussians'), False),
        #(bd.FiveFingers(name='five_fingers'), False), # Covariance matrix decomposition failed
        #(bd.Cauchy(name='cauchy'), False),# pass, check exact
        #(bd.Gamma(name='gamma'), False) # pass
        #(stats.norm(loc=1, scale=2), False),
        #(stats.norm(loc=0, scale=10), False),
        #(stats.lognorm(scale=np.exp(1), s=1), False),    # Quite hard but peak is not so small comparet to the tail.
        #(stats.lognorm(scale=np.exp(-3), s=2), False),  # Extremely difficult to fit due to very narrow peak and long tail.
        # (stats.lognorm(scale=np.exp(-3), s=2), True),    # Still difficult for Lagrange with many moments.
        #(stats.chi2(df=10), False),# Monomial: s1=nan, Fourier: s1= -1.6, Legendre: s1=nan
        #(stats.chi2(df=5), True), # Monomial: s1=-10, Fourier: s1=-1.6, Legendre: OK
        #(stats.weibull_min(c=0.5), False),  # Exponential # Monomial stuck, Fourier stuck
        # (stats.weibull_min(c=1), False),  # Exponential
        #(stats.weibull_min(c=2), False),  # Rayleigh distribution
        #(stats.weibull_min(c=5, scale=4), False),   # close to normal
        # (stats.weibull_min(c=1.5), True),  # Infinite derivative at zero
    ]

    # @pytest.mark.skip
    mom = [
        # moments_class, min and max number of moments, use_covariance flag
        # (moments.Monomial, 3, 10),
        # (moments.Fourier, 5, 61),
        # (moments.Legendre, 7,61, False),
        (moments.Legendre, 25, 25, True),
        #(moments.Spline, 10, 10, True),
        #(moments.Legendre, 10, 15, True)
    ]

    for m in mom:
        for distr in enumerate(distribution_list):
            #compare_spline_vs_max_ent(distr, m)
            splines_indicator_vs_smooth(m, distr)
            #test_spline_approx(m, distr)

            #test_pdf_approx_exact_moments(m, distr)


def compare_spline_vs_max_ent(distr, moments):
    moments_class, n_moments, _, _ = moments
    target_var = 1e-4
    n_moments = 15
    n_levels = 1
    distr_accuracy = 1e-10

    quantiles = np.array([0.00])
    i_distr, distr_ = distr
    distribution, log_flag = distr_

    for quantile in quantiles:
        if 'dist' not in distribution.__dict__:
            quantile = 0

        print("distribution ", distribution)
        print("qunatile ", quantile)

        cut_distr = CutDistribution(distribution, quantile)
        #moments_fn = Legendre(n_moments, cut_distr.domain, log=log_flag, safe_eval=False)
        mc_test = MLMCTest(n_levels, n_moments, cut_distr.distr, log_flag, "_sample_fn", moments_class=moments_class)
        # number of samples on each level
        mc_test.mc.set_initial_n_samples()
        mc_test.mc.refill_samples()
        mc_test.mc.wait_for_simulations()
        mc_test.mc.select_values({"quantity": (b"quantity_1", "="), "time": (0, "=")})
        estimator = mlmc.estimate.Estimate(mc_test.mc, mc_test.moments_fn)

        estimator.target_var_adding_samples(target_var, mc_test.moments_fn)
        mc = mc_test.mc

        mc_test.mc.update_moments(mc_test.moments_fn)
        means, vars = estimator.estimate_moments(mc_test.moments_fn)

        print("means ", means)
        print("vars ", vars)

        expected_value = np.mean(means[:, 1])

        print("expected value ", expected_value)

        reg_param = 0#1e-1

        estimator.construct_density(tol=distr_accuracy, reg_param=reg_param)

        n_samples =[]
        for level in mc.levels:
            n_samples.append(level._n_collected_samples)

        print("N samples ", n_samples)

        int_points_domain = [-1000, 1000]
        polynomial_degree = 5  # r=3
        accuracy = 1e-4

        cut_distr.domain = mc_test.moments_fn.domain

        #X = np.linspace(cut_distr.inter_points_domain[0]-10, cut_distr.inter_points_domain[1]+10, 1000)
        X = np.linspace(cut_distr.domain[0]-10, cut_distr.domain[1]+10, 1000)

        distr_plot = plot.Distribution(exact_distr=cut_distr, title="Spline approximation",
                                       log_x=log_flag, error_plot='kl')

        mc0_samples = np.concatenate(mc.levels[0].sample_values[:, 0])
        distr_plot.add_raw_samples(mc0_samples)
        distr_plot.add_distribution(estimator.distribution)

        # conv = {}
        # case = DistributionDomainCase(moments, distr, quantile)
        # name = case.mlmc_conv.__name__
        # test_results = case.mlmc_conv(mlmc, distr_plot)
        # values = conv.setdefault(name, (case.title, []))
        # values[1].append(test_results)

        #distr_obj = make_spline_approx(int_points_domain, mc, polynomial_degree, distr_accuracy, bspline=False)

        #_test_interpolation_points(cut_distr, distr_obj, moments_fn, X, n_samples, accuracy)

        #_test_polynomial_degrees(cut_distr, distr_obj, mc_test.moments_fn, X, n_samples, distr_accuracy, log_flag, distr_plot)

        distr_plot.show(None)


def run_mlmc(n_levels, n_moments, cut_distr, log_flag, quantile, moments_fn, target_var):
    mc_test = MLMCTest(n_levels, n_moments, cut_distr, log_flag, sim_method='_sample_fn', quantile=quantile)

    mc_test.moments_fn = moments_fn

    #estimator = mlmc.estimate.Estimate(mc_test.mc)
    mc_test.mc.set_initial_n_samples()#[500000])#[10000, 2000, 500, 50])
    mc_test.mc.refill_samples()
    mc_test.mc.wait_for_simulations()
    mc_test.mc.select_values({"quantity": (b"quantity_1", "="), "time": (1, "<")})
    mc_test.estimator.target_var_adding_samples(target_var, moments_fn, sleep=0)
    mc_test.mc.wait_for_simulations()

    mc_test.mc.update_moments(mc_test.moments_fn)
    #
    # moments_mean, moments_var = estimator.estimate_moments(mc_test.moments_fn)
    # moments_mean = np.squeeze(moments_mean)
    # moments_var = np.squeeze(moments_var)
    #
    # print("moments mean ", moments_mean)
    # print("moments var ", moments_var)

    return mc_test.mc


def _test_interpolation_points(cut_distr, distr_obj, moments_fn,  X, n_samples, accuracy):
    interpolation_points = [5, 10, 15]#, 10, 20, 30]#, 15, 20, 25, 30, 35]
    for n_int_points in interpolation_points:
        distribution = distr_obj.mlmc_cdf(X, moments_fn, "smooth", int_points=n_int_points)
        mask = distr_obj.mask
        plt.plot(X[mask], distribution, linestyle="-", label="{}".format(n_int_points))

    plt.plot(X, cut_distr.distr.cdf(X), linestyle="--", label="exact")
    plt.title("Compare interpolation points, MLMC smoothing \n MLMC samples: {} \n accuracy: ".format(n_samples, accuracy))
    plt.legend()
    plt.show()
    exit()


def splines_indicator_vs_smooth(m, distr):
    np.random.seed(1234)
    quantiles = np.array([0.001])
    # i_distr, distr = distr
    # distribution, log_flag = distr
    n_levels = 1
    n_moments = 2

    target_var = 1e-5

    for quantile in quantiles:

        spline_plot = plot.Spline_plot(bspline=True)

        distr_domain_case = DistributionDomainCase(m, distr, quantile)

        i_distr, distribution = distr
        distr, log_flag = distribution

        distr = distr_domain_case.cut_distr.distr  # CutDistribution(distribution, quantile)
        cut_distr = distr_domain_case.cut_distr

        moments_fn = Legendre(n_moments, distr_domain_case.cut_distr.domain, log=log_flag, safe_eval=True)
        mlmc = run_mlmc(n_levels, n_moments, cut_distr.distr, log_flag, quantile, moments_fn, target_var=target_var)

        n_samples = []
        for level in mlmc.levels:
            n_samples.append(level._n_collected_samples)

        int_points_domain = cut_distr.domain

        interpolation_points = 5
        polynomial_degree = 3  # r=3
        accuracy = 1e-6

        # X = np.linspace(cut_distr.inter_points_domain[0]-10, cut_distr.inter_points_domain[1]+10, 1000)
        X = np.linspace(cut_distr.domain[0], cut_distr.domain[1], 1000)

        distr_obj = make_spline_approx(int_points_domain, mlmc, polynomial_degree, accuracy)

        interpolation_points = [10, 20, 30, 50]

        spline_plot.interpolation_points = interpolation_points

        for n_int_points in interpolation_points:
            distr_obj = make_spline_approx(int_points_domain, mlmc, polynomial_degree, accuracy)
            distr_obj.moments_fn = moments_fn
            distr_obj.indicator_method_name = "indicator"
            distr_obj.n_interpolation_points = n_int_points
            cdf = distr_obj.cdf(X)
            spline_plot.add_indicator((X[distr_obj.distr_mask], cdf))

            distr_obj = make_spline_approx(int_points_domain, mlmc, polynomial_degree, accuracy)
            distr_obj.moments_fn = moments_fn
            distr_obj.indicator_method_name = "smooth"
            distr_obj.n_interpolation_points = n_int_points
            cdf = distr_obj.cdf(X)
            spline_plot.add_smooth((X[distr_obj.distr_mask], cdf))

            distr_obj = make_spline_approx(int_points_domain, mlmc, polynomial_degree, accuracy, bspline=True)
            distr_obj.moments_fn = moments_fn
            distr_obj.n_interpolation_points = n_int_points
            cdf = distr_obj.cdf(X)
            spline_plot.add_bspline((X, cdf))

        spline_plot.add_exact_values(X, cut_distr.distr.cdf(X))
        from statsmodels.distributions.empirical_distribution import ECDF
        level = mlmc.levels[0]
        moments = level.evaluate_moments(moments_fn)
        fine_values = np.squeeze(moments[0])[:, 1]
        fine_values = moments_fn.inv_linear(fine_values)
        ecdf = ECDF(fine_values)
        spline_plot.add_ecdf(X, ecdf(X))

        spline_plot.show()


def _test_polynomial_degrees(cut_distr, distr_obj, moments_fn,  X, n_samples, accuracy, log_flag, distr_plot=None, bspline=False, mlmc=None):
    polynomial_degrees = [5]#, 5, 7, 9, 15]#, 10, 20, 30]#, 15, 20, 25, 30, 35]
    n_int_points = 300#1250#1250#500

    if mlmc is not None:
        from statsmodels.distributions.empirical_distribution import ECDF
        level = mlmc.levels[0]
        moments = level.evaluate_moments(moments_fn)
        fine_values = np.squeeze(moments[0])[:, 1]
        fine_values = moments_fn.inv_linear(fine_values)

        print("fine values ", fine_values)
        print("fine values len ", len(fine_values))

        print("fine values sorted ", np.sort(fine_values))


    #interpolation_points = [100, 120, 140, 160]
    interpolation_points = [10, 20, 30]
    if not bspline:
        interpolation_points = [10, 20, 30]
        #interpolation_points = [1250, 1350, 1450]
    #interpolation_points = [500]#[700, 900, 1000, 1200, 1500]

    distr_obj.moments_fn = moments_fn
    distr_obj.indicator_method_name = "indicator"
    distr_obj.n_interpolation_points = n_int_points

    # density = False
    # for index, poly_degree in enumerate(polynomial_degrees):
    #     int_point = interpolation_points[0]
    #     distr_obj.n_interpolation_points = int_point
    #     distr_obj.poly_degree = poly_degree
    #     col = 'C{}'.format(index)
    #
    #     if density:
    #         distribution = distr_obj.density(X)
    #     else:
    #         # distribution, _ = distr_obj.cdf_pdf(X)
    #         # distribution = distr_obj.cdf(X)
    #         distribution = distr_obj.cdf(X)
    #
    #     print("distr_obj.distr_mask ", distr_obj.distr_mask)
    #     distr_obj.mask = None
    #     print("distr_obj.mask  ", distr_obj.mask)
    #     if distr_obj.distr_mask is not None or distr_obj.mask is not None:
    #         if distr_obj.distr_mask is not None:
    #             mask = distr_obj.distr_mask
    #         else:
    #             mask = distr_obj.mask
    #         plt.plot(X[mask], distribution, "--", color=col, label="{}, KS test: {}".format(poly_degree,
    #                                                                                         stats.kstest(distribution,
    #                                                                                                      cut_distr.distr.cdf,
    #                                                                                                      )))
    #     else:
    #         plt.plot(X, distribution, "--", color=col, label="{}, ".format(poly_degree))
    #
    #         # plt.plot(X, distribution, "--", color=col, label="{}, KS test: {}".format(int_point,
    #         #                                                                           stats.kstest(distribution,
    #         #                                                                                        cut_distr.distr.cdf,
    #         #                                                                                        )))
    #
    # if density:
    #     plt.plot(X, cut_distr.distr.pdf(X), color='C{}'.format(index + 1), linestyle="--", label="exact")
    # else:
    #     plt.plot(X, cut_distr.distr.cdf(X), color='C{}'.format(index + 1), linestyle="--", label="exact")

    density = False
    for index, int_point in enumerate(interpolation_points):

        distr_obj.n_interpolation_points = int_point

        col = 'C{}'.format(index)

        if density:
            distribution = distr_obj.density(X)
        else:
            # distribution, _ = distr_obj.cdf_pdf(X)
            #distribution = distr_obj.cdf(X)
            distribution = distr_obj.cdf(X)

        print("distr_obj.distr_mask ", distr_obj.distr_mask)
        if distr_obj.distr_mask is not None:
            distr_obj.mask = None
        #distr_obj.mask = None
        #print("distr_obj.mask  ", distr_obj.mask)
        if distr_obj.distr_mask is not None or distr_obj.mask is not None:
            if distr_obj.distr_mask is not None:
                mask = distr_obj.distr_mask
            else:
                mask = distr_obj.mask
            plt.plot(X[mask], distribution, "--", color=col, label="{} ".format(int_point))
                                                                                            # stats.kstest(distribution, cut_distr.distr.cdf,
                                                                                            #              )))
        else:
            plt.plot(X, distribution, "--", color=col, label="{}, ".format(int_point))

            # plt.plot(X, distribution, "--", color=col, label="{}, KS test: {}".format(int_point,
            #                                                                           stats.kstest(distribution,
            #                                                                                        cut_distr.distr.cdf,
            #                                                                                        )))

    if density:
        plt.plot(X, cut_distr.distr.pdf(X), color='C{}'.format(index+1), label="exact")
    else:
        plt.plot(X, cut_distr.distr.cdf(X), color='C{}'.format(index+1), label="exact")
        # ecdf = ECDF(fine_values)
        # plt.plot(X, ecdf(X), label="ECDF")

    # density = False
    # for poly_degree in polynomial_degrees:
    #     distr_obj.poly_degree = poly_degree
    #
    #     if density:
    #         distribution = distr_obj.density(X)
    #     else:
    #         #distribution, _ = distr_obj.cdf_pdf(X)
    #         distribution = distr_obj.cdf(X)
    #
    #     if distr_obj.distr_mask is not None:
    #         mask = distr_obj.distr_mask
    #         plt.plot(X[mask], distribution, "r:", label="{}".format(poly_degree))
    #     else:
    #         plt.plot(X, distribution, "r:", label="{}".format(poly_degree))
    #
    # if density:
    #     plt.plot(X, cut_distr.distr.pdf(X), linestyle="--", label="exact")
    # else:
    #     plt.plot(X, cut_distr.cdf(X), linestyle="--", label="exact")

    #plt.xlim(-35, 35)

    print("distr obj interpolation points ", distr_obj.interpolation_points)
    #plt.plot(distr_obj.interpolation_points, np.ones(len(distr_obj.interpolation_points)), ":")

    print("cut_distr.cdf(X) ", cut_distr.cdf(X))
    print("approx distribution ", distribution)
    plt.title("Compare smoothing polynomial degrees \n MLMC with smoothing, BSpline={},  samples: {} \n accuracy: {} \n n_inter_points: {} \n inter_points_domain: {} ".format(bspline, n_samples,
                                                                                                                        accuracy, int_point, distr_obj.inter_points_domain))
    plt.legend()
    plt.show()

    exit()


def test_spline_approx(m, distr):
    np.random.seed(1234)
    quantiles = np.array([0.001])
    #i_distr, distr = distr
    #distribution, log_flag = distr
    n_levels = 5
    n_moments = 2

    target_var = 1e-5

    bspline = False

    for quantile in quantiles:
        print("distr ", distr)
        print("qunatile ", quantile)

        distr_domain_case = DistributionDomainCase(m, distr, quantile)

        i_distr, distribution = distr
        distr, log_flag = distribution

        distr = distr_domain_case.cut_distr.distr#CutDistribution(distribution, quantile)
        cut_distr = distr_domain_case.cut_distr

        moments_fn = Legendre(n_moments, distr_domain_case.cut_distr.domain, log=log_flag, safe_eval=True)
        mlmc = run_mlmc(n_levels, n_moments, cut_distr.distr, log_flag, quantile, moments_fn, target_var=target_var)

        n_samples = []
        for level in mlmc.levels:
            n_samples.append(level._n_collected_samples)

        print("N samples ", n_samples)

        int_points_domain = cut_distr.domain

        print("cut distr domain ", cut_distr.domain)

        #if not bspline:
        # int_points_domain = [0, 0]
        # int_points_domain[0] = cut_distr.domain[0] - 100
        # int_points_domain[1] = cut_distr.domain[1] + 100
            #[-500, 500]
            #int_points_domain = [-30, 30]
        #inter_points_domain = [-50, 50] # not good

        # Remove data standardisation
        #moments_fn.ref_domain = cut_distr.inter_points_domain
        # moments_fn = Legendre(2, cut_distr.inter_points_domain, safe_eval=True, log=log_flag)
        # print("moments_fn.inter_points_domain ", moments_fn.inter_points_domain)
        #
        # moments = moments_fn.eval_all(data)
        # data = moments[:, 1]

        print("int points domain ", int_points_domain)

        interpolation_points = 5
        polynomial_degree = 3  # r=3
        accuracy = 1e-6

        #X = np.linspace(cut_distr.inter_points_domain[0]-10, cut_distr.inter_points_domain[1]+10, 1000)
        X = np.linspace(cut_distr.domain[0], cut_distr.domain[1], 1000)

        #X = np.linspace(int_points_domain[0]+10, int_points_domain[1]-10, 1000)

        # mlmc_1 = run_mlmc(1, n_moments, cut_distr, log_flag, quantile, moments_fn)
        # distr_obj = make_spline_approx(cut_distr, mlmc_1, polynomial_degree, accurency)
        # distribution = distr_obj.cdf(X, cut_distr.distr.rvs(100))
        # mask = distr_obj.mask
        # plt.plot(X[mask], distribution, linestyle="-", label="MC without smoothing")

        #mlmc_1 = run_mlmc(1, n_moments, cut_distr, log_flag, quantile, moments_fn)
        # distr_obj = make_spline_approx(cut_distr, mlmc_1, polynomial_degree, accuracy)
        # distribution = distr_obj.mlmc_cdf(X, moments_fn, "indicator", int_points=interpolation_points)
        # mask = distr_obj.mask
        # plt.plot(X[mask], distribution, linestyle="-", label="MC without smoothing")
        # print("Kolmogorov-Smirnov test, 1LMC", stats.kstest(cut_distr.distr.rvs, distr_obj.cdf))
        # #
        distr_obj = make_spline_approx(int_points_domain, mlmc, polynomial_degree, accuracy, bspline=bspline)

        #_test_interpolation_points(cut_distr, distr_obj, moments_fn, X, n_samples, accuracy)
        _test_polynomial_degrees(cut_distr, distr_obj, moments_fn, X, n_samples, accuracy, log_flag, bspline=bspline, mlmc=mlmc)

        # distribution = distr_obj.mlmc_cdf(X, moments_fn, "indicator", int_points=interpolation_points)
        # mask = distr_obj.mask
        # plt.plot(X[mask], distribution, linestyle="-", label="MLMC without smoothing")
        # print("Kolmogorov-Smirnov test, MLMC without smoothing", stats.kstest(cut_distr.distr.rvs, distr_obj.cdf))
        #
        # distr_obj = make_spline_approx(cut_distr, mlmc, polynomial_degree, accuracy)
        # distribution = distr_obj.mlmc_cdf(X, moments_fn, "smooth", int_points=interpolation_points)
        # mask = distr_obj.mask
        # plt.plot(X[mask], distribution, linestyle="-", label="MLMC with smoothing")
        # print("Kolmogorov-Smirnov test, MLMC with smoothing ", stats.kstest(cut_distr.distr.rvs, distr_obj.cdf))

        #print("len interpolation points ", len(distr_obj.interpolation_points))

        #plt.plot(distr_obj.interpolation_points, np.ones(len(distr_obj.interpolation_points)) * 0.5, linestyle=":")

        # plt.title("\n".join(wrap("Distribution, interpolation points: {}, accuracy: {}, polynomial degree: {}, n evaluation points: {}".
        #           format(interpolation_points, accuracy, polynomial_degree, len(X)))))


        #plt.plot(X, distribution, linestyle="--", label="MLMC without smoothing")
        #X = np.linspace(-1, 1, 500)

        #distr_sorted, mask = distr_obj.cdf(X)


        # distribution = distr_obj.cdf(X)
        # mask = distr_obj.mask
        # plt.plot(X[mask], distribution, linestyle="--", label="without smoothing")
        #plt.plot(X, distribution, linestyle="--", label="approx")

        # distr_obj = make_spline_approx(cut_distr, data)
        # distribution = distr_obj.cdf_smoothing(X)
        # mask = distr_obj.mask
        # plt.plot(X[mask], distribution, linestyle="--", label="with smoothing")
        # plt.plot(X, cut_distr.distr.cdf(X), linestyle="--", label="exact")
        # plt.legend()
        # plt.show()
        #
        # print()


def make_spline_approx(domain, mlmc, polynomial_degree=7, accuracy=0.01, bspline=False):
    if bspline is False:
        spline_approx_instance = spline_approx.SplineApproximation(mlmc, domain, poly_degree=polynomial_degree,
                                                                   accuracy=accuracy)
    else:
        spline_approx_instance = spline_approx.BSplineApproximation(mlmc, domain, poly_degree=polynomial_degree,
                                                                    accuracy=accuracy)
    return spline_approx_instance


    # a, b = cut_distr.inter_points_domain
    # result.kl = mlmc.simple_distribution.KL_divergence(cut_distr.distr.pdf, distr_obj.density, a, b)
    # result.l2 = mlmc.simple_distribution.L2_distance(cut_distr.distr.pdf, distr_obj.density, a, b)
    # result.tv = mlmc.simple_distribution.total_variation_int(distr_obj.density_derivation, a, b)
    # print(result)
    # X = np.linspace(cut_distr.inter_points_domain[0], cut_distr.inter_points_domain[1], 10)
    # density_vals = distr_obj.density(X)
    # exact_vals = cut_distr.distr.pdf(X)
    # #print("vals: ", density_vals)
    # #print("exact: ", exact_vals)
    # return result, distr_obj


if __name__ == "__main__":
    # import scipy as sc
    # sc.linalg.norm([1], 2)

    #plot_derivatives()
    #test_total_variation()

    # import time as t
    # zacatek = t.time()
    #run_distr()
    # print("celkový čas ", t.time() - zacatek)

    import cProfile
    import pstats
    pr = cProfile.Profile()
    pr.enable()

    my_result = run_distr()

    pr.disable()
    ps = pstats.Stats(pr).sort_stats('cumtime')
    ps.print_stats()
