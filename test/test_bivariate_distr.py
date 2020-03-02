from scipy.stats import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt


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
import mlmc.bivariate_simple_distr
from mlmc import moments
import test.benchmark_distributions as bd
import mlmc.tool.plot as plot
from test.fixtures.mlmc_test_run import MLMCTest
import mlmc.spline_approx as spline_approx
from mlmc.moments import Legendre, BivariateMoments
from textwrap import wrap

import pandas as pd


class CutDistribution:
    """
    Renormalization of PDF, CDF for exact distribution
    restricted to given finite domain.
    """

    def __init__(self, distr, quantile):
        """

        :param distr: scipy.stat distribution object.
        :param quantile: float, define lower bound for approx. distr. domain.
        """
        self.distr = distr
        self.quantile = quantile
        self.domain, self.force_decay = self.domain_for_quantile(distr, quantile)

        print("self domain ", self.domain)
        p0, p1 = distr.cdf(self.domain)

        self.shift = p0
        self.scale = 1 / (p1 - p0)

    @staticmethod
    def domain_for_quantile(distr, quantile):
        """
        Determine domain from quantile. Detect force boundaries.
        :param distr: exact distr
        :param quantile: lower bound quantile, 0 = domain from random sampling
        :return: (lower bound, upper bound), (force_left, force_right)
        """
        # print("quantile ", quantile)
        # print("distr ", distr)
        # data = distr.rvs(size=100000)
        # X = data[:, 0]
        # Y = data[:, 1]
        # print("X ", X)
        # print("len X ", len(X))
        #
        # p_90 = np.percentile(X, 99)
        # p_01 = np.percentile(X, 1)
        # x_domain = (p_01, p_90)
        # print("x_domain ", x_domain)
        #
        # quantile_x_domain = (np.quantile(X, .01), np.quantile(X, .99))
        #
        # print("quantile x_domain ", quantile_x_domain)
        #
        # quantile_x_domain = (np.quantile(X, .001), np.quantile(X, .999))
        #
        # print("quantile x_domain ", quantile_x_domain)
        #
        # p_90 = np.percentile(Y, 99)
        # p_01 = np.percentile(Y, 1)
        # y_domain = (p_01, p_90)
        # print("y_domain ", y_domain)
        #
        # quantile_y_domain = (np.quantile(Y, .01), np.quantile(Y, .99))
        #
        # print("quantile y_domain ", quantile_y_domain)
        #
        # quantile_y_domain = (np.quantile(Y, .001), np.quantile(Y, .999))
        #
        # print("quantile y_domain ", quantile_y_domain)
        #
        # exit()

        if quantile == 0:
            # Determine domain by MC sampling.
            if hasattr(distr, "domain"):
                domain = distr.domain
            else:
                X = distr.rvs(size=1000)
                err = stats.norm.rvs(size=1000)
                #X = X * (1 + 0.1 * err)
                domain = (np.min(X), np.max(X))
            # p_90 = np.percentile(X, 99)
            # p_01 = np.percentile(X, 1)
            # domain = (p_01, p_90)

            #domain = (-20, 20)
        else:
            domain = distr.ppf([quantile, 1 - quantile])

        print("domain ", domain)

        # Detect PDF decay on domain boundaries, test that derivative is positive/negative for left/right side.
        eps = 1e-10
        force_decay = [False, False]
        for side in [0, 1]:
            print("domain[side] ", domain[side])
            print("distr.pdf(domain[side]) ", distr.pdf(domain[side]))
            diff = (distr.pdf(domain[side]) - distr.pdf(np.array(domain[side]) - np.array([eps, eps]))) / eps
            if side:
                diff = -diff
            if diff > 0:
                force_decay[side] = True
        return domain, force_decay

    def pdf(self, x, y=None):
        if y is None:
            return self.distr.pdf(x)# * self.scale

        return self.distr.pdf((x, y))# * self.scale

    def cdf(self, x):
        return (self.distr.cdf(x) - self.shift)# * self.scale

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
    Class to perform tests for fully specified exact distribution (distr. + domain) and moment functions.
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

        # domain_str = "({:5.2}, {:5.2})".format(*self.domain)
        self.eigenvalues_plot = None

    @property
    def title(self):
        cov = "_cov" if self.use_covariance else ""
        return "distr: {} moment_fn: {}{}".format(self.distr_name, self.fn_name, cov)

    def pdfname(self, subtitle):
        return "{}_{}.pdf".format(self.title, subtitle)

    @property
    def domain(self):
        return self.cut_distr.domain

    def pdf(self, x, y=None):
        return self.cut_distr.pdf(x, y)

    def setup_moments(self, moments_data, noise_level, reg_param=0, orth_method=1):
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

        print("self domain ", self.domain)

        moments_x = Legendre(max_n_moments, self.domain[0], log=log, safe_eval=False)
        moments_y = Legendre(max_n_moments, self.domain[1], log=log, safe_eval=False)

        self.moments_fn = BivariateMoments(moments_x, moments_y)

        size = self.moments_fn.size
        base_moments = self.moments_fn

        print("self pdf ", self.pdf)

        exact_cov, reg_matrix = mlmc.bivariate_simple_distr.compute_semiexact_cov_2(base_moments, self.pdf,
                                                                                 reg_param=reg_param)

        # exact_cov, reg_matrix = mlmc.bivariate_simple_distr.compute_exact_cov(base_moments, self.pdf,
        #                                                                             reg_param=reg_param)

        print("exact cov ")
        print(pd.DataFrame(exact_cov))
        print("reg matrix ", reg_matrix)

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

        print("cov matrix with noise")
        print(pd.DataFrame(cov))

        self.moments_fn, info, cov_centered = mlmc.bivariate_simple_distr.construct_orthogonal_moments(base_moments,
                                                                                        cov,
                                                                                        noise_level,
                                                                                        reg_param=reg_param,
                                                                                        orth_method=orth_method)
        self._cov_with_noise = cov
        self._cov_centered = cov_centered
        original_evals, evals, threshold, L = info
        self.L = L

        print("threshold: ", threshold, " from N: ", size)
        if self.eigenvalues_plot:
            threshold = evals[threshold]
            noise_label = "{:5.2e}".format(noise_level)
            self.eigenvalues_plot.add_values(evals, threshold=threshold, label=noise_label)

            # noise_label = "original evals, {:5.2e}".format(noise_level)
            # self.eigenvalues_plot.add_values(original_evals, threshold=threshold, label=noise_label)

        self.tol_density_approx = 0.01

        self.exact_moments = mlmc.bivariate_simple_distr.compute_semiexact_moments(self.moments_fn,
                                                                                  self.pdf, tol=tol_exact_moments)

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

    def make_approx(self, distr_class, noise, moments_data, tol, reg_param=0, regularization=None):
        result = ConvResult()

        distr_obj = distr_class(self.moments_fn, moments_data,
                                domain=self.domain, force_decay=self.cut_distr.force_decay, reg_param=reg_param,
                                regularization=regularization)

        t0 = time.time()
        min_result = distr_obj.estimate_density_minimize(tol=tol, multipliers=None)

        moments = mlmc.bivariate_simple_distr.compute_semiexact_moments(self.moments_fn, distr_obj.density)

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
        result.kl = mlmc.bivariate_simple_distr.KL_divergence(self.pdf, distr_obj.density, a, b)
        #result.kl_2 = mlmc.bivariate_simple_distr.KL_divergence_2(self.pdf, distr_obj.density, a, b)
        #result.l2 = mlmc.bivariate_simple_distr.L2_distance(self.pdf, distr_obj.density, a, b)
        result.tv = 0#mlmc.simple_distribution.total_variation_int(distr_obj.density_derivation, a, b)
        print(result)
        # X = np.linspace(self.cut_distr.domain[0], self.cut_distr.domain[1], 10)
        # Y = np.linspace(self.cut_distr.domain[0], self.cut_distr.domain[1], 10)
        # density_vals = distr_obj.density(X, Y)
        # exact_vals = self.pdf(X, Y)
        #print("vals: ", density_vals)
        #print("exact: ", exact_vals)
        return result, distr_obj

    def exact_conv(self):
        """
        Test density approximation for varying number of exact moments.
        :return:
        """
        results = []

        mom_class, min_mom, max_mom, log_flag = self.moments_data
        moments_num = [max_mom]

        distr_plot = plot.BivariateDistribution(exact_distr=self.cut_distr,
                                                title=self.title + "_exact_moments: {}".format(max_mom))

        for i_m, n_moments in enumerate(moments_num):
            self.moments_data = (mom_class, n_moments, n_moments, log_flag)
            # Setup moments.
            self.setup_moments(self.moments_data, noise_level=0, orth_method=2)

            if n_moments > self.moments_fn.size:
                continue

            #n_moments = (self.moments_fn._origin.moment_x.size * self.moments_fn._origin.moment_y.size)
            moments_data = np.empty((len(self.exact_moments), 2))

            moments_data[:, 0] = self.exact_moments
            moments_data[:, 1] = 1.0

            result, distr_obj = self.make_approx(mlmc.bivariate_simple_distr.SimpleDistribution, 0.0, moments_data,
                                                 tol=1e-7)

            distr_plot.add_distribution(distr_obj, label="#{}".format(n_moments) +
                                                         "\n total variation {:6.2g}".format(result.tv))
            results.append(result)

            print("KL divergence: {}, L2 norm: {}".format(result.kl, result.l2))

        # mult_tranform_back = distr_obj.multipliers  # @ np.linalg.inv(self.L)
        # final_jac = distr_obj._calculate_jacobian_matrix(mult_tranform_back)

        final_jac = distr_obj.final_jac

        #distr_obj_exact_conv_int = mlmc.bivariate_simple_distr.compute_exact_cov(distr_obj.moments_fn, distr_obj.density)
        M = np.eye(len(self._cov_with_noise[0]))
        M[:, 0] = -self._cov_with_noise[:, 0]

        # plt.contourf(x, y, density_values, 20, cmap='RdGy')
        # #plt.contourf(x, y, rv.pdf(pos), 20, cmap='RdGy')
        # plt.show()

        # print("M @ L-1 @ H @ L.T-1 @ M.T")
        # print(pd.DataFrame(
        #     M @ (np.linalg.inv(self.L) @ final_jac @ np.linalg.inv(self.L.T)) @ M.T))
        #
        # print("orig cov centered")
        # print(pd.DataFrame(self._cov_centered))

        distr_plot._title += " KL div: {}".format(result.kl)

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

    def inexact_conv(self):
        """
        Test density approximation for maximal number of moments
        and varying amount of noise added to covariance matrix.
        :return:
        """
        min_noise = 1e-6
        max_noise = 1e-2
        results = []

        distr_plot = plot.BivariateDistribution(exact_distr=self.cut_distr, title=self.title+"_inexact")

        geom_seq = np.exp(np.linspace(np.log(min_noise), np.log(max_noise), 5))
        noise_levels = np.flip(np.concatenate(([0.0], geom_seq)), axis=0)

        noise_levels = noise_levels[:1]

        print("noise levels ", noise_levels)

        mom_class, min_mom, max_mom, log_flag = self.moments_data

        moments_num = [5]#, 10, 20, 30]

        self.use_covariance = True
        orth_method = 2

        for noise in noise_levels:
            print("noise ", noise)
            for m in moments_num:

                self.moments_data = (mom_class, m, m, log_flag)
                info, moments_with_noise = self.setup_moments(self.moments_data, noise_level=noise,
                                                              orth_method=orth_method)

                original_evals, evals, threshold, L = info
                new_moments = np.matmul(moments_with_noise, L.T)
                n_moments = len(new_moments)

                moments_data = np.empty((n_moments, 2))
                moments_data[:, 0] = new_moments
                moments_data[:, 1] = noise ** 2
                moments_data[0, 1] = 1.0

                print("moments data ", moments_data)

                modif_cov, reg_mat = mlmc.bivariate_simple_distr.compute_semiexact_cov_2(self.moments_fn, self.pdf)

                print("modif_cov ", modif_cov)

                diff_norm = np.linalg.norm(modif_cov - np.eye(*modif_cov.shape)) / n_moments
                ref_moments = np.zeros(n_moments)
                ref_moments[0] = 1.0
                mom_err = np.linalg.norm(self.exact_moments[:n_moments] - ref_moments) / np.sqrt(n_moments)
                print("noise: {:6.2g} error of natural cov: {:6.2g} natural moments: {:6.2g}".format(
                    noise, diff_norm, mom_err))

                #assert mom_err/(noise + 1e-10) < 50  - 59 for five fingers dist

                result, distr_obj = self.make_approx(mlmc.bivariate_simple_distr.SimpleDistribution, noise,
                                                     moments_data,
                                                     tol=1e-5)

                distr_plot.add_distribution(distr_obj,
                                            label="{} moments, {} threshold".format(n_moments, threshold))
                results.append(result)

                print("KL divergence: {}, L2 norm: {}".format(result.kl, result.l2))

        distr_plot._title += "noise: {}, KL div: {}".format(noise, result.kl)

        #self.check_convergence(results)
        #self.eigenvalues_plot.show(None)#file=self.pdfname("_eigenvalues"))
        distr_plot.show(None)#"PDF aprox")#file=self.pdfname("_pdf_iexact"))
        distr_plot.reset()
        plt.show()
        return results

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
        distr_plot = plot.BivariateDistribution(exact_distr=self.cut_distr, title=self.title+"_exact")

        geom_seq = np.exp(np.linspace(np.log(min_noise), np.log(max_noise), 20))
        noise_levels = np.flip(np.concatenate(([0.0], geom_seq)), axis=0)

        noise_levels = noise_levels[:1]

        plot_mom_indices = None

        kl_total = []
        kl_2_total = []
        l2 = []

        self.use_covariance = True

        for noise in noise_levels:
            print("noise ", noise)
            kl_2 = []
            kl = []

            orth_method = 1

            regularization = mlmc.bivariate_simple_distr.Regularization1()

            #reg_parameters = [10, 1e-5] # 10 is suitable for Splines
            #reg_parameters = [5e-6] # is suitable for Legendre

            reg_parameters = [7e-5]
            #reg_parameters = [2.1842105263157896e-05]#, 1.6631578947368423e-05, 3.7473684210526315e-05, 3.226315789473684e-05,1.1421052631578948e-05]

            reg_parameters = [0, 9.081632653061225e-06, 1.5142857142857144e-05]#, 7.061224489795918e-06, 1.3122448979591838e-05,
             #1.716326530612245e-05]

            reg_parameters = [8.255102040816327e-05, 0.001, 0.00046991836734693883]

            #reg_parameters = [0, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 1e-7]

            #reg_parameters = [8.255102040816327e-07, 1e-05, 4.699183673469388e-06, 1.6410204081632653e-06, 2.456530612244898e-06]

            #reg_parameters = [5e-6]
            reg_parameters = [1e-8, 1e-9, 1e-10]
            #reg_parameters = [0]

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
                info, moments_with_noise = self.setup_moments(self.moments_data, noise_level=noise,
                                                              reg_param=reg_param, orth_method=orth_method)

                original_evals, evals, threshold, L = info
                new_moments = np.matmul(moments_with_noise, L.T)
                n_moments = len(new_moments)

                moments_data = np.empty((n_moments, 2))
                moments_data[:, 0] = new_moments
                moments_data[:, 1] = noise ** 2
                moments_data[0, 1] = 1.0

                print("moments data ", moments_data)

                result, distr_obj = self.make_approx(mlmc.bivariate_simple_distr.SimpleDistribution, noise, moments_data,
                                                     tol=1e-8, reg_param=reg_param, regularization=regularization)

                # m = mlmc.bivariate_simple_distr.compute_exact_moments(self.moments_fn, distr_obj.density)
                # e_m = mlmc.bivariate_simple_distr.compute_exact_moments(self.moments_fn, self.pdf)
                # moments.append(m)
                # all_exact_moments.append(e_m)

                print("DISTR OBJ reg param {}, MULTIPLIERS {}".format(reg_param, distr_obj.multipliers))

                distr_plot.add_distribution(distr_obj,
                                            label="noise: {}, threshold: {}, reg param: {}".format(noise, threshold,
                                                                                                   reg_param),
                                           size=n_moments, mom_indices=plot_mom_indices, reg_param=reg_param)

                results.append(result)

                final_jac = distr_obj.final_jac

                # print("ORIGINAL COV CENTERED")
                # print(pd.DataFrame(self._cov_centered))
                #
                # M = np.eye(len(self._cov_with_noise[0]))
                # M[:, 0] = -self._cov_with_noise[:, 0]
                #
                # print("M-1 @ L-1 @ H @ L.T-1 @ M.T-1")
                # print(pd.DataFrame(
                #     np.linalg.inv(M) @ (
                #                 np.linalg.inv(L) @ final_jac @ np.linalg.inv(L.T)) @ np.linalg.inv(M.T)))


                tv.append(result.tv)
                l2.append(result.l2)
                kl.append(result.kl)
                kl_2.append(result.kl_2)

                # distr_obj._update_quadrature(distr_obj.multipliers)
                # q_density = distr_obj._density_in_quads(distr_obj.multipliers)
                # q_gradient = distr_obj._quad_moments.T * q_density
                # integral = np.dot(q_gradient, distr_obj._quad_weights) / distr_obj._moment_errs
                #
                # int_density.append(abs(sum(integral)-1))

                distr_plot._title = self.title+" noise: {}, reg param: {} KL div: {}".format(noise, reg_param, result.kl)

                distr_plot.show(None)  # file="determine_param {}".format(self.title))#file=os.path.join(dir, self.pdfname("_pdf_iexact")))
                distr_plot.reset()

            kl_total.append(np.mean(kl))
            kl_2_total.append(np.mean(kl_2))



        return results

    def find_regularization_param(self):
        np.random.seed(1234)
        noise_level = 1e-2
        reg_params = np.linspace(1e-9, 1e-6, num=50)

        orth_method = 1

        min_results = []
        print("reg params ", reg_params)

        moment_class, min_n_moments, max_n_moments, self.use_covariance = self.moments_data
        log = self.log_flag
        if min_n_moments == max_n_moments:
            self.moment_sizes = np.array(
                [max_n_moments])  # [36, 38, 40, 42, 44, 46, 48, 50, 52, 54])+1#[max_n_moments])#10, 18, 32, 64])
        else:
            self.moment_sizes = np.round(
                np.exp(np.linspace(np.log(min_n_moments), np.log(max_n_moments), 8))).astype(int)

        #self.moments_fn = moment_class(max_n_moments, self.domain, log=log, safe_eval=False)

        moments_x = Legendre(max_n_moments, self.domain[0], log=log, safe_eval=False)
        moments_y = Legendre(max_n_moments, self.domain[1], log=log, safe_eval=False)

        self.moments_fn = BivariateMoments(moments_x, moments_y)

        _, _, n_moments, _ = self.moments_data

        size = 50
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
            exact_cov, reg_matrix = mlmc.bivariate_simple_distr.compute_semiexact_cov_2(self.moments_fn, self.pdf,
                                                                                        reg_param=reg_param)

            self.exact_moments = exact_cov[:, 0]
            # Add noise and regularization
            cov = exact_cov + fine_noise[:len(exact_cov), :len(exact_cov)]
            cov += reg_matrix

            moments_with_noise = cov[:, 0]
            self.moments_fn, info, cov_centered = mlmc.bivariate_simple_distr.construct_orthogonal_moments(self.moments_fn,
                                                                                                        cov,
                                                                                                        noise_level,
                                                                                                       reg_param=reg_param,
                                                                                                       orth_method=orth_method)
            original_evals, evals, threshold, L = info
            fine_moments = np.matmul(moments_with_noise, L.T)

            moments_data = np.empty((len(fine_moments), 2))
            moments_data[:, 0] = fine_moments  # self.exact_moments
            moments_data[:, 1] = 1  # noise ** 2
            moments_data[0, 1] = 1.0

            regularization = mlmc.bivariate_simple_distr.Regularization1()

            _, distr_obj = self.make_approx(mlmc.bivariate_simple_distr.SimpleDistribution, noise, moments_data,
                                            tol=1e-8, reg_param=reg_param, regularization=regularization)

            estimated_density_covariance, reg_matrix = mlmc.bivariate_simple_distr.compute_semiexact_cov_2(self.moments_fn,
                                                                                                distr_obj.density)

            # M = np.eye(len(cov[0]))
            # M[:, 0] = -cov[:, 0]
            #
            # print("cov centered")
            # print(pd.DataFrame(cov_centered))
            #
            # print("M-1 @ L-1 @ H @ L.T-1 @ M.T-1")
            # print(pd.DataFrame(
            #     M @ (np.linalg.inv(L) @ distr_obj.final_jac @ np.linalg.inv(L.T)) @ M.T))
            #
            # print("cov")
            # print(pd.DataFrame(cov))
            #
            # print("L-1 @ H @ L.T-1")
            # print(pd.DataFrame(
            #     (np.linalg.inv(L) @ distr_obj.final_jac @ np.linalg.inv(L.T))))

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

        #self.determine_regularization_param(best_params)

        return best_params



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
        #tests = [case.exact_conv]
        #tests = [case.inexact_conv]
        #tests = [case.determine_regularization_param]
        tests = [case.find_regularization_param]
        #tests = [case.mlmc_conv]
        #tests = [case.exact_conv]

        for test_fn in tests:
            name = test_fn.__name__
            test_results = test_fn()
            values = conv.setdefault(name, (case.title, []))
            values[1].append(test_results)

    # for key, values in conv.items():
    #     title, results = values
    #     title = "{}_conv_{}".format(title, key)
    #     if results[0] is not None:
    #         plot.plot_convergence(quantiles, results, title=title)

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
#         domain, est_domain, mc_test = mlmc.estimate.compute_results(mlmc_list[0], n_moments, test_mc)
#         mlmc.estimate.plot_pdf_approx(ax1, ax2, mc0_samples, mc_test, domain, est_domain)
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
        #(bd.TwoGaussians(name='two_gaussians'), False),
        #(bd.FiveFingers(name='five_fingers'), False), # Covariance matrix decomposition failed
        #(bd.Cauchy(name='cauchy'), False),# pass, check exact
        #(bd.Gamma(name='gamma'), False) # pass
        #(stats.norm(loc=1, scale=2), False),
        #(stats.norm(loc=0, scale=1), False),
        (bd.BivariateNorm(name='Bivariate_norm'), False),
        #(bd.BivariateTwoGaussians(name='Bivariate_TwoGaussians'), False)
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
        (moments.Legendre, 5, 5, True),
        #(moments.Spline, 5, 5, True),
    ]

    for m in mom:
        for distr in enumerate(distribution_list):
            #test_spline_approx(m, distr)
            test_pdf_approx_exact_moments(m, distr)


if __name__ == "__main__":

    import cProfile
    import pstats
    pr = cProfile.Profile()
    pr.enable()

    my_result = run_distr()

    pr.disable()
    ps = pstats.Stats(pr).sort_stats('cumtime')
    ps.print_stats()