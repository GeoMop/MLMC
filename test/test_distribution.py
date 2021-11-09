import os
import shutil
import time
import pytest
import copy

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.patheffects as mpe
from scipy.interpolate import interp1d

import mlmc.tool.plot
import mlmc.archive.estimate
import mlmc.tool.simple_distribution
#import mlmc.tool.simple_distribution_total_var
from mlmc import moments
import test.benchmark_distributions as bd
import mlmc.tool.plot as plot
import test.fixtures.mlmc_test_run
import mlmc.spline_approx as spline_approx
from mlmc.moments import Legendre
from mlmc import estimator
from mlmc.quantity_spec import ChunkSpec
import mlmc.quantity
import pandas as pd
import pickle
import test.plot_numpy
from cachier import cachier
from memoization import cached
from test.fixtures import benchmark_distribution as benchmark
from mlmc.tool.restrict_distribution import RestrictDistribution


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


distr_names = {'_norm': "norm", '_lognorm': "lognorm", '_two_gaussians': "two_gaussians", "_five_fingers": "five_fingers",
               "_cauchy": "cauchy", "_discontinuous": "discontinuous"}


class DistributionDomainCase:
    """
    Class to perform tests for fully specified exact distribution (distr. + domain) and moment functions.
    Exact moments and other objects are created once. Then various tests are performed.
    """

    def __init__(self, moments, distribution):
        # Setup distribution.
        i_distr, (distr_class, quantile, log_flag) = distribution

        self.log_flag = log_flag
        self.quantile = quantile
        self._name = None

        self.restrict_distr = RestrictDistribution.from_quantiles(distr_class, quantile)
        self.moments_data = moments
        moment_class, min_n_moments, max_n_moments, self.use_covariance = moments
        self.fn_name = str(moment_class.__name__)

        self.eigenvalues_plot = None

    @property
    def title(self):
        cov = "_cov" if self.use_covariance else ""
        return "distr: {} quantile: {} moment_fn: {}{}".format(self.distr_name, self.quantile, self.fn_name, cov)

    def pdfname(self, subtitle):
        return "{}_{}.pdf".format(self.title, subtitle)

    @property
    def distr_name(self):
        return self.restrict_distr.distr_name

    @property
    def domain(self):
        return RestrictDistribution.domain_for_quantile(self.restrict_distr.distr, lower=self.quantile, upper=1-self.quantile)

    def pdf(self, x):
        return self.restrict_distr.pdf(x)

    def setup_moments(self, moments_data, noise_level, orth_method=2):
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

            exact_cov = mlmc.tool.simple_distribution.compute_semiexact_cov(base_moments, self.pdf)

            self.moments_without_noise = exact_cov[:, 0]
            exact_without_reg = exact_cov

            #np.random.seed(4567)
            noises = []
            n_rep = 1
            for _ in range(n_rep):
                noise = np.random.randn(size**2).reshape((size, size))
                noise += noise.T
                noise *= 0.5 * noise_level
                noise[0, 0] = 0

                noises.append(noise)
            noise = np.mean(noises, axis=0)

            cov = exact_cov + noise
            moments = cov[:, 0]

            self.moments_fn, info, cov_centered = mlmc.tool.simple_distribution.construct_orthogonal_moments(base_moments,
                                                                                                        cov,
                                                                                                        tol=noise_level**2,
                                                                                                        orth_method=orth_method,
                                                                                                        exact_cov=exact_without_reg)
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
            # if self.eigenvalues_plot:
            #     threshold = original_evals[threshold]
            #     noise_label = "{:5.2e}".format(noise_level)
            #     self.eigenvalues_plot.add_values(original_evals, threshold=threshold, label=noise_label)
            #
            #     # noise_label = "original evals, {:5.2e}".format(noise_level)
            #     # self.eigenvalues_plot.add_values(original_evals, threshold=threshold, label=noise_label)

            self.tol_density_approx = 0.01

            self.exact_moments = mlmc.tool.simple_distribution.compute_semiexact_moments(self.moments_fn,
                                                                                      self.pdf, tol=tol_exact_moments)

        else:
            self.exact_moments = mlmc.tool.simple_distribution.compute_semiexact_moments(self.moments_fn,
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

    def make_approx(self, distr_class, noise, moments_data, tol):
        result = ConvResult()

        distr_obj = distr_class(self.moments_fn, moments_data,
                                domain=self.domain, force_decay=self.restrict_distr.force_decay)

        t0 = time.time()
        min_result = distr_obj.estimate_density_minimize(tol=tol, multipliers=None)

        # result = profile(lambda : distr_obj.estimate_density_minimize(tol_exact_moments))
        t1 = time.time()
        result.size = moments_data.shape[0]
        result.noise = noise
        result.time = t1 - t0
        result.residual_norm = min_result.fun_norm
        result.success = min_result.success
        result.success = min_result.success
        result.nit = min_result.nit

        a, b = self.domain
        print("a: {}, b: {}".format(a, b))
        result.kl = mlmc.tool.simple_distribution.KL_divergence(self.pdf, distr_obj.density, a, b)
        result.kl_2 = mlmc.tool.simple_distribution.KL_divergence_2(self.pdf, distr_obj.density, a, b)
        result.l2 = mlmc.tool.simple_distribution.L2_distance(self.pdf, distr_obj.density, a, b)
        #result.tv = mlmc.tool.simple_distribution.total_variation_int(distr_obj.density_derivation, a, b)
        print(result)
        X = np.linspace(self.domain[0], self.domain[1], 10)
        density_vals = distr_obj.density(X)
        exact_vals = self.pdf(X)
        #print("vals: ", density_vals)
        #print("exact: ", exact_vals)
        return result, distr_obj

    def _plot_kl_div(self, x, kl):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        # ax.plot(noise_levels, tv, label="total variation")
        ax.plot(x, kl, 'o', c='r')
        #ax.set_xlabel("noise level")
        ax.set_xlabel("noise level")
        ax.set_ylabel("KL divergence")
        # ax.plot(noise_levels, l2, label="l2 norm")
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.legend()

        plt.show()

    def _compute_exact_kl(self, n_moments, moments_fn, orth_method, tol_density=1e-5, tol_exact_cov=1e-10):
        """
        Compute KL divergence truncation error of given number of moments
        :param n_moments: int
        :param moments_fn: moments object instance
        :param tol_density: minimization tolerance
        :param tol_exact_cov: covariance matrix, integration tolerance
        :return: KL divegence, SimpleDistribution instance
        """
        exact_cov = mlmc.tool.simple_distribution.compute_semiexact_cov(moments_fn, self.pdf)
        self.moments_fn, info, _ = mlmc.tool.simple_distribution.construct_orthogonal_moments(moments_fn, exact_cov, tol=0,
                                                                                         orth_method=orth_method)
        orig_evals, evals, threshold, L = info

        exact_moments = mlmc.tool.simple_distribution.compute_semiexact_moments(self.moments_fn, self.pdf, tol=tol_exact_cov)

        moments_data = np.empty((self.moments_fn.size, 2))
        moments_data[:, 0] = exact_moments[:self.moments_fn.size]
        moments_data[:, 1] = 1.0

        result, distr_obj = self.make_approx(mlmc.tool.simple_distribution.SimpleDistribution, 0.0, moments_data, tol=tol_density)
        return result.kl, distr_obj

    def get_cache_key_cov(self, moments_fn, pdf, tol=1e-15, use_quad=False):
        return (moments_fn.__class__.__name__, moments_fn.size, pdf.__class__.__name__, tol, use_quad)

    def get_cache_key_moments(self, moments_fn, pdf,tol=1e-15, use_quad=False):
        return (moments_fn.__class__.__name__, moments_fn.size, pdf.__class__.__name__, tol, use_quad)

    def hash_moments(self, obj):
        pass
        # Somehow, this allows you to use cache, even though it cannot cache Moments class.

    #@cachier(hash_params=hash_moments)
    def compute_moments(self, moments_fn, pdf, tol=1e-15, use_quad=False):
        if use_quad:
            return mlmc.tool.simple_distribution.compute_semiexact_moments(moments_fn, pdf, tol=tol)
        else:
            return mlmc.tool.simple_distribution.compute_exact_moments(moments_fn, pdf, tol=tol)

    @cachier(hash_params=hash_moments)
    def compute_cov(self, moments_fn, pdf, tol, use_quad, size):
        if use_quad:
            return mlmc.tool.simple_distribution.compute_semiexact_cov(moments_fn, pdf, tol=tol)
        else:
            return mlmc.tool.simple_distribution.compute_exact_cov(moments_fn, pdf, tol=tol)

    def plot_KL_div_exact(self):
        """
        Plot KL divergence for different number of exact moments
        :return:
        """
        # Clear cache
        #DistributionDomainCase.compute_cov.clear_cache()
        noise_level = 0
        tol_exact_moments = 1e-15
        tol_density = 1e-8
        results = []
        orth_method = 2
        distr_plot = plot.Distribution(exact_distr=self.restrict_distr, title=self.title + "_exact", cdf_plot=False,
                                       log_x=True, error_plot=False)

        dir_name = "KL_div_exact_{}".format(orth_method)
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

        work_dir = os.path.join(dir_name, self.distr_name)

        #########################################
        # Set moments objects
        moment_class, min_n_moments, max_n_moments, self.use_covariance = self.moments_data
        log = self.log_flag
        if min_n_moments == max_n_moments:
            self.moment_sizes = np.array(
                [max_n_moments])
        else:
            self.moment_sizes = np.round(np.exp(np.linspace(np.log(min_n_moments), np.log(max_n_moments), 10))).astype(
                int)
        self.moments_fn = moment_class(max_n_moments, self.domain, log=log, safe_eval=False)

        # print("moment sizes ", self.moment_sizes)
        self.moment_sizes = [1, 5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 100]
        self.moment_sizes = [5]

        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)
            # raise FileExistsError
        # else:
        os.mkdir(work_dir)
        np.save(os.path.join(work_dir, "moment_sizes"), self.moment_sizes)

        ##########################################
        # Orthogonalize moments
        # self.moments_fn = moment_class(max_n_moments, self.domain, log=log, safe_eval=False)
        # base_moments = self.moments_fn
        # exact_cov = mlmc.tool.simple_distribution.compute_semiexact_cov(base_moments, self.pdf)
        # self.moments_fn, info, _ = mlmc.tool.simple_distribution.construct_orthogonal_moments(base_moments, exact_cov,
        #                                                                                  tol=noise_level**2, orth_method=orth_method)
        # orig_eval, evals, threshold, L = info
        # eye_approx = L @ exact_cov @ L.T
        # test that the decomposition is done well
        # assert np.linalg.norm(
        #     eye_approx - np.eye(*eye_approx.shape)) < 1e-9  # 1e-10 failed with Cauchy for more moments

        # print("threshold: ", threshold, " from N: ", self.moments_fn.size)
        # if self.eigenvalues_plot:
        #     threshold = evals[threshold]
        #     noise_label = "{:5.2e}".format(noise_level)
        #     self.eigenvalues_plot.add_values(evals, threshold=threshold, label=noise_label)
        # self.exact_moments = mlmc.tool.simple_distribution.compute_semiexact_moments(self.moments_fn, self.pdf, tol=tol_exact_moments)
        # self.exact_moments = mlmc.tool.simple_distribution.compute_exact_moments(self.moments_fn, self.pdf,
        #                                                                              tol=tol_exact_moments)

        kl_plot = plot.KL_divergence(log_y=True, iter_plot=True, kl_mom_err=False,
                                     title="Kullback-Leibler divergence, {}".format(self.title),
                                     xlabel="number of moments", ylabel="KL divergence")

        ###############################################
        # For each moment size compute density
        for i_m, n_moments in enumerate(self.moment_sizes):

            print("self domain ", self.domain)
            self.moments_fn = moment_class(n_moments, self.domain, log=log, safe_eval=True)
            base_moments = self.moments_fn
            # exact_cov = mlmc.tool.simple_distribution.compute_exact_cov(base_moments, self.pdf,
            #                                                                tol=tol_exact_moments)

            # exact_cov = mlmc.tool.simple_distribution.compute_semiexact_cov(base_moments, self.pdf)
            exact_cov = mlmc.tool.simple_distribution.compute_exact_cov(base_moments, self.pdf,
                                                                           tol=tol_exact_moments)
            self.moments_fn, info, _ = mlmc.tool.simple_distribution.construct_orthogonal_moments(base_moments,
                                                                                                  exact_cov,
                                                                                                  tol=noise_level ** 2,
                                                                                                  orth_method=orth_method)

            self.exact_moments = mlmc.tool.simple_distribution.compute_semiexact_moments(self.moments_fn, self.pdf,
                                                                                         tol=tol_exact_moments)
            threshold = 0
            # orig_eval, evals, threshold, L = info

            # print("n moments ", n_moments)
            # print("self. moments_fn size ", self.moments_fn.size)
            # if n_moments > self.moments_fn.size:
            #
            #     continue
            n_moments = self.moments_fn.size

            # moments_fn = moment_fn(n_moments, domain, log=log_flag, safe_eval=False )
            moments_data = np.empty((n_moments, 2))
            moments_data[:, 0] = self.exact_moments[:n_moments]
            moments_data[:, 1] = 1.0

            # modif_cov = mlmc.tool.simple_distribution.compute_semiexact_cov(self.moments_fn, self.pdf)
            # diff_norm = np.linalg.norm(modif_cov - np.eye(*modif_cov.shape))
            # print("#{} cov mat norm: {}".format(n_moments, diff_norm))

            result, distr_obj = self.make_approx(mlmc.tool.simple_distribution.SimpleDistribution, 0.0, moments_data,
                                                 tol=tol_density)

            iter_res_mom = []
            for iteration in distr_obj._monitor.iterations:
                # print("exact moments ", self.exact_moments[:n_moments])
                # print("it mom ", it_mom)
                # print("self it mom shape ", it_mom.shape)
                # print("(self.exact_moments[:n_moments] - it_mom) ", (self.exact_moments[:n_moments] - it_mom))
                iter_res_mom.append(np.sum((self.exact_moments[:n_moments] - iteration.moments_by_quad) ** 2))

            # print("iter res mom ", iter_res_mom)

            np.save('{}/{}_{}.npy'.format(work_dir, n_moments, "res_moments"), iter_res_mom)

            distr_plot.add_distribution(distr_obj, label="#{}, KL div: {}".format(n_moments, result.kl))
            results.append(result)

            self._save_distr_data(distr_obj, distr_plot, work_dir, n_moments, result)

            kl_plot.add_value((n_moments, result.kl))
            kl_plot.add_iteration(x=n_moments, n_iter=result.nit, failed=not result.success)

            self._save_kl_data_exact(work_dir, n_moments, result.kl, result.nit, not result.success, threshold)

        # self.check_convergence(results)
        # kl_plot.show(None)
        # distr_plot.show(None)#file=self.pdfname("_pdf_exact"))
        distr_plot.reset()

        return results, dir_name

    def _save_kl_data_exact(self, work_dir, n_moments, kl_div, nit, success, threshold):
        np.save('{}/{}_{}.npy'.format(work_dir, n_moments, "add-value"), (n_moments, kl_div))
        np.save('{}/{}_{}.npy'.format(work_dir, n_moments, "add-iteration"), (n_moments, nit, success))
        np.save('{}/{}_{}.npy'.format(work_dir, n_moments, "threshold"), threshold)

    def _save_distr_data(self, distr_object, distr_plot, work_dir, noise_level, result, L=None, name=""):
        domain = distr_object.domain
        distr_plot.adjust_domain(domain)
        X = distr_plot._grid(100000, domain=domain)

        np.save('{}/{}_{}.npy'.format(work_dir, noise_level, "result" + name), (result.kl, result.kl_2, result.l2,
                                                                                result.residual_norm, result.time))
        np.save('{}/{}_{}.npy'.format(work_dir, noise_level, "domain" + name), distr_object.domain)
        np.save('{}/{}_{}.npy'.format(work_dir, noise_level, "X" + name), X)
        np.save('{}/{}_{}.npy'.format(work_dir, noise_level, "Y_pdf" + name), distr_object.density(X))
        np.save('{}/{}_{}.npy'.format(work_dir, noise_level, "Y_cdf" + name), distr_object.cdf(X))
        np.save('{}/{}_{}.npy'.format(work_dir, noise_level, "Y_pdf_exact" + name), self.restrict_distr.pdf(X))
        np.save('{}/{}_{}.npy'.format(work_dir, noise_level, "Y_cdf_exact" + name), self.restrict_distr.cdf(X))
        np.save('{}/{}_{}.npy'.format(work_dir, noise_level, "Y_pdf_log" + name), distr_object.density_log(X))
        np.save('{}/{}_{}.npy'.format(work_dir, noise_level, "Y_der_1" + name), distr_object.mult_mom_der(X, degree=1))
        np.save('{}/{}_{}.npy'.format(work_dir, noise_level, "Y_der_2" + name), distr_object.mult_mom_der(X, degree=2))
        if L is not None:
            np.save('{}/{}_{}.npy'.format(work_dir, noise_level, "L" + name), L)

    def plot_KL_div_inexact(self):
        """
        Plot KL divergence for different noise level of exact moments
        """
        min_noise = 1e-6
        max_noise = 1e-1
        geom_seq = np.exp(np.linspace(np.log(min_noise), np.log(max_noise), 20))
        noise_levels = np.flip(np.concatenate(([0.0], geom_seq)), axis=0)

        # noise_levels = noise_levels[:1]

        # noise_levels = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 1e-8]

        min_noise = 1e-1
        max_noise = 1e-12
        geom_seq = np.exp(np.linspace(np.log(min_noise), np.log(max_noise), 50))
        noise_levels = np.flip(np.concatenate(([0.0], geom_seq)), axis=0)

        # noise_levels = [1e-1, 1e-2, 1e-3, 1e-4,  1e-5, 1e-6, 1e-8]

        # noise_levels = [1e-2]

        # noise_levels = [1e-4, 1e-5, 1e-6, 1e-8, 1e-10, 1e-12]

        noise_levels = [1e-5]
        noise_level = 0
        tol_exact_moments = 1e-15
        tol_exact_cov = 1e-15
        tol_density = 1e-8
        results = []
        n_moments = 35
        orth_method = 2

        distr_plot = plot.Distribution(exact_distr=self.restrict_distr, title=self.title + "_inexact", cdf_plot=False,
                                       log_x=self.log_flag, error_plot=False)

        dir_name = "KL_div_inexact_{}_{}".format(orth_method, noise_levels[0])
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        # else:
        #     shutil.rmtree(dir_name)
        #     os.mkdir(dir_name)

        work_dir = os.path.join(dir_name, self.distr_name)
        if os.path.exists(work_dir):
            raise FileExistsError
        else:
            os.mkdir(work_dir)
            np.save(os.path.join(work_dir, "noise_levels"), noise_levels)
            np.save(os.path.join(work_dir, "n_moments"), n_moments)

        kl_plot = plot.KL_divergence(iter_plot=True, log_y=True, log_x=True,
                                     title=self.title + "_n_mom_{}".format(n_moments), xlabel="noise std",
                                     ylabel="KL divergence", truncation_err_label="trunc. err, m: {}".format(n_moments))

        ##########################################
        # Set moments objects
        moment_class, _, _, self.use_covariance = self.moments_data
        log = self.log_flag

        print("n moments ", n_moments)

        # print("self moments fn size ", self.moments_fn.size)

        ##########################################
        # Orthogonalize moments
        self.moments_fn = moment_class(n_moments, self.domain, log=log, safe_eval=False)

        base_moments = copy.deepcopy(self.moments_fn)
        exact_cov = mlmc.tool.simple_distribution.compute_semiexact_cov(base_moments, self.pdf)

        kl_plot.truncation_err, distr_obj_exact = self._compute_exact_kl(n_moments, base_moments, orth_method,
                                                                         tol_density, tol_exact_cov)

        np.save(os.path.join(work_dir, "truncation_err"), kl_plot.truncation_err)

        # exact_moments_orig = mlmc.tool.simple_distribution.compute_semiexact_moments(self.moments_fn, self.pdf, tol=1e-10)
        exact_moments_orig = exact_cov[:, 0]
        # print("original exact moments ", exact_moments_orig)
        # print("exact cov[:, 0] ", exact_cov[:, 0])

        self.moments_fn = moment_class(n_moments, self.domain, log=log, safe_eval=False)

        ###############################################
        # For each moment size compute density
        for i_m, noise_level in enumerate(noise_levels):
            print("NOISE LEVEL ", noise_level)
            # Add noise to exact covariance matrix
            # np.random.seed(4567)
            noises = []
            n_rep = 1
            for _ in range(n_rep):
                noise = np.random.randn(base_moments.size ** 2).reshape((base_moments.size, base_moments.size))
                noise += noise.T
                noise *= 0.5 * noise_level
                noise[0, 0] = 0

                noises.append(noise)
            noise = np.mean(noises, axis=0)
            cov = exact_cov + noise
            threshold = 0
            L = None

            # # Change base
            tol = noise_level
            #tol = None
            self.moments_fn, info, _ = mlmc.tool.simple_distribution.construct_orthogonal_moments(base_moments, cov,
                                                                                                  tol,
                                                                                                  orth_method=orth_method)
            # Tests
            original_evals, evals, threshold, L = info
            # eye_approx = L @ exact_cov @ L.T
            # test that the decomposition is done well
            # assert np.linalg.norm(
            #     eye_approx - np.eye(*eye_approx.shape)) < 1e-9  # 1e-10 failed with Cauchy for more moments_fn
            # print("threshold: ", threshold, " from N: ", self.moments_fn.size)
            # modif_cov = mlmc.tool.simple_distribution.compute_semiexact_cov(self.moments_fn, self.pdf, tol=tol_exact_cov)
            # diff_norm = np.linalg.norm(modif_cov - np.eye(*modif_cov.shape))
            # print("#{} cov mat norm: {}".format(n_moments, diff_norm))

            # Set moments data
            n_moments = self.moments_fn.size

            if L is None:
                transformed_moments = cov[:, 0]
            else:
                # print("cov moments ", cov[:, 0])
                transformed_moments = np.matmul(cov[:, 0], L.T)
                # print("transformed moments ", transformed_moments)

            # print("transformed moments len ", len(transformed_moments))

            moments_data = np.empty((len(transformed_moments), 2))
            moments_data[:, 0] = transformed_moments
            moments_data[:, 1] = 1
            moments_data[0, 1] = 1.0

            exact_moments = exact_moments_orig[:len(transformed_moments)]

            result, distr_obj = self.make_approx(mlmc.tool.simple_distribution.SimpleDistribution, 0.0, moments_data,
                                                 tol=tol_density)
            distr_plot.add_distribution(distr_obj,
                                        label="noise: {:f}, th: {}, KL div: {:f}".format(noise_level, threshold,
                                                                                         result.kl))
            results.append(result)

            iter_res_mom = []
            for iteration in distr_obj._monitor.iterations:
                # print("exact moments ", self.exact_moments[:n_moments])
                # print("it mom ", it_mom)
                # print("self it mom shape ", it_mom.shape)
                # print("(self.exact_moments[:n_moments] - it_mom) ", (self.exact_moments[:n_moments] - it_mom))

                res = moments_data[:, 0] - iteration.moments_by_quad
                if L is not None:
                    L_inv = np.linalg.pinv(L)
                    res = np.matmul(L_inv, res)
                # print("res ", res)
                # print("sum res**2 ", np.sum(res**2))
                # print("np.sum((moments_data[:, 0] - it_mom) ** 2 ", np.sum((moments_data[:, 0] - it_mom) ** 2))
                # exit()
                iter_res_mom.append(np.sum(res ** 2))

            # print("iter res mom ", iter_res_mom)

            np.save('{}/{}.npy'.format(work_dir, "res_moments"), iter_res_mom)

            self._save_distr_data(distr_obj, distr_plot, work_dir, noise_level, result, L)

            print("RESULT ", result.success)

            kl_div = mlmc.tool.simple_distribution.KL_divergence(distr_obj_exact.density, distr_obj.density,
                                                                 self.domain[0], self.domain[1])
            # total_variation = mlmc.tool.simple_distribution.total_variation_int(distr_obj.density, self.domain[0], self.domain[1])

            kl_plot.add_value((noise_level, kl_div))
            kl_plot.add_iteration(x=noise_level, n_iter=result.nit, failed=not result.success)

            # print("exact moments ", exact_moments[:len(moments_data[:, 0])])
            # print("moments data ", moments_data[:, 0])
            # print("difference ", np.array(exact_moments) - np.array(moments_data[:, 0]))
            print("difference orig", np.array(exact_moments_orig) - np.array(cov[:, 0][:len(exact_moments_orig)]))

            diff_orig = np.array(exact_moments_orig) - np.array(cov[:, 0][:len(exact_moments_orig)])

            kl_plot.add_moments_l2_norm((noise_level, np.linalg.norm(diff_orig) ** 2))

            self._save_kl_data(work_dir, noise_level, kl_div, result.nit, not result.success,
                               np.linalg.norm(diff_orig) ** 2, threshold, total_variation=result.tv)

        kl_plot.show(None)
        distr_plot.show(None)
        distr_plot.reset()

        return results, dir_name

    def _save_kl_data(self, work_dir, noise_level, kl_div, nit, success, mom_err, threshold, total_variation=0, name=""):
        np.save('{}/{}_{}.npy'.format(work_dir, noise_level, "add-value" + name), (noise_level, kl_div))
        np.save('{}/{}_{}.npy'.format(work_dir, noise_level, "add-iteration" + name), (noise_level, nit, success))
        np.save('{}/{}_{}.npy'.format(work_dir, noise_level, "add-moments" + name), (noise_level, mom_err))
        np.save('{}/{}_{}.npy'.format(work_dir, noise_level, "threshold" + name), threshold)
        np.save('{}/{}_{}.npy'.format(work_dir, noise_level, "total_variation" + name), total_variation)

    def eval_plot(self, noise=None):
        """
        Test density approximation for maximal number of moments
        and varying amount of noise added to covariance matrix.
        :return:
        """
        min_noise = 1e-6
        max_noise = 1e-2
        results = []
        orth_method = 2
        #np.random.seed(8888)

        _, _, n_moments, _ = self.moments_data
        distr_plot = plot.Distribution(exact_distr=self.restrict_distr, title="Preconditioning reg, {},  n_moments: {}, noise: {}".format(self.title, n_moments, max_noise),
                                       log_x=self.log_flag, error_plot=None, reg_plot=False, cdf_plot=False, log_density=True)

        self.eigenvalues_plot = plot.Eigenvalues(title="", log_y=False)
        eval_plot_obj = plot.EvalPlot(title="")

        geom_seq = np.exp(np.linspace(np.log(min_noise), np.log(max_noise), 20))
        noise_levels = np.flip(np.concatenate(([0.0], geom_seq)), axis=0)

        noise_levels = [1e-1, 1e-2, 1e-3]
        plot_mom_indices = None

        moments = []
        all_exact_moments = []

        for noise in noise_levels:

            dir = self.title + "noise: ".format(noise)
            if not os.path.exists(dir):
                os.makedirs(dir)

            info, moments_with_noise = self.setup_moments(self.moments_data, noise_level=noise, orth_method=orth_method)
            n_moments = len(self.exact_moments)

            original_evals, evals, threshold, L = info
            new_moments = np.matmul(moments_with_noise, L.T)

            moments_data = np.empty((n_moments, 2))
            moments_data[:, 0] = new_moments
            moments_data[:, 1] = noise ** 2
            moments_data[0, 1] = 1.0

            print("moments data ", moments_data)

            result, distr_obj = self.make_approx(mlmc.tool.simple_distribution.SimpleDistribution, noise, moments_data,
                                                 tol=1e-7)

            m = mlmc.tool.simple_distribution.compute_exact_moments(self.moments_fn, distr_obj.density)
            e_m = mlmc.tool.simple_distribution.compute_exact_moments(self.moments_fn, self.pdf)
            moments.append(m)
            all_exact_moments.append(e_m)

            distr_plot.add_distribution(distr_obj,
                                        label="n: {:0.4g}, th: {}, KL_div: {:0.4g}".format(noise, threshold, result.kl),
                                       size=n_moments, mom_indices=plot_mom_indices)

            results.append(result)

            final_jac = distr_obj.final_jac

            eval, evec = np.linalg.eigh(final_jac)
            eval[::-1].sort()


            cond_num = distr_obj.cond_number
            eval_plot_obj.add_values(eval, label=r'$\sigma$' + "={:0.3g}, ".format(noise) +
                                           r'$\kappa_2$' + "={:0.4g}".format(cond_num))


            print("original evals ", original_evals)

            self.eigenvalues_plot.add_values(original_evals, threshold=noise**2, label=r'$\sigma^2$' + "={:0.0e}".format(noise**2))

            print("final jac ")
            print(pd.DataFrame(final_jac))

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
            #

        print("FINAL moments ", moments)
        print("exact moments ", all_exact_moments)

        # for exact, estimated in zip(moments, all_exact_moments):
        #     print("(exact-estimated)**2", (exact-estimated)**2)
        #     print("sum(exact-estimated)**2", np.sum((exact - estimated) ** 2))

        distr_plot.show(file="determine_param {}".format(self.title))#file=os.path.join(dir, self.pdfname("_pdf_iexact")))
        distr_plot.reset()
        #self.check_convergence(results)
        self.eigenvalues_plot.show(file="eigenvalues")
        self.eigenvalues_plot.show(file=None)#self.pdfname("_eigenvalues"))
        eval_plot_obj.show(file=None)

        return results


    def plot_gradients(self, gradients):
        print("gradients ", gradients)
        print("gradients LEN ", len(gradients))
        gradients = [np.linalg.norm(gradient) for gradient in gradients]

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(gradients)
        plt.show()


def run_distr():
    quantile = 0.001
    distribution_list = [
        (benchmark.TwoGaussians(), quantile, False),
        (benchmark.FiveFingers(), quantile, False),  # Covariance matrix decomposition failed
        (benchmark.Cauchy(), quantile, False),  # pass, check exact
        (stats.lognorm(scale=np.exp(1), s=1), quantile, False),
        (benchmark.Abyss(), quantile, False),
        (benchmark.ZeroValue(), quantile, False)
    ]

    # @pytest.mark.skip
    mom = [
        # moments_class, min and max number of moments, use_covariance flag
        (moments.Legendre, 25, 100, True),
    ]

    for m in mom:
        for distr in enumerate(distribution_list):
            name, dir_name = test_pdf_approx_exact_moments(m, distr)

    if name == "plot_KL_div_exact":
        test.plot_numpy.plot_KL_div_exact_iter(os.path.abspath(dir_name))
    else:
        test.plot_numpy.plot_KL_div_inexact_iter(os.path.abspath(dir_name))

    # plot_requirements = {
    #                      'sqrt_kl': False,
    #                      'sqrt_kl_Cr': True,
    #                      'tv': True,
    #                      'sqrt_tv_Cr': True, # TV
    #                      'reg_term': False,
    #                      'l2': False,
    #                      'barron_diff_mu_line': False,
    #                      '1_eig0_diff_mu_line': False}
    #
    # #
    # test_kl_estimates(mom[0], distribution_list, plot_requirements)
    # #test_gauss_degree(mom[0], distribution_list[0], plot_requirements, degrees=[210, 220, 240, 260, 280, 300]) #  degrees=[10, 20, 40, 60, 80, 100], [110, 120, 140, 160, 180, 200]
    # test_gauss_degree(mom[0], distribution_list[0], plot_requirements, degrees=[10, 20, 40, 60, 80, 100])


def test_pdf_approx_exact_moments(moments, distribution):
    """
    Test reconstruction of the density function from exact moments.
    - various distributions
    - various moments functions
    - test convergency with increasing number of moments
    :return:
    """
    #quantiles = np.array([0.001])
    #quantiles = np.array([0.01])
    conv = {}
    np.random.seed(1234)
    case = DistributionDomainCase(moments, distribution)

    #tests = [case.plot_KL_div_exact]
    tests = [case.plot_KL_div_inexact]

    for test_fn in tests:
        name = test_fn.__name__
        test_results, dir_name = test_fn()

        values = conv.setdefault(name, (case.title, []))
        values[1].append(test_results)

    return name, dir_name


@pytest.mark.skip
def test_gauss_degree(moments, distr, plot_requirements, degrees=[100]):
    shape = (2, 3)
    fig, axes = plt.subplots(*shape, sharex=True, sharey=True, figsize=(15, 10))
    # fig.suptitle("Mu -> Lambda")
    axes = axes.flatten()

    if degrees is not None:
        for gauss_degree, ax in zip(degrees, axes[:len(degrees)]):
            kl_estimates((0,distr), moments, ax, plot_requirements, gauss_degree)
    plt.tight_layout()
    # mlmc.plot._show_and_save(fig, "", "mu_to_lambda_lim")
    mlmc.tool.plot._show_and_save(fig, None, "mu_to_alpha")
    mlmc.tool.plot._show_and_save(fig, "", "mu_to_alpha")


@pytest.mark.skip
def test_kl_estimates(moments, distribution_list, plot_requirements):
    shape = (2, 3)
    fig, axes = plt.subplots(*shape, sharex=True, sharey=True,
                             figsize=(15, 10))
    # fig.suptitle("Mu -> Lambda")
    axes = axes.flatten()
    for distr, ax in zip(enumerate(distribution_list), axes[:len(distribution_list)]):
        kl_estimates(distr, moments, ax, plot_requirements)
    plt.tight_layout()

    legend = plt.legend()
    # ax = legend.axes
    from matplotlib.lines import Line2D

    # handles, labels = ax.get_legend_handles_labels()
    # from matplotlib.patches import Patch
    #
    # handles.append(Patch(facecolor='red'))
    # labels.append(r'$\\alpha_0|\lambda_0 - \lambda_r|$"')
    #
    # handles.append(Patch(facecolor='blue'))
    # labels.append(r'$\sqrt{D(\\rho || \\rho_{R}) / C_R}$"')
    #
    # handles.append(Patch(facecolor='orange'))
    # labels.append(r'$|\lambda_0 - \lambda_r| / \sqrt{C_R}$"')
    #
    # print("handles ", handles)
    # print("labels ", labels)
    #
    # legend._legend_box = None
    # legend._init_legend_box(handles, labels)
    # legend._set_loc(legend._loc)
    # legend.set_title(legend.get_title().get_text())

    # mlmc.plot._show_and_save(fig, "", "mu_to_lambda_lim")
    mlmc.tool.plot._show_and_save(fig, None, "mu_to_alpha")
    mlmc.tool.plot._show_and_save(fig, "", "mu_to_alpha")


def kl_estimates(distribution, moments, ax, plot_req, gauss_degree=None):
    idx, distr_cfg = distribution

    if gauss_degree is not None:
        mlmc.tool.simple_distribution.GUASS_DEGREE = gauss_degree

    case = DistrTestCase(distr_cfg, moments)

    title = case.restrict_distr.distr_name
    if title == "norm":
        title = "normal"

    # if gauss_degree is not None:
    #     title = case.title + " gauss degree: {}".format(gauss_degree)
    orto_moments, moment_data = case.make_orto_moments(0)
    exact_distr = mlmc.tool.simple_distribution.SimpleDistribution(orto_moments, moment_data,
                                                            domain=case.domain,
                                                            force_decay=case.restrict_distr.force_decay)

    # true_pdf = case.distr.pdf
    # a, b = case.distr.domain
    tolerance = 1e-8
    min_result = exact_distr.estimate_density_minimize(tol=tolerance)
    # exact_tol = max(min_result.res_norm, tolerance)
    exact_mu = case.exact_orto_moments
    exact_eval_0, exact_eval_max = exact_distr.jacobian_spectrum()[[0, -1]]
    mu_diffs, l_diffs, eigs, total_vars = [], [], [], []
    #ratio_distribution = stats.lognorm(s=0.1)

    scale = 0.01
    #scale = 0.1
    #scale=0.0001

    ratio_distribution = stats.norm(scale=scale*np.linalg.norm(exact_distr.multipliers[1:]))
    ratio_distribution = stats.norm(scale=scale)
    raw_distr = mlmc.tool.simple_distribution.SimpleDistribution(orto_moments, moment_data,
                                                            domain=case.domain,
                                                            force_decay=case.restrict_distr.force_decay)
    size = len(exact_distr.multipliers)
    linf_log_approx_error = np.max(np.log(case.restrict_distr.pdf(exact_distr._quad_points))
                                   - np.log(exact_distr.density(exact_distr._quad_points)))
    b_factor_estimate = np.exp(linf_log_approx_error)
    linf_inv_distr = np.max(1/case.restrict_distr.pdf(exact_distr._quad_points))
    Am_factor_estimate = (orto_moments.size + 1) * np.sqrt(linf_inv_distr)

    kl_divs = []
    L2_dist = []
    TV_distr_diff = []
    dot_l_diff_mu_diff = []

    reg_terms = []

    for _ in range(200):
        s = 3 * stats.uniform.rvs(size=1)[0]
        lambda_inex = exact_distr.multipliers + s*ratio_distribution.rvs(size)
        raw_distr._initialize_params(size)
        raw_distr.multipliers = lambda_inex
        raw_distr.set_quadrature(exact_distr)
        raw_distr.moments = raw_distr.moments_by_quadrature()
        raw_distr._quad_moments_2nd_der = raw_distr.moments_by_quadrature(der=2)
        raw_eval_0, raw_eval_max = raw_distr.jacobian_spectrum()[[0, -1]]
        raw_distr.multipliers[0] += np.log(raw_distr.moments[0])

        lambda_diff = -(exact_distr.multipliers - raw_distr.multipliers)

        l_diff_norm = np.linalg.norm(lambda_diff[:])
        mu_diff = exact_mu - raw_distr.moments
        mu_diff_norm = np.linalg.norm(mu_diff[:])
        dot_l_diff_mu_diff.append(np.dot(exact_mu, lambda_diff))

        l_diffs.append(l_diff_norm)
        mu_diffs.append(mu_diff_norm)
        eigs.append((raw_eval_0, raw_eval_max))

        if plot_req['tv']:
            total_vars.append(mlmc.tool.simple_distribution.total_variation_distr_diff(exact_distr, raw_distr))

        if plot_req['l2']:
            L2_dist.append(mlmc.tool.simple_distribution.L2_distance(exact_distr.density, raw_distr.density, *case.domain))

        kl_divs.append(mlmc.tool.simple_distribution.KL_divergence(exact_distr.density, raw_distr.density, *case.domain))
        if plot_req['sqrt_tv_Cr']:
            TV_distr_diff.append(mlmc.tool.simple_distribution.TV_distr_diff(exact_distr, raw_distr))

        if plot_req['reg_term']:
            reg_terms.append(mlmc.tool.simple_distribution.reg_term_distr_diff(exact_distr, raw_distr))

    plot_mu_to_lambda_lim = False
    plot_kl_lambda_diff = False

    size = 5
    scatter_size = size ** 2

    barron_coef = 2 * b_factor_estimate * np.exp(1)

    if plot_mu_to_lambda_lim:
        Y = np.array(l_diffs) * np.array(np.array(eigs)[:, 0]) / np.array(mu_diffs)
        ax, lx = plot_scatter(ax, mu_diffs, Y, title, ('log', 'linear'), color='red')
        ax.set_ylabel("$\\alpha_0|\lambda_0 - \lambda_r| / |\mu_0 - \mu_r|$")
        ax.set_xlabel("$|\mu_0 - \mu_r|$")
        ax.axhline(y=1.0, color='red', alpha=0.3)

    elif plot_kl_lambda_diff:
        plot_scatter(ax, mu_diffs, np.array(l_diffs) * np.array(np.array(eigs)[:, 0]), title, ('log', 'log'), color='red', s=scatter_size,
                     )#label="$\\alpha_0|\lambda_0 - \lambda_r|$")

        plot_scatter(ax, mu_diffs, np.sqrt(np.array(kl_divs) / barron_coef), title, ('log', 'log'), color='blue',
                     s=scatter_size)#, label="$\sqrt{D(\\rho || \\rho_{R}) / C_R}$")

        plot_scatter(ax, mu_diffs, np.sqrt(np.array(l_diffs)**2 / barron_coef), title, ('log', 'log'), color='orange',
                       s=scatter_size)#, label="$|\lambda_0 - \lambda_r| / \sqrt{C_R}$")


        #plot_scatter(ax, mu_diffs, np.sqrt(dot_l_diff_mu_diff/ barron_coef), title, ('log', 'log'), color='black', s=scatter_size)

    else:
        Y = np.array(l_diffs) * np.array(np.array(eigs)[:, 0]) / np.array(mu_diffs)
        #Y = np.array(eigs)

        #ax, lx = plot_scatter(ax, l_diffs, mu_diffs, title, ('log', 'log'), color='red')
        ax, lx = plot_scatter(ax, mu_diffs, np.array(l_diffs) * np.array(np.array(eigs)[:, 0]),
                              title, ('log', 'log'), color='red', s=scatter_size)

        #if plot_req['tv']:
            # rescale
            #total_vars = np.array(total_vars) / 10000000

            #ax, lx = plot_scatter(ax, mu_diffs, total_vars, title, ('log', 'log'), color='green', s=size**2)
            #ax, lx = plot_scatter(ax, mu_diffs,  np.sqrt(2 * np.log(np.exp(1)) * np.array(total_vars) ** 2 / barron_coef), title, ('log', 'log'), color='magenta', s=size ** 2)
        #plot_scatter(ax, mu_diffs, Y[:, 1], title, ('log', 'log'), color='blue')
        ax.set_xlabel("$|\mu_0 - \mu_r|$")
        #ax.set_xlabel("$|\lambda_0 - \lambda_r|$")

        outline = mpe.withStroke(linewidth=size, foreground='black')

        ax.plot(lx, lx, color='black', lw=size-3,
                path_effects=[outline])
        #ax.plot(lx, lx, color='m', lw=5.0)

        #ax.plot(lx, lx, color='red', label="raw $1/\\alpha_0$", alpha=0.3)


        # if plot_req['sqrt_kl_Cr']:
        #     plot_scatter(ax, mu_diffs, np.sqrt(np.array(kl_divs) / barron_coef), title, ('log', 'log'),
        #                  color='blue',
        #                  s=scatter_size)

        #kl_divs = np.array(l_diffs)**2

        if plot_req['sqrt_kl_Cr']:
            plot_scatter(ax, mu_diffs, np.sqrt(np.array(dot_l_diff_mu_diff) / barron_coef), title, ('log', 'log'), color='blue', s=scatter_size)

        # kl_divs = np.array(l_diffs)**2
        #
        # if plot_req['sqrt_kl']:
        #     plot_scatter(ax, mu_diffs, np.sqrt(np.array(kl_divs)), title, ('log', 'log'), color='blue',
        #                  s=scatter_size)

        # if plot_req['sqrt_kl_Cr']:
        #     plot_scatter(ax, mu_diffs, np.sqrt(np.array(kl_divs)/barron_coef), title, ('log', 'log'), color='blue',
        #                  s=scatter_size)

        if plot_req['barron_diff_mu_line']:
            ax.plot(mu_diffs, np.array(mu_diffs) * barron_coef, color='blue', lw=size - 3,
                    path_effects=[outline])

        if plot_req['1_eig0_diff_mu_line']:
            ax.plot(mu_diffs, np.array(mu_diffs) * 1/np.array(np.array(eigs)[:, 0]), color='red', lw=size - 3,
                    path_effects=[outline])

        if plot_req['l2']:
            plot_scatter(ax, mu_diffs, np.array(L2_dist), title, ('log', 'log'), color='orange',
                         s=scatter_size)

        if plot_req['sqrt_tv_Cr']:
            ax, lx = plot_scatter(ax, mu_diffs,
                                  np.sqrt(2 * np.log(np.exp(1)) * np.array(TV_distr_diff) ** 2 / barron_coef),
                                  title, ('log', 'log'), color='green', s=scatter_size)

        if plot_req['reg_term']:
            plot_scatter(ax, mu_diffs, np.array(reg_terms), title, ('log', 'log'), color='brown',
                         s=scatter_size)

        # shaw_mu_lim = 1 / (4 * np.exp(1) * b_factor_estimate * Am_factor_estimate)

        #ax.plot(lx, lx * barron_coef, color='blue', label="shaw", alpha=0.3)
        # ax.plot(lx, np.sqrt(lx * shaw_coef), color='blue', label="raw $1/\\alpha_0$", lw=size - 3,
        #         path_effects=[outline])
        # ax.plot(mu_diffs, np.sqrt(kl_divs / shaw_coef), color='blue', label="raw $1/\\alpha_0$", lw=size - 3,
        #         path_effects=[outline])

        # #ax.axvline(x=shaw_mu_lim, color='blue', alpha=0.3)
        # case.eigenvalues_plot.show("")

    # def plot_mineig_by_lambda():
    #     plt.suptitle(case.title)
    #     lx = np.geomspace(1e-10, 0.1, 100)
    #     Y = exact_eval_0 * np.ones_like(lx)
    #     plt.plot(lx, Y, color='red')
    #
    #     plt.scatter(l_diffs, eigs, marker='.')
    #     #plt.ylim((1e-5, 0.1))
    #     plt.xlim((1e-5, 0.1))
    #     # #lx = np.linspace(1e-10, 0.1, 100)
    #     # plt.plot(lx, lx / raw_eval_0, color='orange')
    #     # #plt.plot(lx, lx / raw_eval_max, color='green')
    #     plt.xscale('log')
    #     # plt.yscale('log')
    #     plt.show()


def plot_scatter(ax, X, Y, title, xy_scale, xlim=None, ylim=None, **kw):
    ax.set_title(title)
    ax.set_xscale(xy_scale[0])
    ax.set_yscale(xy_scale[1])
    if xy_scale[0] == 'log':
        if xlim is None:
            ax.set_xlim((1e-5, 1e1))
        else:
            ax.set_xlim(xlim)
        lx = np.geomspace(1e-5, 1e1, 100)
    else:
        #ax.set_xlim((0, 1))
        pass
    if xy_scale[1] == 'log':
        if ylim is None:
            ax.set_ylim((1e-5, 1e1))
        else:
            ax.set_ylim(ylim)
    else:
        if ylim is None:
            ax.set_ylim((0, 1.2))
        else:
            ax.set_ylim(ylim)
    ax.scatter(X, Y, edgecolors='none', **kw)
    return ax, lx


class DistrTestCase:
    """
    Common code for single combination of cut distribution and moments configuration.
    """
    def __init__(self, distr_cfg, moments_cfg):
        print("distr cfg" , distr_cfg)
        (distr_class, quantile, log_flag) = distr_cfg
        self.restrict_distr = RestrictDistribution.from_quantiles(distr_class, quantile)

        self.domain = RestrictDistribution.domain_for_quantile(self.restrict_distr.distr, lower=quantile, upper=1-quantile)

        self.moment_class, self.min_n_moments, self.max_n_moments, self.log_flag = moments_cfg
        self.moments_fn = self.moment_class(self.max_n_moments, self.domain, log=log_flag, safe_eval=False)

        self.exact_covariance = mlmc.tool.simple_distribution.compute_semiexact_cov(self.moments_fn, self.restrict_distr.pdf)
        self.eigenvalues_plot = mlmc.tool.plot.Eigenvalues(title="Eigenvalues, " + self.title)

    @property
    def title(self):
        fn_name = str(self.moment_class.__name__)
        return "distr: {} moment_fn: {}".format(self.restrict_distr.distr_name, fn_name)

    def noise_cov(self, noise):
        noise_mat = np.random.randn(self.moments_fn.size, self.moments_fn.size)
        noise_mat = 0.5 * noise * (noise_mat + noise_mat.T)
        noise_mat[0, 0] = 0
        return self.exact_covariance + noise_mat

    def make_orto_moments(self, noise):
        cov = self.noise_cov(noise)
        orto_moments_fn, info, cov_centered = mlmc.tool.simple_distribution.construct_orthogonal_moments(self.moments_fn, cov, tol=noise)
        original_evals, evals, threshold, L = info
        self.L = L

        print("threshold: ", threshold, " from N: ", self.moments_fn.size)
        self.eigenvalues_plot.add_values(evals, threshold=evals[threshold], label="{:5.2e}".format(noise))
        eye_approx = L @ cov @ L.T
        # test that the decomposition is done well
        assert np.linalg.norm(eye_approx - np.eye(*eye_approx.shape)) < 1e-8
        # TODO: test deviance from exact covariance in some sense
        self.exact_orto_moments = mlmc.tool.simple_distribution.compute_semiexact_moments(orto_moments_fn, self.restrict_distr.pdf, tol=1e-13)

        tol_density_approx = 0.01
        moments_data = np.ones((orto_moments_fn.size, 2))
        moments_data[1:, 0] = 0.0
        #moments_data[0,1] = 0.01
        return orto_moments_fn, moments_data



if __name__ == "__main__":
    run_distr()