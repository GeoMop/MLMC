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
        p0, p1 = distr.cdf(self.domain)
        self.shift = p0
        self.scale = 1 / (p1 - p0)

        if 'dist' in distr.__dict__:
            self.distr_name = distr.dist.name
        else:
            self.distr_name = distr.name

    @staticmethod
    def domain_for_quantile(distr, quantile):
        """
        Determine domain from quantile. Detect force boundaries.
        :param distr: exact distr
        :param quantile: lower bound quantile, 0 = domain from random sampling
        :return: (lower bound, upper bound), (force_left, force_right)
        """
        if quantile == 0:
            # Determine domain by MC sampling.
            if hasattr(distr, "domain"):
                domain = distr.domain
            else:
                X = distr.rvs(size=100000)
                err = stats.norm.rvs(size=100000)
                #X = X * (1 + 0.1 * err)
                domain = (np.min(X), np.max(X))
            # p_90 = np.percentile(X, 99)
            # p_01 = np.percentile(X, 1)
            # domain = (p_01, p_90)

            #domain = (-20, 20)
        else:
            domain = distr.ppf([quantile, 1 - quantile])

        # Detect PDF decay on domain boundaries, test that derivative is positive/negative for left/right side.
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


distr_names = {'_norm': "norm", '_lognorm': "lognorm", '_two_gaussians': "two_gaussians", "_five_fingers": "five_fingers",
               "_cauchy": "cauchy", "_discontinuous": "discontinuous"}


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
        self._name = None

        if 'dist' in distr.__dict__:
            self.d_name = distr.dist.name
            self.distr_name = "{:02}_{}".format(i_distr, distr.dist.name)
            self.cut_distr = CutDistribution(distr, quantile)
        else:
            self.d_name = distr.name
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
        return "distr: {} quantile: {} moment_fn: {}{}".format(self.distr_name, self.quantile, self.fn_name, cov)

    def pdfname(self, subtitle):
        return "{}_{}.pdf".format(self.title, subtitle)

    @property
    def name(self):
        if self._name is None:
            for distr_name, name in distr_names.items():
                if distr_name in self.distr_name:
                    self._name = name

        return self._name

    @property
    def domain(self):
        return self.cut_distr.domain

    def pdf(self, x):
        return self.cut_distr.pdf(x)

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
                                domain=self.domain, force_decay=self.cut_distr.force_decay)

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
        X = np.linspace(self.cut_distr.domain[0], self.cut_distr.domain[1], 10)
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
        distr_plot = plot.Distribution(exact_distr=self.cut_distr, title=self.title+"_exact", cdf_plot=False,
                                       log_x=self.log_flag, error_plot=False)

        dir_name = "KL_div_exact_{}".format(orth_method)
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

        work_dir = os.path.join(dir_name, self.d_name)

        #########################################
        # Set moments objects
        moment_class, min_n_moments, max_n_moments, self.use_covariance = self.moments_data
        log = self.log_flag
        if min_n_moments == max_n_moments:
            self.moment_sizes = np.array(
                [max_n_moments])
        else:
            self.moment_sizes = np.round(np.exp(np.linspace(np.log(min_n_moments), np.log(max_n_moments), 10))).astype(int)
        self.moments_fn = moment_class(max_n_moments, self.domain, log=log, safe_eval=False)

        #print("moment sizes ", self.moment_sizes)
        self.moment_sizes = [1, 5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 100]
        self.moment_sizes = [25]

        if os.path.exists(work_dir):
            #shutil.rmtree(work_dir)
            raise FileExistsError
        else:
            os.mkdir(work_dir)
        np.save(os.path.join(work_dir, "moment_sizes"), self.moment_sizes)

        kl_plot = plot.KL_divergence(log_y=True, iter_plot=True, kl_mom_err=False, title="Kullback-Leibler divergence, {}".format(self.title),
                                     xlabel="number of moments", ylabel="KL divergence")

        ###############################################
        # For each moment size compute density
        for i_m, n_moments in enumerate(self.moment_sizes):
            self.moments_fn = moment_class(n_moments, self.domain, log=log, safe_eval=False)
            base_moments = self.moments_fn
            exact_cov = self.compute_cov(self.moments_fn, self.pdf, tol_exact_moments, True, self.moments_fn.size)

            self.moments_fn, info, _ = mlmc.tool.simple_distribution.construct_orthogonal_moments(base_moments,
                                                                                                  exact_cov,
                                                                                                  tol=tol_exact_moments**2,
                                                                                                  orth_method=orth_method)

            self.exact_moments = self.compute_moments(self.moments_fn, self.pdf, tol=tol_exact_moments, use_quad=True)
            orig_eval, evals, threshold, L = info
            n_moments = self.moments_fn.size

            # moments_fn = moment_fn(n_moments, domain, log=log_flag, safe_eval=False )
            moments_data = np.empty((n_moments, 2))
            moments_data[:, 0] = self.exact_moments[:n_moments]
            moments_data[:, 1] = 1.0

            # modif_cov = mlmc.tool.simple_distribution.compute_semiexact_cov(self.moments_fn, self.pdf)
            # diff_norm = np.linalg.norm(modif_cov - np.eye(*modif_cov.shape))
            # print("#{} cov mat norm: {}".format(n_moments, diff_norm))

            result, distr_obj = self.make_approx(mlmc.tool.simple_distribution.SimpleDistribution, 0.0, moments_data, tol=tol_density)

            ##### Moments SSE
            iter_res_mom = []
            for it_mom in distr_obj._iter_moments:
                iter_res_mom.append(np.sum((self.exact_moments[:n_moments] - it_mom) ** 2))
            np.save('{}/{}_{}.npy'.format(work_dir, n_moments, "res_moments"), iter_res_mom)

            distr_plot.add_distribution(distr_obj, label="#{}, KL div: {}".format(n_moments, result.kl))
            results.append(result)

            self._save_distr_data(distr_obj, distr_plot, work_dir, n_moments, result)

            kl_plot.add_value((n_moments, result.kl))
            kl_plot.add_iteration(x=n_moments, n_iter=result.nit, failed=not result.success)

            self._save_kl_data_exact(work_dir, n_moments, result.kl, result.nit, not result.success, threshold)

        #self.check_convergence(results)
        #kl_plot.show(None)
        distr_plot.show(None)#file=self.pdfname("_pdf_exact"))
        distr_plot.reset()

        return results, dir_name

    def _save_kl_data_exact(self, work_dir, n_moments, kl_div, nit, success, threshold):
        np.save('{}/{}_{}.npy'.format(work_dir, n_moments, "add-value"), (n_moments, kl_div))
        np.save('{}/{}_{}.npy'.format(work_dir, n_moments, "add-iteration"), (n_moments, nit, success))
        np.save('{}/{}_{}.npy'.format(work_dir, n_moments, "threshold"), threshold)

    def _save_distr_data(self, distr_object, distr_plot, work_dir, noise_level, result, name=""):
        domain = distr_object.domain
        distr_plot.adjust_domain(domain)
        X = distr_plot._grid(10000, domain=domain)

        np.save('{}/{}_{}.npy'.format(work_dir, noise_level, "result" + name), (result.kl, result.kl_2, result.l2,
                                                                         result.residual_norm, result.time))
        np.save('{}/{}_{}.npy'.format(work_dir, noise_level, "domain" + name), distr_object.domain)
        np.save('{}/{}_{}.npy'.format(work_dir, noise_level, "X" + name), X)
        np.save('{}/{}_{}.npy'.format(work_dir, noise_level, "Y_pdf" + name), distr_object.density(X))
        np.save('{}/{}_{}.npy'.format(work_dir, noise_level, "Y_cdf" + name), distr_object.cdf(X))
        np.save('{}/{}_{}.npy'.format(work_dir, noise_level, "Y_pdf_exact" + name), self.cut_distr.pdf(X))
        np.save('{}/{}_{}.npy'.format(work_dir, noise_level, "Y_cdf_exact" + name), self.cut_distr.cdf(X))
        np.save('{}/{}_{}.npy'.format(work_dir, noise_level, "Y_pdf_log" + name), distr_object.density_log(X))
        np.save('{}/{}_{}.npy'.format(work_dir, noise_level, "Y_der_1" + name), distr_object.mult_mom_der(X, degree=1))
        np.save('{}/{}_{}.npy'.format(work_dir, noise_level, "Y_der_2" + name), distr_object.mult_mom_der(X, degree=2))

    def plot_KL_div_inexact(self):
        """
        Plot KL divergence for different noise level of exact moments
        """
        min_noise = 1e-6
        max_noise = 1e-1
        geom_seq = np.exp(np.linspace(np.log(min_noise), np.log(max_noise), 20))
        noise_levels = np.flip(np.concatenate(([0.0], geom_seq)), axis=0)

        # noise_levels = noise_levels[:1]
        min_noise = 1e-1
        max_noise = 1e-12
        geom_seq = np.exp(np.linspace(np.log(min_noise), np.log(max_noise), 50))
        noise_levels = np.flip(np.concatenate(([0.0], geom_seq)), axis=0)

        # noise_levels = [1e-1, 1e-2, 1e-3, 1e-4,  1e-5, 1e-6, 1e-8]
        # noise_levels = [1e-2]
        # noise_levels = [1e-4, 1e-5, 1e-6, 1e-8, 1e-10, 1e-12]

        noise_levels = [1e-3]
        noise_level = 0
        tol_exact_moments = 1e-15
        tol_exact_cov = 1e-15
        tol_density = 1e-8
        results = []
        n_moments = 15
        orth_method = 2

        distr_plot = plot.Distribution(exact_distr=self.cut_distr, title=self.title + "_inexact", cdf_plot=False,
                                       log_x=self.log_flag, error_plot=False)

        dir_name = "KL_div_inexact_{}".format(orth_method)
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        else:
            shutil.rmtree(dir_name)
            os.mkdir(dir_name)

        work_dir = os.path.join(dir_name, self.d_name)
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
            tol = noise_level ** 2
            tol = None
            # self.moments_fn, info, _ = mlmc.tool.simple_distribution.construct_orthogonal_moments(base_moments, cov, tol,
            #                                                                               orth_method=orth_method)
            #
            # # Tests
            # original_evals, evals, threshold, L = info
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

            print("n moments ", n_moments)

            if L is None:
                transformed_moments = cov[:, 0]
            else:
                # print("cov moments ", cov[:, 0])
                transformed_moments = np.matmul(cov[:, 0], L.T)
                # print("transformed moments ", transformed_moments)

            print("transformed moments len ", len(transformed_moments))

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
            for it_mom in distr_obj._iter_moments:
                # print("exact moments ", self.exact_moments[:n_moments])
                # print("it mom ", it_mom)
                # print("self it mom shape ", it_mom.shape)
                # print("(self.exact_moments[:n_moments] - it_mom) ", (self.exact_moments[:n_moments] - it_mom))
                iter_res_mom.append(np.sum((moments_data[:, 0] - it_mom) ** 2))

            # print("iter res mom ", iter_res_mom)

            np.save('{}/{}.npy'.format(work_dir, "res_moments"), iter_res_mom)

            self._save_distr_data(distr_obj, distr_plot, work_dir, noise_level, result)

            kl_div = mlmc.tool.simple_distribution.KL_divergence(distr_obj_exact.density, distr_obj.density,
                                                                 self.domain[0], self.domain[1])
            # total_variation = mlmc.tool.simple_distribution.total_variation_int(distr_obj.density, self.domain[0], self.domain[1])

            kl_plot.add_value((noise_level, kl_div))
            kl_plot.add_iteration(x=noise_level, n_iter=result.nit, failed=not result.success)
            diff_orig = np.array(exact_moments_orig) - np.array(cov[:, 0][:len(exact_moments_orig)])
            kl_plot.add_moments_l2_norm((noise_level, np.linalg.norm(diff_orig) ** 2))

            self._save_kl_data(work_dir, noise_level, kl_div, result.nit, not result.success,
                               np.linalg.norm(diff_orig) ** 2, threshold, total_variation=result.tv)

        #kl_plot.show(None)
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
        distr_plot = plot.Distribution(exact_distr=self.cut_distr, title="Preconditioning reg, {},  n_moments: {}, noise: {}".format(self.title, n_moments, max_noise),
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


@pytest.mark.skip
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

        tests = [case.plot_KL_div_exact]
        #tests = [case.plot_KL_div_inexact]

        for test_fn in tests:
            name = test_fn.__name__
            test_results, dir_name = test_fn()

            values = conv.setdefault(name, (case.title, []))
            values[1].append(test_results)

    return name, dir_name

def run_distr():
    distribution_list = [
        # distibution, log_flag
        # (stats.dgamma(1,1), False) # not good
        # (stats.beta(0.5, 0.5), False) # Looks great

        # (stats.lognorm(loc=0, scale=10), False),

        (bd.TwoGaussians(name='two-gaussians'), False),
        (bd.FiveFingers(name='five-fingers'), False),  # Covariance matrix decomposition failed
        (bd.Cauchy(name='cauchy'), False),  # pass, check exact
        # # # # # # # ##(bd.Discontinuous(name='discontinuous'), False),
        (stats.lognorm(scale=np.exp(1), s=1), False),
        (bd.Abyss(name="abyss"), False),
        (bd.ZeroValue(name='zero-value'), False),
        # # # # # # # # # # # # # # # # # # # #(bd.Gamma(name='gamma'), False) # pass
        # # # # # # # # # # # # # # # # # # # #(stats.norm(loc=1, scale=2), False),

        # Quite hard but peak is not so small comparet to the tail.
        # # (stats.lognorm(scale=np.exp(-3), s=2), False),  # Extremely difficult to fit due to very narrow peak and long tail.
        # (stats.lognorm(scale=np.exp(-3), s=2), True),    # Still difficult for Lagrange with many moments.
        # (stats.chi2(df=10), False),# Monomial: s1=nan, Fourier: s1= -1.6, Legendre: s1=nan
        # (stats.chi2(df=5), True), # Monomial: s1=-10, Fourier: s1=-1.6, Legendre: OK
        # (stats.weibull_min(c=0.5), False),  # Exponential # Monomial stuck, Fourier stuck
        # (stats.weibull_min(c=1), False),  # Exponential
        # (stats.weibull_min(c=2), False),  # Rayleigh distribution
        # (stats.weibull_min(c=5, scale=4), False),   # close to normal
        # (stats.weibull_min(c=1.5), True),  # Infinite derivative at zero
    ]

    # @pytest.mark.skip
    mom = [
        # moments_class, min and max number of moments, use_covariance flag
        #.(moments.Monomial, 10, 10, True),
        # (moments.Fourier, 5, 61),
        # (moments.Legendre, 7,61, False),
        (moments.Legendre, 25, 100, True),
        #(moments.Spline, 10, 10, True),
    ]

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

    for m in mom:
        for distr in enumerate(distribution_list):
            #test_spline_approx(m, distr)
            #splines_indicator_vs_smooth(m, distr)
            name, dir_name = test_pdf_approx_exact_moments(m, distr)

    print("name ", name)
    if name == "plot_KL_div_exact":
        test.plot_numpy.plot_KL_div_exact_iter(os.path.abspath(dir_name))
    else:
        test.plot_numpy.plot_KL_div_inexact_iter(os.path.abspath(dir_name))


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
    quantile = 0.01
    idx, distr_cfg = distribution

    if gauss_degree is not None:
        mlmc.tool.simple_distribution.GUASS_DEGREE = gauss_degree

    case = DistrTestCase(distr_cfg, quantile, moments)

    title = case.distr.distr_name#case.title
    if title == "norm":
        title = "normal"

    # if gauss_degree is not None:
    #     title = case.title + " gauss degree: {}".format(gauss_degree)
    orto_moments, moment_data = case.make_orto_moments(0)
    exact_distr = mlmc.tool.simple_distribution.SimpleDistribution(orto_moments, moment_data,
                                                            domain=case.distr.domain,
                                                            force_decay=case.distr.force_decay)

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
                                                            domain=case.distr.domain,
                                                            force_decay=case.distr.force_decay)
    size = len(exact_distr.multipliers)
    linf_log_approx_error = np.max(np.log(case.distr.pdf(exact_distr._quad_points))
                                   - np.log(exact_distr.density(exact_distr._quad_points)))
    b_factor_estimate = np.exp(linf_log_approx_error)
    linf_inv_distr = np.max(1/case.distr.pdf(exact_distr._quad_points))
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
            L2_dist.append(mlmc.tool.simple_distribution.L2_distance(exact_distr.density, raw_distr.density, *case.distr.domain))

        kl_divs.append(mlmc.tool.simple_distribution.KL_divergence(exact_distr.density, raw_distr.density, *case.distr.domain))
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
    def __init__(self, distr_cfg, quantile, moments_cfg):
        distr, log_flag = distr_cfg
        self.distr = CutDistribution(distr, quantile)

        self.moment_class, self.min_n_moments, self.max_n_moments, self.log_flag = moments_cfg
        self.moments_fn = self.moment_class(self.max_n_moments, self.distr.domain, log=log_flag, safe_eval=False)

        self.exact_covariance = mlmc.tool.simple_distribution.compute_semiexact_cov(self.moments_fn, self.distr.pdf)
        self.eigenvalues_plot = mlmc.tool.plot.Eigenvalues(title="Eigenvalues, " + self.title)

    @property
    def title(self):
        fn_name = str(self.moment_class.__name__)
        return "distr: {} moment_fn: {}".format(self.distr.distr_name, fn_name)

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
        print("np.linalg.norm(eye_approx - np.eye(*eye_approx.shape)) ", np.linalg.norm(eye_approx - np.eye(*eye_approx.shape)))

        assert np.linalg.norm(eye_approx - np.eye(*eye_approx.shape)) < 1e-8
        # TODO: test deviance from exact covariance in some sense
        self.exact_orto_moments = mlmc.tool.simple_distribution.compute_semiexact_moments(orto_moments_fn, self.distr.pdf, tol=1e-13)

        tol_density_approx = 0.01
        moments_data = np.ones((orto_moments_fn.size, 2))
        moments_data[1:, 0] = 0.0
        #moments_data[0,1] = 0.01
        return orto_moments_fn, moments_data


def run_mlmc(n_levels, n_moments, cut_distr, log_flag, quantile, moments_fn, target_var, mlmc_file=None):
    mc_test = test.fixtures.mlmc_test_run.MLMCTest(n_levels, n_moments, cut_distr, log_flag, sim_method='_sample_fn', quantile=quantile,
                       mlmc_file=mlmc_file)

    mc_test.moments_fn = moments_fn

    #estimator = mlmc.archive.estimate.Estimate(mc_test.mc)
    mc_test.mc.set_initial_n_samples()#[500000])#[10000, 2000, 500, 50])
    mc_test.mc.refill_samples()
    mc_test.mc.wait_for_simulations()
    mc_test.mc.select_values({"quantity": (b"quantity_1", "="), "time": (1, "<")})
    if mlmc_file is None:
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


def save_mlmc(mlmc, path):
    with open(path, "wb") as writer:
        pickle.dump(mlmc, writer)


def load_mlmc(path):
    with open(path, "rb") as writer:
        mlmc = pickle.load(writer)
    return mlmc


def splines_indicator_vs_smooth(m, distr):
    np.random.seed(1234)
    quantiles = np.array([0.001])
    # i_distr, distr = distr
    # distribution, log_flag = distr
    n_levels = 1
    n_moments = 2

    target_var = 1e-4

    orth_method = 2

    interpolation_points = [200, 220, 240, 260, 280]#[10, 20, 30]

    for quantile in quantiles:

        distr_domain_case = DistributionDomainCase(m, distr, quantile)

        dir_name = "MEM_spline_L:{}_M:{}_TV:{}_q:{}_:int_point".format(n_levels, n_moments, target_var, quantile, interpolation_points)
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

        work_dir = os.path.join(dir_name, distr_domain_case.name)
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)
            os.mkdir(work_dir)
            #raise FileExistsError
        else:
            os.mkdir(work_dir)
            np.save(os.path.join(work_dir, "noise_levels"), target_var)
            np.save(os.path.join(work_dir, "n_moments"), n_moments)

        i_distr, distribution = distr
        distr, log_flag = distribution

        distr = distr_domain_case.cut_distr.distr  # CutDistribution(distribution, quantile)
        cut_distr = distr_domain_case.cut_distr

        moments_fn = Legendre(n_moments, distr_domain_case.cut_distr.domain, log=log_flag, safe_eval=True)


        mlmc_file = None
        #mlmc_file = "/home/martin/Documents/MLMC_spline/data/target_var_1e-2/mlmc_{}.hdf5".format(n_levels)
        mlmc = run_mlmc(n_levels, n_moments, cut_distr.distr, log_flag, quantile, moments_fn, target_var=target_var,
                        mlmc_file=mlmc_file)

        #save_mlmc(mlmc, os.path.join(work_dir, "saved_mlmc"))

        n_samples = []
        for level in mlmc.levels:
            n_samples.append(level._n_collected_samples)

        int_points_domain = cut_distr.domain
        np.save(os.path.join(work_dir, "int_points_domain"), int_points_domain)

        # int_points_domain = [0, 0]
        # int_points_domain[0] = cut_distr.domain[0] - 1000
        # int_points_domain[1] = cut_distr.domain[1] + 1000

        density = True
        spline_plot = plot.Spline_plot(bspline=True,
                                       title="levels: {}, int_points_domain: {}".format(n_levels, int_points_domain),
                                       density=density)

        #interpolation_points = 5
        polynomial_degree = 3  # r=3
        accuracy = 1e-6

        # X = np.linspace(cut_distr.domain[0]-10, cut_distr.domain[1]+10, 1000)
        X = np.linspace(cut_distr.domain[0], cut_distr.domain[1], 10000)

        #distr_obj = make_spline_approx(int_points_domain, mlmc, polynomial_degree, accuracy)

        #interpolation_points = [300, 500, 750, 1000, 1250]
        np.save(os.path.join(work_dir, "polynomial_degree"), polynomial_degree)
        np.save(os.path.join(work_dir, "interpolation_points"), interpolation_points)
        np.save(os.path.join(work_dir, "X"), X)
        np.save(os.path.join(work_dir, "accuracy"), accuracy)
        np.save(os.path.join(work_dir, "density"), density)

        spline_plot.interpolation_points = interpolation_points

        #interpolation_points = [interpolation_points]
        for n_int_points in interpolation_points:
            # distr_obj = make_spline_approx(int_points_domain, mlmc, polynomial_degree, accuracy)
            # distr_obj.moments_fn = moments_fn
            # distr_obj.indicator_method_name = "indicator"
            # distr_obj.n_interpolation_points = n_int_points
            # if density:
            #     distr_obj.density(X)
            #     cdf, pdf = distr_obj.cdf_pdf(X)
            #     np.save(os.path.join(work_dir, "indicator_pdf"), pdf)
            #     np.save(os.path.join(work_dir, "indicator_pdf_X"), X[distr_obj.mask])
            #     spline_plot.add_indicator_density((X[distr_obj.mask], pdf))
            # else:
            #     cdf = distr_obj.cdf(X)
            # np.save(os.path.join(work_dir, "indicator_cdf"), cdf)
            # np.save(os.path.join(work_dir, "indicator_cdf_X"), X[distr_obj.distr_mask])
            # spline_plot.add_indicator((X[distr_obj.distr_mask], cdf))
            #
            # distr_obj = make_spline_approx(int_points_domain, mlmc, polynomial_degree, accuracy)
            # distr_obj.moments_fn = moments_fn
            # distr_obj.indicator_method_name = "smooth"
            # distr_obj.n_interpolation_points = n_int_points
            # if density:
            #     distr_obj.density(X)
            #     cdf, pdf = distr_obj.cdf_pdf(X)
            #     np.save(os.path.join(work_dir, "smooth_pdf"), pdf)
            #     np.save(os.path.join(work_dir, "smooth_pdf_X"), X[distr_obj.mask])
            #     spline_plot.add_smooth_density((X[distr_obj.mask], pdf))
            # else:
            #     cdf = distr_obj.cdf(X)
            # np.save(os.path.join(work_dir, "smooth_cdf"), cdf)
            # np.save(os.path.join(work_dir, "smooth_cdf_X"), X[distr_obj.distr_mask])
            # spline_plot.add_smooth((X[distr_obj.distr_mask], cdf))

            distr_obj = make_spline_approx(int_points_domain, mlmc, polynomial_degree, accuracy, bspline=True)
            distr_obj.moments_fn = moments_fn
            distr_obj.n_interpolation_points = n_int_points
            cdf = distr_obj.cdf(X)
            if density:
                pdf = distr_obj.density(X)
                np.save(os.path.join(work_dir, "spline_pdf"), pdf)
                np.save(os.path.join(work_dir, "spline_pdf_X"), X)
                spline_plot.add_bspline_density((X, pdf))

            np.save(os.path.join(work_dir, "spline_cdf"), cdf)
            np.save(os.path.join(work_dir, "spline_cdf_X"), X)
            spline_plot.add_bspline((X, cdf))

        spline_plot.add_exact_values(X, cut_distr.distr.cdf(X))
        np.save(os.path.join(work_dir, "exact_cdf"), cut_distr.distr.cdf(X))
        if density:
            spline_plot.add_density_exact_values(X, cut_distr.distr.pdf(X))
            np.save(os.path.join(work_dir, "exact_pdf"), cut_distr.distr.pdf(X))

        from statsmodels.distributions.empirical_distribution import ECDF
        level = mlmc.levels[0]
        moments = level.evaluate_moments(moments_fn)
        fine_values = np.squeeze(moments[0])[:, 1]
        fine_values = moments_fn.inv_linear(fine_values)
        ecdf = ECDF(fine_values)
        np.save(os.path.join(work_dir, "ecdf"), ecdf(X))
        np.save(os.path.join(work_dir, "ecdf_X"), X)
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
    plt.title("Compare smoothing polynomial degrees \n MLMC with smoothing, BSpline={},  samples: {} \n accuracy: {} \n n_inter_points: {} \n domain: {} ".format(bspline, n_samples,
                                                                                                                        accuracy, int_point, distr_obj.inter_points_domain))
    plt.legend()
    plt.show()

    exit()

@pytest.mark.skip
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
        int_points_domain = cut_distr.domain

        #if not bspline:
        # int_points_domain = [0, 0]
        # int_points_domain[0] = cut_distr.domain[0] - 100
        # int_points_domain[1] = cut_distr.domain[1] + 100
            #[-500, 500]
            #int_points_domain = [-30, 30]
        #domain = [-50, 50] # not good

        # Remove data standardisation
        #moments_fn.ref_domain = cut_distr.domain
        # moments_fn = Legendre(2, cut_distr.domain, safe_eval=True, log=log_flag)
        # print("moments_fn.domain ", moments_fn.domain)
        #
        # moments = moments_fn.eval_all(data)
        # data = moments[:, 1]

        interpolation_points = 5
        polynomial_degree = 3  # r=3
        accuracy = 1e-6

        #X = np.linspace(cut_distr.domain[0]-10, cut_distr.domain[1]+10, 1000)
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


    # a, b = cut_distr.domain
    # result.kl = mlmc.tool.simple_distribution.KL_divergence(cut_distr.distr.pdf, distr_obj.density, a, b)
    # result.l2 = mlmc.tool.simple_distribution.L2_distance(cut_distr.distr.pdf, distr_obj.density, a, b)
    # result.tv = mlmc.tool.simple_distribution.total_variation_int(distr_obj.density_derivation, a, b)
    # print(result)
    # X = np.linspace(cut_distr.domain[0], cut_distr.domain[1], 10)
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
    run_distr()
    # print("celkový čas ", t.time() - zacatek)

    # import cProfile
    # import pstats
    # pr = cProfile.Profile()
    # pr.enable()

    # my_result = run_distr()
    #
    # pr.disable()
    # ps = pstats.Stats(pr).sort_stats('cumtime')
    # ps.print_stats()
