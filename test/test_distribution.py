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
import sys
import time
import pytest

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
import mlmc.quantity
import pandas as pd
import pickle


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

    def setup_moments(self, moments_data, noise_level, reg_param=0, orth_method=2, regularization=None):
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
            exact_cov, reg_matrix = mlmc.tool.simple_distribution.compute_semiexact_cov_2(base_moments, self.pdf,
                                                                                     regularization=regularization, reg_param=reg_param)

            self.moments_without_noise = exact_cov[:, 0]
            exact_without_reg = exact_cov

            # Add regularization
            exact_cov += reg_matrix

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
                                                                                                        reg_param=reg_param,
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
            if self.eigenvalues_plot:
                threshold = original_evals[threshold]
                noise_label = "{:5.2e}".format(noise_level)
                self.eigenvalues_plot.add_values(original_evals, threshold=threshold, label=noise_label)

                # noise_label = "original evals, {:5.2e}".format(noise_level)
                # self.eigenvalues_plot.add_values(original_evals, threshold=threshold, label=noise_label)

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
        #         domain[0], domain[1], n_failed[-1], tot_nit, cumtime / tot_nit, s1, s0, min_err, max_err))

    def make_approx(self, distr_class, noise, moments_data, tol, reg_param=0, regularization=None):
        result = ConvResult()

        distr_obj = distr_class(self.moments_fn, moments_data,
                                domain=self.domain, force_decay=self.cut_distr.force_decay, reg_param=reg_param,
                                regularization=regularization)

        # multipliers = None
        # if prior_distr_obj is not None:
        #     multipliers = prior_distr_obj.multipliers
        #     distr_obj.reg_domain = [distr_obj.moments_fn.domain[0], 0]

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
        result.kl = mlmc.tool.simple_distribution.KL_divergence(self.pdf, distr_obj.density, a, b)
        result.kl_2 = mlmc.tool.simple_distribution.KL_divergence_2(self.pdf, distr_obj.density, a, b)
        result.l2 = mlmc.tool.simple_distribution.L2_distance(self.pdf, distr_obj.density, a, b)
        result.tv = mlmc.tool.simple_distribution.total_variation_int(distr_obj.density_derivation, a, b)
        print(result)
        X = np.linspace(self.cut_distr.domain[0], self.cut_distr.domain[1], 10)
        density_vals = distr_obj.density(X)
        exact_vals = self.pdf(X)
        #print("vals: ", density_vals)
        #print("exact: ", exact_vals)
        return result, distr_obj

    def mlmc_conv(self, distr_plot=None):
        results = []
        kl_divergences = []
        target_vars = [1e-6]
        distr_accuracy = 1e-8
        reg_params = [0]
        mom_class, min_mom, max_mom, mom_log = self.moments_data
        n_levels = [1]#, 3, 5]
        log_flag = self.log_flag
        a, b = self.domain

        for level in n_levels:
            for target_var in target_vars:
                if distr_plot is None:
                    distr_plot = plot.Distribution(exact_distr=self.cut_distr,
                                                   title="Density, {},  n_moments: {}, target_var: {}".format(self.title,
                                                                                                              max_mom,
                                                                                                              target_var),
                                                   log_x=self.log_flag, error_plot=None, reg_plot=False, cdf_plot=False,
                                                   log_density=True)

                mc_test = test.fixtures.mlmc_test_run.MLMCTest(level, max_mom, self.cut_distr, log_flag, "_sample_fn", moments_class=mom_class,
                                   domain=self.cut_distr.domain)

                quantity = mlmc.quantity.make_root_quantity(storage=mc_test.sampler.sample_storage,
                                                            q_specs=mc_test.result_format())
                length = quantity['length']
                time = length[1]
                location = time['10']
                value_quantity = location[0]

                # number of samples on each level
                mc_test.set_estimator(value_quantity)
                mc_test.generate_samples(target_var=target_var)

                estimator = mlmc.estimator.Estimate(quantity=value_quantity,
                                                    sample_storage=mc_test.sampler.sample_storage,
                                                    moments_fn=mc_test.moments_fn)

                for reg_param in reg_params:
                    distr_obj, info, result, moments_fn = estimator.construct_density(
                                                                             tol=distr_accuracy,
                                                                             reg_param=reg_param,
                                                                             orth_moments_tol=target_var,
                                                                             exact_pdf=self.cut_distr.pdf)

                    original_evals, evals, threshold, L = info

                    if level == 1:
                        samples = value_quantity.samples(level_id=0,
                                                         n_samples=mc_test.sampler.sample_storage.get_n_collected()[0])[..., 0]
                        distr_plot.add_raw_samples(np.squeeze(samples))

                    distr_plot.add_distribution(distr_obj, label="n_l: {}, reg_param: {}, th: {}".
                                                format(level, reg_param, threshold),
                                                size=max_mom, reg_param=reg_param)

                    kl =mlmc.tool.simple_distribution.KL_divergence(self.cut_distr.pdf, distr_obj.density,
                                                                    self.cut_distr.domain[0], self.cut_distr.domain[1])
                    kl_divergences.append(kl)
                    #l2 = mlmc.tool.simple_distribution.L2_distance(self.cut_distr.pdf, estimator.distribution, a, b)

        distr_plot.show(None)
        #self._plot_kl_div(target_vars, kl_divergences)
        return results

    def exact_conv(self):
        """
        Test density approximation for varying number of exact moments.
        :return:
        """
        results = []
        distr_plot = plot.Distribution(exact_distr=self.cut_distr, title=self.title+"_exact", cdf_plot=False,
                                            log_x=self.log_flag, error_plot='kl')

        mom_class, min_mom, max_mom, log_flag = self.moments_data
        moments_num = [max_mom]

        for i_m, n_moments in enumerate(moments_num):
            self.moments_data = (mom_class, n_moments, n_moments, log_flag)
            # Setup moments.
            self.setup_moments(self.moments_data, noise_level=0)

            if n_moments > self.moments_fn.size:
                continue
            # moments_fn = moment_fn(n_moments, domain, log=log_flag, safe_eval=False )
            # print(i_m, n_moments, domain, force_decay)
            moments_data = np.empty((n_moments, 2))
            moments_data[:, 0] = self.exact_moments[:n_moments]
            moments_data[:, 1] = 1.0

            if self.use_covariance:
                # modif_cov, reg = mlmc.tool.simple_distribution.compute_exact_cov(self.moments_fn, self.pdf)
                # diff_norm = np.linalg.norm(modif_cov - np.eye(*modif_cov.shape))
                # print("#{} cov mat norm: {}".format(n_moments, diff_norm))

                result, distr_obj = self.make_approx(mlmc.tool.simple_distribution.SimpleDistribution, 0.0, moments_data,
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
        #
        # distr_obj_exact_conv_int = mlmc.tool.simple_distribution.compute_exact_cov(distr_obj.moments_fn, distr_obj.density)
        M = np.eye(len(self._cov_with_noise[0]))
        M[:, 0] = -self._cov_with_noise[:, 0]

        print("M @ L-1 @ H @ L.T-1 @ M.T")
        print(pd.DataFrame(
            M @ (np.linalg.inv(self.L) @ final_jac @ np.linalg.inv(self.L.T)) @ M.T))

        print("orig cov centered")
        print(pd.DataFrame(self._cov_centered))

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

    def find_best_spline(self, mc_test, distr_accuracy, poly_degree, work_dir, target_var):
        # two gausians (best 20-22) all_n_int_points = [10, 12, 14, 16, 18, 20, 22, 24]
        # five fingers (best 200 -220) all_n_int_points = [180, 200, 220, 240, 260, 280]
        # cauchy  - 25, 18,  21 , 16 all_n_int_points = [10, 12, 14, 16, 18, 20, 22, 24]
        # discontinuous - 11, 14, 13
        # norm - 11, 17, 8, 22, 19, 31
        # lognorm 16, 13, 11 10, 12

        interpolation_points = {"norm": range(5, 30, 1), "lognorm": range(5, 25, 1),
                                "two_gaussians": range(10, 30, 1), "five_fingers": range(180, 220, 1),
                                "cauchy": range(10, 30, 1), "discontinuous": range(5, 30, 1)}


        all_n_int_points = interpolation_points[self.name]

        mc_test.set_moments_fn(moments.Legendre)  # mean values for spline approximation
        mc_test.mc.update_moments(mc_test.moments_fn)

        kl_divs = {}
        spline_distr_objects = {}

        for n_int_points in all_n_int_points:
            print("n int points ", n_int_points)
            spline_distr = spline_approx.BSplineApproximation(mc_test.mc, self.domain, poly_degree=poly_degree,
                                                              accuracy=distr_accuracy)

            spline_distr.moments_fn = mc_test.moments_fn
            spline_distr.n_interpolation_points = n_int_points

            spline_kl = mlmc.tool.simple_distribution.KL_divergence(self.cut_distr.pdf, spline_distr.density, self.domain[0],
                                                               self.domain[1])

            kl_divs[n_int_points] = spline_kl
            spline_distr_objects[n_int_points] = spline_distr

            # distr_plot.add_distribution(spline_distr,
            #                             label="BSpline, degree: {}, n_int_points: {}, KL: {}".format(poly_degree,
            #                                                                                          n_int_points,
            #                                                                                          spline_kl))

        np.save('{}/{}_{}.npy'.format(work_dir, target_var, "spline_int_points"), all_n_int_points)

        keys = []
        values = []
        for key, value in kl_divs.items():
            print("key ", key)
            print("value ", value)
            keys.append(key)
            values.append(value)

        print("keys ", keys)
        print("values ", values)
        np.save('{}/{}_{}.npy'.format(work_dir, target_var, "spline_kl_divs"), (keys, values))

        # info = []
        # for index, distr in distr_objects.items():
        #     info.append((distr[1].kl, distr[1].nit, not distr[1].success, distr[2]))
        #
        # np.save('{}/{}_{}.npy'.format(work_dir, noise_level, "info"), info)
        # self._save_distr_spline(spline_distr, distr_plot, work_dir, target_var, spline_kl, "bspline")

        best_kl_divs = sorted(kl_divs, key=lambda par: kl_divs[par])

        return spline_distr_objects[best_kl_divs[0]], kl_divs[best_kl_divs[0]]

    def mc_find_regularization_param(self, plot_res=True, work_dir=None, orth_method=4, n_mom=None,
                                     target_var=1e-4, n_levels=1, mlmc_obj=None, estimator_obj=None):
        #n_levels = 1
        distr_accuracy = 1e-6
        #orth_method = 4

        if work_dir == None:
            dir_name = "mc_find_reg_param"
            if not os.path.exists(dir_name):
                os.mkdir(dir_name)
            else:
                shutil.rmtree(dir_name)
                os.mkdir(dir_name)

            work_dir = os.path.join(dir_name, self.name)
            if os.path.exists(work_dir):
                raise FileExistsError
            else:
                os.mkdir(work_dir)

        rep = 1
        n_reg_params = 20
        reg_params = np.geomspace(1e-11, 1e-4, num=n_reg_params)

        reg_params = np.append(reg_params, [0])
        # reg_params = [1e-9, 1e-7, 1e-6, 1e-5, 1e-4]

        #reg_params = []
        min_results = []

        moment_class, min_n_moments, max_n_moments, self.use_covariance = self.moments_data
        log = self.log_flag
        if min_n_moments == max_n_moments:
            self.moment_sizes = np.array(
                [max_n_moments])  # [36, 38, 40, 42, 44, 46, 48, 50, 52, 54])+1#[max_n_moments])#10, 18, 32, 64])
        else:
            self.moment_sizes = np.round(
                np.exp(np.linspace(np.log(min_n_moments), np.log(max_n_moments), 8))).astype(int)

        if n_mom is not None:
            max_n_moments = n_moments = n_mom

        self.moments_fn = moment_class(max_n_moments, self.domain, log=log, safe_eval=False)

        _, _, n_moments, _ = self.moments_data

        size = 1

        ####################################
        # Run MLMC                         #
        ####################################
        if mlmc_obj is None:
            mc_test = test.fixtures.mlmc_test_run.MLMCTest(n_levels, max_n_moments, self.cut_distr.distr, log, "_sample_fn",
                               moments_class=moment_class,
                               domain=self.cut_distr.domain)
            # number of samples on each level

            mc_test.mc.set_initial_n_samples()
            mc_test.mc.refill_samples()
            mc_test.mc.wait_for_simulations()
            mc_test.mc.select_values({"quantity": (b"quantity_1", "="), "time": (0, "=")})
            estimator = mlmc.archive.estimate.Estimate(mc_test.mc, mc_test.moments_fn)

            estimator.target_var_adding_samples(target_var, mc_test.moments_fn)
            mc = mc_test.mc

            for level in mc.levels:
                print("level sample values ", level._sample_values)
                np.save(os.path.join(work_dir, "level_{}_values".format(level._level_idx)), level._sample_values)

            mc_test.mc.update_moments(mc_test.moments_fn)
            means, vars = estimator.estimate_moments(mc_test.moments_fn)
            print("means ", means)
            print("vars ", vars)

            mlmc_obj = mc_test.mc
            estimator_obj = estimator

        exact_moments = mlmc.tool.simple_distribution.compute_exact_moments(estimator_obj.moments, self.pdf)
        print("exact moments: {}".format(exact_moments))

        num_moments = self.moments_fn.size
        used_reg_params = []

        distr_objects = {}
        kl_divs = {}
        cond_numbers = {}
        all_moments_from_density = {}

        all_num_moments = []
        all_result_norm = []

        for index, reg_param in enumerate(reg_params):
            mlmc_obj.clean_subsamples()
            print("REG PARAMETER ", reg_param)
            regularization = mlmc.tool.simple_distribution.Regularization2ndDerivation()
            #regularization = mlmc.tool.simple_distribution.RegularizationInexact2()

            # self.moments_fn = moment_class(max_n_moments, self.domain, log=log, safe_eval=False)
            # #size = self.moments_fn.size
            # base_moments = self.moments_fn

            ####################################
            # MaxEnt method                    #
            ####################################
            result = ConvResult()
            info, min_result = estimator_obj.construct_density(tol=distr_accuracy, reg_param=reg_param,
                                                           orth_moments_tol=np.sqrt(target_var),
                                                           exact_pdf=self.pdf, orth_method=orth_method)

            original_evals, evals, threshold, L = info

            a, b = self.domain[0], self.domain[1]
            max_ent_kl = mlmc.tool.simple_distribution.KL_divergence(self.cut_distr.pdf, estimator_obj.distribution.density, a, b)

            # distr_plot.add_distribution(estimator._distribution,
            #                             label="reg param: {}, threshold: {}, KL: {}".format(reg_param, threshold,
            #                                                                                 max_ent_kl),
            #                             size=max_mom, reg_param=reg_param)

            t1 = time.time()
            result.residual_norm = min_result.fun_norm
            result.success = min_result.success
            result.success = min_result.success
            result.nit = min_result.nit

            a, b = self.domain
            result.kl = mlmc.tool.simple_distribution.KL_divergence(self.pdf, estimator_obj._distribution.density, a, b)
            result.kl_2 = mlmc.tool.simple_distribution.KL_divergence_2(self.pdf, estimator_obj._distribution.density, a, b)
            result.l2 = mlmc.tool.simple_distribution.L2_distance(self.pdf, estimator_obj._distribution.density, a, b)
            result.tv = 0  # mlmc.tool.simple_distribution.total_variation_int(distr_obj.density_derivation, a, b)

            #fine_means, fine_vars = estimator.estimate_moments(moments_fn=self.moments_fn)
            #print("fine moments ", fine_moments)

            moments_from_density = (np.linalg.pinv(L) @ estimator_obj._distribution.final_jac @ np.linalg.pinv(L.T))[:, 0]

            print("L @ jac @ L.T ", moments_from_density)
            distr_exact_cov, distr_reg_matrix = mlmc.tool.simple_distribution.compute_semiexact_cov_2(estimator_obj.moments,
                                                                                                 estimator_obj._distribution.density,
                                                                                                 reg_param=0,
                                                                                                 regularization=regularization)


            # print("moments approx error: ", np.linalg.norm(moments_from_density - exact_moments[len(moments_from_density)-1]),
            #       "m0: ", moments_from_density[0])

            # print("num moments ", num_moments)
            # print("moments_from_density[:num_moments-1] ", moments_from_density[:num_moments-1])
            # print("self cov centered ", self._cov_centered)

            # print("distr final jac")
            # print(pd.DataFrame(estimator._distribution.final_jac))
            #
            # # print("distr object moment means ", distr_obj.moment_means)
            #
            # print("distr cov moments ", distr_exact_cov[:, 0])

            print("L @ jac @ L.T ", moments_from_density)
            moments_from_density = distr_exact_cov[:, 0]

            print("exact moments from denstiy ", moments_from_density)

            all_moments_from_density[reg_param] = moments_from_density

            if not result.success:
                continue

            used_reg_params.append(reg_param)

            distr_objects[reg_param] = (estimator_obj._distribution, result, threshold, L)

            kl_divs[reg_param] = result.kl

            cond_numbers[reg_param] = estimator_obj._distribution.cond_number

            final_jac = estimator_obj._distribution.final_jac
            #
            # distr_obj_exact_conv_int = mlmc.tool.simple_distribution.compute_exact_cov(distr_obj.moments_fn, distr_obj.density)
            # M = np.eye(len(self._cov_with_noise[0]))
            # M[:, 0] = -self._cov_with_noise[:, 0]

            print("size ", size)

            n_subsamples = 100

            result = []
            result_norm = []
            for _ in range(n_subsamples):
                mlmc_obj.clean_subsamples()
                n_samples = mlmc_obj.n_samples

                subsamples = [int(n_sam * 0.8) for n_sam in n_samples]

                # print("n samples ", n_samples)
                # print("subsamples ", subsamples)

                mlmc_obj.subsample(sub_samples=subsamples)

                # for level in mlmc_obj.levels:
                #     print("level.last_moments_eval ", len(level.last_moments_eval[0]))

                coarse_means, coarse_vars = estimator_obj.estimate_moments(moments_fn=self.moments_fn)

                # print("moments from density ", moments_from_density)
                # print("coarse means ", coarse_means)

                num_moments = len(moments_from_density)

                # print("moments_from_density[:num_moments-1] ", moments_from_density[:num_moments])
                # print("coarse_moments[:num_moments-1] ", coarse_means[:num_moments])

                res = (moments_from_density[:num_moments] - coarse_means[:num_moments]) ** 2

                # res = (moments_from_density - coarse_moments) ** 2
                # res = ((moments_from_density[:num_moments] - coarse_moments[:num_moments])/num_moments) ** 2
                #
                #
                # res = np.linalg.norm(moments_from_density[:num_moments] - coarse_moments[:num_moments])

                # res = res * rations[:num_moments-1]

                result_norm.append(np.array(res) / num_moments)

                print("res to result ", res)
                result.append(res)

            # distr_plot.add_distribution(distr_obj,
            #                             label="noise: {}, threshold: {}, reg param: {}".format(noise_level, threshold,
            #                                                                                    reg_param),
            #                             size=len(coarse_moments), reg_param=reg_param)

            # print("norm result ", result)

            all_num_moments.append(num_moments)
            min_results.append(np.sum(result))  # np.sum(result))
            all_result_norm.append(np.sum(result_norm))

        reg_params = used_reg_params

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        zipped = zip(reg_params, min_results)

        for reg_param, min_result in zip(reg_params, min_results):
            print("reg_param: {}, min_result: {}".format(reg_param, min_result))

        sorted_zip = sorted(zipped, key=lambda x: x[1])

        best_params = []
        #best_params.append(0)
        min_best = None
        for s_tuple in sorted_zip:
            if min_best is None:
                min_best = s_tuple
            print(s_tuple)
            if len(best_params) < 5:
                best_params.append(s_tuple[0])

        best_kl_divs = sorted(kl_divs, key=lambda par: kl_divs[par])
        best_params = best_kl_divs[:5]

        print("best params ", best_params)

        kl_div_to_plot = [kl_divs[r_par] for r_par in reg_params]

        if work_dir is not None:
            if n_mom is not None:
                self._save_reg_param_data(work_dir, n_mom, reg_params, min_results, distr_objects)
            else:
                self._save_reg_param_data(work_dir, target_var, reg_params, min_results, distr_objects, cond_numbers)

        if plot_res:

            res_norm_2 = []
            for res, used_moments in zip(min_results, all_num_moments):
                res_norm_2.append(res * (used_moments / max_n_moments))

            fig, ax = plt.subplots()
            ax.plot(reg_params, min_results, 'o', label="MSE")
            ax.plot(reg_params, kl_div_to_plot, 'v', label="kl div")
            ax.plot(min_best[0], min_best[1], 'x', color='red')
            ax.set_ylabel("MSE")
            ax.set_xlabel(r"$\log(\alpha)$")
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.legend(loc='best')
            logfmt = matplotlib.ticker.LogFormatterExponent(base=10.0, labelOnlyBase=True)
            ax.xaxis.set_major_formatter(logfmt)

            plt.show()

            distr_plot = plot.Distribution(exact_distr=self.cut_distr,
                                           title="Preconditioning reg, {},  n_moments: {}, target var: {}".format(self.title,
                                                                                                             n_moments,
                                                                                                             target_var),
                                           log_x=self.log_flag, error_plot=None, reg_plot=False, cdf_plot=False,
                                           log_density=True)

            if "0" in distr_objects:
                best_params.append(0)
            for reg_par in best_params:
                #print("distr_objects[reg_par] ", distr_objects[reg_par])
                distr_plot.add_distribution(distr_objects[reg_par][0],
                                            label="var: {:0.4g}, th: {}, alpha: {:0.4g},"
                                                  " KL_div: {:0.4g}".format(target_var, distr_objects[reg_par][2], reg_par,
                                                                            distr_objects[reg_par][1].kl),
                                            size=n_moments, mom_indices=False, reg_param=reg_par)

            #self.determine_regularization_param(best_params, regularization, noise=noise_level)
            distr_plot.show(None)

            for reg_par, kl_div in kl_divs.items():
                print("KL: {} reg_param: {}".format(kl_div, reg_par))

            return best_params

        else:

            # exact_cov, reg_matrix = mlmc.tool.simple_distribution.compute_semiexact_cov_2(base_moments, self.pdf,
            #                                                                          reg_param=best_params[0],
            #                                                                          regularization=regularization)
            #
            # cov = exact_cov + reg_matrix

            return best_params, distr_objects[best_params[0]], exact_moments, all_moments_from_density[best_params[0]]

    def compare_spline_max_ent_save(self):
        n_levels = 1
        target_var = 1e-4
        distr_accuracy = 1e-6
        tol_exact_cov = 1e-10
        poly_degree = 3
        n_int_points = 220
        reg_param = 0 # posibly estimate by find_regularization_param()
        orth_method = 2
        mom_class, min_mom, max_mom, _ = self.moments_data

        log_flag = self.log_flag
        a, b = self.domain

        target_vars = [1e-3, 1e-4]

        dir_name = "MEM_spline_orth:{}_L:{}_M:{}".format(orth_method, n_levels, max_mom)
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

        work_dir = os.path.join(dir_name, self.name)
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)
            os.mkdir(work_dir)
            np.save(os.path.join(work_dir, "target_vars"), target_vars)
            np.save(os.path.join(work_dir, "n_moments"), max_mom)
            np.save(os.path.join(work_dir, "int_points_domain"), self.domain)
            #raise FileExistsError
        else:
            os.mkdir(work_dir)
            np.save(os.path.join(work_dir, "target_vars"), target_vars)
            np.save(os.path.join(work_dir, "n_moments"), max_mom)
            np.save(os.path.join(work_dir, "int_points_domain"), self.domain)

        for target_var in target_vars:

            distr_plot = plot.Distribution(exact_distr=self.cut_distr,
                                           title="Density, {},  n_moments: {}, target_var: {}".format(self.title,
                                                                                                      max_mom,
                                                                                                      target_var),
                                           log_x=self.log_flag, error_plot=None, reg_plot=False, cdf_plot=True,
                                           log_density=True, multipliers_plot=False)

            ####################################
            # Run MLMC                         #
            ####################################
            mc_test = test.fixtures.mlmc_test_run.MLMCTest(n_levels, max_mom, self.cut_distr.distr, log_flag, "_sample_fn", moments_class=mom_class,
                               domain=self.cut_distr.domain)
            # number of samples on each level
            if mom_class.__name__ == "Spline":
                mc_test.moments_fn.poly_degree = poly_degree
                print("mc_test.moments_fn ", mc_test.moments_fn)

            mc_test.mc.set_initial_n_samples()
            mc_test.mc.refill_samples()
            mc_test.mc.wait_for_simulations()
            mc_test.mc.select_values({"quantity": (b"quantity_1", "="), "time": (0, "=")})
            estimator = mlmc.archive.estimate.Estimate(mc_test.mc, mc_test.moments_fn)

            estimator.target_var_adding_samples(target_var, mc_test.moments_fn)
            mc = mc_test.mc

            for level in mc.levels:
                print("level sample values ", level._sample_values)
                np.save(os.path.join(work_dir, "level_{}_values".format(level._level_idx)), level._sample_values)

            mc_test.mc.update_moments(mc_test.moments_fn)
            means, vars = estimator.estimate_moments(mc_test.moments_fn)
            print("means ", means)
            print("vars ", vars)
            exact_moments = mlmc.tool.simple_distribution.compute_exact_moments(mc_test.moments_fn, self.pdf)
            print("exact moments: {}".format(exact_moments))


            ####################################
            # MaxEnt method                    #
            ####################################
            result = ConvResult()
            truncation_err, distr_obj_exact = self._compute_exact_kl(max_mom, mc_test.moments_fn, orth_method,
                                                                     distr_accuracy, tol_exact_cov)

            info, min_result = estimator.construct_density(tol=distr_accuracy, reg_param=reg_param,
                                                           orth_moments_tol=np.sqrt(target_var),
                                                           exact_pdf=self.pdf, orth_method=orth_method)

            original_evals, evals, threshold, L = info

            max_ent_kl = mlmc.tool.simple_distribution.KL_divergence(self.cut_distr.pdf, estimator.distribution.density, a,b)

            # distr_plot.add_distribution(estimator._distribution,
            #                             label="reg param: {}, threshold: {}, KL: {}".format(reg_param, threshold,
            #                                                                                 max_ent_kl),
            #                             size=max_mom, reg_param=reg_param)

            t1 = time.time()
            result.residual_norm = min_result.fun_norm
            result.success = min_result.success
            result.success = min_result.success
            result.nit = min_result.nit

            a, b = self.domain
            result.kl = mlmc.tool.simple_distribution.KL_divergence(self.pdf, estimator._distribution.density, a, b)
            result.kl_2 = mlmc.tool.simple_distribution.KL_divergence_2(self.pdf, estimator._distribution.density, a, b)
            result.l2 = mlmc.tool.simple_distribution.L2_distance(self.pdf, estimator._distribution.density, a, b)
            result.tv = 0  # mlmc.tool.simple_distribution.total_variation_int(distr_obj.density_derivation, a, b)

            kl_div = mlmc.tool.simple_distribution.KL_divergence(distr_obj_exact.density, estimator.distribution.density,
                                                            self.domain[0],
                                                            self.domain[1])

            estimated_moments = mlmc.tool.simple_distribution.compute_exact_moments(mc_test.moments_fn, estimator.distribution.density)
            diff_orig = np.array(exact_moments) - np.array(estimated_moments)

            self._save_kl_data(work_dir, target_var, kl_div, result.nit, not result.success,
                               np.linalg.norm(diff_orig) ** 2, threshold)
            self._save_distr_data(estimator._distribution, distr_plot, work_dir, target_var, result)

            ############################
            ##### With regularization ##
            ############################

            _, distr_obj, exact_moments, estimated_moments = self.mc_find_regularization_param(plot_res=False, target_var=target_var,
                                                                             work_dir=work_dir, orth_method=orth_method,
                                                                             mlmc_obj=mc_test.mc, estimator_obj=estimator)
            #exact_moments_orig = exact_cov[:, 0]

            distr_plot.add_distribution(distr_obj[0], label="tar var: {:f}, th: {}, KL div: {:f}".format(target_var,
                                                                                                       distr_obj[2],
                                                                                                       distr_obj[1].kl))

            kl_div = mlmc.tool.simple_distribution.KL_divergence(distr_obj_exact.density, distr_obj[0].density, self.domain[0],
                                                            self.domain[1])
            max_ent_kl = mlmc.tool.simple_distribution.KL_divergence(self.cut_distr.pdf, estimator.distribution.density, a, b)
            np.save('{}/{}_{}.npy'.format(work_dir, target_var, "max_ent_kl"), (target_var, max_ent_kl))

            diff_orig = np.array(exact_moments) - np.array(estimated_moments)

            self._save_kl_data(work_dir, target_var, kl_div, distr_obj[1].nit, not distr_obj[1].success,
                               np.linalg.norm(diff_orig) ** 2, distr_obj[2], name="_reg")

            self._save_distr_data(distr_obj[0], distr_plot, work_dir, target_var, distr_obj[1], name="_reg")


            ####################################
            # Spline approximation             #
            ####################################

            #all_n_int_points = [13]#range(5, 40, 1)

            mc_test.mc.clean_subsamples()
            mc_test.set_moments_fn(moments.Legendre)  # mean values for spline approximation
            mc_test.mc.update_moments(mc_test.moments_fn)

            print("spline domain ", self.domain)

            spline_distr, spline_kl = self.find_best_spline(mc_test, distr_accuracy, poly_degree, work_dir, target_var)


            # kl_divs = []
            #
            # for n_int_points in all_n_int_points:
            #     print("n int points ", n_int_points)
            #     spline_distr = spline_approx.BSplineApproximation(mc_test.mc, self.domain, poly_degree=poly_degree,
            #                                                       accuracy=distr_accuracy)
            #
            #     spline_distr.moments_fn = mc_test.moments_fn
            #     spline_distr.n_interpolation_points = n_int_points
            #
            #     spline_kl = mlmc.tool.simple_distribution.KL_divergence(self.cut_distr.pdf, spline_distr.density, a, b)
            #
            #     kl_divs.append((n_int_points, spline_kl))
            #
            #     distr_plot.add_distribution(spline_distr,
            #                                 label="BSpline, degree: {}, n_int_points: {}, KL: {}".format(poly_degree,
            #                                                                                              n_int_points,
            #                                                                                              spline_kl))

            self._save_distr_spline(spline_distr, distr_plot, work_dir, target_var, spline_kl, "_bspline")


            #best_kl_divs = sorted(kl_divs, key=lambda x: x[1])

            #print("BEST KL divs ", best_kl_divs)

            # # #####################################
            # # #### Indicator interpolation  #
            # # #####################################
            indicator_kl = 10
            # mc_test.set_moments_fn(moments.Legendre)  # mean values for spline approximation
            # mc_test.mc.update_moments(mc_test.moments_fn)
            #
            # spline_distr = spline_approx.SplineApproximation(mc_test.mc, self.domain, poly_degree=poly_degree,
            #                                                  accuracy=distr_accuracy)
            #
            # spline_distr.moments_fn = mc_test.moments_fn
            # spline_distr.indicator_method_name = "indicator"
            # spline_distr.n_interpolation_points = n_int_points
            #
            # indicator_kl = mlmc.tool.simple_distribution.KL_divergence(self.cut_distr.pdf, spline_distr.density, a, b)
            #
            # distr_plot.add_spline_distribution(spline_distr,
            #                             label="Indicator, degree: {}, n_int_points: {}, KL: {}".format(poly_degree,
            #                                                                                          n_int_points,
            #                                                                                          indicator_kl))
            #
            # self._save_distr_spline(spline_distr, distr_plot, work_dir, target_var, indicator_kl, "indicator")
            #
            # #####################################
            # #### Smooth interpolation           #
            # #####################################
            smooth_kl=10
            # mc_test.set_moments_fn(moments.Legendre)  # mean values for spline approximation
            # mc_test.mc.update_moments(mc_test.moments_fn)
            #
            # spline_distr = spline_approx.SplineApproximation(mc_test.mc, self.domain, poly_degree=poly_degree,
            #                                                            accuracy=distr_accuracy)
            #
            # spline_distr.moments_fn = mc_test.moments_fn
            # spline_distr.indicator_method_name = "smooth"
            # spline_distr.n_interpolation_points = n_int_points
            #
            # smooth_kl = mlmc.tool.simple_distribution.KL_divergence(self.cut_distr.pdf, spline_distr.density, a, b)
            #
            # distr_plot.add_spline_distribution(spline_distr,
            #                             label="Smooth, degree: {}, n_int_points: {}, KL: {}".format(poly_degree,
            #                                                                                          n_int_points,
            #                                                                                          smooth_kl))
            # self._save_distr_spline(spline_distr, distr_plot, work_dir, target_var, smooth_kl, "smooth")

            ####################################
            # KL divergences                   #
            ####################################

            distr_plot.show(None)
            plt.show()

            print("KL div - MEM:{:0.4g}, BSpline:{:0.4g}, smooth:{:0.4g}, indicator:{:0.4g}".format(max_ent_kl, spline_kl,
                                                                                                    smooth_kl, indicator_kl))

    def compare_spline_max_ent(self):
        n_levels = 1
        target_var = 1e-4
        distr_accuracy = 1e-6
        poly_degree = 3
        n_int_points = 220
        reg_param = 0 # posibly estimate by find_regularization_param()
        orth_method = 2
        mom_class, min_mom, max_mom, _ = self.moments_data

        log_flag = self.log_flag
        a, b = self.domain

        dir_name = "MEM_spline_L:{}_M:{}_TV:{}_:int_point".format(n_levels, max_mom, target_var, n_int_points)
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

        work_dir = os.path.join(dir_name, self.name)
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)
            os.mkdir(work_dir)
            #raise FileExistsError
        else:
            os.mkdir(work_dir)
            np.save(os.path.join(work_dir, "noise_levels"), target_var)
            np.save(os.path.join(work_dir, "n_moments"), max_mom)
            np.save(os.path.join(work_dir, "int_points_domain"), self.domain)

        distr_plot = plot.Distribution(exact_distr=self.cut_distr,
                                       title="Density, {},  n_moments: {}, target_var: {}".format(self.title,
                                                                                                  max_mom,
                                                                                                  target_var),
                                       log_x=self.log_flag, error_plot=None, reg_plot=False, cdf_plot=True,
                                       log_density=True, multipliers_plot=False)

        ####################################
        # Run MLMC                         #
        ####################################
        mc_test = test.fixtures.mlmc_test_run.MLMCTest(n_levels, max_mom, self.cut_distr.distr, log_flag, "_sample_fn", moments_class=mom_class,
                           domain=self.cut_distr.domain)
        # number of samples on each level
        if mom_class.__name__ == "Spline":
            mc_test.moments_fn.poly_degree = poly_degree
            print("mc_test.moments_fn ", mc_test.moments_fn)

        mc_test.mc.set_initial_n_samples()
        mc_test.mc.refill_samples()
        mc_test.mc.wait_for_simulations()
        mc_test.mc.select_values({"quantity": (b"quantity_1", "="), "time": (0, "=")})
        estimator = mlmc.archive.estimate.Estimate(mc_test.mc, mc_test.moments_fn)

        estimator.target_var_adding_samples(target_var, mc_test.moments_fn)
        mc = mc_test.mc

        for level in mc.levels:
            print("level sample values ", level._sample_values)
            np.save(os.path.join(work_dir, "level_{}_values".format(level._level_idx)), level._sample_values)

        mc_test.mc.update_moments(mc_test.moments_fn)
        means, vars = estimator.estimate_moments(mc_test.moments_fn)
        print("means ", means)
        print("vars ", vars)
        exact_moments = mlmc.tool.simple_distribution.compute_exact_moments(mc_test.moments_fn, self.pdf)
        print("exact moments: {}".format(exact_moments))


        ####################################
        # MaxEnt method                    #
        ####################################
        result = ConvResult()
        info, min_result = estimator.construct_density(tol=distr_accuracy, reg_param=reg_param,
                                                       orth_moments_tol=np.sqrt(target_var),
                                           exact_pdf=self.pdf, orth_method=orth_method)

        original_evals, evals, threshold, L = info

        max_ent_kl = mlmc.tool.simple_distribution.KL_divergence(self.cut_distr.pdf, estimator.distribution.density, a, b)

        distr_plot.add_distribution(estimator._distribution,
                                    label="reg param: {}, threshold: {}, KL: {}".format(reg_param, threshold, max_ent_kl),
                                    size=max_mom, reg_param=reg_param)

        t1 = time.time()
        result.residual_norm = min_result.fun_norm
        result.success = min_result.success
        result.success = min_result.success
        result.nit = min_result.nit

        a, b = self.domain
        result.kl = mlmc.tool.simple_distribution.KL_divergence(self.pdf, estimator._distribution.density, a, b)
        result.kl_2 = mlmc.tool.simple_distribution.KL_divergence_2(self.pdf, estimator._distribution.density, a, b)
        result.l2 = mlmc.tool.simple_distribution.L2_distance(self.pdf, estimator._distribution.density, a, b)
        result.tv = 0  # mlmc.tool.simple_distribution.total_variation_int(distr_obj.density_derivation, a, b)

        self._save_distr_data(estimator._distribution, distr_plot, work_dir, target_var, result)

        # if n_levels == 1:
        #     mc0_samples = np.concatenate(mc.levels[0].sample_values[:, 0])
        #     distr_plot.add_raw_samples(mc0_samples)

        ####################################
        # Spline approximation             #
        ####################################
        # two gausians (best 20-22) all_n_int_points = [10, 12, 14, 16, 18, 20, 22, 24]
        # five fingers (best 200 -220) all_n_int_points = [180, 200, 220, 240, 260, 280]
        # cauchy  - 25, 18,  21 , 16 all_n_int_points = [10, 12, 14, 16, 18, 20, 22, 24]
        # discontinuous - 11, 14, 13
        # norm - 11, 17, 8, 22, 19, 31
        # lognorm 16, 13, 11 10, 12
        all_n_int_points = [13]#range(5, 40, 1)

        mc_test.set_moments_fn(moments.Legendre)  # mean values for spline approximation
        mc_test.mc.update_moments(mc_test.moments_fn)

        print("spline domain ", self.domain)

        kl_divs = []

        for n_int_points in all_n_int_points:
            print("n int points ", n_int_points)
            spline_distr = spline_approx.BSplineApproximation(mc_test.mc, self.domain, poly_degree=poly_degree,
                                                              accuracy=distr_accuracy)

            spline_distr.moments_fn = mc_test.moments_fn
            spline_distr.n_interpolation_points = n_int_points

            spline_kl = mlmc.tool.simple_distribution.KL_divergence(self.cut_distr.pdf, spline_distr.density, a, b)

            kl_divs.append((n_int_points, spline_kl))

            distr_plot.add_distribution(spline_distr,
                                        label="BSpline, degree: {}, n_int_points: {}, KL: {}".format(poly_degree,
                                                                                                     n_int_points,
                                                                                                     spline_kl))

            self._save_distr_spline(spline_distr, distr_plot, work_dir, target_var, spline_kl, "bspline")


        best_kl_divs = sorted(kl_divs, key=lambda x: x[1])

        print("BEST KL divs ", best_kl_divs)

        # # #####################################
        # # #### Indicator interpolation  #
        # # #####################################
        # indicator_kl = 10
        # mc_test.set_moments_fn(moments.Legendre)  # mean values for spline approximation
        # mc_test.mc.update_moments(mc_test.moments_fn)
        #
        # spline_distr = spline_approx.SplineApproximation(mc_test.mc, self.domain, poly_degree=poly_degree,
        #                                                  accuracy=distr_accuracy)
        #
        # spline_distr.moments_fn = mc_test.moments_fn
        # spline_distr.indicator_method_name = "indicator"
        # spline_distr.n_interpolation_points = n_int_points
        #
        # indicator_kl = mlmc.tool.simple_distribution.KL_divergence(self.cut_distr.pdf, spline_distr.density, a, b)
        #
        # distr_plot.add_spline_distribution(spline_distr,
        #                             label="Indicator, degree: {}, n_int_points: {}, KL: {}".format(poly_degree,
        #                                                                                          n_int_points,
        #                                                                                          indicator_kl))
        #
        # self._save_distr_spline(spline_distr, distr_plot, work_dir, target_var, indicator_kl, "indicator")
        #
        # #####################################
        # #### Smooth interpolation           #
        # #####################################
        # smooth_kl=10
        # mc_test.set_moments_fn(moments.Legendre)  # mean values for spline approximation
        # mc_test.mc.update_moments(mc_test.moments_fn)
        #
        # spline_distr = spline_approx.SplineApproximation(mc_test.mc, self.domain, poly_degree=poly_degree,
        #                                                            accuracy=distr_accuracy)
        #
        # spline_distr.moments_fn = mc_test.moments_fn
        # spline_distr.indicator_method_name = "smooth"
        # spline_distr.n_interpolation_points = n_int_points
        #
        # smooth_kl = mlmc.tool.simple_distribution.KL_divergence(self.cut_distr.pdf, spline_distr.density, a, b)
        #
        # distr_plot.add_spline_distribution(spline_distr,
        #                             label="Smooth, degree: {}, n_int_points: {}, KL: {}".format(poly_degree,
        #                                                                                          n_int_points,
        #                                                                                          smooth_kl))
        # self._save_distr_spline(spline_distr, distr_plot, work_dir, target_var, smooth_kl, "smooth")

        ####################################
        # KL divergences                   #
        ####################################

        distr_plot.show(None)
        plt.show()

        print("KL div - MEM:{:0.4g}, BSpline:{:0.4g}, smooth:{:0.4g}, indicator:{:0.4g}".format(max_ent_kl, spline_kl,
                                                                                                smooth_kl, indicator_kl))

    def _save_distr_spline(self, distr_object, distr_plot, work_dir, noise_level, kl_div, name=""):
        domain = distr_object.domain
        distr_plot.adjust_domain(domain)
        X = distr_plot._grid(10000, domain=domain)

        np.save('{}/{}_{}.npy'.format(work_dir, noise_level, "result" + name), kl_div)
        np.save('{}/{}_{}.npy'.format(work_dir, noise_level, "domain" + name), distr_object.domain)
        np.save('{}/{}_{}.npy'.format(work_dir, noise_level, "X"+ name), X)
        np.save('{}/{}_{}.npy'.format(work_dir, noise_level, "Y_pdf" +name), distr_object.density(X))
        np.save('{}/{}_{}.npy'.format(work_dir, noise_level, "Y_cdf" + name), distr_object.cdf(X))
        np.save('{}/{}_{}.npy'.format(work_dir, noise_level, "Y_pdf_exact" + name), self.cut_distr.pdf(X))
        np.save('{}/{}_{}.npy'.format(work_dir, noise_level, "Y_cdf_exact" + name), self.cut_distr.cdf(X))
        #np.save('{}/{}_{}.npy'.format(work_dir, noise_level, "Y_pdf_log" + name), distr_object.density_log(X))

    def find_regularization_param(self, plot_res=True, noise_level=0.01, work_dir=None, orth_method=2, n_mom=None):
        if work_dir == None:
            dir_name = "find_reg_param"
            if not os.path.exists(dir_name):
                os.mkdir(dir_name)
            else:
                shutil.rmtree(dir_name)
                os.mkdir(dir_name)

            work_dir = os.path.join(dir_name, self.name)
            if os.path.exists(work_dir):
                raise FileExistsError
            else:
                os.mkdir(work_dir)

        reg_params = np.linspace(1e-12, 1e-5, num=50)  # Legendre
        #reg_params = np.linspace(10, 1e-2, num=25)  # BSpline

        reg_params = np.geomspace(1e-8, 1e-5, num=80)  # two gaussians 2nd der
        #reg_params = np.geomspace(1e-12, 1e-6, num=60) # two gaussians 3rd der
        # reg_params = np.geomspace(1e-12, 1e-9, num=60) # cauchy 3rd der
        # reg_params = np.geomspace(1e-12, 1e-9, num=60)  # cauchy 3rd der
        reg_params = np.geomspace(1e-9, 1e-4, num=60) # five fingers 2nd derivative
        #reg_params = np.geomspace(1e-12, 4e-9, num=50) # lognorm 2nd derivative
        #reg_params = np.geomspace(1e-10*2, 1e-9, num=10)
        #reg_params = np.geomspace(2e-10, 1e-9, num=30)
        #reg_params = np.geomspace(1e-9, 1e-5, num=6)
        #reg_params = [0]

        rep = 1

        n_reg_params = 100

        reg_params = np.geomspace(1e-7, 1e-5, num=n_reg_params)
        reg_params = np.geomspace(1e-9, 1e-4, num=n_reg_params)

        reg_params = [7.391e-8]

        #reg_params = [0, 6.691189901715622e-9]


        #reg_params = [5.590810182512222e-11, 5.590810182512222e-10, 5.590810182512222e-9]

        #reg_params = [1e-12, 5e-12, 1e-11, 5e-11, 1e-10, 5e-10, 1e-9, 5e-9,
        #              1e-8, 5e-8, 1e-7, 5e-7]

        #reg_params = np.geomspace(1e-7, 1e-5, num=100)

        #reg_params = [3.16227766e-07]

        #reg_params = [4.893900918477499e-10, 5.736152510448681e-10]

        #reg_params = [1e-3, 1e-5]

        #reg_params = [1e-8, 1e-3]

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

        if n_mom is not None:
            max_n_moments = n_moments = n_mom

        self.moments_fn = moment_class(max_n_moments, self.domain, log=log, safe_eval=False)

        _, _, n_moments, _ = self.moments_data

        size = 100

        fine_noises = []
        for _ in range(rep):
            noise = np.random.randn(self.moments_fn.size ** 2).reshape((self.moments_fn.size, self.moments_fn.size))
            print("fine noise ")
            print(pd.DataFrame(noise))
            noise += noise.T
            noise *= 0.5 * noise_level
            noise[0, 0] = 0

            fine_noises.append(noise)

        fine_noise = np.mean(fine_noises, axis=0)

        distr_objects = {}
        kl_divs = {}

        all_noises = []
        for _ in range(rep):
            noises = []
            for i in range(size):
                noise = np.random.randn(self.moments_fn.size ** 2).reshape((self.moments_fn.size, self.moments_fn.size))
                print("coarse noise ", noise)
                noise += noise.T
                noise *= 0.5 * noise_level * 1.2
                noise[0, 0] = 0
                noises.append(noise)
            #print("coarse noises shape ", np.array(noises).shape)
            all_noises.append(noises)
            #print("coarse all noises shape ", np.array(all_noises).shape)

        #print("np.array(all_noises).shape ", np.array(all_noises).shape)
        noises = np.mean(all_noises, axis=0)
        noises_var = np.var(all_noises, axis=0)


        num_moments = self.moments_fn.size
        used_reg_params = []
        for index, reg_param in enumerate(reg_params):
            print("REG PARAMETER ", reg_param)
            regularization = mlmc.tool.simple_distribution.Regularization2ndDerivation()
            #regularization = mlmc.tool.simple_distribution.RegularizationInexact2()

            self.moments_fn = moment_class(max_n_moments, self.domain, log=log, safe_eval=False)
            #size = self.moments_fn.size
            base_moments = self.moments_fn
            exact_cov, reg_matrix = mlmc.tool.simple_distribution.compute_semiexact_cov_2(base_moments, self.pdf,
                                                                                     reg_param=reg_param,
                                                                                     regularization=regularization)
            self.original_exact_cov = exact_cov
            self.moments_without_noise = exact_cov[:, 0]

            print("reg matrix")
            print(pd.DataFrame(reg_matrix))

            self.exact_moments = exact_cov[0, :] #mlmc.tool.simple_distribution.compute_semiexact_moments(self.moments_fn,
                                                                                   # self.pdf)

            # Add regularization
            exact_cov += reg_matrix

            # np.random.seed(1234)
            # noise = np.random.randn(size ** 2).reshape((size, size))
            # noise += noise.T
            # noise *= 0.5 * noise_level
            # noise[0, 0] = 0

            print("noise ")
            print(pd.DataFrame(noise))
            cov = exact_cov + fine_noise
            moments = cov[:, 0]

            self.moments_fn, info, cov_centered = mlmc.tool.simple_distribution.construct_orthogonal_moments(
                                                                                                    base_moments,
                                                                                                    cov,
                                                                                                    noise_level**2,
                                                                                                    reg_param=reg_param,
                                                                                          orth_method=orth_method)
            self._cov_with_noise = cov
            self._cov_centered = cov_centered
            original_evals, evals, threshold, L = info
            self.L = L
            self.tol_density_approx = 1e-7



            moments_with_noise = moments

            #info, moments_with_noise = self.setup_moments(self.moments_data, noise_level=noise_level)

            n_moments = len(moments_with_noise)

            original_evals, evals, threshold, L = info
            fine_moments = np.matmul(moments_with_noise, L.T)

            # print("n moments ", n_moments)
            # print("self.moments_fn.size ", self.moments_fn.size)
            # print("fine moments_fn.shape ", fine_moments.shape)

            n_moments = self.moments_fn.size

            # if n_moments > self.moments_fn.size:
            #     continue
            # moments_fn = moment_fn(n_moments, domain, log=log_flag, safe_eval=False )
            # print(i_m, n_moments, domain, force_decay)

            moments_data = np.empty((n_moments, 2))
            moments_data[:, 0] = fine_moments[:n_moments]
            moments_data[:, 1] = 1.0

            # original_evals, evals, threshold, L = info
            # fine_moments = np.matmul(moments, L.T)
            #
            # moments_data = np.empty((len(fine_moments), 2))
            # moments_data[:, 0] = fine_moments  # self.exact_moments
            # moments_data[:, 1] = 1  # noise ** 2
            # moments_data[0, 1] = 1.0

            #regularization = mlmc.tool.simple_distribution.Regularization3rdDerivation()

            result, distr_obj = self.make_approx(mlmc.tool.simple_distribution.SimpleDistribution, noise, moments_data,
                                            tol=1e-7, reg_param=reg_param, regularization=regularization)

            if not result.success:
                continue

            used_reg_params.append(reg_param)

            estimated_density_covariance, reg_matrix = mlmc.tool.simple_distribution.compute_semiexact_cov_2(self.moments_fn,
                                                                                                distr_obj.density)


            distr_objects[reg_param] = (distr_obj, result, threshold, L)

            kl_divs[reg_param] = result.kl

            # M = np.eye(len(cov[0]))
            # M[:, 0] = -cov[:, 0]
            #
            # print("cov centered")
            # print(pd.DataFrame(cov_centered))
            #
            # print("M-1 @ L-1 @ H @ L.T-1 @ M.T-1")
            # print(pd.DataFrame(
            #     M @ (np.linalg.inv(L) @ distr_obj.final_jac @ np.linalg.inv(L.T)) @ M.T))

            final_jac = distr_obj.final_jac
            #
            # distr_obj_exact_conv_int = mlmc.tool.simple_distribution.compute_exact_cov(distr_obj.moments_fn, distr_obj.density)
            M = np.eye(len(self._cov_with_noise[0]))
            M[:, 0] = -self._cov_with_noise[:, 0]

            # print("M @ L-1 @ H @ L.T-1 @ M.T")
            # print(pd.DataFrame(
            #     M @ (np.linalg.inv(self.L) @ final_jac @ np.linalg.inv(self.L.T)) @ M.T))
            #
            # print("orig cov centered")
            # print(pd.DataFrame(self._cov_centered))


            # print("cov")
            # print(pd.DataFrame(cov))
            #
            # print("L-1 @ H @ L.T-1")
            # print(pd.DataFrame(
            #     (np.linalg.inv(L) @ distr_obj.final_jac @ np.linalg.inv(L.T))))

            # print(pd.DataFrame(
            #     M @ (np.linalg.inv(L) @ estimated_density_covariance @ np.linalg.inv(L.T)) @ M.T))

            # print("np.linalg.inv(L).shape ", np.linalg.inv(L).shape)
            # print("distr_obj.final_jac.shape ", distr_obj.final_jac.shape)
            # print("np.linalg.inv(L.T).shape ", np.linalg.inv(L.T).shape)

            # if len(distr_obj.multipliers) < num_moments:
            #     num_moments = len(distr_obj.multipliers)
            #     print("NUM MOMENTS ", num_moments)

            print("size ", size)

        reg_params = used_reg_params

        # eval, evec = np.linalg.eigh(self._cov_centered)
        # print("eval ", eval)
        # print("evec ", evec)
        #
        # tot = sum(eval)
        # var_exp = [(i / tot) * 100 for i in sorted(eval, reverse=True)]
        # print("var_exp ", var_exp)
        # cum_var_exp = np.cumsum(var_exp)
        # print("cum_var_exp ", cum_var_exp)
        #
        # rations = []
        # for i in range(len(cum_var_exp)):
        #     if i == 0:
        #         rations.append(cum_var_exp[i])
        #     else:
        #         rations.append(cum_var_exp[i] - cum_var_exp[i - 1])
        #
        # rations = np.array(rations)

        all_num_moments = []
        all_result_norm = []

        for index, distr in distr_objects.items():
            L = distr[3]
            distr_obj = distr[0]
            moments_from_density = (np.linalg.pinv(L) @ distr_obj.final_jac @ np.linalg.pinv(L.T))[:, 0]

            distr_exact_cov, distr_reg_matrix = mlmc.tool.simple_distribution.compute_semiexact_cov_2(base_moments, distr_obj.density,
                                                                                     reg_param=0,
                                                                                     regularization=regularization)

            moments_from_density = distr_exact_cov[:, 0]


            result = []
            result_norm = []
            for i in range(size):
                cov = self.original_exact_cov + noises[i][:len(self.original_exact_cov), :len(self.original_exact_cov)]
                #print("Cov ", cov)
                coarse_moments = cov[:, 0]
                coarse_moments[0] = 1
                #coarse_moments = np.matmul(coarse_moments, L.T)

                # _, distr_obj = self.make_approx(mlmc.tool.simple_distribution.SimpleDistribution, noise, moments_data,
                #                                      tol=1e-7, reg_param=reg_param)
                #
                # estimate_density_exact_moments = mlmc.tool.simple_distribution.compute_semiexact_moments(self.moments_fn,
                #                                                                                     distr_obj.density)

                num_moments = len(moments_from_density)
                res = (moments_from_density[:num_moments] - coarse_moments[:num_moments])**2

                #res = (moments_from_density - coarse_moments) ** 2

                # res = ((moments_from_density[:num_moments] - coarse_moments[:num_moments])/num_moments) ** 2
                #
                #
                # res = np.linalg.norm(moments_from_density[:num_moments] - coarse_moments[:num_moments])

                # res = res * rations[:num_moments-1]

                result_norm.append(np.array(res) / num_moments)
                result.append(res)

            # distr_plot.add_distribution(distr_obj,
            #                             label="noise: {}, threshold: {}, reg param: {}".format(noise_level, threshold,
            #                                                                                    reg_param),
            #                             size=len(coarse_moments), reg_param=reg_param)

            all_num_moments.append(num_moments)
            min_results.append(np.sum(result))#np.sum(result))
            all_result_norm.append(np.sum(result_norm))


        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        # ax.plot(noise_levels, tv, label="total variation")

        zipped = zip(reg_params, min_results)

        for reg_param, min_result in zip(reg_params, min_results):
            print("reg_param: {}, min_result: {}".format(reg_param, min_result))

        sorted_zip = sorted(zipped, key=lambda x: x[1])

        best_params = []
        #best_params.append(0)
        min_best = None
        for s_tuple in sorted_zip:
            if min_best is None:
                min_best = s_tuple
            if len(best_params) < 5:
                best_params.append(s_tuple[0])

        # ax.plot(reg_params, min_results, 'o', c='r')
        # # ax.set_xlabel("noise level")
        # ax.set_xlabel("regularization param (alpha)")
        # ax.set_ylabel("min")
        # # ax.plot(noise_levels, l2, label="l2 norm")
        # # ax.plot(reg_parameters, int_density, label="abs(density-1)")
        # #ax.set_yscale('log')
        # #ax.set_xscale('log')
        # ax.legend()
        #
        # plt.show()
        kl_div_to_plot = [kl_divs[r_par] for r_par in reg_params]

        if work_dir is not None:
            if n_mom is not None:
                self._save_reg_param_data(work_dir, n_mom, reg_params, min_results, distr_objects)
            else:
                self._save_reg_param_data(work_dir, noise_level, reg_params, min_results, distr_objects)

        if plot_res:

            res_norm_2 = []
            for res, used_moments in zip(min_results, all_num_moments):
                res_norm_2.append(res * (used_moments / max_n_moments))

            fig, ax = plt.subplots()
            ax.plot(reg_params, min_results, 'o', label="MSE")
            #ax.plot(reg_params, all_result_norm, 's', label="MSE norm")
            #ax.plot(reg_params, res_norm_2, '>', label="MSE norm 2")
            ax.plot(reg_params, kl_div_to_plot, 'v', label="kl div")
            ax.plot(min_best[0], min_best[1], 'x', color='red')
            ax.set_ylabel("MSE")
            ax.set_xlabel(r"$\log(\alpha)$")
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.legend(loc='best')
            logfmt = matplotlib.ticker.LogFormatterExponent(base=10.0, labelOnlyBase=True)
            ax.xaxis.set_major_formatter(logfmt)

            plt.show()

            distr_plot = plot.Distribution(exact_distr=self.cut_distr,
                                           title="Preconditioning reg, {},  n_moments: {}, noise: {}".format(self.title,
                                                                                                             n_moments,
                                                                                                             noise_level),
                                           log_x=self.log_flag, error_plot=None, reg_plot=False, cdf_plot=False,
                                           log_density=True)

            if "0" in distr_objects:
                best_params.append(0)
            for reg_par in best_params:
                #print("distr_objects[reg_par] ", distr_objects[reg_par])
                distr_plot.add_distribution(distr_objects[reg_par][0],
                                            label="n: {:0.4g}, th: {}, alpha: {:0.4g},"
                                                  " KL_div: {:0.4g}".format(noise_level, distr_objects[reg_par][2], reg_par,
                                                                            distr_objects[reg_par][1].kl),
                                            size=n_moments, mom_indices=False, reg_param=reg_par)

            #self.determine_regularization_param(best_params, regularization, noise=noise_level)
            distr_plot.show(None)

            for reg_par, kl_div in kl_divs.items():
                print("KL: {} reg_param: {}".format(kl_div, reg_par))

            return best_params
        else:
            exact_cov, reg_matrix = mlmc.tool.simple_distribution.compute_semiexact_cov_2(base_moments, self.pdf,
                                                                                     reg_param=best_params[0],
                                                                                     regularization=regularization)
            cov += reg_matrix + fine_noise

            return best_params, distr_objects[best_params[0]], exact_cov, cov

    def _save_reg_param_data(self, work_dir, noise_level, reg_params, min_results, distr_objects, cond_numbers=None):
        np.save('{}/{}_{}.npy'.format(work_dir, noise_level, "reg-params"), reg_params)
        np.save('{}/{}_{}.npy'.format(work_dir, noise_level, "min-results"), min_results)

        if cond_numbers is not None:
            np.save('{}/{}_{}.npy'.format(work_dir, noise_level, "cond-numbers"), cond_numbers)

        info = []
        for index, distr in distr_objects.items():
            info.append((distr[1].kl, distr[1].nit, not distr[1].success, distr[2]))

        np.save('{}/{}_{}.npy'.format(work_dir, noise_level, "info"), info)

    def find_regularization_param_tv(self):
        np.random.seed(1234)
        noise_level = 1e-2

        reg_params = np.linspace(1e-12, 1e-5, num=50)  # Legendre
        # reg_params = np.linspace(10, 1e-2, num=25)  # BSpline

        reg_params = np.geomspace(1e-12, 1e-6, num=60)  # two gaussians 3rd der
        reg_params = np.geomspace(1e-12, 1e-9, num=60)  # cauchy 3rd der
        #reg_params = np.geomspace(5*1e-6, 6*1e-5, num=60) # two gaussian total variation
        #reg_params = np.geomspace(3e-5, 7e-5, num=60) # norm tv not good
        reg_params = np.geomspace(1e-4, 5e-2, num=60)
        # reg_params = [0]

        # reg_params = [3.16227766e-07]

        min_results = []
        orth_method = 1

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

        size = 60
        noises = []

        noise = np.random.randn(self.moments_fn.size ** 2).reshape((self.moments_fn.size, self.moments_fn.size))
        noise += noise.T
        noise *= 0.5 * noise_level
        noise[0, 0] = 0
        fine_noise = noise

        for i in range(size):
            noise = np.random.randn(self.moments_fn.size ** 2).reshape((self.moments_fn.size, self.moments_fn.size))
            noise += noise.T
            noise *= 0.5 * noise_level * 1.1
            noise[0, 0] = 0
            noises.append(noise)

        distr_objects = {}
        kl_divs = {}

        for reg_param in reg_params:
            regularization = None#mlmc.tool.simple_distribution.Regularization2ndDerivation()

            self.moments_fn = moment_class(max_n_moments, self.domain, log=log, safe_eval=False)
            # size = self.moments_fn.size
            base_moments = self.moments_fn
            exact_cov, reg_matrix = mlmc.tool.simple_distribution_total_var.compute_semiexact_cov_2(base_moments, self.pdf,
                                                                                     reg_param=reg_param)
            self.original_exact_cov = exact_cov
            self.moments_without_noise = exact_cov[:, 0]

            # Add regularization
            exact_cov += reg_matrix
            cov = exact_cov + fine_noise
            moments = cov[:, 0]

            self.moments_fn, info, cov_centered = mlmc.tool.simple_distribution_total_var.construct_orthogonal_moments(
                base_moments,
                cov,
                noise_level**2,
                reg_param=reg_param,
                orth_method=orth_method)
            self._cov_with_noise = cov
            self._cov_centered = cov_centered
            original_evals, evals, threshold, L = info
            self.L = L
            self.tol_density_approx = 0.01

            self.exact_moments = mlmc.tool.simple_distribution_total_var.compute_semiexact_moments(self.moments_fn,
                                                                                    self.pdf)

            moments_with_noise = moments


            original_evals, evals, threshold, L = info
            fine_moments = np.matmul(moments_with_noise, L.T)

            n_moments = self.moments_fn.size


            moments_data = np.empty((n_moments, 2))
            moments_data[:, 0] = fine_moments[:n_moments]
            moments_data[:, 1] = 1.0

            # regularization = mlmc.tool.simple_distribution.Regularization3rdDerivation()

            result, distr_obj = self.make_approx(mlmc.tool.simple_distribution_total_var.SimpleDistribution, noise, moments_data,
                                            tol=1e-7, reg_param=reg_param, regularization=regularization)

            estimated_density_covariance, reg_matrix = mlmc.tool.simple_distribution_total_var.compute_semiexact_cov_2(self.moments_fn,
                                                                                                        distr_obj.density)
            distr_objects[reg_param] = (distr_obj, result, threshold)

            kl_divs[reg_param] = result.kl

            final_jac = distr_obj.final_jac
            #
            # distr_obj_exact_conv_int = mlmc.tool.simple_distribution.compute_exact_cov(distr_obj.moments_fn, distr_obj.density)
            M = np.eye(len(self._cov_with_noise[0]))
            M[:, 0] = -self._cov_with_noise[:, 0]

            # print("M @ L-1 @ H @ L.T-1 @ M.T")
            # print(pd.DataFrame(
            #     M @ (np.linalg.inv(self.L) @ final_jac @ np.linalg.inv(self.L.T)) @ M.T))
            #
            # print("orig cov centered")
            # print(pd.DataFrame(self._cov_centered))

            # print("cov")
            # print(pd.DataFrame(cov))
            #
            # print("L-1 @ H @ L.T-1")
            # print(pd.DataFrame(
            #     (np.linalg.inv(L) @ distr_obj.final_jac @ np.linalg.inv(L.T))))

            # print(pd.DataFrame(
            #     M @ (np.linalg.inv(L) @ estimated_density_covariance @ np.linalg.inv(L.T)) @ M.T))

            # print("np.linalg.inv(L).shape ", np.linalg.inv(L).shape)
            # print("distr_obj.final_jac.shape ", distr_obj.final_jac.shape)
            # print("np.linalg.inv(L.T).shape ", np.linalg.inv(L.T).shape)

            moments_from_density = (np.linalg.inv(L) @ distr_obj.final_jac @ np.linalg.inv(L.T))[:, 0]

            result = []
            for i in range(size):
                cov = self.original_exact_cov + noises[i][:len(self.original_exact_cov), :len(self.original_exact_cov)]
                coarse_moments = cov[:, 0]
                coarse_moments[0] = 1
                res = (moments_from_density - coarse_moments) ** 2
                result.append(res)

            min_results.append(np.sum(result))  # np.sum(result))

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        # ax.plot(noise_levels, tv, label="total variation")
        zipped = zip(reg_params, min_results)

        # for reg_param, min_result in zip(reg_params, min_results):
        #     print("reg_param: {}, min_result: {}".format(reg_param, min_result))

        sorted_zip = sorted(zipped, key=lambda x: x[1])

        best_params = []
        if '0' in distr_objects:
            best_params.append(0)

        for s_tuple in sorted_zip:
            print(s_tuple)
            if len(best_params) < 5:
                best_params.append(s_tuple[0])

        #
        # xnew = np.linspace(np.min(reg_params), np.max(reg_params), num=41, endpoint=True)
        plt.plot(reg_params, min_results, 'o')

        plt.xscale('log')
        plt.legend(loc='best')
        plt.show()

        distr_plot = plot.Distribution(exact_distr=self.cut_distr,
                                       title="Preconditioning reg, {},  n_moments: {}, noise: {}".format(self.title,
                                                                                                         n_moments,
                                                                                                         noise_level),
                                       log_x=self.log_flag, error_plot=None, reg_plot=False, cdf_plot=False,
                                       log_density=True)

        if "0" in distr_objects:
            best_params.append(0)
        for reg_par in best_params:
            distr_plot.add_distribution(distr_objects[reg_par][0],
                                        label="n: {:0.4g}, th: {}, alpha: {:0.4g},"
                                              " KL_div: {:0.4g}".format(noise_level, distr_objects[reg_par][2], reg_par,
                                                                        distr_objects[reg_par][1].kl),
                                        size=n_moments, mom_indices=False, reg_param=reg_par)

        # self.determine_regularization_param(best_params, regularization, noise=noise_level)
        distr_plot.show(None)

        # for reg_par, kl_div in kl_divs.items():
        #     print("KL: {} reg_param: {}".format(kl_div, reg_par))

        #self.determine_regularization_param_tv(best_params, regularization, noise=noise_level)

        return best_params

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
        self.moments_fn, info, _ = mlmc.tool.simple_distribution.construct_orthogonal_moments(moments_fn, exact_cov, 0,
                                                                                         orth_method=orth_method)
        orig_evals, evals, threshold, L = info

        exact_moments = mlmc.tool.simple_distribution.compute_semiexact_moments(self.moments_fn, self.pdf, tol=tol_exact_cov)

        moments_data = np.empty((n_moments, 2))
        moments_data[:, 0] = exact_moments[:n_moments]
        moments_data[:, 1] = 1.0

        result, distr_obj = self.make_approx(mlmc.tool.simple_distribution.SimpleDistribution, 0.0, moments_data, tol=tol_density)
        return result.kl, distr_obj

    def plot_KL_div_exact(self):
        """
        Plot KL divergence for different number of exact moments
        :return:
        """
        noise_level = 0
        tol_exact_moments = 1e-6
        tol_density = 1e-5
        results = []
        orth_method = 4
        distr_plot = plot.Distribution(exact_distr=self.cut_distr, title=self.title+"_exact", cdf_plot=False,
                                       log_x=self.log_flag, error_plot=False)

        dir_name = "KL_div_exact_numpy_{}_five_fingers".format(orth_method)
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

        work_dir = os.path.join(dir_name, self.name)

        #########################################
        # Set moments objects
        moment_class, min_n_moments, max_n_moments, self.use_covariance = self.moments_data
        log = self.log_flag
        if min_n_moments == max_n_moments:
            self.moment_sizes = np.array(
                [max_n_moments])
        else:
            self.moment_sizes = np.round(np.exp(np.linspace(np.log(min_n_moments), np.log(max_n_moments), 3))).astype(int)
        self.moments_fn = moment_class(max_n_moments, self.domain, log=log, safe_eval=False)

        if os.path.exists(work_dir):
            raise FileExistsError
        else:
            os.mkdir(work_dir)
            np.save(os.path.join(work_dir, "moment_sizes"), self.moment_sizes)


        ##########################################
        # Orthogonalize moments

        base_moments = self.moments_fn
        exact_cov = mlmc.tool.simple_distribution.compute_semiexact_cov(base_moments, self.pdf)
        self.moments_fn, info, _ = mlmc.tool.simple_distribution.construct_orthogonal_moments(base_moments, exact_cov,
                                                                                         noise_level**2, orth_method=orth_method)
        orig_eval, evals, threshold, L = info
        #eye_approx = L @ exact_cov @ L.T
        # test that the decomposition is done well
        # assert np.linalg.norm(
        #     eye_approx - np.eye(*eye_approx.shape)) < 1e-9  # 1e-10 failed with Cauchy for more moments

        print("threshold: ", threshold, " from N: ", self.moments_fn.size)
        if self.eigenvalues_plot:
            threshold = evals[threshold]
            noise_label = "{:5.2e}".format(noise_level)
            self.eigenvalues_plot.add_values(evals, threshold=threshold, label=noise_label)
        self.exact_moments = mlmc.tool.simple_distribution.compute_semiexact_moments(self.moments_fn, self.pdf, tol=tol_exact_moments)

        kl_plot = plot.KL_divergence(log_y=True, iter_plot=True, kl_mom_err=False, title="Kullback-Leibler divergence, {}, threshold: {}".format(self.title, threshold),
                                     xlabel="number of moments", ylabel="KL divergence")

        ###############################################
        # For each moment size compute density
        for i_m, n_moments in enumerate(self.moment_sizes):
            if n_moments > self.moments_fn.size:
                continue

            # moments_fn = moment_fn(n_moments, domain, log=log_flag, safe_eval=False )
            moments_data = np.empty((n_moments, 2))
            moments_data[:, 0] = self.exact_moments[:n_moments]
            moments_data[:, 1] = 1.0

            # modif_cov = mlmc.tool.simple_distribution.compute_semiexact_cov(self.moments_fn, self.pdf)
            # diff_norm = np.linalg.norm(modif_cov - np.eye(*modif_cov.shape))
            # print("#{} cov mat norm: {}".format(n_moments, diff_norm))

            result, distr_obj = self.make_approx(mlmc.tool.simple_distribution.SimpleDistribution, 0.0, moments_data, tol=tol_density)
            distr_plot.add_distribution(distr_obj, label="#{}, KL div: {}".format(n_moments, result.kl))
            results.append(result)

            self._save_distr_data(distr_obj, distr_plot, work_dir, n_moments, result)

            kl_plot.add_value((n_moments, result.kl))
            kl_plot.add_iteration(x=n_moments, n_iter=result.nit, failed=not result.success)

            self._save_kl_data_exact(work_dir, n_moments, result.kl, result.nit, not result.success, threshold)

        #self.check_convergence(results)
        kl_plot.show(None)
        distr_plot.show(None)#file=self.pdfname("_pdf_exact"))
        distr_plot.reset()
        return results

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

        #noise_levels = noise_levels[:1]

        #noise_levels = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 1e-8]

        min_noise = 1e-1
        max_noise = 1e-12
        geom_seq = np.exp(np.linspace(np.log(min_noise), np.log(max_noise), 50))
        noise_levels = np.flip(np.concatenate(([0.0], geom_seq)), axis=0)

        #noise_levels = [1e-1, 1e-2, 1e-3, 1e-4,  1e-5, 1e-6, 1e-8]

        #noise_levels = [1e-2]

        #noise_levels = [1e-4, 1e-5, 1e-6, 1e-8, 1e-10, 1e-12]

        #noise_levels = [1e-1]

        tol_exact_cov = 1e-10
        tol_density = 1e-5
        results = []
        n_moments = 35  # 25 is not enough for TwoGaussians
        orth_method = 2

        distr_plot = plot.Distribution(exact_distr=self.cut_distr, title=self.title+"_inexact", cdf_plot=False,
                                       log_x=self.log_flag, error_plot=False)

        dir_name = "KL_div_inexact_for_reg_{}_all".format(orth_method)
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        else:
            shutil.rmtree(dir_name)
            os.mkdir(dir_name)

        work_dir = os.path.join(dir_name, self.name)
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

        self.moments_fn = moment_class(n_moments, self.domain, log=log, safe_eval=False)

        ##########################################
        # Orthogonalize moments

        base_moments = self.moments_fn
        exact_cov = mlmc.tool.simple_distribution.compute_semiexact_cov(base_moments, self.pdf)

        kl_plot.truncation_err, distr_obj_exact = self._compute_exact_kl(n_moments, base_moments, orth_method,
                                                                         tol_density, tol_exact_cov)

        np.save(os.path.join(work_dir, "truncation_err"), kl_plot.truncation_err)

        # exact_moments_orig = mlmc.tool.simple_distribution.compute_semiexact_moments(self.moments_fn, self.pdf, tol=1e-10)
        exact_moments_orig = exact_cov[:, 0]
        # print("original exact moments ", exact_moments_orig)
        # print("exact cov[:, 0] ", exact_cov[:, 0])

        ###############################################
        # For each moment size compute density
        for i_m, noise_level in enumerate(noise_levels):
            print("NOISE LEVEL ", noise_level)
            # Add noise to exact covariance matrix
            #np.random.seed(4567)
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

            # Change base
            self.moments_fn, info, _ = mlmc.tool.simple_distribution.construct_orthogonal_moments(base_moments, cov, noise_level**2,
                                                                                          orth_method=orth_method)

            # Tests
            original_evals, evals, threshold, L = info
            eye_approx = L @ exact_cov @ L.T
            # test that the decomposition is done well
            # assert np.linalg.norm(
            #     eye_approx - np.eye(*eye_approx.shape)) < 1e-9  # 1e-10 failed with Cauchy for more moments_fn
            # print("threshold: ", threshold, " from N: ", self.moments_fn.size)
            # modif_cov = mlmc.tool.simple_distribution.compute_semiexact_cov(self.moments_fn, self.pdf, tol=tol_exact_cov)
            # diff_norm = np.linalg.norm(modif_cov - np.eye(*modif_cov.shape))
            # print("#{} cov mat norm: {}".format(n_moments, diff_norm))

            # Set moments data
            n_moments = self.moments_fn.size
            print("cov moments ", cov[:, 0])
            transformed_moments = np.matmul(cov[:, 0], L.T)
            #print("transformed moments ", transformed_moments)

            moments_data = np.empty((n_moments, 2))
            moments_data[:, 0] = transformed_moments
            moments_data[:, 1] = 1
            moments_data[0, 1] = 1.0

            exact_moments = exact_moments_orig[:len(transformed_moments)]

            result, distr_obj = self.make_approx(mlmc.tool.simple_distribution.SimpleDistribution, 0.0, moments_data, tol=tol_density)
            distr_plot.add_distribution(distr_obj, label="noise: {:f}, th: {}, KL div: {:f}".format(noise_level, threshold, result.kl))
            results.append(result)

            self._save_distr_data(distr_obj, distr_plot, work_dir, noise_level, result)

            print("RESULT ", result.success)

            kl_div = mlmc.tool.simple_distribution.KL_divergence(distr_obj_exact.density, distr_obj.density, self.domain[0], self.domain[1])
            #total_variation = mlmc.tool.simple_distribution.total_variation_int(distr_obj.density, self.domain[0], self.domain[1])

            kl_plot.add_value((noise_level, kl_div))
            kl_plot.add_iteration(x=noise_level, n_iter=result.nit, failed=not result.success)

            # print("exact moments ", exact_moments[:len(moments_data[:, 0])])
            # print("moments data ", moments_data[:, 0])
            # print("difference ", np.array(exact_moments) - np.array(moments_data[:, 0]))
            print("difference orig", np.array(exact_moments_orig) - np.array(cov[:, 0][:len(exact_moments_orig)]))

            diff_orig = np.array(exact_moments_orig) - np.array(cov[:, 0][:len(exact_moments_orig)])

            kl_plot.add_moments_l2_norm((noise_level, np.linalg.norm(diff_orig)**2))

            self._save_kl_data(work_dir, noise_level, kl_div, result.nit, not result.success,
                               np.linalg.norm(diff_orig)**2, threshold, total_variation=result.tv)

        kl_plot.show(None)
        distr_plot.show(None)
        distr_plot.reset()
        return results

    def _save_kl_data(self, work_dir, noise_level, kl_div, nit, success, mom_err, threshold, total_variation=0, name=""):
        np.save('{}/{}_{}.npy'.format(work_dir, noise_level, "add-value" + name), (noise_level, kl_div))
        np.save('{}/{}_{}.npy'.format(work_dir, noise_level, "add-iteration" + name), (noise_level, nit, success))
        np.save('{}/{}_{}.npy'.format(work_dir, noise_level, "add-moments" + name), (noise_level, mom_err))
        np.save('{}/{}_{}.npy'.format(work_dir, noise_level, "threshold" + name), threshold)
        np.save('{}/{}_{}.npy'.format(work_dir, noise_level, "total_variation" + name), total_variation)

    def plot_KL_div_inexact_reg_mom(self):
        """
        Plot KL divergence for different noise level of exact moments
        """
        min_noise = 1e-6
        max_noise = 1e-1
        geom_seq = np.exp(np.linspace(np.log(min_noise), np.log(max_noise), 10))
        noise_levels = np.flip(np.concatenate(([0.0], geom_seq)), axis=0)

        #noise_levels = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6]

        noise_level = 1e-2  #, 1e-3, 1e-4]

        #noise_levels = noise_levels[:2]

        tol_exact_cov = 1e-10
        tol_density = 1e-5
        results = []
        orth_method = 4
        n_moments = [10, 23, 35, 47, 60, 75]  # 25 is not enough for TwoGaussians

        distr_plot = plot.Distribution(exact_distr=self.cut_distr, title=self.title+"_inexact", cdf_plot=False,
                                       log_x=self.log_flag, error_plot=False)

        dir_name = "reg_KL_div_inexact_35_{}_mom".format(orth_method)
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        else:
            # @TODO: rm ASAP
            shutil.rmtree(dir_name)
            os.mkdir(dir_name)

        work_dir = os.path.join(dir_name, self.name)
        if os.path.exists(work_dir):
            raise FileExistsError
        else:
            os.mkdir(work_dir)
            np.save(os.path.join(work_dir, "noise_levels"), noise_levels)
            np.save(os.path.join(work_dir, "n_moments"), n_moments)

        kl_plot = plot.KL_divergence(iter_plot=True, log_y=True, log_x=True,
                                     title=self.title + "_noise_{}".format(noise_level), xlabel="noise std",
                                     ylabel="KL divergence", truncation_err_label="trunc. err, m: {}".format(n_moments))

        ##########################################
        # # Set moments objects
        moment_class, _, _, self.use_covariance = self.moments_data
        log = self.log_flag
        #
        # self.moments_fn = moment_class(n_moments, self.domain, log=log, safe_eval=False)
        # #
        # # ##########################################
        # # # Orthogonalize moments
        # #
        # base_moments = self.moments_fn
        # exact_cov = mlmc.tool.simple_distribution.compute_semiexact_cov(base_moments, self.pdf)

        # kl_plot.truncation_err, distr_obj_exact = self._compute_exact_kl(n_moments, base_moments, orth_method,
        #                                                                  tol_density, tol_exact_cov)
        #
        # np.save(os.path.join(work_dir, "truncation_err"), kl_plot.truncation_err)
        # exact_moments_orig = mlmc.tool.simple_distribution.compute_semiexact_moments(self.moments_fn, self.pdf, tol=1e-10)
        #exact_moments_orig = exact_cov[:, 0]
        # print("original exact moments ", exact_moments_orig)
        # print("exact cov[:, 0] ", exact_cov[:, 0])

        ###############################################
        # For each moment size compute density
        for i_m, n_mom in enumerate(n_moments):
            self.moments_fn = moment_class(n_mom, self.domain, log=log, safe_eval=False)
            #
            # ##########################################
            # # Orthogonalize moments
            #
            base_moments = self.moments_fn
            # exact_cov = mlmc.tool.simple_distribution.compute_semiexact_cov(base_moments, self.pdf)

            kl_plot.truncation_err, distr_obj_exact = self._compute_exact_kl(n_mom, base_moments, orth_method,
                                                                             tol_density, tol_exact_cov)

            np.save(os.path.join(work_dir, "truncation_err_{}".format(n_mom)), kl_plot.truncation_err)


            #print("NOISE LEVEL ", noise_level)
            _, distr_obj, exact_cov, cov = self.find_regularization_param(plot_res=False, noise_level=noise_level,
                                                                          work_dir=work_dir, orth_method=orth_method,
                                                                          n_mom=n_mom)
            exact_moments_orig = exact_cov[:, 0]

            distr_plot.add_distribution(distr_obj[0], label="noise: {:f}, mom: {} th: {}, KL div: {:f}".format(noise_level,
                                                                                                               n_mom,
                                                                                                    distr_obj[2],
                                                                                                    distr_obj[1].kl))

            self._save_distr_data(distr_obj[0], distr_plot, work_dir, n_mom, distr_obj[1])

            kl_div = mlmc.tool.simple_distribution.KL_divergence(distr_obj_exact.density, distr_obj[0].density, self.domain[0],
                                                            self.domain[1])

            kl_plot.add_value((n_mom, kl_div))
            kl_plot.add_iteration(x=n_mom, n_iter=distr_obj[1].nit, failed=not distr_obj[1].success)

            # print("exact moments ", exact_moments[:len(moments_data[:, 0])])
            # print("moments data ", moments_data[:, 0])
            # print("difference ", np.array(exact_moments) - np.array(moments_data[:, 0]))
            print("difference orig", np.array(exact_moments_orig) - np.array(cov[:, 0][:len(exact_moments_orig)]))

            diff_orig = np.array(exact_moments_orig) - np.array(cov[:, 0][:len(exact_moments_orig)])

            kl_plot.add_moments_l2_norm((n_mom, np.linalg.norm(diff_orig)**2))

            self._save_kl_data(work_dir, n_mom, kl_div, distr_obj[1].nit, not distr_obj[1].success,
                               np.linalg.norm(diff_orig) ** 2, distr_obj[2])

        kl_plot.show(None)
        distr_plot.show(None)
        distr_plot.reset()
        return results

    def plot_KL_div_inexact_reg(self):
        """
        Plot KL divergence for different noise level of exact moments
        """
        min_noise = 1e-6
        max_noise = 1e-1
        geom_seq = np.exp(np.linspace(np.log(min_noise), np.log(max_noise), 10))
        noise_levels = np.flip(np.concatenate(([0.0], geom_seq)), axis=0)

        noise_levels = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6]

        noise_levels = [1e-2]#, 1e-3, 1e-4]

        #noise_levels = noise_levels[:2]

        tol_exact_cov = 1e-10
        tol_density = 1e-5
        results = []
        orth_method = 1
        n_moments = 35  # 25 is not enough for TwoGaussians

        distr_plot = plot.Distribution(exact_distr=self.cut_distr, title=self.title+"_inexact", cdf_plot=False,
                                       log_x=self.log_flag, error_plot=False)

        dir_name = "reg_KL_div_inexact_35_{}_five_fingers_1e-2_density".format(orth_method)
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

        work_dir = os.path.join(dir_name, self.name)
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
        # # Set moments objects
        moment_class, _, _, self.use_covariance = self.moments_data
        log = self.log_flag
        #
        self.moments_fn = moment_class(n_moments, self.domain, log=log, safe_eval=False)
        #
        # ##########################################
        # # Orthogonalize moments
        #
        base_moments = self.moments_fn
        # exact_cov = mlmc.tool.simple_distribution.compute_semiexact_cov(base_moments, self.pdf)

        kl_plot.truncation_err, distr_obj_exact = self._compute_exact_kl(n_moments, base_moments, orth_method,
                                                                         tol_density, tol_exact_cov)

        np.save(os.path.join(work_dir, "truncation_err"), kl_plot.truncation_err)
        # exact_moments_orig = mlmc.tool.simple_distribution.compute_semiexact_moments(self.moments_fn, self.pdf, tol=1e-10)
        #exact_moments_orig = exact_cov[:, 0]
        # print("original exact moments ", exact_moments_orig)
        # print("exact cov[:, 0] ", exact_cov[:, 0])

        ###############################################
        # For each moment size compute density
        for i_m, noise_level in enumerate(noise_levels):
            #print("NOISE LEVEL ", noise_level)
            _, distr_obj, exact_cov, cov = self.find_regularization_param(plot_res=False, noise_level=noise_level,
                                                                          work_dir=work_dir, orth_method=orth_method)
            exact_moments_orig = exact_cov[:, 0]

            distr_plot.add_distribution(distr_obj[0], label="noise: {:f}, th: {}, KL div: {:f}".format(noise_level,
                                                                                                    distr_obj[2],
                                                                                                    distr_obj[1].kl))

            self._save_distr_data(distr_obj[0], distr_plot, work_dir, noise_level, distr_obj[1])

            kl_div = mlmc.tool.simple_distribution.KL_divergence(distr_obj_exact.density, distr_obj[0].density, self.domain[0],
                                                            self.domain[1])

            kl_plot.add_value((noise_level, kl_div))
            kl_plot.add_iteration(x=noise_level, n_iter=distr_obj[1].nit, failed=not distr_obj[1].success)

            diff_orig = np.array(exact_moments_orig) - np.array(cov[:, 0][:len(exact_moments_orig)])

            kl_plot.add_moments_l2_norm((noise_level, np.linalg.norm(diff_orig)**2))

            self._save_kl_data(work_dir, noise_level, kl_div, distr_obj[1].nit, not distr_obj[1].success,
                               np.linalg.norm(diff_orig) ** 2, distr_obj[2])

        kl_plot.show(None)
        distr_plot.show(None)
        distr_plot.reset()
        return results

    def determine_regularization_param(self, reg_params=None, regularization=None, noise=None):
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

        #noise = 1e-1

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

        if noise is not None:
            noise_levels = [noise]

        #noise_levels = [5e-2, 1e-2, 5e-3]#, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6]

        noise_levels = [1e-2, 1e-18]
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

            if regularization is None:
                regularization = mlmc.tool.simple_distribution.Regularization2ndDerivation()

            #regularization = mlmc.tool.simple_distribution.RegularizationInexact()
            #reg_parameters = [0]#[1e-6]

            #regularization = mlmc.tool.simple_distribution.Regularization2ndDerivation()

            #reg_parameters = [10, 1e-5] # 10 is suitable for Splines
            #reg_parameters = [5e-6] # is suitable for Legendre

            #reg_parameters = [5e-6, 3.65810502e-06]
            #reg_parameters = [1e-12]

            #reg_parameters = [2.682695795279722e-05, 1.9306977288832498e-06, 1e-05, 2.6826957952797274e-06, 0.0002682695795279722]
            #reg_parameters = [5e-6, 1.519911082952933e-06, 1.788649529057435e-06, 1.2915496650148827e-06, 1.0974987654930544e-06, 2.104904144512022e-06]

            #reg_parameters = [0, 1e-12, 5e-9, 5e-8]

            #reg_parameters = [1.5361749466718295e-07]#, 5e-7]

            #reg_parameters = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
            #reg_parameters = [1e-12, 1e-11, 1e-10, 1e-9]

            ##########################
            # CUT EIGENVALUES params #
            ##########################
            # two gaussians
            #reg_parameters = [0, 1e-7, 1e-6, 5e-6, 1e-5]
            # norm
            # reg_parameters = [0, 1e-8, 1e-7, 1e-6, 1e-5]
            # five fingers
            #reg_parameters = [0, 1e-8, 1e-7, 1e-6, 1e-5]
            reg_parameters = [0, 1e-7, 1e-6]
            #reg_parameters = [0, 1e-7, 1e-6, 1e-5]
            reg_parameters = [0, 5e-7, 1e-6]
            #reg_parameters = [0, 5e-8, 1e-6]#[1e-9, 1e-7]
            reg_parameters = [1e-8, 1e-7, 1e-6]
            reg_parameters = [1.3848863713938746e-05, 1.6681005372000593e-05, 2.0092330025650498e-05, 2.4201282647943835e-05, 2.9150530628251818e-05, 3.511191734215135e-05, 4.2292428743895077e-05, 5.0941380148163855e-05, 6.135907273413175e-05, 7.39072203352579e-05]
            reg_parameters = [1.3848863713938746e-06, 1.6681005372000593e-06, 2.0092330025650498e-06,
                              2.4201282647943835e-06, 2.9150530628251818e-06, 3.511191734215135e-06,
                              4.2292428743895077e-06, 5.0941380148163855e-06, 6.135907273413175e-06,
                              7.39072203352579e-06]

            reg_parameters = [1.3848863713938746e-07, 1.6681005372000593e-07,
                              2.0092330025650498e-07,
                              2.4201282647943835e-07, 2.9150530628251818e-07,
                              3.511191734215135e-07,
                              4.2292428743895077e-07, 5.0941380148163855e-07,
                              6.135907273413175e-07,
                              7.39072203352579e-07]

            reg_parameters = [0, 5.590810182512222e-11, 5.590810182512222e-10]


            # two gaussians
            #reg_parameters = [8.66882444e-06]# orth 2
            reg_parameters = [2.1964e-5] # orth 2
            #reg_parameters = [2.7879e-7]# orth 4

            # lognorm
            #reg_parameters = [1.11096758e-04] # orth 2
            reg_parameters = [0, 5e-7]#[5.4789e-06]
            #reg_parameters = [5.292e-9]

            # norm
            reg_parameters = [0, 3.2557e-6]  # orth 2
            reg_parameters = [0, 2.327e-6]  # orth 4

            # lognorm
            reg_parameters = [0, 3.2557e-6]  # orth 2
            reg_parameters = [0, 2.327e-6]  # orth 4

            # TWO gaussians
            reg_parameters = [6e-7, 7e-7, 8e-7, 9e-7, 1e-6]#, 2e-6, 5e-6, 7e-6, 9e-6] # orth 2
            #reg_parameters = [5.41918e-7]  # orth 2
            #reg_parameters = [1.54956e-7] # orth 4

            reg_parameters = [1.676e-7]

            reg_parameters = [1e-6] # Twogaussians
            reg_parameters = [7e-6] # NORM orth 2
            reg_parameters = [1e-6]

            reg_parameters = [5e-7]
            reg_parameters = [2e-7] # NORM

            reg_parameters = [7e-10] # lognorm
            reg_parameters = [7e-11, 6e-9, 7e-9, 2e-7]

            reg_parameters = [3e-7]
            reg_parameters = [5e-7] # five fingers orth 4
            reg_parameters = [1e-9] # five fingers orth 2

            # five fingers
            # reg_parameters = [0, 6.14735e-7]  # orth 2
            # reg_parameters = [0, 3.11859e-7]  # orth 4


            # ORTH 2
            #cauchy
            reg_parameters = [2.082e-8] # 1e-2
            # lognorm
            #reg_parameters = [7.118e-6]
            reg_parameters = [0]


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
                                                              reg_param=reg_param, orth_method=orth_method,
                                                              regularization=regularization)
                n_moments = len(self.exact_moments)

                original_evals, evals, threshold, L = info
                new_moments = np.matmul(moments_with_noise, L.T)

                moments_data = np.empty((n_moments, 2))
                moments_data[:, 0] = new_moments
                moments_data[:, 1] = noise ** 2
                moments_data[0, 1] = 1.0

                print("moments data ", moments_data)


                result, distr_obj = self.make_approx(mlmc.tool.simple_distribution.SimpleDistribution, noise, moments_data,
                                                     tol=1e-7, reg_param=reg_param, regularization=regularization)

                m = mlmc.tool.simple_distribution.compute_exact_moments(self.moments_fn, distr_obj.density)
                e_m = mlmc.tool.simple_distribution.compute_exact_moments(self.moments_fn, self.pdf)
                moments.append(m)
                all_exact_moments.append(e_m)

                # if reg_param > 0:
                #     distr_obj._analyze_reg_term_jacobian([reg_param])

                # result, distr_obj = self.make_approx(mlmc.tool.simple_distribution.SimpleDistribution, noise, moments_data,
                #                                      tol=1e-10, reg_param=reg_param, prior_distr_obj=distr_obj)

                print("DISTR OBJ reg param {}, MULTIPLIERS {}".format(reg_param, distr_obj.multipliers))

                distr_plot.add_distribution(distr_obj,
                                            label="n: {:0.4g}, th: {}, alpha: {:0.4g},"
                                                  " KL_div: {:0.4g}".format(noise, threshold, reg_param, result.kl),
                                           size=n_moments, mom_indices=plot_mom_indices, reg_param=reg_param)

                results.append(result)

                final_jac = distr_obj.final_jac

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

            print("kl ", kl)
            print("tv ", tv)
            print("l2 ", l2)
            # print("density ", int_density)

        print("FINAL moments ", moments)
        print("exact moments ", all_exact_moments)

        # for exact, estimated in zip(moments, all_exact_moments):
        #     print("(exact-estimated)**2", (exact-estimated)**2)
        #     print("sum(exact-estimated)**2", np.sum((exact - estimated) ** 2))

        distr_plot.show(file="determine_param {}".format(self.title))#file=os.path.join(dir, self.pdfname("_pdf_iexact")))
        distr_plot.reset()

        print("kl divergence", kl)

        #self._plot_kl_div(noise_levels, kl_total)

        #self.plot_gradients(distr_obj.gradients)
        #self._plot_kl_div(noise_levels, kl_2_total)
        plt.show()

        #self.check_convergence(results)
        #self.eigenvalues_plot.show(file=None)#self.pdfname("_eigenvalues"))

        return results

    def determine_regularization_param_tv(self, reg_params=None):
        """
        Test density approximation for maximal number of moments
        and varying amount of noise added to covariance matrix.
        :return:
        """
        np.random.seed(1234)
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
        regularization = None

        moments = []
        all_exact_moments = []

        moment_class, min_n_moments, max_n_moments, self.use_covariance = self.moments_data
        log = self.log_flag
        if min_n_moments == max_n_moments:
            self.moment_sizes = np.array(
                [max_n_moments])  # [36, 38, 40, 42, 44, 46, 48, 50, 52, 54])+1#[max_n_moments])#10, 18, 32, 64])
        else:
            self.moment_sizes = np.round(np.exp(np.linspace(np.log(min_n_moments), np.log(max_n_moments), 8))).astype(
                int)
        # self.moment_sizes = [3,4,5,6,7]

        self.moments_fn = moment_class(max_n_moments, self.domain, log=log, safe_eval=False)

        for noise in noise_levels:
            kl_2 = []
            kl = []

            #regularization = mlmc.tool.simple_distribution.Regularization1()
            #regularization = mlmc.tool.simple_distribution.RegularizationTV()

            #reg_parameters = [10, 1e-5] # 10 is suitable for Splines
            #reg_parameters = [5e-6] # is suitable for Legendre

            #reg_parameters = [1e-8]
            #reg_parameters = [6*1e-2, 7*1e-2, 1e-1]  # find reg param between 5*1e-2 and 1e-1

            reg_parameters = [0.0001]
            reg_parameters = [9.47421052631579e-05]
            reg_parameters = [7e-6] # 10 momentů
            reg_parameters = [1e-4] # 20 momentů
            reg_parameters = [1e-5] # 20 momentů
            reg_parameters = [1e-5] # TwoGaussians 35 moments, 0.01 noise
            reg_parameters = [0.0003, 0.00035, 0.0004, 0.00010163898118064394]

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
                # info, moments_with_noise = self.setup_moments(self.moments_data, noise_level=noise,
                #                                               reg_param=reg_param)

                exact_cov, reg_matrix = mlmc.tool.simple_distribution_total_var.compute_semiexact_cov_2(self.moments_fn,
                                                                                                   self.pdf,
                                                                                                   reg_param=reg_param)
                size = self.moments_fn.size

                self.exact_moments = exact_cov[:, 0]

                cov_noise = np.random.randn(size ** 2).reshape((size, size))
                cov_noise += cov_noise.T
                cov_noise *= 0.5 * noise
                cov_noise[0, 0] = 0

                print("cov noise ")
                print(pd.DataFrame(cov_noise))
                cov = exact_cov + cov_noise

                # Add noise and regularization
                #cov = exact_cov + fine_noise[:len(exact_cov), :len(exact_cov)]
                cov += reg_matrix

                moments_with_noise = cov[:, 0]
                self.moments_fn, info, cov_centered = mlmc.tool.simple_distribution_total_var.construct_orthogonal_moments(
                    self.moments_fn,
                    cov,
                    noise**2,
                    reg_param=reg_param,
                    orth_method=1)
                original_evals, evals, threshold, L = info
                fine_moments = np.matmul(moments_with_noise, L.T)

                cov_with_noise = cov

                moments_data = np.empty((len(fine_moments), 2))
                moments_data[:, 0] = fine_moments  # self.exact_moments
                moments_data[:, 1] = 1  # noise ** 2
                moments_data[0, 1] = 1.0

                self.exact_moments = exact_cov[:, 0][:len(fine_moments)]

                # n_moments = len(self.exact_moments)
                #
                # original_evals, evals, threshold, L = info
                # new_moments = np.matmul(moments_with_noise, L.T)
                #
                # moments_data = np.empty((n_moments, 2))
                # moments_data[:, 0] = new_moments
                # moments_data[:, 1] = noise ** 2
                # moments_data[0, 1] = 1.0

                print("moments data ", moments_data)

                # modif_cov, reg_matrix = mlmc.tool.simple_distribution.compute_semiexact_cov_2(self.moments_fn, self.pdf, reg_param=reg_param,
                #                                                              reg_param_beta=reg_param_beta)
                #
                # #modif_cov += reg_matrix
                # # print("modif cov")
                # # print(pd.DataFrame(modif_cov))
                # # print("modif cov inv")
                # # print(np.linalg.inv(pd.DataFrame(modif_cov)))
                #
                # diff_norm = np.linalg.norm(modif_cov - np.eye(*modif_cov.shape)) / n_moments
                # ref_moments = np.zeros(n_moments)
                # ref_moments[0] = 1.0
                #
                # print("ref moments ", ref_moments)
                # mom_err = np.linalg.norm(self.exact_moments - ref_moments) / np.sqrt(n_moments)
                # print("noise: {:6.2g} error of natural cov: {:6.2g} natural moments: {:6.2g}".format(
                #     noise, diff_norm, mom_err))

                # distr_plot = plot.Distribution(exact_distr=self.cut_distr, title="Density, " + self.title,
                #                                      log_x=self.log_flag, error_plot='kl')
                result, distr_obj = self.make_approx(mlmc.tool.simple_distribution_total_var.SimpleDistribution, noise,
                                                     moments_data,
                                                     tol=1e-7, reg_param=reg_param, regularization=regularization)

                m = mlmc.tool.simple_distribution_total_var.compute_exact_moments(self.moments_fn, distr_obj.density)
                e_m = mlmc.tool.simple_distribution_total_var.compute_exact_moments(self.moments_fn, self.pdf)
                moments.append(m)
                all_exact_moments.append(e_m)

                # if reg_param > 0:
                #     distr_obj._analyze_reg_term_jacobian([reg_param])

                # result, distr_obj = self.make_approx(mlmc.tool.simple_distribution.SimpleDistribution, noise, moments_data,
                #                                      tol=1e-10, reg_param=reg_param, prior_distr_obj=distr_obj)

                print("DISTR OBJ reg param {}, MULTIPLIERS {}".format(reg_param, distr_obj.multipliers))

                distr_plot.add_distribution(distr_obj,
                                            label="noise: {}, threshold: {}, reg param: {}, KL_div: {}".format(noise, threshold,
                                                                                                   reg_param, result.kl),
                                           size=n_moments, mom_indices=plot_mom_indices, reg_param=reg_param)

                results.append(result)

                final_jac = distr_obj.final_jac

                print("final jac")
                print(pd.DataFrame(final_jac))

                print("ORIGINAL COV CENTERED")
                print(pd.DataFrame(cov_centered))

                #print("np.linalg.inv(L) ", np.linalg.inv(L))

                M = np.eye(len(cov_with_noise[0]))
                M[:, 0] = -cov_with_noise[:, 0]

                # print("M")
                # print(pd.DataFrame(M))
                #
                # print("np.linalg.inv(M) ", np.linalg.inv(M))

                # print("M-1 @ L-1 @ H @ L.T-1 @ M.T-1")
                # print(pd.DataFrame(
                #     np.linalg.inv(M) @ (
                #                 np.linalg.inv(L) @ final_jac @ np.linalg.inv(L.T)) @ np.linalg.inv(M.T)))
                #

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


        print("FINAL moments ", moments)
        print("exact moments ", all_exact_moments)

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

    def compare_orthogonalization(self):
        """
        Test density approximation for maximal number of moments
        and varying amount of noise added to covariance matrix.
        :return:
        """
        min_noise = 1e-6
        max_noise = 0.01
        results = []

        orth_methods = [4]  # 1 - add constant to all eigenvalues,
                                 # 2 - cut eigenvalues below threshold,
                                 # 3 - add const to eigenvalues below threshold

        titles = {1: "add noise-min(eval, 0) to eigenvalues",
                  2: "cut eigenvalues",
                  3: "add const to eigenvalues below threshold",
                  4: "pca"}

        geom_seq = np.exp(np.linspace(np.log(min_noise), np.log(max_noise), 5))
        noise_levels = np.flip(np.concatenate(([0.0], geom_seq)), axis=0)

        noise_levels = noise_levels[:1]
        print("noise levels ", noise_levels)

        reg_param = 0  # NOT works with regularization too

        mom_class, min_mom, max_mom, log_flag = self.moments_data
        self.use_covariance = True

        for orth_method in orth_methods:
            distr_plot = plot.Distribution(exact_distr=self.cut_distr, title=titles[orth_method], cdf_plot=False,
                                           log_x=self.log_flag, error_plot='kl')

            self.eigenvalues_plot = plot.Eigenvalues(title="Eigenvalues, " + self.title)

            for noise in noise_levels:

                self.moments_data = (mom_class, max_mom, max_mom, log_flag)
                info, moments_with_noise = self.setup_moments(self.moments_data, noise_level=noise,
                                                              reg_param=reg_param, orth_method=orth_method)

                original_evals, evals, threshold, L = info
                new_moments = np.matmul(moments_with_noise, L.T)

                n_moments = len(moments_with_noise)

                moments_data = np.empty((len(new_moments), 2))
                moments_data[:, 0] = new_moments
                moments_data[:, 1] = noise ** 2
                moments_data[0, 1] = 1.0

                print("momentsdata ", moments_data)

                modif_cov = mlmc.tool.simple_distribution.compute_semiexact_cov(self.moments_fn, self.pdf)

                print("modif_cov ", modif_cov)

                # diff_norm = np.linalg.norm(modif_cov - np.eye(*modif_cov.shape)) / n_moments
                # ref_moments = np.zeros(n_moments)
                # ref_moments[0] = 1.0
                # mom_err = np.linalg.norm(self.exact_moments[:n_moments] - ref_moments) / np.sqrt(n_moments)
                # print("noise: {:6.2g} error of natural cov: {:6.2g} natural moments: {:6.2g}".format(
                #     noise, diff_norm, mom_err))

                # assert mom_err/(noise + 1e-10) < 50  - 59 for five fingers dist

                regularization = mlmc.tool.simple_distribution.Regularization2ndDerivation()

                result, distr_obj = self.make_approx(mlmc.tool.simple_distribution.SimpleDistribution, noise,
                                                     moments_data, reg_param=reg_param, tol=1e-5,
                                                     regularization=regularization)

                distr_plot.add_distribution(distr_obj,
                                            label="m: {}, th: {}, noise: {}, KL: {}".format(n_moments, threshold,
                                                                                            noise, result.kl))
                results.append(result)


            # self.check_convergence(results)
            #self.eigenvalues_plot.show(None)  # file=self.pdfname("_eigenvalues"))
            distr_plot.show(None)  # "PDF aprox")#file=self.pdfname("_pdf_iexact"))
            distr_plot.reset()
            plt.show()
        return results

    def inexact_conv(self):
        """
        Test density approximation for maximal number of moments
        and varying amount of noise added to covariance matrix.
        :return:
        """
        min_noise = 1e-6
        max_noise = 1e-2
        results = []

        distr_plot = plot.Distribution(exact_distr=self.cut_distr, title="", cdf_plot=False,
                                            log_x=self.log_flag, error_plot='kl')

        self.eigenvalues_plot = plot.Eigenvalues(title="Eigenvalues, " + self.title)

        geom_seq = np.exp(np.linspace(np.log(min_noise), np.log(max_noise), 5))
        noise_levels = np.flip(np.concatenate(([0.0], geom_seq)), axis=0)

        noise_levels = noise_levels[:1]

        print("noise levels ", noise_levels)

        orth_method = 2  # cut eigenvalues
        mom_class, min_mom, max_mom, log_flag = self.moments_data

        #moments_num = [5, 10, 15, 20]#, 10, 20, 30]
        moments_num = [35]
        regularization = None
        reg_param = 0
        res_mom = []
        res_mom_norm = []
        norm_coefs = []

        for noise in noise_levels:
            for m in moments_num:#np.arange(min_mom, max_mom, 5):

                for self.use_covariance in [True]:
                    print("self use covariance ", self.use_covariance)

                    # regularization = mlmc.tool.simple_distribution.RegularizationInexact()
                    # reg_param = 1e-3

                    self.moments_data = (mom_class, m, m, log_flag)
                    info, moments_with_noise = self.setup_moments(self.moments_data, noise_level=noise,
                                                                  orth_method=orth_method, regularization=regularization,
                                                                  reg_param=reg_param)

                    n_moments = len(moments_with_noise)

                    original_evals, evals, threshold, L = info
                    new_moments = np.matmul(moments_with_noise, L.T)
                    n_moments = len(new_moments)

                    moments_data = np.empty((n_moments, 2))
                    moments_data[:, 0] = new_moments
                    moments_data[:, 1] = noise ** 2
                    moments_data[0, 1] = 1.0

                    print("moments data ", moments_data)

                    if self.use_covariance:
                        print("if use covariance ", self.use_covariance)

                        modif_cov, reg_matrix = mlmc.tool.simple_distribution.compute_semiexact_cov_2(self.moments_fn, self.pdf,
                                                                                     regularization=regularization,
                                                                                     reg_param=reg_param)

                        print("modif_cov ", modif_cov)

                        diff_norm = np.linalg.norm(modif_cov - np.eye(*modif_cov.shape)) / n_moments
                        ref_moments = np.zeros(n_moments)
                        ref_moments[0] = 1.0
                        mom_err = np.linalg.norm(self.exact_moments[:n_moments] - ref_moments) / np.sqrt(n_moments)
                        print("noise: {:6.2g} error of natural cov: {:6.2g} natural moments: {:6.2g}".format(
                            noise, diff_norm, mom_err))

                        #assert mom_err/(noise + 1e-10) < 50  - 59 for five fingers dist

                        result, distr_obj = self.make_approx(mlmc.tool.simple_distribution.SimpleDistribution, noise,
                                                             moments_data,
                                                             tol=1e-8, regularization=regularization, reg_param=reg_param)

                        distr_plot.add_distribution(distr_obj,
                                                    label="moments {}, threshold: {}, noise: {:0.3g}, kl: {:0.3g}".
                                                    format(n_moments, threshold, noise, result.kl))
                        results.append(result)

                    else:

                        # TODO:
                        # Use SimpleDistribution only as soon as it use regularization that improve convergency even without
                        # cov matrix. preconditioning.
                        result, distr_obj = self.make_approx(mlmc.tool.simple_distribution.SimpleDistribution, noise, moments_data, tol=1e-5)
                        distr_plot.add_distribution(distr_obj, label="{} moments, kl: {}".format(n_moments, result.kl))
                        results.append(result)

                    print("ORIGINAL COV CENTERED")
                    print(pd.DataFrame(self._cov_centered))

                    M = np.eye(len(self._cov_with_noise[0]))
                    M[:, 0] = -self._cov_with_noise[:, 0]

                    final_jac = distr_obj.final_jac

                    print("result jacobian")
                    print(pd.DataFrame(distr_obj.final_jac))

                    # print("M-1 @ L-1 @ H @ L.T-1 @ M.T-1")
                    # print(pd.DataFrame(
                    #     np.linalg.inv(M) @ (
                    #                 np.linalg.inv(L) @ final_jac @ np.linalg.inv(L.T)) @ np.linalg.inv(M.T)))

                    num_moments = m

                    moments_from_density = (np.linalg.pinv(L) @ distr_obj.final_jac @ np.linalg.pinv(L.T))[:, 0]

                    res = (moments_from_density[:num_moments - 1] - self.moments_without_noise[:num_moments - 1]) ** 2

                    norm_coef = np.max(moments_num) - m
                    if norm_coef == 0:
                        norm_coef = 1

                    norm_coefs.append(norm_coef)

                    print("norm coef ", norm_coef)
                    res_mom_norm.append(np.array(res_mom) / norm_coef)

                    res_mom.append(res)

        print("res mom ", res_mom)
        print("res mom norm ", res_mom_norm)
        for res, res_n, n_coef in zip(res_mom, res_mom_norm, norm_coefs):
            print("res sum ", np.sum(res))
            print("res norm sum ", np.sum(res_n))

            print("res sum / norm coef  ", np.sum(res)/n_coef)

        for res in res_mom:
            print("NORMED res ", np.sum(res) * ((np.max(moments_num)) / len(res)))

        #self.check_convergence(results)
        self.eigenvalues_plot.show(None)#file=self.pdfname("_eigenvalues"))
        distr_plot.show(None)#"PDF aprox")#file=self.pdfname("_pdf_iexact"))
        distr_plot.reset()
        plt.show()
        return results

    def inexact_conv_test(self):
        """
        Test density approximation for maximal number of moments
        and varying amount of noise added to covariance matrix.
        :return:
        """
        min_noise = 1e-6
        max_noise = 1e-2
        results = []

        distr_plot = plot.Distribution(exact_distr=self.cut_distr, title="", cdf_plot=False,
                                            log_x=self.log_flag, error_plot='kl')

        self.eigenvalues_plot = plot.Eigenvalues(title="Eigenvalues, " + self.title)

        geom_seq = np.exp(np.linspace(np.log(min_noise), np.log(max_noise), 5))
        noise_levels = np.flip(np.concatenate(([0.0], geom_seq)), axis=0)

        noise_levels = noise_levels[:1]

        print("noise levels ", noise_levels)

        orth_method = 2
        mom_class, min_mom, max_mom, log_flag = self.moments_data

        #moments_num = [5, 10, 15, 20]#, 10, 20, 30]
        moments_num = [max_mom]
        regularization = None
        reg_param = 0

        res_mom = []

        res_mom_norm = []

        norm_coefs = []

        for noise in noise_levels:
            for m in moments_num:#np.arange(min_mom, max_mom, 5):
                multipliers = []
                rep_size = 1
                multipliers = np.zeros((rep_size, m))

                self.setup_moments(self.moments_data, noise_level=0)
                exact_moments = self.exact_moments
                exact_moments_orig = self.moments_without_noise
                moments_data = np.empty((m, 2))
                moments_data[:, 0] = self.exact_moments[:m]
                moments_data[:, 1] = 1.0

                exact_result, exact_distr_obj = self.make_approx(mlmc.tool.simple_distribution.SimpleDistribution, 0.0,
                                                     moments_data,
                                                     tol=1e-10)
                exact_L = self.L

                for i in range(rep_size):
                    #np.random.seed(i)

                    for self.use_covariance in [True]:
                        self.moments_data = (mom_class, m, m, log_flag)
                        info, moments_with_noise = self.setup_moments(self.moments_data, noise_level=noise,
                                                                      orth_method=orth_method, regularization=regularization,
                                                                      reg_param=1e-3)

                        original_evals, evals, threshold, L = info
                        new_moments = np.matmul(moments_with_noise, L.T)
                        n_moments = len(new_moments)
                        moments_data = np.empty((n_moments, 2))
                        moments_data[:, 0] = new_moments
                        moments_data[:, 1] = noise ** 2
                        moments_data[0, 1] = 1.0


                        # modif_cov, reg_matrix = mlmc.tool.simple_distribution.compute_semiexact_cov_2(self.moments_fn, self.pdf,
                        #                                                              regularization=regularization,
                        #                                                              reg_param=reg_param)
                        #
                        # diff_norm = np.linalg.norm(modif_cov - np.eye(*modif_cov.shape)) / n_moments
                        # ref_moments = np.zeros(n_moments)
                        # ref_moments[0] = 1.0
                        # mom_err = np.linalg.norm(self.exact_moments[:n_moments] - ref_moments) / np.sqrt(n_moments)
                        # print("noise: {:6.2g} error of natural cov: {:6.2g} natural moments: {:6.2g}".format(
                        #     noise, diff_norm, mom_err))

                        result, distr_obj = self.make_approx(mlmc.tool.simple_distribution.SimpleDistribution, noise,
                                                             moments_data,
                                                             tol=1e-8, regularization=regularization, reg_param=reg_param)

                        multipliers[i, :len(distr_obj.multipliers)] = distr_obj.multipliers

                        distr_plot.add_distribution(distr_obj,
                                                    label="{} moments, {} threshold, noise: {}, kl: {}".
                                                    format(n_moments, threshold, noise, result.kl))
                        results.append(result)



                        # print("ORIGINAL COV CENTERED")
                        # print(pd.DataFrame(self._cov_centered))
                        #
                        # M = np.eye(len(self._cov_with_noise[0]))
                        # M[:, 0] = -self._cov_with_noise[:, 0]
                        #
                        # final_jac = distr_obj.final_jac
                        #
                        # print("result jacobian")
                        # print(pd.DataFrame(distr_obj.final_jac))
                        #
                        # # print("M-1 @ L-1 @ H @ L.T-1 @ M.T-1")
                        # # print(pd.DataFrame(
                        # #     np.linalg.inv(M) @ (
                        # #                 np.linalg.inv(L) @ final_jac @ np.linalg.inv(L.T)) @ np.linalg.inv(M.T)))

                        #===================================================
                        new_moments = np.matmul(exact_moments, L.T)
                        n_moments = len(new_moments)
                        moments_data = np.empty((n_moments, 2))
                        moments_data[:, 0] = new_moments
                        moments_data[:, 1] = noise ** 2
                        moments_data[0, 1] = 1.0

                        result, exact_distr_obj = self.make_approx(mlmc.tool.simple_distribution.SimpleDistribution, noise,
                                                             moments_data,
                                                             tol=1e-8, regularization=regularization,
                                                             reg_param=reg_param)


                        #===================================================

                        num_moments = m

                        moments_from_density = (np.linalg.pinv(L) @ distr_obj.final_jac @ np.linalg.pinv(L.T))[:, 0]

                        print("moments from density ", moments_from_density)
                        print("distr obj multipliers ", distr_obj.multipliers)


                        res = (moments_from_density[:num_moments - 1] - self.moments_without_noise[:num_moments - 1]) ** 2

                        norm_coef = np.max(moments_num) - m
                        if norm_coef == 0:
                            norm_coef = 1

                        norm_coefs.append(norm_coef)

                        print("norm coef ", norm_coef)
                        res_mom_norm.append(np.array(res_mom) / norm_coef)

                        res_mom.append(res)

                        a, b = self.domain
                        kl = mlmc.tool.simple_distribution.KL_divergence(exact_distr_obj.density, distr_obj.density, a, b)

                        print("KL divergence ", kl)

                        # moments = np.linalg.inv(exact_L) @ exact_moments
                        # print("moments ", moments)
                        # print("exact moments orig ", exact_moments_orig)
                        # exact_multipliers = exact_distr_obj.multipliers @ np.linalg.inv(exact_L)
                        # multipliers = distr_obj.multipliers @ np.linalg.inv(self.L)

                        moments = new_moments
                        exact_multipliers = exact_distr_obj.multipliers
                        multipliers = distr_obj.multipliers

                        mu_lambda_kl = np.dot(moments[:len(multipliers)],
                                              -(exact_multipliers[:len(multipliers)] - multipliers))

                        print("mu_lambda_kl ", mu_lambda_kl)


                average_multipliers = np.mean(np.array(multipliers), axis=0)

                #distr_obj.multipliers = average_multipliers

                distr_plot.add_distribution(distr_obj, label="average multipliers")


        # print("res mom ", res_mom)
        # print("res mom norm ", res_mom_norm)
        # for res, res_n, n_coef in zip(res_mom, res_mom_norm, norm_coefs):
        #     print("res sum ", np.sum(res))
        #     print("res norm sum ", np.sum(res_n))
        #
        #     print("res sum / norm coef  ", np.sum(res)/n_coef)
        #
        # for res in res_mom:
        #     print("NORMED res ", np.sum(res) * ((np.max(moments_num)) / len(res)))
        #
        # #self.check_convergence(results)
        # self.eigenvalues_plot.show(None)#file=self.pdfname("_eigenvalues"))
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
        tests = [case.mlmc_conv]
        #tests = [case.exact_conv]
        #tests = [case.inexact_conv]
        tests = [case.inexact_conv_test]
        #tests = [case.plot_KL_div_exact]
        #tests = [case.plot_KL_div_inexact_reg]
        #tests = [case.plot_KL_div_inexact_reg_mom]
        #tests = [case.plot_KL_div_inexact]
        #tests = [case.determine_regularization_param]
        # #tests = [case.determine_regularization_param_tv]
        #tests = [case.find_regularization_param]
        #tests = [case.find_regularization_param_tv]
        #tests = [case.compare_orthogonalization]
        #tests = [case.compare_spline_max_ent]
        #tests = [case.mc_find_regularization_param]
        #tests = [case.compare_spline_max_ent_save]

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
    #     kl_collected[i_q, :], l2_collected[i_q, :] = exact_conv(cut_distr, moments, tol_exact_moments, title)
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
#         domain, est_domain, mc_test = mlmc.archive.estimate.compute_results(mlmc_list[0], n_moments, test_mc)
#         mlmc.archive.estimate.plot_pdf_approx(ax1, ax2, mc0_samples, mc_test, domain, est_domain)
#     ax1.legend()
#     ax2.legend()
#     fig.savefig('compare_distributions.pdf')
#     plt.show()

@pytest.mark.skip
def test_total_variation():
    function = lambda x: np.sin(x)
    lower_bound, higher_bound = 0, 2 * np.pi
    total_variation = mlmc.tool.simple_distribution.total_variation_vec(function, lower_bound, higher_bound)
    tv = mlmc.tool.simple_distribution.total_variation_int(function, lower_bound, higher_bound)

    assert np.isclose(total_variation, 4, rtol=1e-2, atol=0)
    assert np.isclose(tv, 4, rtol=1e-1, atol=0)

    function = lambda x: x**2
    lower_bound, higher_bound = -5, 5
    total_variation = mlmc.tool.simple_distribution.total_variation_vec(function, lower_bound, higher_bound)
    tv = mlmc.tool.simple_distribution.total_variation_int(function, lower_bound, higher_bound)

    assert np.isclose(total_variation, lower_bound**2 + higher_bound**2, rtol=1e-2, atol=0)
    assert np.isclose(tv, lower_bound ** 2 + higher_bound ** 2, rtol=1e-2, atol=0)

    function = lambda x: x
    lower_bound, higher_bound = -5, 5
    total_variation = mlmc.tool.simple_distribution.total_variation_vec(function, lower_bound, higher_bound)
    tv = mlmc.tool.simple_distribution.total_variation_int(function, lower_bound, higher_bound)
    assert np.isclose(total_variation, abs(lower_bound) + abs(higher_bound), rtol=1e-2, atol=0)
    assert np.isclose(tv, abs(lower_bound) + abs(higher_bound), rtol=1e-2, atol=0)


def plot_derivatives():
    function = lambda x: x
    lower_bound, higher_bound = -5, 5
    x = np.linspace(lower_bound, higher_bound, 1000)
    y = mlmc.tool.simple_distribution.l1_norm(function, x)
    hubert_y = mlmc.tool.simple_distribution.hubert_norm(function, x)

    plt.plot(x, y, '--')
    plt.plot(x, hubert_y, linestyle=':')
    plt.show()


def run_distr():
    distribution_list = [
        # distibution, log_flag
        # (stats.dgamma(1,1), False) # not good
        # (stats.beta(0.5, 0.5), False) # Looks great
        #(bd.TwoGaussians(name='two_gaussians'), False),
        # (bd.FiveFingers(name='five_fingers'), False), # Covariance matrix decomposition failed
        # (bd.Cauchy(name='cauchy'), False),# pass, check exact
        # (bd.Discontinuous(name='discontinuous'), False),
        #(bd.Abyss(), False),
        # # # # # # # # # # # # # # # # # # # #(bd.Gamma(name='gamma'), False) # pass
        # # # # # # # # # # # # # # # # # # # #(stats.norm(loc=1, scale=2), False),
        (stats.norm(loc=0, scale=10), False),
        #(stats.lognorm(scale=np.exp(1), s=1), False),    # Quite hard but peak is not so small comparet to the tail.
        # # (stats.lognorm(scale=np.exp(-3), s=2), False),  # Extremely difficult to fit due to very narrow peak and long tail.
        # (stats.lognorm(scale=np.exp(-3), s=2), True),    # Still difficult for Lagrange with many moments.
        #(stats.chi2(df=10), False),# Monomial: s1=nan, Fourier: s1= -1.6, Legendre: s1=nan
        #(stats.chi2(df=5), True), # Monomial: s1=-10, Fourier: s1=-1.6, Legendre: OK
        #(stats.weibull_min(c=0.5), False),  # Exponential # Monomial stuck, Fourier stuck
        #(stats.weibull_min(c=1), False),  # Exponential
        #(stats.weibull_min(c=2), False),  # Rayleigh distribution
        #(stats.weibull_min(c=5, scale=4), False),   # close to normal
        # (stats.weibull_min(c=1.5), True),  # Infinite derivative at zero
    ]

    # @pytest.mark.skip
    mom = [
        # moments_class, min and max number of moments, use_covariance flag
        #.(moments.Monomial, 10, 10, True),
        # (moments.Fourier, 5, 61),
        # (moments.Legendre, 7,61, False),
        (moments.Legendre, 25, 25, True),
        #(moments.Spline, 10, 10, True),
    ]

    # plot_requirements = {
    #                      'sqrt_kl': False,
    #                      'sqrt_kl_Cr': False,
    #                      'tv': False,
    #                      'sqrt_tv_Cr': False, # TV
    #                      'reg_term': False,
    #                      'l2': False,
    #                      'barron_diff_mu_line': False,
    #                      '1_eig0_diff_mu_line': False}
    #
    #
    # test_kl_estimates(mom[0], distribution_list, plot_requirements)
    # #test_gauss_degree(mom[0], distribution_list[0], plot_requirements, degrees=[210, 220, 240, 260, 280, 300]) #  degrees=[10, 20, 40, 60, 80, 100], [110, 120, 140, 160, 180, 200]
    # test_gauss_degree(mom[0], distribution_list[0], plot_requirements, degrees=[10, 20, 40, 60, 80, 100])
    for m in mom:
        for distr in enumerate(distribution_list):
            #test_spline_approx(m, distr)
            #splines_indicator_vs_smooth(m, distr)
            test_pdf_approx_exact_moments(m, distr)

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

    title = case.title
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

    for _ in range(1000):
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
        # dot_l_diff_mu_diff.append(np.dot(mu_diff, lambda_diff)) # good

        print("exact mu ", exact_mu)
        print("original exact mu ", np.matmul(exact_mu, np.linalg.inv(case.L.T)))
        print("lambda diff ", lambda_diff)

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
    plot_kl_lambda_diff = True

    size = 5
    scatter_size = size ** 2

    if plot_mu_to_lambda_lim:
        Y = np.array(l_diffs) * np.array(np.array(eigs)[:, 0]) / np.array(mu_diffs)
        ax, lx = plot_scatter(ax, mu_diffs, Y, title, ('log', 'linear'), color='red')
        ax.set_ylabel("$\\alpha_0|\lambda_0 - \lambda_r| / |\mu_0 - \mu_r|$")
        ax.set_xlabel("$|\mu_0 - \mu_r|$")
        ax.axhline(y=1.0, color='red', alpha=0.3)

    elif plot_kl_lambda_diff:
        plot_scatter(ax, mu_diffs, np.array(l_diffs) * np.array(np.array(eigs)[:, 0]), title, ('log', 'log'), color='red', s=scatter_size,
                     )#label="$\\alpha_0|\lambda_0 - \lambda_r|$")

        barron_coef = 2 * b_factor_estimate * np.exp(1)
        plot_scatter(ax, mu_diffs, np.sqrt(np.array(kl_divs) / barron_coef), title, ('log', 'log'), color='blue',
                     s=scatter_size)#, label="$\sqrt{D(\\rho || \\rho_{R}) / C_R}$")

        plot_scatter(ax, mu_diffs, np.sqrt(np.array(l_diffs)**2 / barron_coef), title, ('log', 'log'), color='orange',
                       s=scatter_size)#, label="$|\lambda_0 - \lambda_r| / \sqrt{C_R}$")


        plot_scatter(ax, mu_diffs, np.sqrt(dot_l_diff_mu_diff/ barron_coef), title, ('log', 'log'), color='black', s=scatter_size)

    else:
        Y = np.array(l_diffs) * np.array(np.array(eigs)[:, 0]) / np.array(mu_diffs)
        #Y = np.array(eigs)

        #ax, lx = plot_scatter(ax, l_diffs, mu_diffs, title, ('log', 'log'), color='red')
        ax, lx = plot_scatter(ax, mu_diffs, np.array(l_diffs) * np.array(np.array(eigs)[:, 0]),
                              title, ('log', 'log'), color='red', s=scatter_size)

        if plot_req['tv']:
            ax, lx = plot_scatter(ax, mu_diffs, total_vars, title, ('log', 'log'), color='green', s=size**2)
        #plot_scatter(ax, mu_diffs, Y[:, 1], title, ('log', 'log'), color='blue')
        ax.set_xlabel("$|\mu_0 - \mu_r|$")
        #ax.set_xlabel("$|\lambda_0 - \lambda_r|$")

        outline = mpe.withStroke(linewidth=size, foreground='black')

        ax.plot(lx, lx, color='black', lw=size-3,
                path_effects=[outline])
        #ax.plot(lx, lx, color='m', lw=5.0)

        #ax.plot(lx, lx, color='red', label="raw $1/\\alpha_0$", alpha=0.3)

        barron_coef = 2 * b_factor_estimate * np.exp(1)

        # if plot_req['sqrt_kl_Cr']:
        #     plot_scatter(ax, mu_diffs, np.sqrt(np.array(kl_divs) / barron_coef), title, ('log', 'log'),
        #                  color='blue',
        #                  s=scatter_size)

        #kl_divs = np.array(l_diffs)**2

        if plot_req['sqrt_kl_Cr']:
            plot_scatter(ax, mu_diffs, dot_l_diff_mu_diff, title, ('log', 'log'), color='blue', s=scatter_size)

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
