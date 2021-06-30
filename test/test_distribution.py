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
import time
import pytest
import numpy as np
import scipy.stats as stats
import mlmc.plot.plots
import mlmc.tool.simple_distribution
from mlmc import moments


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
            X = distr.rvs(size=1000)
            err = stats.norm.rvs(size=1000)
            X = X * (1 + 0.1 * err)
            domain = (np.min(X), np.max(X))
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


class ConvResult:
    """
    Results of a convergency calculation.
    """
    def __init__(self):
        self.size = 0
        # Moment sizes used
        self.noise = 0.0
        # Noise level used in Covariance and moments.
        self.kl = np.nan
        # KL divergence KL(exact, approx) for individual sizes
        self.l2 = np.nan
        # L2 distance of densities
        self.time = 0
        # times of calculations of density approx.
        self.nit = 0
        # number of iterations in minimization problems
        self.residual_norm = 0
        # norm of residual of non-lin solver
        self.success = False

    def __str__(self):
        return "#{} it:{} err:{} kl:{:6.2g} l2:{:6.2g}".format(self.size, self.nit, self.residual_norm,
                                                               self.kl, self.l2)

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
        self.distr_name = "{:02}_{}".format(i_distr, distr.dist.name)
        self.cut_distr = CutDistribution(distr, quantile)
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
    def domain(self):
        return self.cut_distr.domain

    def pdf(self, x):
        return self.cut_distr.pdf(x)

    def setup_moments(self, moments_data, noise_level):
        """
        Setup moments without transformation.
        :param moments_data: config tuple
        :param noise_level: magnitude of added Gauss noise.
        :return:
        """
        tol_exact_moments = 1e-6
        moment_class, min_n_moments, max_n_moments, self.use_covariance = moments_data
        log = self.log_flag
        self.moment_sizes = np.round(np.exp(np.linspace(np.log(min_n_moments), np.log(max_n_moments), 8))).astype(int)
        #self.moment_sizes = [3,4,5,6,7]
        self.moments_fn = moment_class(max_n_moments, self.domain, log=log, safe_eval=False)


        if self.use_covariance:
            size = self.moments_fn.size
            base_moments = self.moments_fn
            exact_cov = mlmc.tool.simple_distribution.compute_semiexact_cov(base_moments, self.pdf)
            noise = np.random.randn(size**2).reshape((size, size))
            noise += noise.T
            noise *= 0.5 * noise_level
            noise[0, 0] = 0
            cov = exact_cov + noise
            self.moments_fn, info = mlmc.tool.simple_distribution.construct_ortogonal_moments(base_moments, cov, noise_level)
            evals, threshold, L = info

            eye_approx = L @ cov @ L.T
            # test that the decomposition is done well
            assert np.linalg.norm(eye_approx - np.eye(*eye_approx.shape)) < 1e-10

            print("threshold: ", threshold, " from N: ", size)
            if self.eigenvalues_plot:
                threshold = evals[threshold]
                noise_label = "{:5.2e}".format(noise_level)
                self.eigenvalues_plot.add_values(evals, threshold=threshold, label=noise_label)
            self.tol_density_approx = 0.01
        else:
            self.exact_moments += noise_level * np.random.randn(self.moments_fn.size)
            self.tol_density_approx = 1e-4

        self.exact_moments = mlmc.tool.simple_distribution.compute_semiexact_moments(self.moments_fn,
                                self.pdf, tol=tol_exact_moments)



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


    def make_approx(self, distr_class, noise, moments_data, tol):
        result = ConvResult()

        distr_obj = distr_class(self.moments_fn, moments_data,
                                domain=self.domain, force_decay=self.cut_distr.force_decay)
        t0 = time.time()
        min_result = distr_obj.estimate_density_minimize(tol=tol)
        moments = mlmc.tool.simple_distribution.compute_semiexact_moments(self.moments_fn, distr_obj.density)
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
        result.kl = mlmc.tool.simple_distribution.KL_divergence(self.pdf, distr_obj.density, a, b)
        result.l2 = mlmc.tool.simple_distribution.L2_distance(self.pdf, distr_obj.density, a, b)
        print(result)
        X = np.linspace(self.cut_distr.domain[0], self.cut_distr.domain[1] , 10)
        density_vals = distr_obj.density(X)
        exact_vals = self.pdf(X)
        #print("vals: ", density_vals)
        #print("exact: ", exact_vals)
        return result, distr_obj

    def exact_conv(self):
        """
        Test density approximation for varying number of exact moments.
        :return:
        """
        # Setup moments.
        self.setup_moments(self.moments_data, noise_level=0)

        results = []
        distr_plot = mlmc.plot.plots.Distribution(exact_distr=self.cut_distr, title=self.title+"_exact",
                                            log_x=self.log_flag, error_plot='kl')
        for i_m, n_moments in enumerate(self.moment_sizes):
            if n_moments > self.moments_fn.size:
                continue
            print("======================")
            print("EXACT CONV - ", self.title)

            # moments_fn = moment_fn(n_moments, domain, log=log_flag, safe_eval=False )
            # print(i_m, n_moments, domain, force_decay)
            moments_data = np.empty((n_moments, 2))
            moments_data[:, 0] = self.exact_moments[:n_moments]
            moments_data[:, 1] = 1.0

            if self.use_covariance:
                modif_cov = mlmc.tool.simple_distribution.compute_semiexact_cov(self.moments_fn, self.pdf)
                diff_norm = np.linalg.norm(modif_cov - np.eye(*modif_cov.shape))
                print("#{} cov mat norm: {}".format(n_moments, diff_norm))

                result, distr_obj = self.make_approx(mlmc.tool.simple_distribution.SimpleDistribution, 0.0, moments_data,
                                                     tol=1e-5)
            else:
                # TODO:
                # Use SimpleDistribution only as soon as it use regularization that improve convergency even without
                # cov matrix. preconditioning.
                result, distr_obj = self.make_approx(mlmc.distribution.Distribution, 0.0, moments_data)
            distr_plot.add_distribution(distr_obj, label="#{}".format(n_moments))
            results.append(result)

        #self.check_convergence(results)
        distr_plot.show(file=self.pdfname("_pdf_exact"))
        distr_plot.reset()
        return results

    def inexact_conv(self):
        """
        Test density approximation for maximal number of moments
        and varying amount of noise added to covariance matrix.
        :return:
        """
        min_noise = 1e-6
        max_noise = 0.01
        results = []
        distr_plot = mlmc.plot.plots.Distribution(exact_distr=self.cut_distr, title="Density, " + self.title,
                                            log_x=self.log_flag, error_plot='kl')
        self.eigenvalues_plot = mlmc.plot.plots.Eigenvalues(title = "Eigenvalues, " + self.title)

        geom_seq = np.exp(np.linspace(np.log(min_noise), np.log(max_noise), 5))
        noise_levels = np.flip(np.concatenate(([0.0], geom_seq)), axis=0)
        for noise in noise_levels:
            print("======================")
            print("INEXACT CONV - ", self.title)
            self.setup_moments(self.moments_data, noise_level=noise)
            n_moments = len(self.exact_moments)

            moments_data = np.empty((n_moments, 2))
            moments_data[:, 0] = self.exact_moments
            moments_data[:, 1] = 1.0

            if self.use_covariance:
                modif_cov = mlmc.tool.simple_distribution.compute_semiexact_cov(self.moments_fn, self.pdf)
                diff_norm = np.linalg.norm(modif_cov - np.eye(*modif_cov.shape)) / n_moments
                ref_moments = np.zeros(n_moments)
                ref_moments[0] = 1.0
                mom_err = np.linalg.norm(self.exact_moments - ref_moments) / np.sqrt(n_moments)
                print("noise: {:6.2g} error of natural cov: {:6.2g} natural moments: {:6.2g}".format(
                    noise, diff_norm, mom_err))
                assert mom_err/(noise + 1e-10) < 50

                result, distr_obj = self.make_approx(mlmc.tool.simple_distribution.SimpleDistribution, noise, moments_data,
                                                     tol=1e-5)
            else:
                # TODO:
                # Use SimpleDistribution only as soon as it use regularization that improve convergency even without
                # cov matrix. preconditioning.
                result, distr_obj = self.make_approx(mlmc.distribution.Distribution, noise, moments_data)
            distr_plot.add_distribution(distr_obj, label="noise {}".format(noise))
            results.append(result)

        #self.check_convergence(results)
        self.eigenvalues_plot.show(file = self.pdfname("_eigenvalues"))
        distr_plot.show(file=self.pdfname("_pdf_iexact"))
        distr_plot.reset()
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
    quantiles = np.array([0.01])
    #quantiles = np.array([0.01])
    conv = {}
    # Dict of result matricies (n_quantiles, n_moments) for every performed kind of test.
    for i_q, quantile in enumerate(quantiles):
        np.random.seed(1234)
        case = DistributionDomainCase(moments, distribution, quantile)
        tests = [case.exact_conv, case.inexact_conv]
        for test_fn in tests:
            name = test_fn.__name__
            test_results = test_fn()
            values = conv.setdefault(name, (case.title, []))
            values[1].append(test_results)

    for key, values in conv.items():
        title, results = values
        title = "{}_conv_{}".format(title, key)
        if results[0] is not None:
            mlmc.plot.plots.plot_convergence(quantiles, results, title=title)

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
#
#
# test_distributions()
