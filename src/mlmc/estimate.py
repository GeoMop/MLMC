import mlmc
import numpy as np
import scipy.stats as st
import scipy.integrate as integrate
from mlmc import simple_distribution
from mlmc import plot





def compute_results(mlmc_l0, n_moments, mlmc_wrapper):
    """
    Compute density and moments domains
    TODO: remove completely or move into test_distribution
    :param mlmc_l0: One level Monte-Carlo method
    :param n_moments: int, Number of moments
    :param mc_wrapper: Object with mlmc instance, must contains also moments function object
    :return: domain - tuple, domain from 1LMC
             est_domain - tuple, domain estimated by mlmc instance
             mc_wrapper - with current distr_obj (object estimating distribution)
    """
    mlmc = mlmc_wrapper.mc
    moments_fn = mlmc_wrapper.moments_fn
    domain = mlmc_l0.ref_domain
    est_domain = mlmc.estimate_domain()

    t_var = 1e-5
    ref_diff_vars, _ = mlmc.estimate_diff_vars(moments_fn)
    # ref_moments, ref_vars = mc.estimate_moments(moments_fn)
    # ref_std = np.sqrt(ref_vars)
    # ref_diff_vars_max = np.max(ref_diff_vars, axis=1)
    # ref_n_samples = mc.set_target_variance(t_var, prescribe_vars=ref_diff_vars)
    # ref_n_samples = np.max(ref_n_samples, axis=1)
    # ref_cost = mc.estimate_cost(n_samples=ref_n_samples)
    # ref_total_std = np.sqrt(np.sum(ref_diff_vars / ref_n_samples[:, None]) / n_moments)
    # ref_total_std_x = np.sqrt(np.mean(ref_vars))

    est_moments, est_vars = mlmc.estimate_moments(moments_fn)

    # def describe(arr):
    #     print("arr ", arr)
    #     q1, q3 = np.percentile(arr, [25, 75])
    #     print("q1 ", q1)
    #     print("q2 ", q3)
    #     return "{:f8.2} < {:f8.2} | {:f8.2} | {:f8.2} < {:f8.2}".format(
    #         np.min(arr), q1, np.mean(arr), q3, np.max(arr))

    moments_data = np.stack((est_moments, est_vars), axis=1)

    distr_obj = simple_distribution.SimpleDistribution(moments_fn, moments_data)
    distr_obj.domain = domain
    distr_obj.estimate_density_minimize(1)
    mlmc_wrapper.distr_obj = distr_obj

    return domain, est_domain, mlmc_wrapper



class Estimate:
    """
    Base of future class dedicated to all kind of processing of the collected samples
    MLMC should only collect the samples.

    TODO: try to move plotting methods into separate file, allowing independent usage of the plots for
    explicitely provided datasets.
    """


    def _variance_of_variance(self, n_samples = None):
        """
        Approximate variance of log(X) where
        X is from ch-squared with df=n_samples - 1.
        Return array of variances for actual n_samples array.

        :param n_samples: Optional array with n_samples.
        :return: array of variances of variance estimate.
        """
        if n_samples is None:
            n_samples = self.n_samples
        if hasattr(self, "_saved_var_var"):
            ns, var_var = self._saved_var_var
            if np.sum(np.abs(ns - n_samples)) == 0:
                return var_var

        vars = []
        for ns in n_samples:
            df = ns - 1

            def log_chi_pdf(x):
                return np.exp(x) * df * st.chi2.pdf(np.exp(x) * df, df=df)

            def compute_moment(moment):
                std_est = np.sqrt(2 / df)
                fn = lambda x, m=moment: x ** m * log_chi_pdf(x)
                return integrate.quad(fn, -100 * std_est, 100 * std_est)[0]

            mean = compute_moment(1)
            second = compute_moment(2)
            vars.append(second - mean ** 2)

        self._saved_var_var = (n_samples, np.array(vars))
        return np.array(vars)


    @classmethod
    def estimate_domain(cls, mlmc, quantile=None):
        """
        Estimate density domain from MLMC samples.
        :parameter mlmc: MLMC object that provides the samples
        :parameter quantile: float in interval (0, 1), None means whole sample range.
        :return: lower_bound, upper_bound
        """
        ranges = np.array([l.sample_domain(quantile) for l in mlmc.levels])
        return np.min(ranges[:, 0]), np.max(ranges[:, 1])



    def __init__(self, mlmc, moments=None):
        self.mlmc = mlmc
        self.moments = moments

        # Distribution aproximation, created by method 'construct_density'
        self._distribution = None

        # Bootstrap estimates of variances of MLMC estimators.
        # Created by method 'ref_estimates_bootstrap'.
        # BS estimate of variance of MLMC mean estimate. For every moment.
        self._bs_mean_variance = None
        # BS estimate of variance of MLMC variance estimate. For every moment.
        self._bs_var_variance = None
        # BS estimate of variance of MLMC level mean estimate. For every level, every moment,
        self._bs_level_mean_variance = None
        # BS estimate of variance of MLMC level variance estimate. For every level, every moment,
        self._bs_level_var_variance = None

    @property
    def n_moments(self):
        return self.moments.size

    @property
    def n_levels(self):
        return self.mlmc.n_levels

    @property
    def n_samples(self):
        return self.mlmc.n_samples

    @property
    def levels(self):
        return self.mlmc.levels

    @property
    def distribution(self):
        assert self._distribution is not None, "Need to call construct_density before."
        return self._distribution

    @property
    def sim_steps(self):
        return self.mlmc.sim_steps

    def approx_pdf(self, x):
        return self.distribution.density(x)

    def approx_cdf(self, x):
        return self.distribution.cdf(x)



    def estimate_level_vars(self, moments_fn=None):
        """
        Estimate variances for moments of X approximations on individual levels.
        i.t. Var \phi_r( X^l ).
        :param moments_fn:
        :return:
        """
        if moments_fn is None:
            moments_fn = self.moments
        sim_steps = self.sim_steps
        #n_samples = mlmc.n_samples
        vars = []
        steps = []
        for il, level in enumerate(self.levels):
            var_coarse, var_fine = level.estimate_level_var(moments_fn)
            if il > 0:
                vars.append(var_coarse)
                steps.append(sim_steps[il-1])
            vars.append(var_fine)
            steps.append(sim_steps[il])
        return np.array(steps), np.array(vars)

    def estimate_diff_vars_regression(self, moments_fn=None, raw_vars=None):
        """
        Estimate variances using linear regression model.
        Assumes increasing variance with moments, use only two moments with highest average variance.
        :param moments_fn: Moment evaluation function
        :return: array of variances, shape  L
        """
        # vars shape L x R
        if raw_vars is None:
            assert moments_fn is not None
            raw_vars, n_samples = self.estimate_diff_vars(moments_fn)
        sim_steps = self.sim_steps
        #vars = self._varinace_regression(raw_vars, sim_steps)
        vars = self._all_moments_varinace_regression(raw_vars, sim_steps)
        return vars


    def _variance_of_variance(self, n_samples = None):
        """
        Approximate variance of log(X) where
        X is from ch-squared with df=n_samples - 1.
        Return array of variances for actual n_samples array.

        :param n_samples: Optional array with n_samples.
        :return: array of variances of variance estimate.
        """
        if n_samples is None:
            n_samples = self.n_samples
        if hasattr(self, "_saved_var_var"):
            ns, var_var = self._saved_var_var
            if np.sum(np.abs(ns - n_samples)) == 0:
                return var_var

        vars = []
        for ns in n_samples:
            df = ns - 1

            def log_chi_pdf(x):
                return np.exp(x) * df * st.chi2.pdf(np.exp(x) * df, df=df)

            def compute_moment(moment):
                std_est = np.sqrt(2 / df)
                fn = lambda x, m=moment: x ** m * log_chi_pdf(x)
                return integrate.quad(fn, -100 * std_est, 100 * std_est)[0]

            mean = compute_moment(1)
            second = compute_moment(2)
            vars.append(second - mean ** 2)

        self._saved_var_var = (n_samples, np.array(vars))
        return np.array(vars)

    def _moment_varinace_regression(self, raw_vars, sim_steps):
        """
        Estimate level variance using separate model for every moment.

        log(var_l) = A + B * log(h_l) + C * log^2(hl),
                                            for l = 0, .. L-1
        :param raw_vars: moments variances raws, shape (L,)
        :param sim_steps: simulation steps, shape (L,)
        :return: np.array  (L, )
        """
        L, = raw_vars.shape
        L1 = L - 1
        if L < 3:
            return raw_vars

        # estimate of variances of variances, compute scaling
        W = 1.0 / np.sqrt(self._variance_of_variance())
        W = W[1:]   # ignore level 0
        W = np.ones((L - 1,))

        # Use linear regresion to improve estimate of variances V1, ...
        # model log var_{r,l} = a_r  + b * log step_l
        # X_(r,l), j = dirac_{r,j}

        K = 3 # number of parameters

        X = np.zeros((L1, K))
        log_step = np.log(sim_steps[1:])
        X[:, 0] = np.ones(L1)
        X[:, 1] = np.full(L1, log_step)
        X[:, 2] = np.full(L1, log_step ** 2)


        WX = X * W[:, None]    # scale

        log_vars = np.log(raw_vars[1:])     # omit first variance
        log_vars = W * log_vars       # scale RHS

        params, res, rank, sing_vals = np.linalg.lstsq(WX, log_vars)
        new_vars = raw_vars.copy()
        new_vars[1:] = np.exp(np.dot(X, params))
        return new_vars

    def _all_moments_varinace_regression(self, raw_vars, sim_steps):
        reg_vars = raw_vars.copy()
        n_moments = raw_vars.shape[1]
        for m in range(1, n_moments):
            reg_vars[:, m] = self._moment_varinace_regression(raw_vars[:, m], sim_steps)
        assert np.allclose( reg_vars[:, 0], 0.0)
        return reg_vars

    def estimate_diff_vars(self, moments_fn=None):
        """
        Estimate moments variance from samples
        :param moments_fn: Moment evaluation functions
        :return: (diff_variance, n_samples);
            diff_variance - shape LxR, variances of diffs of moments
            n_samples -  shape L, num samples for individual levels.

            Returns simple variance for level 0.
        """
        if moments_fn is None:
            moments_fn = self.moments
        vars = []
        n_samples = []

        for level in self.levels:
            v, n = level.estimate_diff_var(moments_fn)
            vars.append(v)
            n_samples.append(n)
        return np.array(vars), np.array(n_samples)

    def estimate_level_means(self, moments_fn):
        """
        Estimate means on individual levels.
        :param moments_fn: moments object of size R
        :return: shape (L, R)
        """
        means = []
        for level in self.mlmc.levels:
            means.append(level.estimate_diff_mean(moments_fn))
        return np.array(means)

    def estimate_n_samples_for_target_variance(self, target_variance, moments_fn=None, prescribe_vars=None):
        """
        Estimate optimal number of samples for individual levels that should provide a target variance of
        resulting moment estimate. Number of samples are directly set to levels.
        This also set given moment functions to be used for further estimates if not specified otherwise.
        TODO: separate target_variance per moment
        :param target_variance: Constrain to achieve this variance.
        :param moments_fn: moment evaluation functions
        :param prescribe_vars: vars[ L, M] for all levels L and moments M safe the (zeroth) constant moment with zero variance.
        :return: np.array with number of optimal samples for individual levels and moments, array (LxR)
        """
        if moments_fn is None:
            moments_fn = self.moments
        if prescribe_vars is None:
            vars = self.estimate_diff_vars_regression(moments_fn)
        else:
            vars = prescribe_vars

        n_ops = np.array([lvl.n_ops_estimate for lvl in self.levels])

        sqrt_var_n = np.sqrt(vars.T * n_ops)  # moments in rows, levels in cols
        total = np.sum(sqrt_var_n, axis=1)  # sum over levels
        n_samples_estimate = np.round((sqrt_var_n / n_ops).T * total / target_variance).astype(int)  # moments in cols

        # Limit maximal number of samples per level
        n_samples_estimate_safe = np.maximum(np.minimum(n_samples_estimate, vars * self.n_levels / target_variance), 2)
        n_samples = np.max(n_samples_estimate_safe, axis=1).astype(int)

        return n_samples


    def estimate_moments(self, moments_fn):
        """
        Use collected samples to estimate moments and variance of this estimate.
        :param moments_fn: Vector moment function, gives vector of moments for given sample or sample vector.
        :return: estimate_of_moment_means, estimate_of_variance_of_estimate ; arrays of length n_moments
        """
        means = []
        vars = []
        n_samples = []
        for level in self.levels:
            means.append(level.estimate_diff_mean(moments_fn))
            l_vars, ns = level.estimate_diff_var(moments_fn)
            vars.append(l_vars)
            n_samples.append(ns)
        means = np.sum(np.array(means), axis=0)
        n_samples = np.array(n_samples, dtype=int)

        vars = np.sum(np.array(vars) / n_samples[:, None], axis=0)

        return np.array(means), np.array(vars)

    def estimate_level_cost(self):
        """
        For every level estimate of cost of evaluation of a single coarse-fine simulation pair.
        TODO: Estimate simulation cost from collected times + regression similar to variance
        :return:
        """
        return np.array([lvl.n_ops_estimate for lvl in self.mlmc.levels])

    def estimate_cost(self, level_times=None, n_samples=None):
        """
        Estimate total cost of mlmc
        :param level_times: Cost estimate for single simulation for every level.
        :param n_samples: Number of samples on each level
        :return: total cost
        """
        if level_times is None:
            level_times = self.estimate_level_cost()
        if n_samples is None:
            n_samples = self.mlmc.n_samples
        return np.sum(level_times * n_samples)




    def estimate_covariance(self, moments_fn, levels, stable=False, mse=False):
        """
        MLMC estimate of covariance matrix of moments.
        :param stable: use formula with better numerical stability
        :param mse: Mean squared error??
        :return:
        """
        cov_mat = np.zeros((moments_fn.size, moments_fn.size))

        for level in levels:
            cov_mat += level.estimate_covariance(moments_fn, stable)
        if mse:
            mse_diag = np.zeros(moments_fn.size)
            for level in levels:
                mse_diag += level.estimate_cov_diag_err(moments_fn)/level.n_samples
            return cov_mat, mse_diag
        else:
            return cov_mat

    def construct_density(self, tol=1.95, reg_param=0.01):
        """
        Construct approximation of the density using given moment functions.
        Args:
            moments_fn: Moments object, determines also domain and n_moments.
            tol: Tolerance of the fitting problem, with account for variances in moments.
                 Default value 1.95 corresponds to the two tail confidency 0.95.
            reg_param: Regularization parameter.
        """
        cov = self.estimate_covariance(self.moments, self.mlmc.levels)
        moments_obj, info = simple_distribution.construct_ortogonal_moments(self.moments, cov, tol=0.0001)
        print("n levels: ", self.n_levels, "size: ", moments_obj.size)
        est_moments, est_vars = self.estimate_moments(moments_obj)
        #est_moments = np.zeros(moments_obj.size)
        #est_moments[0] = 1.0
        est_vars = np.ones(moments_obj.size)
        min_var, max_var = np.min(est_vars[1:]), np.max(est_vars[1:])
        print("min_err: {} max_err: {} ratio: {}".format(min_var, max_var, max_var / min_var))
        moments_data = np.stack((est_moments, est_vars), axis=1)
        distr_obj = simple_distribution.SimpleDistribution(moments_obj, moments_data, domain=moments_obj.domain)
        distr_obj.estimate_density_minimize(tol, reg_param)  # 0.95 two side quantile
        self._distribution = distr_obj


    def _bs_get_estimates(self):
        moments_fn = self.moments
        #mean_est, var_est = self.mlmc.estimate_moments(moments_fn)
        level_var_est, _ = self.estimate_diff_vars(moments_fn)
        level_mean_est = self.estimate_level_means(moments_fn)
        return level_mean_est, level_var_est

    def _bs_get_estimates_regression(self):
        moments_fn = self.moments
        #mean_est, var_est = self.mlmc.estimate_moments(moments_fn)
        level_var_est, _ = self.estimate_diff_vars(moments_fn)
        level_mean_est = self.estimate_level_means(moments_fn)
        level_var_est = self.estimate_diff_vars_regression(moments_fn, level_var_est)
        #var_est = np.sum(level_var_est[:, :]/self.n_samples[:,  None], axis=0)
        return level_mean_est, level_var_est

    def check_bias(self, a, b, var, label):
        diff = np.abs(a - b)
        tol = 2*np.sqrt(var) + 1e-20
        if np.any( diff > tol):
            print("Bias ", label)
            it = np.nditer(diff, flags=['multi_index'])
            while not it.finished:
                midx = it.multi_index
                sign = "<" if diff[midx] < tol[midx] else ">"
                print("{:6} {:8.2g} {} {:8.2g} {:8.3g} {:8.3g}".format(
                    str(midx), diff[midx], sign, tol[midx], a[midx], b[midx]))
                it.iternext()

    def ref_estimates_bootstrap(self, n_subsamples=100, sample_vector=None, regression=False, log=False):
        """
        Use current MLMC sample_vector to compute reference estimates for: mean, var, level_means, leval_vars.

        Estimate error of these MLMC estimators using a bootstrapping (with replacements).
        Bootstrapping samples MLMC estimators using provided 'sample_vector'.
        Reference estimates are used as mean values of the estimators.
        :param n_subsamples: Number of subsamples to perform. Default is 1000. This should guarantee at least
                             first digit to be correct.
        :param sample_vector: By default same as the original sampling.
        :return: None. Set reference and BS estimates.
        """
        moments_fn = self.moments
        if sample_vector is None:
            sample_vector = self.mlmc.n_samples
        if len(sample_vector) > self.n_levels:
            sample_vector = sample_vector[:self.n_levels]
        sample_vector = np.array(sample_vector)

        level_estimate_fn = self._bs_get_estimates
        if regression:
            level_estimate_fn = self._bs_get_estimates_regression

        def _estimate_fn():
            lm, lv =level_estimate_fn()
            return (np.sum(lm, axis=0), np.sum(lv[:, :] / sample_vector[:, None], axis=0), lm, lv)
        if log:
            def estimate_fn():
                (m, v, lm, lv) = _estimate_fn()
                return (m, np.log(np.maximum(v, 1e-10)), lm, np.log(np.maximum(lv, 1e-10)))
        else:
            estimate_fn = _estimate_fn

        self.mlmc.update_moments(moments_fn)
        estimates = estimate_fn()
        est_samples = [ np.zeros(n_subsamples) * est[..., None]  for est in estimates]
        for i in range(n_subsamples):
            self.mlmc.subsample(sample_vector)
            sub_estimates = estimate_fn()
            for es, se in zip(est_samples, sub_estimates):
                es[..., i] = se

        bs_mean_est = [np.mean(est, axis=-1) for est in est_samples]
        bs_err_est = [np.var(est, axis=-1, ddof=1) for est in est_samples]
        mvar, vvar, lmvar, lvvar = bs_err_est
        m, v, lm, lv = estimates

        self._ref_mean = m
        self._ref_var = v
        self._ref_level_mean = lm
        self._ref_level_var = lv

        self._bs_n_samples = self.n_samples
        self._bs_mean_variance = mvar
        self._bs_var_variance = vvar

        # Var dX_l =  n * Var[ mean dX_l ] = n * (1 / n^2) * n * Var dX_l
        self._bs_level_mean_variance = lmvar * self.n_samples[:, None]
        self._bs_level_var_variance = lvvar

        # Check bias
        mmean, vmean, lmmean, lvmean = bs_mean_est
        self.check_bias(m,   mmean, mvar,   "Mean")
        self.check_bias(v,   vmean, vvar,   "Variance")
        self.check_bias(lm, lmmean, lmvar,  "Level Mean")
        self.check_bias(lv, lvmean, lvvar,  "Level Varaince")

        self.mlmc.clean_subsamples()









class CompareLevels:
    """
    Class to compare MLMC for different number of levels.
    """

    def __init__(self, mlmc_list, **kwargs):
        """
        Args:
            List of MLMC instances with collected data.
        """
        self._mlmc_list = mlmc_list
        # Directory for plots.
        self.output_dir = kwargs.get('output_dir', "")
        # Optional quantity name used in plots
        self.quantity_name = kwargs.get('quantity_name', 'X')

        self.reinit(**kwargs)

    def reinit(self, **kwargs):
        """
        Re-create new Estimate objects from same original MLMC list.
        Set new parameters in particular for moments.
        :return:
        """
        # Default moments, type, number.
        self.log_scale = kwargs.get('log_scale', False)
        # Name of Moments class to use.
        self.moment_class = kwargs.get('moment_class', mlmc.moments.Legendre)
        # Number of moments.
        self.n_moments = kwargs.get('n_moments', 21)

        # Set domain to union of domains  of all mlmc:
        self.domain = kwargs.get('domain', self.common_domain())

        self._moments = self.moment_class(self.n_moments, self.domain, self.log_scale)

        self.mlmc = [Estimate(mc, self._moments) for mc in self._mlmc_list]
        self.mlmc_dict = {mc.n_levels: mc for mc in self.mlmc}
        self._moments_params = None

    def common_domain(self):
        L = +np.inf
        U = -np.inf
        for mc in self._mlmc_list:
            l, u = Estimate.estimate_domain(mc)
            L = min(l, L)
            U = max(u, U)
        return (L, U)

    def __getitem__(self, n_levels):
        return self.mlmc_dict[n_levels]

    @property
    def moments(self):
        return self._moments

    @property
    def moments_uncut(self):
        return self.moment_class(self.n_moments, self.domain, log=self.log_scale, safe_eval=False)

    def collected_report(self):
        """
        Print a record about existing levels, their collected samples, etc.
        """

        print("\n#Levels |     N collected samples")
        print("\n        |     Average sample time")
        for mlmc in self.mlmc:
            samples_tabs = ["{:8}".format(n) for n in mlmc.n_samples]
            times_tabs = ["{:8.2f}s".format(t) for t in mlmc.mlmc.get_sample_times()]
            print("{:7} | {}".format(mlmc.n_levels, " ".join(samples_tabs)))
            print("{:7} | {}".format(mlmc.n_levels, " ".join(times_tabs)))
        print("\n")


    def set_common_domain(self, i_mlmc, domain=None):
        if domain is not None:
            self.domain = domain
        self.domain = Estimate.estimate_domain(self._mlmc_list[i_mlmc])

    def plot_means(self, moments_fn):
        pass

    def construct_densities(self, tol=1.95, reg_param=0.01):
        for mc_est in self.mlmc:
            mc_est.construct_density(tol, reg_param)



    def plot_densities(self, i_sample_mlmc=0):
        """
        Plot constructed densities (see construct densities)
        Args:
            i_sample_mlmc: Index of MLMC used to construct histogram and ecdf.

        Returns:
        """
        distr_plot = plot.Distribution(title="Approx. density", quantity_name=self.quantity_name, legend_title="Number of levels",
                 log_density=False, cdf_plot=True, log_x=True, error_plot='kl')


        if i_sample_mlmc is not None:
            mc0_samples = self.mlmc[i_sample_mlmc].levels[0].sample_values[:, 0]
            distr_plot.add_raw_samples(mc0_samples)

        for mc in self.mlmc:
            if mc._distribution is None:
                continue
            label = "L = {}".format(mc.n_levels)
            distr_plot.add_distribution(mc._distribution, label=label)

        distr_plot.show('compare_distributions.pdf')

    def plot_variances(self):
        var_plot = plot.VarianceBreakdown()
        for mc in self.mlmc:
            #sample_vec = [5000, 5000, 1700, 600, 210, 72, 25, 9, 3]
            sample_vec = mc.estimate_n_samples_for_target_variance(0.0001)
            print("L", mc.n_levels, sample_vec)

            mc.ref_estimates_bootstrap(300, sample_vector=sample_vec)
            #sample_vec = [10000, 10000, 3000, 1200, 400, 140, 50, 18, 6]
            mc.mlmc.subsample(sample_vec)
            mc.mlmc.update_moments(self.moments)

            vars, n_samples = mc.estimate_diff_vars()
            var_plot.add_variances(vars, n_samples, ref_level_vars=mc._bs_level_mean_variance)
        var_plot.show()

    def plot_level_variances(self):
        var_plot = plot.Variance()
        moments = np.array([1, 2, 3, 4, 6, 8, 12, 20, 28, 44])
        moments = moments[moments < self.n_moments]
        for mc in self.mlmc:
            steps, vars = mc.estimate_level_vars()
            var_plot.add_level_variances(steps, vars[:, moments], mc.n_levels)
        var_plot.show()

    def ref_estimates_bootstrap(self, n_samples, sample_vector=None):
        for mc in self.mlmc:
            mc.ref_estimates_bootstrap(self.moments, n_subsamples=n_samples, sample_vector=sample_vector)

    def plot_var_compare(self, nl):
        self[nl].plot_bootstrap_variance_compare(self.moments)

    def plot_var_var(self, nl):
        self[nl].plot_bootstrap_var_var(self.moments)
