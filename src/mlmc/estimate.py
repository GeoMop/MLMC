import mlmc
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

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

    def estimate_domain(self):
        return self.mlmc.estimate_domain()

    def set_moments_color_bar(self, ax):
        self._moments_cmap = create_color_bar(self.n_moments, "Moments", ax)


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

    def estimate_diff_vars(self, moments_fn):
        """
        Estimate moments variance from samples
        :param moments_fn: Moment evaluation functions
        :return: (diff_variance, n_samples);
            diff_variance - shape LxR, variances of diffs of moments
            n_samples -  shape L, num samples for individual levels.

            Returns simple variance for level 0.
        """
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

    def estimate_domain(self):
        """
        Estimate domain of the density function.
        TODO: compute mean and variance and use quantiles of normal or lognormal distribution (done in Distribution)
        :return:
        """
        ranges = np.array([l.sample_range() for l in self.levels])

        return np.min(ranges[:, 0]), np.max(ranges[:, 1])

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
        level_var_est, _ = self.mlmc.estimate_diff_vars(moments_fn)
        level_mean_est = self.mlmc.estimate_level_means(moments_fn)
        return level_mean_est, level_var_est

    def _bs_get_estimates_regression(self):
        moments_fn = self.moments
        #mean_est, var_est = self.mlmc.estimate_moments(moments_fn)
        level_var_est, _ = self.mlmc.estimate_diff_vars(moments_fn)
        level_mean_est = self.mlmc.estimate_level_means(moments_fn)
        level_var_est = self.mlmc.estimate_diff_vars_regression(moments_fn, level_var_est)
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





    def _scatter_level_moment_data(self, ax, values, i_moments=None, marker='o'):
        """
        Scatter plot of given table of data for moments and levels.
        X coordinate is given by level, and slight shift is applied to distinguish the moments.
        Moments are colored using self._moments_cmap.
        :param ax: Axis where to add the scatter.
        :param values: data to plot, array n_levels x len(i_moments)
        :param i_moments: Indices of moments to use, all moments grater then 0 are used.
        :param marker: Scatter marker to use.
        :return:
        """
        cmap = self._moments_cmap
        if i_moments is None:
            i_moments = range(1, self.n_moments)
        values = values[:, i_moments[:]]
        n_levels = values.shape[0]
        n_moments = values.shape[1]

        moments_x_step = 0.5/n_moments
        for m in range(n_moments):
            color = cmap(i_moments[m])
            X = np.arange(n_levels) + moments_x_step * m
            Y = values[:, m]
            col = np.ones(n_levels)[:, None] * np.array(color)[None, :]
            ax.scatter(X, Y, c=col, marker=marker, label="var, m=" + str(i_moments[m]))

    def plot_bootstrap_variance_compare(self):
        """
        Plot fraction (MLMC var est) / (BS var set) for the total variance and level variances.
        :param moments_fn:
        :return:
        """
        moments_fn = self.moments
        mean, var, l_mean, l_var = self._bs_get_estimates(moments_fn)
        l_var = l_var / self.n_samples[: , None]
        est_variances = np.concatenate((var[None, 1:], l_var[:, 1:]), axis=0)

        bs_var = self._bs_mean_variance
        bs_l_var = self._bs_level_mean_variance / self.n_samples[:, None]
        bs_variances = np.concatenate((bs_var[None, 1:], bs_l_var[:, 1:]), axis=0)

        fraction = est_variances / bs_variances

        fig = plt.figure(figsize=(30, 10))
        ax = fig.add_subplot(1, 1, 1)

        #self._scatter_level_moment_data(ax, bs_variances, marker='.')
        #self._scatter_level_moment_data(ax, est_variances, marker='d')
        self._scatter_level_moment_data(ax, fraction, marker='o')

        #ax.legend(loc=6)
        lbls = ['Total'] + [ 'L{:2d}'.format(l+1) for l in range(self.n_levels)]
        ax.set_xticks(ticks = np.arange(self.n_levels + 1))
        ax.set_xticklabels(lbls)
        ax.set_yscale('log')
        ax.set_ylim((0.3, 3))

        self.color_bar(moments_fn.size, 'moments')

        fig.savefig('bs_var_vs_var.pdf')
        plt.show()

    def plot_bs_variances(self, variances, y_label=None, log=True, y_lim=None):
        """
        Plot BS estimate of error of variances of other related quantities.
        :param variances: Data, shape: (n_levels + 1, n_moments).
        :return:
        """
        if y_lim is None:
            y_lim = (np.min(variances[:, 1:]), np.max(variances[:, 1:]))
        if y_label is None:
            y_label = "Error of variance estimates"

        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(1, 1, 1)
        self.set_moments_color_bar(ax)
        self._scatter_level_moment_data(ax, variances, marker='.')

        lbls = ['Total'] + ['L{:2d}\n{}\n{}'.format(l + 1, nsbs, ns)
                            for l, (nsbs, ns) in enumerate(zip(self._bs_n_samples, self.n_samples))]
        ax.set_xticks(ticks = np.arange(self.n_levels + 1))
        ax.set_xticklabels(lbls)
        if log:
            ax.set_yscale('log')
        ax.set_ylim(y_lim)
        ax.set_ylabel(y_label)

        fig.savefig('bs_var_var.pdf')
        plt.show()

    def plot_bs_var_error_contributions(self):
        """
        MSE of total variance and contribution of individual levels.
        """
        bs_var_var = self._bs_var_variance[:]
        bs_l_var_var = self._bs_level_var_variance[:, :]
        bs_l_var_var[:, 1:] /= self._bs_n_samples[:, None]**2

        bs_variances = np.concatenate((bs_var_var[None, :], bs_l_var_var[:, :]), axis=0)
        self.plot_bs_variances(bs_variances, log=True,
                               y_label="MSE of total variance and contributions from individual levels.",
                               )

    def plot_bs_level_variances_error(self):
        """
        Plot error of estimates of V_l. Scaled as V_l^2 / N_l
        """
        l_var = self._ref_level_var

        l_var_var_scale = l_var[:, 1:] ** 2 * 2 / (self._bs_n_samples[:, None] - 1)
        total_var_var_scale = np.sum(l_var_var_scale[:, :] / self._bs_n_samples[:, None]**2, axis=0 )

        bs_var_var = self._bs_var_variance[:]
        bs_var_var[1:] /= total_var_var_scale

        bs_l_var_var = self._bs_level_var_variance[:, :]
        bs_l_var_var[:, 1:] /= l_var_var_scale

        bs_variances = np.concatenate((bs_var_var[None, :], bs_l_var_var[:, :]), axis=0)
        self.plot_bs_variances(bs_variances, log=True,
                               y_label="MSE of level variances estimators scaled by $V_l^2/N_l$.")

    def plot_bs_var_log_var(self):
        """
        Test that  MSE of log V_l scales as variance of log chi^2_{N-1}, that is approx. 2 / (n_samples-1).
        """
        #vv = 1/ self.mlmc._variance_of_variance(self._bs_n_samples)
        vv = self._bs_n_samples
        bs_l_var_var = np.sqrt((self._bs_level_var_variance[:, :]) * vv[:, None])
        bs_var_var = self._bs_var_variance[:]  # - np.log(total_var_var_scale)
        bs_variances = np.concatenate((bs_var_var[None, :], bs_l_var_var[:, :]), axis=0)
        self.plot_bs_variances(bs_variances, log=True,
                               y_label="BS est. of var. of $\hat V^r$, $\hat V^r_l$ estimators.",
                               )#y_lim=(0.1, 20))

    # def plot_bs_var_reg_var(self):
    #     """
    #     Test that  MSE of log V_l scales as variance of log chi^2_{N-1}, that is approx. 2 / (n_samples-1).
    #     """
    #     vv = self.mlmc._variance_of_variance(self._bs_n_samples)
    #     bs_l_var_var = (self._bs_level_var_variance[:, :]) / vv[:, None]
    #     bs_var_var = self._bs_var_variance[:]  # - np.log(total_var_var_scale)
    #     bs_variances = np.concatenate((bs_var_var[None, :], bs_l_var_var[:, :]), axis=0)
    #     self.plot_bs_variances(bs_variances, log=True,
    #                            y_label="BS est. of var. of $\hat V^r$, $\hat V^r_l$ estimators.",
    #                            y_lim=(0.1, 20))


    def plot_means_and_vars(self, moments_mean, moments_var, n_levels, exact_moments):
        """
        Plot means with variance whiskers to given axes.
        :param moments_mean: array, moments mean
        :param moments_var: array, moments variance
        :param n_levels: array, number of levels
        :param exact_moments: array, moments from distribution
        :param ex_moments: array, moments from distribution samples
        :return:
        """
        colors = iter(plt.cm.rainbow(np.linspace(0, 1, len(moments_mean) + 1)))
        # print("moments mean ", moments_mean)
        # print("exact momentss ", exact_moments)

        x = np.arange(0, len(moments_mean[0]))
        x = x - 0.3
        default_x = x

        for index, means in enumerate(moments_mean):
            if index == int(len(moments_mean) / 2) and exact_moments is not None:
                plt.plot(default_x, exact_moments, 'ro', label="Exact moments")
            else:
                x = x + (1 / (len(moments_mean) * 1.5))
                plt.errorbar(x, means, yerr=moments_var[index], fmt='o', capsize=3, color=next(colors),
                             label = "%dLMC" % n_levels[index])
        if ex_moments is not None:
                plt.plot(default_x - 0.125, ex_moments, 'ko', label="Exact moments")
        plt.legend()
        #plt.show()
        #exit()








    def plot_var_regression(self, i_moments = None):
        """
        Plot total and level variances and their regression and errors of regression.
        :param i_moments: List of moment indices to plot. If it is an int M, the range(M) is used.
                       If None, self.moments.size is used.
        """
        moments_fn = self.moments

        fig = plt.figure(figsize=(30, 10))
        ax = fig.add_subplot(1, 2, 1)
        ax_err = fig.add_subplot(1, 2, 2)

        if i_moments is None:
            i_moments = moments_fn.size
        if type(i_moments) is int:
            i_moments = list(range(i_moments))
        i_moments = np.array(i_moments, dtype=int)

        self.set_moments_color_bar(ax=ax)

        est_diff_vars, n_samples = self.mlmc.estimate_diff_vars(moments_fn)
        reg_diff_vars = self.mlmc.estimate_diff_vars_regression(moments_fn) #/ self.n_samples[:, None]
        ref_diff_vars = self._ref_level_var #/ self.n_samples[:, None]

        self._scatter_level_moment_data(ax,  ref_diff_vars, i_moments, marker='o')
        self._scatter_level_moment_data(ax, est_diff_vars, i_moments, marker='d')
        # add regression curves
        moments_x_step = 0.5 / self.n_moments
        for m in i_moments:
            color = self._moments_cmap(m)
            X = np.arange(self.n_levels) + moments_x_step * m
            Y = reg_diff_vars[1:, m]
            ax.plot(X[1:], Y, c=color)
            ax_err.plot(X[:], reg_diff_vars[:, m]/ref_diff_vars[:,m], c=color)

        ax.set_yscale('log')
        ax.set_ylabel("level variance $V_l$")
        ax.set_xlabel("step h_l")

        ax_err.set_yscale('log')
        ax_err.set_ylabel("regresion var. / reference var.")

        #ax.legend(loc=2)
        fig.savefig('level_vars_regression.pdf')
        plt.show()




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
            l, u = mc.estimate_domain()
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
        self.domain = self.mlmc[i_mlmc].estimate_domain()

    def plot_means(self, moments_fn):
        pass

    def construct_densities(self, tol=1.95, reg_param=0.01):
        for mc in self.mlmc:
            mc.construct_density(tol, reg_param)



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

        distr_plot.show(save='compare_distributions.pdf')

    def ref_estimates_bootstrap(self, n_samples, sample_vector=None):
        for mc in self.mlmc:
            mc.ref_estimates_bootstrap(self.moments, n_subsamples=n_samples, sample_vector=sample_vector)

    def plot_var_compare(self, nl):
        self[nl].plot_bootstrap_variance_compare(self.moments)

    def plot_var_var(self, nl):
        self[nl].plot_bootstrap_var_var(self.moments)
