import numpy as np
import scipy.stats as st
import scipy.integrate as integrate
from mlmc.tool import plot
from mlmc.quantity_spec import ChunkSpec
import mlmc.quantity_estimate as qe
import mlmc.tool.simple_distribution


def estimate_n_samples_for_target_variance(target_variance, prescribe_vars, n_ops, n_levels):
    """
    Estimate optimal number of samples for individual levels that should provide a target variance of
    resulting moment estimate.
    This also set given moment functions to be used for further estimates if not specified otherwise.
    :param target_variance: Constrain to achieve this variance.
    :param prescribe_vars: vars[ L, M] for all levels L and moments_fn M safe the (zeroth) constant moment with zero variance.
    :param n_ops: number of operations at each level
    :param n_levels: number of levels
    :return: np.array with number of optimal samples for individual levels and moments_fn, array (LxR)
    """
    # print("vars ", prescribe_vars)
    # print("n ops ", n_ops)

    vars = prescribe_vars
    #print("vars.T * n_ops ", vars.T * n_ops)

    sqrt_var_n = np.sqrt(vars.T * n_ops)  # moments_fn in rows, levels in cols
    total = np.sum(sqrt_var_n, axis=1)  # sum over levels
    #print("total ", total)

    n_samples_estimate = np.round((sqrt_var_n / n_ops).T * total / target_variance).astype(int)  # moments_fn in cols
    # Limit maximal number of samples per level
    n_samples_estimate_safe = np.maximum(
        np.minimum(n_samples_estimate, vars * n_levels / target_variance), 2)

    #print("vars ", vars)


    # return np.max(n_samples_estimate, axis=1).astype(int)
    #
    # print("n samples estimate ", n_samples_estimate)
    # print("n samples estaimte safe ", n_samples_estimate_safe)
    #print("N: ", np.max(n_samples_estimate_safe, axis=1).astype(int))
    return np.max(n_samples_estimate_safe, axis=1).astype(int)


def calc_level_params(step_range, n_levels):
    assert step_range[0] > step_range[1]
    level_parameters = []
    for i_level in range(n_levels):
        if n_levels == 1:
            level_param = 1
        else:
            level_param = i_level / (n_levels - 1)
        level_parameters.append([step_range[0] ** (1 - level_param) * step_range[1] ** level_param])

    return level_parameters


def determine_sample_vec(n_collected_samples, n_levels, sample_vector=None):
    if sample_vector is None:
        sample_vector = n_collected_samples
    if len(sample_vector) > n_levels:
        sample_vector = sample_vector[:n_levels]
    return np.array(sample_vector)


class Estimate:
    def __init__(self, quantity, sample_storage, moments_fn=None):
        self._quantity = quantity
        self._sample_storage = sample_storage
        self._moments_fn = moments_fn

    @property
    def quantity(self):
        return self._quantity

    @quantity.setter
    def quantity(self, quantity):
        self._quantity = quantity

    @property
    def n_moments(self):
        return self._moments_fn.size

    def estimate_moments(self, moments_fn=None):
        """
        Use collected samples to estimate moments and variance of this estimate.
        :param moments_fn: moments function
        :return: estimate_of_moment_means, estimate_of_variance_of_estimate ; arrays of length n_moments
        """
        if moments_fn is None:
            moments_fn = self._moments_fn

        moments_mean = qe.estimate_mean(qe.moments(self._quantity, moments_fn))
        self.moments_mean = moments_mean
        return moments_mean.mean, moments_mean.var

    def estimate_covariance(self, moments_fn=None):
        """
        Use collected samples to estimate covariance matrix and variance of this estimate.
        :param moments_fn: moments function
        :return: estimate_of_moment_means, estimate_of_variance_of_estimate ; arrays of length n_moments
        """
        if moments_fn is None:
            moments_fn = self._moments_fn

        cov_mean = qe.estimate_mean(qe.covariance(self._quantity, moments_fn))
        return cov_mean.mean, cov_mean.var

    def estimate_diff_vars_regression(self, n_created_samples, moments_fn=None, raw_vars=None):
        """
        Estimate variances using linear regression model.
        Assumes increasing variance with moments_fn, use only two moments_fn with highest average variance.
        :param n_created_samples: number of created samples on each level
        :param moments_fn: Moment evaluation function
        :return: array of variances, n_ops_estimate
        """
        self._n_created_samples = n_created_samples
        # vars shape L x R
        if raw_vars is None:
            if moments_fn is None:
                moments_fn = self._moments_fn
            raw_vars, n_samples = self.estimate_diff_vars(moments_fn)

        sim_steps = np.squeeze(self._sample_storage.get_level_parameters())
        vars = self._all_moments_variance_regression(raw_vars, sim_steps)

        # We need to get n_ops_estimate from storage
        return vars, self._sample_storage.get_n_ops()

    def estimate_diff_vars(self, moments_fn=None):
        """
        Estimate moments_fn variance from samples
        :param moments_fn: Moment evaluation functions
        :return: (diff_variance, n_samples);
            diff_variance - shape LxR, variances of diffs of moments_fn
            n_samples -  shape L, num samples for individual levels.
        """
        if moments_fn is None:
            moments_fn = self._moments_fn
        moments_mean = qe.estimate_mean(qe.moments(self._quantity, moments_fn))
        return moments_mean.l_vars, moments_mean.n_samples

    def _all_moments_variance_regression(self, raw_vars, sim_steps):
        reg_vars = raw_vars.copy()
        n_moments = raw_vars.shape[1]
        for m in range(1, n_moments):
            reg_vars[:, m] = self._moment_variance_regression(raw_vars[:, m], sim_steps)
        assert np.allclose(reg_vars[:, 0], 0.0)
        return reg_vars

    def _moment_variance_regression(self, raw_vars, sim_steps):
        """
        Estimate level variance using separate model for every moment.

        log(var_l) = A + B * log(h_l) + C * log^2(hl),
                                            for l = 0, .. L-1
        :param raw_vars: moments_fn variances raws, shape (L,)
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

        K = 3  # number of parameters
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

    def _variance_of_variance(self, n_samples=None):
        """
        Approximate variance of log(X) where
        X is from ch-squared with df=n_samples - 1.
        Return array of variances for actual n_samples array.

        :param n_samples: Optional array with n_samples.
        :return: array of variances of variance estimate.
        """
        if n_samples is None:
            n_samples = self._n_created_samples
        if hasattr(self, "_saved_var_var"):
            ns, var_var = self._saved_var_var
            if np.sum(np.abs(np.array(ns) - np.array(n_samples))) == 0:
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

    def est_bootstrap(self, n_subsamples=100, sample_vector=None, moments_fn=None):

        if moments_fn is not None:
            self._moments_fn = moments_fn
        else:
            moments_fn = self._moments_fn

        sample_vector = determine_sample_vec(n_collected_samples=self._sample_storage.get_n_collected(),
                                             n_levels=self._sample_storage.get_n_levels(),
                                             sample_vector=sample_vector)
        bs_mean = []
        bs_var = []
        bs_l_means = []
        bs_l_vars = []
        for i in range(n_subsamples):
            quantity_subsample = self.quantity.select(self.quantity.subsample(sample_vec=sample_vector))
            moments_quantity = qe.moments(quantity_subsample, moments_fn=moments_fn, mom_at_bottom=False)
            q_mean = qe.estimate_mean(moments_quantity)

            bs_mean.append(q_mean.mean)
            bs_var.append(q_mean.var)
            bs_l_means.append(q_mean.l_means)
            bs_l_vars.append(q_mean.l_vars)

        self.mean_bs_mean = np.mean(bs_mean, axis=0)
        self.mean_bs_var = np.mean(bs_var, axis=0)
        self.mean_bs_l_means = np.mean(bs_l_means, axis=0)
        self.mean_bs_l_vars = np.mean(bs_l_vars, axis=0)

        self.var_bs_mean = np.var(bs_mean, axis=0, ddof=1)
        self.var_bs_var = np.var(bs_var, axis=0, ddof=1)
        self.var_bs_l_means = np.var(bs_l_means, axis=0, ddof=1)
        self.var_bs_l_vars = np.var(bs_l_vars, axis=0, ddof=1)

        self._bs_level_mean_variance = self.var_bs_l_means * np.array(self._sample_storage.get_n_collected())[:, None]

    def bs_target_var_n_estimated(self, target_var, sample_vec=None):
        sample_vec = determine_sample_vec(n_collected_samples=self._sample_storage.get_n_collected(),
                                          n_levels=self._sample_storage.get_n_levels(),
                                          sample_vector=sample_vec)

        self.est_bootstrap(n_subsamples=300, sample_vector=sample_vec)

        variances, n_ops = self.estimate_diff_vars_regression(sample_vec, raw_vars=self.mean_bs_l_vars)
        n_estimated = estimate_n_samples_for_target_variance(target_var, variances, n_ops,
                                                             n_levels=self._sample_storage.get_n_levels())

        return n_estimated

    def plot_variances(self, sample_vec=None):
        var_plot = plot.VarianceBreakdown(10)

        sample_vec = determine_sample_vec(n_collected_samples=self._sample_storage.get_n_collected(),
                                          n_levels=self._sample_storage.get_n_levels(),
                                          sample_vector=sample_vec)
        self.est_bootstrap(n_subsamples=100, sample_vector=sample_vec)

        var_plot.add_variances(self.mean_bs_l_vars, sample_vec, ref_level_vars=self._bs_level_mean_variance)
        var_plot.show(None)

    # def plot_level_variances(self):
    #     var_plot = plot.Variance(10)
    #     for mc in self.mlmc:
    #         steps, vars = mc.estimate_level_vars()
    #         var_plot.add_level_variances(steps, vars)
    #     var_plot.show()

    def plot_bs_var_log(self, sample_vec=None):
        sample_vec = determine_sample_vec(n_collected_samples=self._sample_storage.get_n_collected(),
                                          n_levels=self._sample_storage.get_n_levels(),
                                          sample_vector=sample_vec)

        moments_quantity = qe.moments(self._quantity, moments_fn=self._moments_fn, mom_at_bottom=False)
        q_mean = qe.estimate_mean(moments_quantity)

        bs_plot = plot.BSplots(bs_n_samples=sample_vec, n_samples=self._sample_storage.get_n_collected(),
                               n_moments=self._moments_fn.size, ref_level_var=q_mean.l_vars)

        bs_plot.plot_means_and_vars(self.mean_bs_mean[1:], self.mean_bs_var[1:], n_levels=self._sample_storage.get_n_levels())

        bs_plot.plot_bs_variances(self.mean_bs_l_vars)
        #bs_plot.plot_bs_var_log_var()

        bs_plot.plot_var_regression(self, self._sample_storage.get_n_levels(), self._moments_fn)

    def plot_var_compare(self, nl):
        self[nl].plot_bootstrap_variance_compare(self._moments_fn)

    def plot_var_var(self, nl):
        self[nl].plot_bootstrap_var_var(self._moments_fn)

    def fine_coarse_violinplot(self):
        import pandas as pd
        from mlmc.tool import violinplot

        label_n_spaces = 5
        n_levels = self._sample_storage.get_n_levels()
        for level_id in range(n_levels):
            samples = np.squeeze(self._quantity.samples(ChunkSpec(level_id=level_id)), axis=0)
            if level_id == 0:
                label = "{} F{} {} C".format(level_id, ' ' * label_n_spaces, level_id + 1)
                data = {'samples': samples[:, 0], 'type': 'fine', 'level': label}
                dframe = pd.DataFrame(data)
            else:

                data = {'samples': samples[:, 1], 'type': 'coarse', 'level': label}
                dframe = pd.concat([dframe, pd.DataFrame(data)], axis=0)

                if level_id + 1 < n_levels:
                    label = "{} F{} {} C".format(level_id, ' ' * label_n_spaces, level_id + 1)
                    data = {'samples': samples[:, 0], 'type': 'fine', 'level': label}
                    dframe = pd.concat([dframe, pd.DataFrame(data)], axis=0)
        violinplot.fine_coarse_violinplot(dframe)

    @staticmethod
    def estimate_domain(quantity, sample_storage, quantile=None):
        """
        Estimate moments domain from MLMC samples.
        :param quantity: mlmc.quantity.Quantity instance, represents the real quantity
        :param sample_storage: mlmc.sample_storage.SampleStorage instance, provides all the samples
        :param quantile: float in interval (0, 1), None means whole sample range
        :return: lower_bound, upper_bound
        """
        ranges = []
        if quantile is None:
            quantile = 0.01

        for level_id in range(sample_storage.get_n_levels()):
            fine_samples = quantity.samples(ChunkSpec(level_id=level_id,
                                                  n_samples=sample_storage.get_n_collected()[0]))[..., 0]

            fine_samples = np.squeeze(fine_samples)
            ranges.append(np.percentile(fine_samples, [100 * quantile, 100 * (1 - quantile)]))

        ranges = np.array(ranges)
        return np.min(ranges[:, 0]), np.max(ranges[:, 1])

    def construct_density(self, tol=1e-8, reg_param=0.0, orth_moments_tol=1e-4, exact_pdf=None):
        """
        Construct approximation of the density using given moment functions.
        """
        cov_mean = qe.estimate_mean(qe.covariance(self._quantity, self._moments_fn))
        cov_mat = cov_mean.mean
        moments_obj, info = mlmc.tool.simple_distribution.construct_ortogonal_moments(self._moments_fn,
                                                                                                     cov_mat,
                                                                                                     tol=orth_moments_tol)
        moments_mean = qe.estimate_mean(qe.moments(self._quantity, moments_obj))
        est_moments = moments_mean.mean
        est_vars = moments_mean.var

        # if exact_pdf is not None:
        #     exact_moments = mlmc.tool.simple_distribution.compute_exact_moments(moments_obj, exact_pdf)

        est_vars = np.ones(moments_obj.size)
        min_var, max_var = np.min(est_vars[1:]), np.max(est_vars[1:])
        print("min_err: {} max_err: {} ratio: {}".format(min_var, max_var, max_var / min_var))
        moments_data = np.stack((est_moments, est_vars), axis=1)
        distr_obj = mlmc.tool.simple_distribution.SimpleDistribution(moments_obj, moments_data,
                                                                     domain=moments_obj.domain)
        result = distr_obj.estimate_density_minimize(tol, reg_param)  # 0.95 two side quantile

        return distr_obj, info, result, moments_obj

    def get_level_samples(self, level_id):
        return self._quantity.samples(ChunkSpec(level_id=level_id,
                                                n_samples=self._sample_storage.get_n_collected()[level_id]))
