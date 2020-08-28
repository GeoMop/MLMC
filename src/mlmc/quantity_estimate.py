import numpy as np
import scipy.stats as st
import scipy.integrate as integrate


class QuantityEstimate:

    def __init__(self, sample_storage, moments_fn, sim_steps):
        """
        Quantity estimates
        :param sample_storage: SampleStorage instance
        :param moments_fn: moments function
        :param sim_steps: simulation steps on each level
        """
        self._sample_storage = sample_storage
        self._moments_fn = moments_fn
        self._sim_steps = [s_step[0] for s_step in sim_steps]

    @property
    def levels_results(self):
        new_level_results = QuantityEstimate.get_level_results(self._sample_storage)

        return new_level_results

    @staticmethod
    def get_level_results(sample_storage):
        """
        Get sample results split to levels
        :param sample_storage: Storage that provides the samples
        :return: level results, shape: (n_levels, )
        """
        level_results = sample_storage.sample_pairs()

        if len(level_results) == 0:
            raise Exception("No data")

        # @TODO: it does not works with arrays quantities, remove ASAP
        new_level_results = []

        for lev_res in level_results:
            if len(lev_res) == 0:
                continue

            if lev_res[0].shape[0] > 1:
                if isinstance(lev_res, np.ndarray):
                    new_level_results.append(lev_res[0])

        return new_level_results

    def estimate_diff_vars_regression(self, n_created_samples, moments_fn=None, raw_vars=None):
        """
        Estimate variances using linear regression model.
        Assumes increasing variance with moments, use only two moments with highest average variance.
        :param n_created_samples: number of created samples on each level
        :param moments_fn: Moment evaluation function
        :return: array of variances, n_ops_estimate
        """
        # @TODO: try to set it elsewhere
        self._n_created_samples = n_created_samples

        # vars shape L x R
        if raw_vars is None:
            if moments_fn is None:
                moments_fn = self._moments_fn
            raw_vars, n_samples = self.estimate_diff_vars(moments_fn)
        sim_steps = self._sim_steps
        vars = self._all_moments_variance_regression(raw_vars, sim_steps)

        # We need to get n_ops_estimate from storage
        return vars, self._sample_storage.get_n_ops()

    def estimate_diff_vars(self, moments_fn=None):
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

        for level, level_results in enumerate(self.levels_results):
            zero_level = True if level == 0 else False
            v, n = self.estimate_diff_var(moments_fn, level_results, zero_level)
            vars.append(v)
            n_samples.append(n)
        return np.array(vars), np.array(n_samples)

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

    def estimate_moments(self, moments_fn=None):
        """
        Use collected samples to estimate moments and variance of this estimate.
        :param moments_fn: Vector moment function, gives vector of moments for given sample or sample vector.
        :return: estimate_of_moment_means, estimate_of_variance_of_estimate ; arrays of length n_moments
        """
        if moments_fn is None:
            moments_fn = self._moments_fn

        means = []
        vars = []
        n_samples = []
        for level_id, level_result in enumerate(self.levels_results):
            zero_level = True if level_id == 0 else False
            means.append(self.estimate_diff_mean(moments_fn, level_result, zero_level))
            l_vars, ns = self.estimate_diff_var(moments_fn, level_result, zero_level)
            vars.append(l_vars)
            n_samples.append(ns)
        means = np.sum(np.array(means), axis=0)
        n_samples = np.array(n_samples, dtype=int)
        vars = np.sum(np.array(vars) / n_samples[:, None], axis=0)

        return np.array(means), np.array(vars)

    def estimate_diff_var(self, moments_fn, level_results, zero_level=False):
        """
        Estimate moments variance
        :param moments_fn: Moments evaluation function
        :return: tuple (variance vector, length of moments)
        """

        mom_fine, mom_coarse = self.evaluate_moments(moments_fn, level_results, zero_level)
        assert len(mom_fine) == len(mom_coarse)
        assert len(mom_fine) >= 2
        var_vec = np.var(mom_fine - mom_coarse, axis=0, ddof=1)
        ns = level_results.shape[1]
        return var_vec, ns

    def estimate_diff_mean(self, moments_fn, level_result, zero_level=False):
        """
        Estimate moments mean
        :param moments_fn: Function for calculating moments
        :return: np.array, moments mean vector
        """
        mom_fine, mom_coarse = self.evaluate_moments(moments_fn, level_result, zero_level)
        assert len(mom_fine) == len(mom_coarse)
        assert len(mom_fine) >= 1
        mean_vec = np.mean(mom_fine - mom_coarse, axis=0)
        return mean_vec

    def estimate_covariance(self, moments_fn, stable=False):
        """
        Estimate covariance matrix (non central).
        :param moments_fn: Moment functions object.
        :param stable: Use alternative formula with better numerical stability.
        :return: cov covariance matrix  with shape (n_moments, n_moments)
        """
        cov_mat = np.zeros((moments_fn.size, moments_fn.size))

        for level_id, level_result in enumerate(self.levels_results):
            zero_level = True if level_id == 0 else False
            mom_fine, mom_coarse = self.evaluate_moments(moments_fn, level_result, zero_level)
            n_samples = len(mom_fine)
            assert len(mom_fine) == len(mom_coarse)
            assert len(mom_fine) >= 2

            cov_fine = np.matmul(mom_fine.T, mom_fine)
            cov_coarse = np.matmul(mom_coarse.T, mom_coarse)
            cov_mat += (cov_fine - cov_coarse) / n_samples

        return cov_mat

    def evaluate_moments(self, moments_fn, level_results, is_zero_level=False):
        """
        Evaluate level difference for all samples and given moments.
        :param moments_fn: Moment evaluation object.
        :param level_results: sample data
        :param is_zero_level: bool
        :return: (fine, coarse) both of shape (n_samples, n_moments)
        """
        # Current moment functions are different from last moment functions
        samples = np.squeeze(level_results)

        # Moments from fine samples
        moments_fine = moments_fn(samples[:, 0])

        # For first level moments from coarse samples are zeroes
        if is_zero_level:
            moments_coarse = np.zeros((len(moments_fine), moments_fn.size))
        else:
            moments_coarse = moments_fn(samples[:, 1])
        # Set last moments function
        self._last_moments_fn = moments_fn
        # Moments from fine and coarse samples
        self.last_moments_eval = moments_fine, moments_coarse

        self._remove_outliers_moments()
        return self.last_moments_eval

    def _remove_outliers_moments(self):
        """
        Remove moments from outliers from fine and course moments
        :return: None
        """
        # Fine and coarse moments mask
        ok_fine = np.all(np.isfinite(self.last_moments_eval[0]), axis=1)
        ok_coarse = np.all(np.isfinite(self.last_moments_eval[1]), axis=1)

        # Common mask for coarse and fine
        ok_fine_coarse = np.logical_and(ok_fine, ok_coarse)

        # New moments without outliers
        self.last_moments_eval = self.last_moments_eval[0][ok_fine_coarse, :],\
                                 self.last_moments_eval[1][ok_fine_coarse, :]

    @staticmethod
    def estimate_domain(sample_storage, quantile=None):
        """
        Estimate moments domain from MLMC samples.
        :parameter sample_storage: Storage that provides the samples
        :parameter quantile: float in interval (0, 1), None means whole sample range
        :return: lower_bound, upper_bound
        """
        new_level_results = QuantityEstimate.get_level_results(sample_storage)

        ranges = []
        if quantile is None:
            quantile = 0.01

        for lev_res in new_level_results:
            fine_sample = lev_res[:, 0]
            ranges.append(np.percentile(fine_sample, [100 * quantile, 100 * (1 - quantile)]))

        ranges = np.array(ranges)
        return np.min(ranges[:, 0]), np.max(ranges[:, 1])
