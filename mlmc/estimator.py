import numpy as np
import scipy.stats as st
import scipy.integrate as integrate
import mlmc.quantity.quantity_estimate as qe
import mlmc.tool.simple_distribution
from mlmc.quantity.quantity_estimate import mask_nan_samples
from mlmc.quantity.quantity_types import ScalarType
from mlmc.plot import plots
from mlmc.quantity.quantity_spec import ChunkSpec


class Estimate:
    """
    A wrapper class for moment estimation, PDF approximation, and related MLMC post-processing.

    Provides utility methods to:
      - Estimate statistical moments, variances, and covariances
      - Perform regression-based variance estimation
      - Conduct bootstrap resampling
      - Construct approximate probability density functions
      - Visualize and analyze MLMC variance and sample distributions
    """

    def __init__(self, quantity, sample_storage, moments_fn=None):
        """
        Initialize the Estimate instance.

        :param quantity: mlmc.quantity.Quantity
            Quantity object representing the stochastic quantity of interest.
        :param sample_storage: mlmc.sample_storage.SampleStorage
            Storage containing MLMC samples for each level.
        :param moments_fn: callable, optional
            Function defining the statistical moments to be estimated.
        """
        self._quantity = quantity
        self._sample_storage = sample_storage
        self._moments_fn = moments_fn
        self._moments_mean = None

    @property
    def quantity(self):
        """Return the current Quantity object."""
        return self._quantity

    @quantity.setter
    def quantity(self, quantity):
        """Set a new Quantity object."""
        self._quantity = quantity

    @property
    def n_moments(self):
        """Return the number of moment functions defined."""
        return self._moments_fn.size

    @property
    def moments_mean_obj(self):
        """Return the most recently computed mean of the moments."""
        return self._moments_mean

    @moments_mean_obj.setter
    def moments_mean_obj(self, moments_mean):
        """
        Set the estimated mean of the moments.

        :param moments_mean: mlmc.quantity.quantity.QuantityMean
            Object containing mean and variance of the estimated moments.
        :raises TypeError: If the object is not an instance of QuantityMean.
        """
        if not isinstance(moments_mean, mlmc.quantity.quantity.QuantityMean):
            raise TypeError
        self._moments_mean = moments_mean

    def estimate_moments(self, moments_fn=None):
        """
        Estimate the mean and variance of the defined moment functions.

        :param moments_fn: callable, optional
            Function to compute statistical moments. If None, uses the stored function.
        :return: tuple (moment_means, moment_variances)
            Arrays of length n_moments representing estimated means and variances.
        """
        if moments_fn is None:
            moments_fn = self._moments_fn

        moments_mean = qe.estimate_mean(qe.moments(self._quantity, moments_fn))
        self.moments_mean_obj = moments_mean
        return moments_mean.mean, moments_mean.var

    def estimate_covariance(self, moments_fn=None):
        """
        Estimate the covariance matrix and its variance from MLMC samples.

        :param moments_fn: callable, optional
            Function defining moment evaluations. If None, uses the stored one.
        :return: tuple (covariance_matrix, covariance_variance)
        """
        if moments_fn is None:
            moments_fn = self._moments_fn

        cov_mean = qe.estimate_mean(qe.covariance(self._quantity, moments_fn))
        return cov_mean.mean, cov_mean.var

    def estimate_diff_vars_regression(self, n_created_samples, moments_fn=None, raw_vars=None):
        """
        Estimate variances using a linear regression model.

        Assumes that variance increases with moment order. Typically, only two moments
        with the highest average variance are used.

        :param n_created_samples: array-like
            Number of created samples on each MLMC level.
        :param moments_fn: callable, optional
            Moment evaluation function.
        :param raw_vars: ndarray, optional
            Precomputed raw variance estimates.
        :return: tuple (variance_array, n_ops_estimate)
        """
        self._n_created_samples = n_created_samples
        if raw_vars is None:
            if moments_fn is None:
                moments_fn = self._moments_fn
            raw_vars, n_samples = self.estimate_diff_vars(moments_fn)

        sim_steps = np.squeeze(self._sample_storage.get_level_parameters())
        vars = self._all_moments_variance_regression(raw_vars, sim_steps)

        return vars, self._sample_storage.get_n_ops()

    def estimate_diff_vars(self, moments_fn=None):
        """
        Estimate the variance of moment differences between consecutive MLMC levels.

        :param moments_fn: callable, optional
            Moment evaluation functions.
        :return: tuple (diff_variance, n_samples)
            diff_variance - shape (L, R): variances of differences of moments.
            n_samples - shape (L,): number of samples per level.
        """
        moments_mean = qe.estimate_mean(qe.moments(self._quantity, moments_fn))
        return moments_mean.l_vars, moments_mean.n_samples

    def _all_moments_variance_regression(self, raw_vars, sim_steps):
        """Apply variance regression across all moment functions."""
        reg_vars = raw_vars.copy()
        n_moments = raw_vars.shape[1]
        for m in range(1, n_moments):
            reg_vars[:, m] = self._moment_variance_regression(raw_vars[:, m], sim_steps)
        assert np.allclose(reg_vars[:, 0], 0.0)
        return reg_vars

    def _moment_variance_regression(self, raw_vars, sim_steps):
        """
        Perform regression-based smoothing of level variance for a single moment.

        Model:
            log(var_l) = A + B * log(h_l) + C * log^2(h_l)

        :param raw_vars: ndarray, shape (L,)
            Raw variance estimates of a single moment.
        :param sim_steps: ndarray, shape (L,)
            Simulation step sizes or level parameters.
        :return: ndarray, shape (L,)
            Smoothed variance estimates.
        """
        L, = raw_vars.shape
        L1 = L - 1
        if L < 3 or np.allclose(raw_vars, 0):
            return raw_vars

        W = 1.0 / np.sqrt(self._variance_of_variance())
        W = W[1:]
        W = np.ones((L - 1,))

        K = 3
        X = np.zeros((L1, K))
        log_step = np.log(sim_steps[1:])
        X[:, 0] = np.ones(L1)
        X[:, 1] = np.full(L1, log_step)
        X[:, 2] = np.full(L1, log_step ** 2)

        WX = X * W[:, None]
        log_vars = np.log(raw_vars[1:])
        log_vars = W * log_vars
        params, res, rank, sing_vals = np.linalg.lstsq(WX, log_vars)

        new_vars = raw_vars.copy()
        new_vars[1:] = np.exp(np.dot(X, params))
        return new_vars

    def _variance_of_variance(self, n_samples=None):
        """
        Approximate the variance of log(X), where X follows a chi-squared distribution.

        Used to approximate the uncertainty of variance estimates.

        :param n_samples: array-like, optional
            Number of samples per level.
        :return: ndarray
            Variance of variance estimates per level.
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
        """
        Perform bootstrap resampling to estimate uncertainty in MLMC estimators.

        :param n_subsamples: int, default=100
            Number of bootstrap subsamples.
        :param sample_vector: ndarray, optional
            Sampling vector for selecting subsamples.
        :param moments_fn: callable, optional
            Moment evaluation function.
        """
        if moments_fn is not None:
            self._moments_fn = moments_fn
        else:
            moments_fn = self._moments_fn

        sample_vector = determine_sample_vec(
            n_collected_samples=self._sample_storage.get_n_collected(),
            n_levels=self._sample_storage.get_n_levels(),
            sample_vector=sample_vector
        )
        bs_mean, bs_var, bs_l_means, bs_l_vars = [], [], [], []
        for i in range(n_subsamples):
            quantity_subsample = self.quantity.subsample(sample_vec=sample_vector)
            moments_quantity = qe.moments(quantity_subsample, moments_fn=moments_fn, mom_at_bottom=False)
            q_mean = qe.estimate_mean(moments_quantity)

            bs_mean.append(q_mean.mean)
            bs_var.append(q_mean.var)
            bs_l_means.append(q_mean.l_means)
            bs_l_vars.append(q_mean.l_vars)

        self.bs_mean = bs_mean
        self.bs_var = bs_var

        self.mean_bs_mean = np.mean(bs_mean, axis=0)
        self.mean_bs_var = np.mean(bs_var, axis=0)
        self.mean_bs_l_means = np.mean(bs_l_means, axis=0)
        self.mean_bs_l_vars = np.mean(bs_l_vars, axis=0)

        self.var_bs_mean = np.var(bs_mean, axis=0, ddof=1)
        self.var_bs_var = np.var(bs_var, axis=0, ddof=1)
        self.var_bs_l_means = np.var(bs_l_means, axis=0, ddof=1)
        self.var_bs_l_vars = np.var(bs_l_vars, axis=0, ddof=1)

        self._bs_level_mean_variance = (
            self.var_bs_l_means * np.array(self._sample_storage.get_n_collected())[:, None]
        )

    def bs_target_var_n_estimated(self, target_var, sample_vec=None, n_subsamples=100):
        """
        Estimate the number of samples required to achieve a target variance.

        :param target_var: float
            Desired target variance for MLMC estimation.
        :param sample_vec: ndarray, optional
            Sampling vector specifying subsamples per level.
        :param n_subsamples: int, default=100
            Number of bootstrap resamplings to perform.
        :return: ndarray
            Estimated number of samples required at each level.
        """
        sample_vec = determine_sample_vec(
            n_collected_samples=self._sample_storage.get_n_collected(),
            n_levels=self._sample_storage.get_n_levels(),
            sample_vector=sample_vec
        )

        self.est_bootstrap(n_subsamples=n_subsamples, sample_vector=sample_vec)

        variances, n_ops = self.estimate_diff_vars_regression(
            sample_vec, raw_vars=self.mean_bs_l_vars
        )

        n_estimated = estimate_n_samples_for_target_variance(
            target_var, variances, n_ops, n_levels=self._sample_storage.get_n_levels()
        )

        return n_estimated

    def plot_variances(self, sample_vec=None):
        """
        Plot variance breakdown from bootstrap and regression data.

        :param sample_vec: ndarray, optional
            Sampling vector specifying subsamples per level.
        """
        var_plot = plots.VarianceBreakdown(10)

        sample_vec = determine_sample_vec(
            n_collected_samples=self._sample_storage.get_n_collected(),
            n_levels=self._sample_storage.get_n_levels(),
            sample_vector=sample_vec
        )
        self.est_bootstrap(n_subsamples=100, sample_vector=sample_vec)

        var_plot.add_variances(
            self.mean_bs_l_vars,
            sample_vec,
            ref_level_vars=self._bs_level_mean_variance
        )
        var_plot.show(None)

    def plot_bs_var_log(self, sample_vec=None):
        """
        Generate log-scale bootstrap variance plots and variance regression fits.

        :param sample_vec: ndarray, optional
            Sampling vector specifying subsamples per level.
        """
        sample_vec = determine_sample_vec(
            n_collected_samples=self._sample_storage.get_n_collected(),
            n_levels=self._sample_storage.get_n_levels(),
            sample_vector=sample_vec
        )

        moments_quantity = qe.moments(
            self._quantity, moments_fn=self._moments_fn, mom_at_bottom=False
        )
        q_mean = qe.estimate_mean(moments_quantity)

        bs_plot = plots.BSplots(
            bs_n_samples=sample_vec,
            n_samples=self._sample_storage.get_n_collected(),
            n_moments=self._moments_fn.size,
            ref_level_var=q_mean.l_vars
        )

        bs_plot.plot_means_and_vars(
            self.mean_bs_mean[1:],
            self.mean_bs_var[1:],
            n_levels=self._sample_storage.get_n_levels()
        )

        bs_plot.plot_bs_variances(self.mean_bs_l_vars)
        # bs_plot.plot_bs_var_log_var()
        bs_plot.plot_var_regression(self, self._sample_storage.get_n_levels(), self._moments_fn)

    def fine_coarse_violinplot(self):
        """
        Create violin plots comparing fine and coarse samples across levels.

        Uses pandas for data organization and mlmc.plot.violinplot for visualization.
        """
        import pandas as pd
        from mlmc.plot import violinplot

        label_n_spaces = 5
        n_levels = self._sample_storage.get_n_levels()

        if n_levels > 1:
            for level_id in range(n_levels):
                chunk_spec = next(
                    self._sample_storage.chunks(
                        level_id=level_id,
                        n_samples=self._sample_storage.get_n_collected()[level_id]
                    )
                )
                samples = np.squeeze(self._quantity.samples(chunk_spec, axis=0))
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
        Estimate lower and upper bounds of the domain from MLMC samples.

        :param quantity: mlmc.quantity.Quantity
            Quantity object representing the stochastic quantity.
        :param sample_storage: mlmc.sample_storage.SampleStorage
            Storage object containing all level samples.
        :param quantile: float, optional
            Quantile value in (0, 1). None defaults to 0.01.
        :return: tuple (lower_bound, upper_bound)
        """
        ranges = []
        if quantile is None:
            quantile = 0.01

        for level_id in range(sample_storage.get_n_levels()):
            try:
                sample_storage.get_n_collected()[level_id]
            except AttributeError:
                print(f"No collected values for level {level_id}")
                break

            print("sample_storage.get_n_collected() ", type(sample_storage.get_n_collected()[0]))

            if isinstance(sample_storage.get_n_collected()[level_id], AttributeError):
                print("continue")
                continue

            chunk_spec = next(
                sample_storage.chunks(
                    level_id=level_id,
                    n_samples=sample_storage.get_n_collected()[level_id]
                )
            )
            fine_samples = quantity.samples(chunk_spec)[..., 0]
            fine_samples = np.squeeze(fine_samples)
            print("fine samples ", fine_samples)
            fine_samples = fine_samples[~np.isnan(fine_samples)]
            ranges.append(np.percentile(fine_samples, [100 * quantile, 100 * (1 - quantile)]))

        ranges = np.array(ranges)
        return np.min(ranges[:, 0]), np.max(ranges[:, 1])

    def construct_density(self, tol=1e-8, reg_param=0.0, orth_moments_tol=1e-4, exact_pdf=None):
        """
        Construct an approximate probability density function using orthogonal moments.

        :param tol: float, default=1e-8
            Optimization tolerance for density estimation.
        :param reg_param: float, default=0.0
            Regularization parameter to stabilize estimation.
        :param orth_moments_tol: float, default=1e-4
            Tolerance for orthogonalization of moments.
        :param exact_pdf: callable, optional
            Reference exact PDF for validation or comparison.
        :return: tuple (distribution, info, result, moments_object)
        """
        if not isinstance(self._quantity.qtype, ScalarType):
            raise NotImplementedError("Currently, only ScalarType quantities are supported.")

        cov_mean = qe.estimate_mean(qe.covariance(self._quantity, self._moments_fn))
        cov_mat = cov_mean.mean
        moments_obj, info = mlmc.tool.simple_distribution.construct_ortogonal_moments(
            self._moments_fn, cov_mat, tol=orth_moments_tol
        )

        moments_mean = qe.estimate_mean(qe.moments(self._quantity, moments_obj))
        est_moments = moments_mean.mean
        est_vars = moments_mean.var

        est_vars = np.ones(moments_obj.size)
        min_var, max_var = np.min(est_vars[1:]), np.max(est_vars[1:])
        moments_data = np.stack((est_moments, est_vars), axis=1)

        distr_obj = mlmc.tool.simple_distribution.SimpleDistribution(
            moments_obj, moments_data, domain=moments_obj.domain
        )
        result = distr_obj.estimate_density_minimize(tol, reg_param)

        return distr_obj, info, result, moments_obj

    def get_level_samples(self, level_id, n_samples=None):
        """
        Retrieve MLMC samples for a given level.

        :param level_id: int
            Level index to access.
        :param n_samples: int, optional
            Number of samples to retrieve. If None, retrieves all available samples.
        :return: ndarray
            Samples for the specified level.
        """
        chunk_spec = next(self._sample_storage.chunks(level_id=level_id, n_samples=n_samples))
        return self._quantity.samples(chunk_spec=chunk_spec)

    def kurtosis_check(self, quantity=None):
        """
        Compute and return the kurtosis of the given or stored quantity.

        :param quantity: mlmc.quantity.Quantity, optional
            Quantity for which to compute kurtosis. Defaults to the stored quantity.
        :return: float or ndarray
            Computed kurtosis per level.
        """
        if quantity is None:
            quantity = self._quantity
        moments_mean_quantity = qe.estimate_mean(quantity)
        kurtosis = qe.level_kurtosis(quantity, moments_mean_quantity)
        return kurtosis


def consistency_check(quantity, sample_storage=None):
    """
    Check consistency between fine and coarse level samples in MLMC.

    :param quantity: mlmc.quantity.Quantity instance
    :param sample_storage: mlmc.sample_storage.SampleStorage instance
    :return: dict mapping level_id -> consistency metric
    """
    fine_samples = {}
    coarse_samples = {}

    for chunk_spec in quantity.get_quantity_storage().chunks():
        samples = quantity.samples(chunk_spec)
        chunk, _ = mask_nan_samples(samples)

        # Skip empty chunks
        if chunk.shape[1] == 0:
            continue

        fine_samples.setdefault(chunk_spec.level_id, []).extend(chunk[:, :, 0])
        if chunk_spec.level_id > 0:
            coarse_samples.setdefault(chunk_spec.level_id, []).extend(chunk[:, :, 1])

    cons_check_val = {}
    for level_id in range(sample_storage.get_n_levels()):
        if level_id > 0:
            fine_mean = np.mean(fine_samples[level_id])
            coarse_mean = np.mean(coarse_samples[level_id])
            diff_mean = np.mean(np.array(fine_samples[level_id]) - np.array(coarse_samples[level_id]))

            fine_var = np.var(fine_samples[level_id])
            coarse_var = np.var(fine_samples[level_id])
            diff_var = np.var(np.array(fine_samples[level_id]) - np.array(coarse_samples[level_id]))

            val = np.abs(coarse_mean - fine_mean + diff_mean) / (
                    3 * (np.sqrt(coarse_var) + np.sqrt(fine_var) + np.sqrt(diff_var))
            )

            assert np.isclose(coarse_mean - fine_mean + diff_mean, 0)
            assert val < 0.9

            cons_check_val[level_id] = val

    return cons_check_val


def coping_with_high_kurtosis(vars, costs, kurtosis, kurtosis_threshold=100):
    """
    Adjust variance estimates if kurtosis is unusually high to avoid underestimation.

    :param vars: ndarray of shape (L, M) with level variances for moments
    :param costs: cost of computing samples per level
    :param kurtosis: kurtosis of each level
    :param kurtosis_threshold: threshold above which kurtosis is considered "high"
    :return: adjusted vars ndarray
    """
    for l_id in range(2, vars.shape[0]):
        if kurtosis[l_id] > kurtosis_threshold:
            vars[l_id] = np.maximum(vars[l_id], 0.5 * vars[l_id - 1] * costs[l_id - 1] / costs[l_id])
    return vars


def estimate_n_samples_for_target_variance(target_variance, prescribe_vars, n_ops, n_levels, theta=0, kurtosis=None):
    """
    Estimate optimal number of samples per level to reach a target variance.

    :param target_variance: desired variance for MLMC estimator
    :param prescribe_vars: ndarray of level variances (L x M)
    :param n_ops: cost/operations per level
    :param n_levels: number of levels
    :param theta: safety factor (0 ≤ theta < 1)
    :param kurtosis: optional ndarray of kurtosis per level
    :return: ndarray of optimal number of samples per moment function
    """
    vars = prescribe_vars

    if kurtosis is not None and len(vars) == len(kurtosis):
        vars = coping_with_high_kurtosis(vars, n_ops, kurtosis)

    sqrt_var_n = np.sqrt(vars.T * n_ops)  # moments_fn in rows, levels in cols
    total = np.sum(sqrt_var_n, axis=1)
    n_samples_estimate = np.round((sqrt_var_n / n_ops).T * total / target_variance).astype(int)
    n_samples_estimate = 1 / (1 - theta) * n_samples_estimate

    # Limit maximal number of samples per level
    n_samples_estimate_safe = np.maximum(
        np.minimum(n_samples_estimate, vars * n_levels / target_variance),
        2
    )

    return np.max(n_samples_estimate_safe, axis=1).astype(int)


def calc_level_params(step_range, n_levels):
    """
    Compute level-dependent step sizes for MLMC.

    :param step_range: tuple (h_fine, h_coarse)
    :param n_levels: number of levels
    :return: list of step sizes per level
    """
    assert step_range[0] > step_range[1]
    level_parameters = []
    for i_level in range(n_levels):
        level_param = 1 if n_levels == 1 else i_level / (n_levels - 1)
        level_parameters.append([step_range[0] ** (1 - level_param) * step_range[1] ** level_param])
    return level_parameters


def determine_sample_vec(n_collected_samples, n_levels, sample_vector=None):
    """
    Determine the sample vector for bootstrapping or MLMC calculations.
    """
    if sample_vector is None:
        sample_vector = n_collected_samples
    if len(sample_vector) > n_levels:
        sample_vector = sample_vector[:n_levels]
    return np.array(sample_vector)


def determine_level_parameters(n_levels, step_range):
    """
    Wrapper to calculate level parameters (simulation step sizes).
    """
    assert step_range[0] > step_range[1]
    level_parameters = []
    for i_level in range(n_levels):
        level_param = 1 if n_levels == 1 else i_level / (n_levels - 1)
        level_parameters.append([step_range[0] ** (1 - level_param) * step_range[1] ** level_param])
    return level_parameters


def determine_n_samples(n_levels, n_samples=None):
    """
    Generate an array of target sample sizes for each level.

    :param n_levels: number of MLMC levels
    :param n_samples: int or list of 2 ints to define start/end for exponential spacing
    :return: ndarray of sample sizes for each level
    """
    if n_samples is None:
        n_samples = [100, 3]

    n_samples = np.atleast_1d(n_samples)

    if len(n_samples) == 1:
        n_samples = np.array([n_samples[0], 3])

    if len(n_samples) == 2:
        n0, nL = n_samples
        n_samples = np.round(np.exp2(np.linspace(np.log2(n0), np.log2(nL), n_levels))).astype(int)

    return n_samples
