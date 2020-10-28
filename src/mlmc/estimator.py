import numpy as np
from mlmc.tool import plot
from mlmc.quantity import make_root_quantity, estimate_mean, moment, moments, covariance
from mlmc.quantity_estimate import QuantityEstimate
import mlmc.tool.simple_distribution


def estimate_n_samples_for_target_variance(target_variance, prescribe_vars, n_ops, n_levels):
    """
    Estimate optimal number of samples for individual levels that should provide a target variance of
    resulting moment estimate.
    This also set given moment functions to be used for further estimates if not specified otherwise.
    :param target_variance: Constrain to achieve this variance.
    :param prescribe_vars: vars[ L, M] for all levels L and moments M safe the (zeroth) constant moment with zero variance.
    :param n_ops: number of operations at each level
    :param n_levels: number of levels
    :return: np.array with number of optimal samples for individual levels and moments, array (LxR)
    """
    vars = prescribe_vars
    sqrt_var_n = np.sqrt(vars.T * n_ops)  # moments in rows, levels in cols
    total = np.sum(sqrt_var_n, axis=1)  # sum over levels
    n_samples_estimate = np.round((sqrt_var_n / n_ops).T * total / target_variance).astype(int)  # moments in cols
    # Limit maximal number of samples per level
    n_samples_estimate_safe = np.maximum(
        np.minimum(n_samples_estimate, vars * n_levels / target_variance), 2)

    return np.max(n_samples_estimate_safe, axis=1).astype(int)


def construct_density(quantity, moments_fn, tol=1.95, reg_param=0.01):
    """
    Construct approximation of the density using given moment functions.
    Args:
        moments_fn: Moments object, determines also domain and n_moments.
        tol: Tolerance of the fitting problem, with account for variances in moments.
             Default value 1.95 corresponds to the two tail confidency 0.95.
        reg_param: Regularization parameter.
    """
    cov = estimate_mean(covariance(quantity, moments_fn))

    conductivity_cov = cov['conductivity']
    time_cov = conductivity_cov[1]  # times: [1]
    location_cov = time_cov['0']  # locations: ['0']
    values_cov = location_cov[0, 0]  # result shape: (1, 1)
    cov = values_cov()

    moments_obj, info = mlmc.tool.simple_distribution.construct_ortogonal_moments(moments_fn, cov, tol=0.0001)

    #est_moments, est_vars = self.estimate_moments(moments_obj)
    moments_mean = estimate_mean(moments(quantity, moments_obj))
    est_moments = moments_mean.mean()
    est_vars = moments_mean.var()

    print("est moments ", est_moments)
    print("est vars ", est_vars)
    #est_moments = np.zeros(moments_obj.size)
    #est_moments[0] = 1.0
    est_vars = np.ones(moments_obj.size)
    min_var, max_var = np.min(est_vars[1:]), np.max(est_vars[1:])
    print("min_err: {} max_err: {} ratio: {}".format(min_var, max_var, max_var / min_var))
    moments_data = np.stack((est_moments, est_vars), axis=1)
    distr_obj = mlmc.tool.simple_distribution.SimpleDistribution(moments_obj, moments_data, domain=moments_obj.domain)
    distr_obj.estimate_density_minimize(tol, reg_param)  # 0.95 two side quantile

    return distr_obj


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


class Estimate:

    def __init__(self, quantity, sample_storage, moments=None):
        self._quantity = quantity
        self.sample_storage = sample_storage
        self.moments = moments


    @property
    def quantity(self):
        return self._quantity

    @quantity.setter
    def quantity(self, quantity):
        self._quantity = quantity

    @property
    def n_moments(self):
        return self.moments.size

    def _determine_sample_vec(self, sample_vector=None):
        if sample_vector is None:
            sample_vector = self.sample_storage.get_n_collected()
        if len(sample_vector) > len(self.sample_storage.get_level_ids()):
            sample_vector = sample_vector[:len(self.sample_storage.get_level_ids())]
        return np.array(sample_vector)

    def est_bootstrap(self, n_subsamples=100, sample_vector=None, moments_fn=None):

        if moments_fn is not None:
            self.moments = moments_fn
        else:
            moments_fn = self.moments

        sample_vector = self._determine_sample_vec(sample_vector)

        bs_mean = []
        bs_var = []
        bs_l_means = []
        bs_l_vars = []
        for i in range(n_subsamples):
            quantity_subsample = self.quantity.select(self.quantity.subsample(sample_vec=sample_vector))
            moments_quantity = moments(quantity_subsample, moments_fn=moments_fn, mom_at_bottom=False)
            q_mean = estimate_mean(moments_quantity, level_means=True)

            bs_mean.append(q_mean.mean)
            bs_var.append(q_mean.var)
            bs_l_means.append(q_mean.l_means)
            bs_l_vars.append(q_mean.l_vars)

        # print("bs_mean ", bs_mean)
        # print("bs_var ", bs_var)
        # print("bs_l_means ", bs_l_means)
        # print("bs_l_vars ", bs_l_vars)
        # exit()

        self.mean_bs_mean = np.mean(bs_mean, axis=0)
        self.mean_bs_var = np.mean(bs_var, axis=0)
        self.mean_bs_l_means = np.mean(bs_l_means, axis=0)
        self.mean_bs_l_vars = np.mean(bs_l_vars, axis=0)

        print("bs l vars ", bs_l_vars)
        print("bs l vars shape", np.array(bs_l_vars).shape)

        self.var_bs_mean = np.var(bs_mean, axis=0, ddof=1)
        self.var_bs_var = np.var(bs_var, axis=0, ddof=1)
        self.var_bs_l_means = np.var(bs_l_means, axis=0, ddof=1)
        self.var_bs_l_vars = np.var(bs_l_vars, axis=0, ddof=1)

        # print("self.var_bs_l_means.shape ", self.var_bs_l_means)
        # print("self.sample_storage.get_n_collected() ", self.sample_storage.get_n_collected())
        self._bs_level_mean_variance = self.var_bs_l_means * np.array(self.sample_storage.get_n_collected())[:, None]

        #print("self._bs_level_mean_variance ", self._bs_level_mean_variance)

    def bs_target_var_n_estimated(self, target_var, sample_vec=None):
        sample_vec = self._determine_sample_vec(sample_vec)
        self.est_bootstrap(n_subsamples=300, sample_vector=sample_vec)

        q_estimator = QuantityEstimate(sample_storage=self.sample_storage, moments_fn=self.moments,
                                       sim_steps=self.sample_storage.get_level_parameters())

        variances, n_ops = q_estimator.estimate_diff_vars_regression(sample_vec, raw_vars=self.mean_bs_l_vars)

        n_estimated = estimate_n_samples_for_target_variance(target_var, variances, n_ops,
                                                             n_levels=self.sample_storage.get_n_levels())

        print("n estimated ", n_estimated)

        return n_estimated

    def plot_variances(self, sample_vec=None):
        var_plot = plot.VarianceBreakdown(10)

        sample_vec = self._determine_sample_vec(sample_vec)
        self.est_bootstrap(n_subsamples=10, sample_vector=sample_vec)

        var_plot.add_variances(self.mean_bs_l_vars, sample_vec, ref_level_vars=self._bs_level_mean_variance)
        var_plot.show(None)

    def plot_level_variances(self):
        var_plot = plot.Variance(10)
        for mc in self.mlmc:
            steps, vars = mc.estimate_level_vars()
            var_plot.add_level_variances(steps, vars)
        var_plot.show()

    def plot_bs_var_log(self, sample_vec=None):
        sample_vec = self._determine_sample_vec(sample_vec)
        print("sample vec ", sample_vec)
        bs_plot = plot.BSplots(bs_n_samples=sample_vec, n_samples=self.sample_storage.get_n_collected(),
                               n_moments=self.moments.size)

        bs_plot.plot_means_and_vars(self.mean_bs_mean[1:], self.mean_bs_var[1:], n_levels=self.sample_storage.get_n_levels())

        bs_plot.plot_bs_variances(self.mean_bs_l_vars)
        #bs_plot.plot_bs_var_log_var()

        q_estimator = QuantityEstimate(sample_storage=self.sample_storage, moments_fn=self.moments,
                                       sim_steps=self.sample_storage.get_level_parameters())


        #bs_plot.plot_var_regression(q_estimator, self.sample_storage.get_n_levels(), self.moments, ref_level_var)


    def plot_var_compare(self, nl):
        self[nl].plot_bootstrap_variance_compare(self.moments)

    def plot_var_var(self, nl):
        self[nl].plot_bootstrap_var_var(self.moments)



