import numpy as np
import mlmc.tool.simple_distribution
from mlmc.quantity_concept import estimate_mean, covariance, moments


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
    print("n levels: ", self.n_levels, "size: ", moments_obj.size)

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
