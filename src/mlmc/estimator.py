import numpy as np


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
