import numpy as np
from mlmc.quantity import make_root_quantity, estimate_mean, moment, moments, covariance
from mlmc.quantity import Quantity, QuantityStorage, DictType


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


class Estimate:

    def __init__(self, sample_storage, moments=None):
        self.sample_storage = sample_storage
        self.moments = moments

    @property
    def n_moments(self):
        return self.moments.size

    def est_bootstrap(self, quantity, n_subsamples=100, sample_vector=None, moments_fn=None):

        if moments_fn is not None:
            self.moments = moments_fn
        else:
            moments_fn = self.moments

        if sample_vector is None:
            sample_vector = self.sample_storage.get_n_collected()
        if len(sample_vector) > len(self.sample_storage.get_level_ids()):
            sample_vector = sample_vector[:len(self.sample_storage.get_level_ids())]
        sample_vector = np.array(sample_vector)

        bs_moments = []
        for i in range(n_subsamples):
            quantity_subsample = quantity.select(quantity.subsample(sample_vec=sample_vector))
            moments_quantity = moments(quantity_subsample, moments_fn=moments_fn, mom_at_bottom=True)

            estimate_mean(moments_quantity)
            bs_moments.append(moments_quantity)

        bs_mean_est = [np.mean(est, axis=-1) for est in bs_moments]
        bs_err_est = [np.var(est, axis=-1, ddof=1) for est in bs_moments]

        bs_est_mean = estimate_mean(bs_mean_est)
        bs_est_var = estimate_mean(bs_err_est)

        print("bs_est_mean ", bs_est_mean())
        print("bs_est_var ", bs_est_var())
