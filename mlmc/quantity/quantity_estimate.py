import numpy as np
import mlmc.quantity.quantity
import mlmc.quantity.quantity_types as qt


def mask_nan_samples(chunk):
    """
    Remove (mask out) samples containing NaN values in either the fine or coarse part of the result.

    :param chunk: np.ndarray of shape [M, chunk_size, 2]
                  M - quantity size (number of scalar components),
                  chunk_size - number of samples in the chunk,
                  2 - fine and coarse parts of the result.
    :return: (filtered_chunk, n_masked)
             filtered_chunk: np.ndarray with invalid samples removed,
             n_masked: int, number of masked (removed) samples.
    """
    # Identify any sample with NaNs in its fine or coarse component
    mask = np.any(np.isnan(chunk), axis=0).any(axis=1)
    return chunk[..., ~mask, :], np.count_nonzero(mask)


def cache_clear():
    """
    Clear cached Quantity sample evaluations.

    Used before running MLMC estimations to ensure fresh data is fetched from storage.
    """
    mlmc.quantity.quantity.Quantity.samples.cache_clear()
    mlmc.quantity.quantity.QuantityConst.samples.cache_clear()


def estimate_mean(quantity, form="diff", operation_func=None, **kwargs):
    """
    Estimate the MLMC mean (and variance) of a Quantity using multilevel sampling.

    The function computes per-level means and variances from simulation results.
    Supports large datasets via chunked processing and handles NaN-masked samples.

    :param quantity: Quantity instance to estimate.
    :param form: str, type of estimation:
                 - "diff": estimate based on differences (fine - coarse) → standard MLMC approach.
                 - "fine": estimate using fine-level data only.
                 - "coarse": estimate using coarse-level data only.
    :param operation_func: Optional transformation applied to chunk data before accumulation
                           (e.g., for moment or kurtosis computation).
    :param kwargs: Additional keyword arguments passed to operation_func.
    :return: QuantityMean object containing mean, variance, and sample statistics per level.
    """
    # Reset cached quantity evaluations
    cache_clear()

    quantity_vec_size = quantity.size()
    sums = None
    sums_of_squares = None

    # Initialize level-specific storage
    quantity_storage = quantity.get_quantity_storage()
    level_ids = quantity_storage.level_ids()
    n_levels = np.max(level_ids) + 1
    n_samples = [0] * n_levels
    n_rm_samples = [0] * n_levels

    # Iterate through data chunks
    for chunk_spec in quantity_storage.chunks():
        samples = quantity.samples(chunk_spec)
        chunk, n_mask_samples = mask_nan_samples(samples)
        n_samples[chunk_spec.level_id] += chunk.shape[1]
        n_rm_samples[chunk_spec.level_id] += n_mask_samples

        # Skip empty chunks
        if chunk.shape[1] == 0:
            continue
        assert chunk.shape[0] == quantity_vec_size

        # Allocate accumulators at first valid chunk
        if sums is None:
            sums = [np.zeros(chunk.shape[0]) for _ in range(n_levels)]
            sums_of_squares = [np.zeros(chunk.shape[0]) for _ in range(n_levels)]

        # Select appropriate data form for the estimator
        if form == "fine":
            chunk_diff = chunk[:, :, 0]
        elif form == "coarse":
            chunk_diff = np.zeros_like(chunk[:, :, 0]) if chunk_spec.level_id == 0 else chunk[:, :, 1]
        else:
            # Default MLMC difference (fine - coarse)
            chunk_diff = chunk[:, :, 0] if chunk_spec.level_id == 0 else chunk[:, :, 0] - chunk[:, :, 1]

        # Optional user-defined transformation of data
        if operation_func is not None:
            chunk_diff = operation_func(chunk_diff, chunk_spec, **kwargs)

        # Accumulate sums and squared sums for this level
        sums[chunk_spec.level_id] += np.sum(chunk_diff, axis=1)
        sums_of_squares[chunk_spec.level_id] += np.sum(chunk_diff ** 2, axis=1)

    if sums is None:
        raise Exception("All samples were masked (no valid data found).")

    # Compute means and variances for each level
    l_means = []
    l_vars = []
    for s, sp, n in zip(sums, sums_of_squares, n_samples):
        l_means.append(s / n)
        if n > 1:
            l_vars.append((sp - (s ** 2 / n)) / (n - 1))
        else:
            l_vars.append(np.full(len(s), np.inf))

    # Construct QuantityMean object with level statistics
    return mlmc.quantity.quantity.QuantityMean(
        quantity.qtype,
        l_means=l_means,
        l_vars=l_vars,
        n_samples=n_samples,
        n_rm_samples=n_rm_samples
    )


def moment(quantity, moments_fn, i=0):
    """
    Construct a Quantity that represents a single statistical moment.

    :param quantity: Base Quantity instance.
    :param moments_fn: Instance of mlmc.moments.Moments defining the moment computation.
    :param i: Index of the moment to compute.
    :return: New Quantity that computes the i-th moment.
    """
    def eval_moment(x):
        return moments_fn.eval_single_moment(i, value=x)

    return mlmc.quantity.quantity.Quantity(
        quantity_type=quantity.qtype,
        input_quantities=[quantity],
        operation=eval_moment
    )


def moments(quantity, moments_fn, mom_at_bottom=True):
    """
    Construct a Quantity representing all moments defined by a given Moments object.

    :param quantity: Base Quantity.
    :param moments_fn: mlmc.moments.Moments child defining moment evaluations.
    :param mom_at_bottom: bool, if True, moments are added at the lowest (scalar) level of the Quantity type.
    :return: Quantity that computes all defined moments.
    """
    def eval_moments(x):
        if mom_at_bottom:
            mom = moments_fn.eval_all(x).transpose((0, 3, 1, 2))  # [M, R, N, 2]
        else:
            mom = moments_fn.eval_all(x).transpose((3, 0, 1, 2))  # [R, M, N, 2]
        return mom.reshape((np.prod(mom.shape[:-2]), mom.shape[-2], mom.shape[-1]))  # [M, N, 2]

    # Define new Quantity type according to desired hierarchy
    if mom_at_bottom:
        moments_array_type = qt.ArrayType(shape=(moments_fn.size,), qtype=qt.ScalarType())
        moments_qtype = quantity.qtype.replace_scalar(moments_array_type)
    else:
        moments_qtype = qt.ArrayType(shape=(moments_fn.size,), qtype=quantity.qtype)

    return mlmc.quantity.quantity.Quantity(
        quantity_type=moments_qtype,
        input_quantities=[quantity],
        operation=eval_moments
    )


def covariance(quantity, moments_fn, cov_at_bottom=True):
    """
    Construct a Quantity representing covariance matrices of the given moments.

    :param quantity: Base Quantity.
    :param moments_fn: mlmc.moments.Moments child defining moment evaluations.
    :param cov_at_bottom: bool, if True covariance matrices are attached at the scalar level of the Quantity type.
    :return: Quantity that computes covariance matrices.
    """
    def eval_cov(x):
        # Compute all moments (fine and coarse)
        moments = moments_fn.eval_all(x)
        mom_fine = moments[..., 0, :]
        cov_fine = np.einsum('...i,...j', mom_fine, mom_fine)

        if moments.shape[-2] == 1:
            # Single level (no coarse)
            cov = np.array([cov_fine])
        else:
            mom_coarse = moments[..., 1, :]
            cov_coarse = np.einsum('...i,...j', mom_coarse, mom_coarse)
            cov = np.array([cov_fine, cov_coarse])

        # Reshape covariance according to desired data layout
        if cov_at_bottom:
            cov = cov.transpose((1, 3, 4, 2, 0))  # [M, R, R, N, 2]
        else:
            cov = cov.transpose((3, 4, 1, 2, 0))  # [R, R, M, N, 2]
        return cov.reshape((np.prod(cov.shape[:-2]), cov.shape[-2], cov.shape[-1]))

    # Adjust Quantity type for covariance structure
    if cov_at_bottom:
        moments_array_type = qt.ArrayType(shape=(moments_fn.size, moments_fn.size), qtype=qt.ScalarType())
        moments_qtype = quantity.qtype.replace_scalar(moments_array_type)
    else:
        moments_qtype = qt.ArrayType(shape=(moments_fn.size, moments_fn.size), qtype=quantity.qtype)

    return mlmc.quantity.quantity.Quantity(
        quantity_type=moments_qtype,
        input_quantities=[quantity],
        operation=eval_cov
    )


def kurtosis_numerator(chunk_diff, chunk_spec, l_means):
    """
    Compute the numerator for the sample kurtosis:
        E[(Y_l - E[Y_l])^4]
    :param chunk_diff: np.ndarray [quantity shape, number of samples]
    :param chunk_spec: quantity_spec.ChunkSpec describing current level and chunk.
    :param l_means: List of per-level means used for centering.
    :return: np.ndarray of the same shape as input.
    """
    return (chunk_diff - l_means[chunk_spec.level_id]) ** 4


def level_kurtosis(quantity, means_obj):
    """
    Estimate the sample kurtosis for each level:
        E[(Y_l - E[Y_l])^4] / (Var[Y_l])^2, where Y_l = fine_l - coarse_l

    :param quantity: Quantity instance.
    :param means_obj: QuantityMean object containing level means and variances.
    :return: np.ndarray of kurtosis values per level.
    """
    numerator_means_obj = estimate_mean(quantity, operation_func=kurtosis_numerator, l_means=means_obj.l_means)
    kurtosis = numerator_means_obj.l_means / (means_obj.l_vars) ** 2
    return kurtosis
