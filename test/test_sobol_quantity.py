import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

import numpy as np
from SALib.analyze import sobol as salib_sobol

from mlmc.quantity.quantity import make_root_quantity
from mlmc.quantity.quantity_estimate import estimate_mean
from mlmc.quantity.quantity_spec import QuantitySpec
from mlmc.quantity.sobol import (
    SaltelliSchema,
    estimate_sobol_indices,
)
from mlmc.sample_storage import Memory


def _interaction_model(x_values, a1=1.2, a2=0.7, a12=2.5, c=0.3):
    return c + a1 * x_values[..., 0] + a2 * x_values[..., 1] + a12 * x_values[..., 0] * x_values[..., 1]


def _interaction_reference(a1=1.2, a2=0.7, a12=2.5):
    first_0 = (a1 + 0.5 * a12) ** 2 / 12.0
    first_1 = (a2 + 0.5 * a12) ** 2 / 12.0
    second_01 = a12 ** 2 / 144.0
    variance = first_0 + first_1 + second_01
    return {
        "first_order": np.array([first_0, first_1]) / variance,
        "total_order": np.array([first_0 + second_01, first_1 + second_01]) / variance,
        "second_order": np.array([second_01]) / variance,
        "denominator": variance,
    }


def _salib_estimates(term_values, schema):
    values = np.asarray(term_values)
    a_values, b_values, ab_values, ba_values = salib_sobol.separate_output_values(
        values.reshape(-1),
        D=schema.n_parameters,
        N=values.shape[0],
        calc_second_order=True,
    )
    pairs = [
        (i_param, j_param)
        for i_param in range(schema.n_parameters)
        for j_param in range(i_param + 1, schema.n_parameters)
    ]
    denominator = np.var(np.r_[a_values, b_values])
    first_order = np.array([
        salib_sobol.first_order(a_values, ab_values[:, i_param], b_values)
        for i_param in range(schema.n_parameters)
    ])
    total_order = np.array([
        salib_sobol.total_order(a_values, ab_values[:, i_param], b_values)
        for i_param in range(schema.n_parameters)
    ])
    second_order = np.array([
        salib_sobol.second_order(
            a_values,
            ab_values[:, i_param],
            ab_values[:, j_param],
            ba_values[:, i_param],
            b_values,
        )
        for i_param, j_param in pairs
    ])
    return {
        "denominator": denominator,
        "first_order": first_order,
        "total_order": total_order,
        "second_order": second_order,
        "second_order_pairs": pairs,
    }


def _saltelli_model_values(n_samples=200000):
    rng = np.random.default_rng(12345)
    schema = SaltelliSchema.make(n_parameters=2)
    a_matrix = rng.random((n_samples, 2))
    b_matrix = rng.random((n_samples, 2))
    term_inputs = np.array([schema.terms(a_row, b_row) for a_row, b_row in zip(a_matrix, b_matrix)])
    return _interaction_model(term_inputs)


def _make_memory_root(term_values, schema):
    """
    Store term_values into Memory Storage and create the root quantity for
    sobol estimation quantities.
    """
    storage = Memory()
    result_format = [QuantitySpec(name="value", unit="1", shape=(schema.n_terms,), times=[0], locations=["0"])]
    storage.save_global_data(result_format=result_format, level_parameters=[[1.0]])
    storage.save_samples({
        0: [
            ("L00_S{:07d}".format(i_sample), (values, np.zeros_like(values)))
            for i_sample, values in enumerate(term_values)
        ]
    }, {})
    root_quantity = make_root_quantity(storage, result_format)
    return root_quantity["value"][0]["0"]


def test_sample_formulas():
    # Goal: SALib reference formulas match the analytic interaction-model reference.
    schema = SaltelliSchema.make(n_parameters=2)
    estimates = _salib_estimates(_saltelli_model_values(), schema)
    reference = _interaction_reference()

    assert np.allclose(estimates["denominator"], reference["denominator"], rtol=0.02)
    assert np.allclose(estimates["first_order"], reference["first_order"], atol=0.015)
    assert np.allclose(estimates["total_order"], reference["total_order"], atol=0.015)
    assert estimates["second_order_pairs"] == [(0, 1)]
    assert np.allclose(estimates["second_order"], reference["second_order"], atol=0.02)


def test_quantity_builders():
    # Goal: lazy Quantity builders produce the same ratios as SALib formulas.
    schema = SaltelliSchema.make(n_parameters=2)
    term_values = _saltelli_model_values(n_samples=400)
    saltelli_quantity = _make_memory_root(term_values, schema)
    estimate = estimate_sobol_indices(saltelli_quantity, schema)
    salib_estimates = _salib_estimates(term_values, schema)

    denominator_mean = estimate_mean(estimate.denominator_quantity)
    first_mean = estimate_mean(estimate.first_order_numerator_quantity)
    total_mean = estimate_mean(estimate.total_order_numerator_quantity)
    second_mean = estimate_mean(estimate.second_order_numerator_quantity)

    assert estimate.second_order_pairs == [(0, 1)]
    assert np.allclose(denominator_mean.mean, salib_estimates["denominator"])
    assert np.allclose(first_mean.mean / denominator_mean.mean, salib_estimates["first_order"])
    assert np.allclose(total_mean.mean / denominator_mean.mean, salib_estimates["total_order"])
    assert np.allclose(second_mean.mean / denominator_mean.mean, salib_estimates["second_order"])
    assert denominator_mean.l_vars.shape == (1, 1)
    assert first_mean.l_vars.shape == (1, 2)
    assert total_mean.l_vars.shape == (1, 2)
    assert second_mean.l_vars.shape == (1, 1)


def test_index_estimate():
    # Goal: high-level estimator returns ratios and level-variance diagnostics.
    schema = SaltelliSchema.make(n_parameters=2)
    term_values = _saltelli_model_values(n_samples=400)
    saltelli_quantity = _make_memory_root(term_values, schema)
    salib_estimates = _salib_estimates(term_values, schema)

    estimates = estimate_sobol_indices(saltelli_quantity, schema)

    assert np.allclose(estimates.first_order, salib_estimates["first_order"])
    assert np.allclose(estimates.total_order, salib_estimates["total_order"])
    assert np.allclose(estimates.second_order[(0, 1)], salib_estimates["second_order"][0])
    assert estimates.mean_mlmc.mean.shape == (1,)
    assert estimates.denominator_mlmc.l_vars.shape == (1, 1)
    assert estimates.first_order_numerator_mlmc.l_vars.shape == (1, 2)
    assert estimates.total_order_numerator_mlmc.l_vars.shape == (1, 2)
    assert estimates.second_order_numerator_mlmc.l_vars.shape == (1, 1)
    assert estimates.denominator_std.shape == (1,)
    assert np.allclose(
        estimates.first_order_std,
        np.sqrt(estimates.first_order_numerator_mlmc.var) / np.abs(estimates.denominator_mlmc.mean)
    )
    assert np.allclose(
        estimates.total_order_std,
        np.sqrt(estimates.total_order_numerator_mlmc.var) / np.abs(estimates.denominator_mlmc.mean)
    )
    assert np.allclose(
        estimates.second_order_std[(0, 1)],
        np.sqrt(estimates.second_order_numerator_mlmc.var[0]) / np.abs(estimates.denominator_mlmc.mean)
    )
