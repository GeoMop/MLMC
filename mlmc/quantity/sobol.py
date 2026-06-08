from functools import cached_property

import attr
import numpy as np

from mlmc.quantity.quantity import Quantity
from mlmc.quantity.quantity_estimate import estimate_mean
import mlmc.quantity.quantity_types as qt


@attr.s(auto_attribs=True, frozen=True)
class SaltelliSchema:
    """
    Indexing schema for Saltelli A, AB_i, BA_i, B terms.

    The schema contains only term indices and masks. Parameter distributions
    belong to the wrapped forward model, so Sobol indices are indexed by
    parameter position.
    """

    n_parameters: int
    n_terms: int
    a: int
    ab: np.ndarray
    ba: np.ndarray
    b: int

    @classmethod
    def make(cls, n_parameters: int):
        """
        Create a Saltelli schema for the given parameter count.
        """
        n_terms = 2 * (n_parameters + 1)
        return cls(
            n_parameters=n_parameters,
            n_terms=n_terms,
            a=0,
            ab=np.arange(1, 1 + n_parameters),
            ba=np.arange(1 + n_parameters, 1 + 2 * n_parameters),
            b=1 + 2 * n_parameters,
        )

    @property
    def a_mask(self):
        mask = np.zeros((self.n_terms, self.n_parameters), dtype=bool)
        mask[self.a, :] = True
        for i_param in range(self.n_parameters):
            mask[self.ab[i_param], :] = True
            mask[self.ab[i_param], i_param] = False
            mask[self.ba[i_param], i_param] = True
        return mask

    def terms(self, a_row, b_row):
        """
        Build all Saltelli term input vectors for one A/B row pair.
        """
        a_row = np.asarray(a_row)
        b_row = np.asarray(b_row)
        assert a_row.shape == b_row.shape == (self.n_parameters,)
        return np.where(self.a_mask, a_row[None, :], b_row[None, :])


def _require_saltelli_array(quantity: Quantity, schema: SaltelliSchema):
    if not isinstance(quantity.qtype, qt.ArrayType):
        raise TypeError("Saltelli Sobol builders require a Quantity with an ArrayType Saltelli axis")
    if quantity.qtype._shape[0] != schema.n_terms:
        raise ValueError(
            "Expected {} Saltelli terms, got {}".format(schema.n_terms, quantity.qtype._shape[0])
        )


# AGENT: QArray should be used to this
# Resolved: Saltelli term groups are built with Quantity.QArray in estimate_sobol_indices.
@attr.s(auto_attribs=True, frozen=True)
class SobolIndexEstimate:
    """
    Sobol numerator/denominator estimates with cached derived indices.
    """

    a: Quantity
    b: Quantity
    ab: Quantity
    ba: Quantity

    @property
    def mean_quantity(self):
        return 0.5 * (self.a + self.b)

    @property
    def denominator_quantity(self):
        centered_a = self.a - self.mean_value
        centered_b = self.b - self.mean_value
        return 0.5 * (centered_a * centered_a + centered_b * centered_b)

    @property
    def first_order_numerator_quantity(self):
        return (self.ab - self.a) * self.b

    @property
    def total_order_numerator_quantity(self):
        diff = self.ab - self.a
        return diff * diff * 0.5

    @property
    def second_order_numerator_quantity(self):
        first_terms = (self.ab - self.a) * self.b
        numerators = [
            self.ba[i_param] * self.ab[j_param]
            - self.a * self.b
            - first_terms[i_param]
            - first_terms[j_param]
            for i_param, j_param in self.second_order_pairs
        ]
        return Quantity.QArray(numerators)

    @cached_property
    def mean_mlmc(self):
        return estimate_mean(self.mean_quantity)

    @cached_property
    def mean_value(self):
        return self.mean_mlmc.mean

    @cached_property
    def second_order_pairs(self):
        n_parameters = self.ab.qtype._shape[0]
        return [
            (i_param, j_param)
            for i_param in range(n_parameters)
            for j_param in range(i_param + 1, n_parameters)
        ]

    @cached_property
    def denominator_mlmc(self):
        return estimate_mean(self.denominator_quantity)

    @cached_property
    def first_order_numerator_mlmc(self):
        return estimate_mean(self.first_order_numerator_quantity)

    @cached_property
    def total_order_numerator_mlmc(self):
        return estimate_mean(self.total_order_numerator_quantity)

    @cached_property
    def second_order_numerator_mlmc(self):
        return estimate_mean(self.second_order_numerator_quantity)

    @cached_property
    def first_order(self):
        return self.first_order_numerator_mlmc.mean / self.denominator_mlmc.mean

    @cached_property
    def total_order(self):
        return self.total_order_numerator_mlmc.mean / self.denominator_mlmc.mean

    @cached_property
    def second_order(self):
        """
        Return dict mapping: (i,j) -> second order sobol index
        """
        return {
            pair: self.second_order_numerator_mlmc.mean[i_pair] / self.denominator_mlmc.mean
            for i_pair, pair in enumerate(self.second_order_pairs)
        }

    @cached_property
    def denominator_std(self):
        return np.sqrt(self.denominator_mlmc.var)

    @cached_property
    def first_order_numerator_std(self):
        return np.sqrt(self.first_order_numerator_mlmc.var)

    @cached_property
    def total_order_numerator_std(self):
        return np.sqrt(self.total_order_numerator_mlmc.var)

    @cached_property
    def second_order_numerator_std(self):
        return np.sqrt(self.second_order_numerator_mlmc.var)

    @cached_property
    def first_order_std(self):
        return self.first_order_numerator_std / np.abs(self.denominator_mlmc.mean)

    @cached_property
    def total_order_std(self):
        return self.total_order_numerator_std / np.abs(self.denominator_mlmc.mean)

    @cached_property
    def second_order_std(self):
        second_std = self.second_order_numerator_std / np.abs(self.denominator_mlmc.mean)
        return {
            pair: second_std[i_pair]
            for i_pair, pair in enumerate(self.second_order_pairs)
        }


def estimate_sobol_indices(saltelli_quantity: Quantity, schema: SaltelliSchema) -> SobolIndexEstimate:
    """
    Estimate Sobol indices as ratios of MLMC-estimated numerator and denominator means.
    """
    _require_saltelli_array(saltelli_quantity, schema)
    return SobolIndexEstimate(
        a=saltelli_quantity[schema.a, ...],
        b=saltelli_quantity[schema.b, ...],
        ab=Quantity.QArray([saltelli_quantity[i_term, ...] for i_term in schema.ab]),
        ba=Quantity.QArray([saltelli_quantity[i_term, ...] for i_term in schema.ba]),
    )
