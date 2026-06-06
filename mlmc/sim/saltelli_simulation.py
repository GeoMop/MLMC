import copy
from typing import Callable, List, Optional

import attr
import numpy as np

from mlmc.level_simulation import LevelSimulation
from mlmc.quantity.quantity_spec import QuantitySpec
from mlmc.sim.simulation import Simulation


@attr.s(auto_attribs=True)
class SaltelliSchema:
    """
    Indexing schema for Saltelli A, AB_i, BA_i, B terms.

    Parameters
    ----------
    n_parameters : int
        Number of uncertain input parameters.

    Notes
    -----
    The primary representation is ``a_mask``.  For a term ``i_term`` and
    parameter ``i_param``, ``a_mask[i_term, i_param] == True`` means the
    Saltelli term uses the value from matrix A; ``False`` means it uses matrix B.
    The term order is:
    ``A0, AB[0], ..., AB[N-1], BA[0], ..., BA[N-1], B0``.
    """

    n_parameters: int

    @property
    def n_terms(self):
        return 2 * (self.n_parameters + 1)

    @property
    def a_mask(self):
        mask = np.zeros((self.n_terms, self.n_parameters), dtype=bool)
        mask[0, :] = True
        for i_param in range(self.n_parameters):
            mask[1 + i_param, :] = True
            mask[1 + i_param, i_param] = False
            mask[1 + self.n_parameters + i_param, i_param] = True
        return mask

    @property
    def labels(self):
        return (
            ["A0"]
            + ["AB[{}]".format(i) for i in range(self.n_parameters)]
            + ["BA[{}]".format(i) for i in range(self.n_parameters)]
            + ["B0"]
        )

    def terms(self, a_row, b_row):
        """
        Build all Saltelli term input vectors for one A/B row pair.

        Parameters
        ----------
        a_row, b_row : array-like, shape (n_parameters,)
            Parameter vectors from matrices A and B.

        Returns
        -------
        np.ndarray
            Array of shape ``(2 * (n_parameters + 1), n_parameters)`` in the
            schema term order.
        """
        a_row = np.asarray(a_row)
        b_row = np.asarray(b_row)
        assert a_row.shape == b_row.shape == (self.n_parameters,)
        return np.where(self.a_mask, a_row[None, :], b_row[None, :])

# AGENT: I already requested to replace this class by simple function
class SaltelliRowProvider:
    """
    Master-side adapter around an external Saltelli matrix block generator.

    Parameters
    ----------
    block_generator : callable
        Signature:
        ``block_generator(n_rows: int, n_parameters: int) -> matrix``.
        The returned matrix must be array-like with shape
        ``(n_rows, n_parameters)`` and values from the interval ``[0, 1]``.
        The callable owns any external sequence state, for example OpenTurns
        or QMC state.
    """

    def __init__(self, block_generator: Callable[[int, int], np.ndarray]):
        self._block_generator = block_generator
        self.requested_sizes = []

    def reserve(self, n_rows, n_parameters):
        """
        Reserve a block of A/B rows on the master process.

        Parameters
        ----------
        n_rows : int
            Number of Saltelli rows requested by one scheduled level batch.
        n_parameters : int
            Number of uncertain input parameters, i.e. matrix columns.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Matrices ``A`` and ``B`` with shape ``(n_rows, n_parameters)``.
        """
        return (
            self._generate_matrix(n_rows, n_parameters),
            self._generate_matrix(n_rows, n_parameters),
        )

    def _generate_matrix(self, n_rows, n_parameters):
        n_rows = int(n_rows)
        n_parameters = int(n_parameters)
        self.requested_sizes.append((n_rows, n_parameters))
        matrix = np.asarray(self._block_generator(n_rows, n_parameters), dtype=float)
        assert matrix.shape == (n_rows, n_parameters)
        assert np.all((0.0 <= matrix) & (matrix <= 1.0))
        return matrix


class SaltelliSchemaSimulation(Simulation):
    """
    Simulation wrapper evaluating one full Saltelli row per MLMC sample.

    Parameters
    ----------
    forward_simulation : Simulation
        Wrapped forward simulation.  Its ``level_instance`` and ``calculate``
        methods are used for each Saltelli term.
    row_provider : SaltelliRowProvider
        Master-side provider of A/B parameter rows.
    parameter_applier : callable
        Optional compatibility adapter with signature:
        ``parameter_applier(config_dict: dict, input_vector: np.ndarray) -> dict``.
        It receives a deep copy of the wrapped level configuration and one
        parameter vector.  It must return the config passed to
        ``forward_simulation.calculate`` for that forward evaluation.
        If omitted, the wrapped forward simulation is called as
        ``calculate(config_dict, input_vector)``.
    n_parameters : int
        Number of uncertain input parameters.

    Notes
    -----
    This class customizes ``LevelSimulation.prepare_samples(sample_ids)`` to
    reserve row blocks on the master and submit work items shaped as
    ``(sample_id, saltelli_terms)``. Workers then receive those vectors as the
    second argument of the ordinary ``_calculate(config, sample_input)`` call.
    """

    def __init__(self, forward_simulation: Simulation, row_provider: SaltelliRowProvider,
                 n_parameters: int, parameter_applier: Optional[Callable] = None):
        # Possibly the forward simulation could also be just callable not Simulation child
        # (that doesn't exclude a class method)
        self.forward_simulation = forward_simulation
        self.row_provider = row_provider  # pass a callable with defined signature here.
        self.parameter_applier = parameter_applier # this should be part of the SaltelliSimulation not injected
        self.schema = SaltelliSchema(n_parameters=n_parameters)
        self.need_workspace = getattr(forward_simulation, "need_workspace", False)

    def level_instance(self, fine_level_params: List[float], coarse_level_params: List[float]) -> LevelSimulation:
        """
        Create a level simulation that wraps the forward simulation's level.

        The returned object has a master-side ``prepare_samples`` method that
        returns scheduled work items with arrays of shape
        ``(n_saltelli_terms, n_parameters)``.
        """
        forward_level_sim = self.forward_simulation.level_instance(fine_level_params, coarse_level_params)
        level_sim = LevelSimulation(
            config_dict={"forward_config": forward_level_sim.config_dict},
            common_files=forward_level_sim.common_files,
            need_sample_workspace=forward_level_sim.need_sample_workspace,
            task_size=forward_level_sim.task_size
        )
        level_sim.prepare_samples = self._make_prepare_samples()
        return level_sim

    def result_format(self) -> List[QuantitySpec]:
        """
        Return the wrapped result format with a leading Saltelli term axis.

        For each wrapped ``QuantitySpec`` with ``shape=s``, the returned spec has
        ``shape=(n_saltelli_terms, *s)`` and preserves times and locations.
        """
        result_format = []
        for spec in self.forward_simulation.result_format():
            result_format.append(
                QuantitySpec(
                    name=spec.name,
                    unit=spec.unit,
                    shape=(self.schema.n_terms, *tuple(spec.shape)),
                    times=spec.times,
                    locations=spec.locations
                )
            )
        return result_format

    def calculate(self, config_dict, sample_input):
        """
        Evaluate all Saltelli terms for one MLMC sample.

        Parameters
        ----------
        config_dict : dict
            Contains ``forward_config`` produced by the wrapped level instance.
        sample_input : np.ndarray
            Planned input vectors with shape
            ``(2 * (n_parameters + 1), n_parameters)``.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Flattened fine and coarse arrays.  Each is formed by evaluating the
            wrapped model for all Saltelli term vectors in schema order.
        """
        if sample_input is None:
            raise ValueError("Missing planned Saltelli sample input")

        fine_results = []
        coarse_results = []
        for input_vector in sample_input:
            if self.parameter_applier is None:
                fine_result, coarse_result = self.forward_simulation.calculate(
                    config_dict["forward_config"], input_vector
                )
            else:
                forward_config = self.parameter_applier(copy.deepcopy(config_dict["forward_config"]), input_vector)
                fine_result, coarse_result = self.forward_simulation.calculate(forward_config, input_vector)
            fine_results.append(np.asarray(fine_result).flatten())
            coarse_results.append(np.asarray(coarse_result).flatten())

        return np.asarray(fine_results).flatten(), np.asarray(coarse_results).flatten()

    def _make_prepare_samples(self):
        def prepare(sample_ids):
            """
            Signature: prepare(sample_ids: list[str]) -> list[tuple[str, np.ndarray]].

            Reserve A/B rows for the scheduled batch and return full Saltelli
            term vectors together with their sample ids.
            """
            a_matrix, b_matrix = self.row_provider.reserve(len(sample_ids), self.schema.n_parameters)
            return [
                (sample_id, self.schema.terms(a_row, b_row))
                for sample_id, a_row, b_row in zip(sample_ids, a_matrix, b_matrix)
            ]
        return prepare
