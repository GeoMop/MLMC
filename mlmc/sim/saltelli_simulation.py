from typing import Callable, List

import numpy as np

from mlmc.level_simulation import LevelSimulation
from mlmc.quantity.sobol import SaltelliSchema
from mlmc.quantity.quantity_spec import QuantitySpec
from mlmc.sim.simulation import Simulation


class SaltelliSchemaSimulation(Simulation):
    """
    Simulation wrapper evaluating one full Saltelli row per MLMC sample.

    Parameters
    ----------
    forward_simulation : Simulation
        Wrapped forward simulation.  Its ``level_instance`` and ``calculate``
        methods are used for each Saltelli term.
    matrix_generator : callable
        Signature:
        ``matrix_generator(n_rows: int, n_parameters: int) -> matrix``.
        The returned matrix must have shape ``(n_rows, n_parameters)`` and
        values from interval ``[0, 1]``. It is called twice for each scheduled
        row block, once for matrix A and once for matrix B.
    n_parameters : int
        Number of uncertain input parameters.

    Notes
    -----
    This class customizes ``LevelSimulation.prepare_samples(sample_ids)`` to
    reserve row blocks on the master and submit work items shaped as
    ``(sample_id, saltelli_terms)``. Workers then receive those vectors as the
    second argument of the ordinary ``_calculate(config, sample_input)`` call.
    """

    def __init__(self, forward_simulation: Simulation, matrix_generator: Callable[[int, int], np.ndarray],
                 n_parameters: int):
        self.forward_simulation = forward_simulation
        self.matrix_generator = matrix_generator
        self.schema = SaltelliSchema.make(n_parameters=n_parameters)
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
            fine_result, coarse_result = self.forward_simulation.calculate(config_dict["forward_config"], input_vector)
            fine_results.append(np.asarray(fine_result).flatten())
            coarse_results.append(np.asarray(coarse_result).flatten())

        return np.asarray(fine_results).flatten(), np.asarray(coarse_results).flatten()

    def _generate_matrix(self, n_rows):
        n_rows = int(n_rows)
        matrix = np.asarray(self.matrix_generator(n_rows, self.schema.n_parameters), dtype=float)
        assert matrix.shape == (n_rows, self.schema.n_parameters)
        assert np.all((0.0 <= matrix) & (matrix <= 1.0))
        return matrix

    def _make_prepare_samples(self):
        def prepare(sample_ids):
            """
            Signature: prepare(sample_ids: list[str]) -> list[tuple[str, np.ndarray]].

            Reserve A/B rows for the scheduled batch and return full Saltelli
            term vectors together with their sample ids.

            """
            a_matrix = self._generate_matrix(len(sample_ids))
            b_matrix = self._generate_matrix(len(sample_ids))
            return [
                (sample_id, self.schema.terms(a_row, b_row))
                for sample_id, a_row, b_row in zip(sample_ids, a_matrix, b_matrix)
            ]
        return prepare
