import numpy as np

from mlmc.level_simulation import LevelSimulation
from mlmc.quantity.quantity_spec import QuantitySpec
from mlmc.sample_storage import Memory
from mlmc.sampler import Sampler
from mlmc.sampling_pool import OneProcessPool, SamplingPool
from mlmc.sim.saltelli_simulation import SaltelliRowProvider, SaltelliSchema, SaltelliSchemaSimulation
from mlmc.sim.simulation import Simulation


class ForwardModelSimulation(Simulation):
    need_workspace = False

    def level_instance(self, fine_level_params, coarse_level_params):
        config = {
            "fine": {"step": fine_level_params[0]},
            "coarse": {"step": coarse_level_params[0]},
        }
        return LevelSimulation(config_dict=config, task_size=0.0)

    def result_format(self):
        return [QuantitySpec(name="value", unit="1", shape=(1,), times=[0], locations=["0"])]

    @staticmethod
    def calculate(config_dict, input_vector):
        input_vector = np.asarray(input_vector)
        value = 10 * input_vector[0] + input_vector[1]
        fine = np.array([value + config_dict["fine"]["step"]])
        if config_dict["coarse"]["step"] == 0:
            coarse = np.array([0.0])
        else:
            coarse = np.array([value + config_dict["coarse"]["step"]])
        return fine, coarse


def test_saltelli_schema_a_mask_and_terms():
    schema = SaltelliSchema(n_parameters=2)

    expected_mask = np.array([
        [True, True],
        [False, True],
        [True, False],
        [True, False],
        [False, True],
        [False, False],
    ])
    assert np.array_equal(schema.a_mask, expected_mask)
    assert schema.labels == ["A0", "AB[0]", "AB[1]", "BA[0]", "BA[1]", "B0"]

    terms = schema.terms(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    expected_terms = np.array([
        [1.0, 2.0],
        [3.0, 2.0],
        [1.0, 4.0],
        [1.0, 4.0],
        [3.0, 2.0],
        [3.0, 4.0],
    ])
    assert np.array_equal(terms, expected_terms)


def test_saltelli_simulation_forms_forward_evaluation_scenarios():
    blocks = [
        np.array([[0.1, 0.2]]),
        np.array([[0.3, 0.4]]),
    ]
    provider = SaltelliRowProvider(lambda n, n_params: blocks.pop(0))
    simulation = SaltelliSchemaSimulation(
        forward_simulation=ForwardModelSimulation(),
        row_provider=provider,
        n_parameters=2,
    )
    level_sim = simulation.level_instance([0.1], [0.01])
    level_sim._calculate = simulation.calculate
    level_sim._result_format = simulation.result_format
    level_sim._level_id = 0
    scheduled_sample = level_sim.prepare_samples(["L00_S0000000"])[0]

    sample_id, result, err_msg, _ = SamplingPool.calculate_sample(scheduled_sample, level_sim)

    assert sample_id == "L00_S0000000"
    assert err_msg == ""
    expected_values = np.array([1.2, 3.2, 1.4, 1.4, 3.2, 3.4])
    assert np.allclose(result[0], expected_values + 0.1)
    assert np.allclose(result[1], expected_values + 0.01)
    assert provider.requested_sizes == [(1, 2), (1, 2)]


def test_saltelli_simulation_propagates_inputs_through_local_sampler():
    blocks = [
        np.array([[0.1, 0.2], [0.5, 0.6]]),
        np.array([[0.3, 0.4], [0.7, 0.8]]),
    ]
    provider = SaltelliRowProvider(lambda n, n_params: blocks.pop(0)[:n, :n_params])
    simulation = SaltelliSchemaSimulation(
        forward_simulation=ForwardModelSimulation(),
        row_provider=provider,
        n_parameters=2,
    )
    storage = Memory()
    sampler = Sampler(
        sample_storage=storage,
        sampling_pool=OneProcessPool(),
        sim_factory=simulation,
        level_parameters=[[0.1]],
    )

    sampler.set_initial_n_samples([2])
    sampler.schedule_samples()
    sampler.ask_sampling_pool_for_samples()

    assert provider.requested_sizes == [(2, 2), (2, 2)]
    assert len(storage._results[0]) == 2
    first_fine = storage._results[0][0, 0, :]
    second_fine = storage._results[0][1, 0, :]
    assert np.allclose(first_fine, np.array([1.3, 3.3, 1.5, 1.5, 3.3, 3.5]))
    assert np.allclose(second_fine, np.array([5.7, 7.7, 5.9, 5.9, 7.7, 7.9]))
