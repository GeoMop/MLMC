import numpy as np

from mlmc.level_simulation import LevelSimulation
from mlmc.quantity.quantity_spec import QuantitySpec
from mlmc.sample_storage import Memory
from mlmc.sample_storage_hdf import SampleStorageHDF
from mlmc.sampler import Sampler
from mlmc.sampling_pool import OneProcessPool, SamplingPool
from mlmc.quantity.sobol import SaltelliSchema
from mlmc.sim.saltelli_simulation import SaltelliSchemaSimulation
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


class RecordingPool(SamplingPool):
    def __init__(self):
        self.scheduled = []

    def schedule_sample(self, sample_input, level_sim):
        self.scheduled.append((sample_input, level_sim))

    def have_permanent_samples(self, sample_inputs):
        return False

    def get_finished(self):
        return {}, {}, 0, []


def test_saltelli_schema_a_mask_and_terms():
    schema = SaltelliSchema.make(n_parameters=2)

    expected_mask = np.array([
        [True, True],
        [False, True],
        [True, False],
        [True, False],
        [False, True],
        [False, False],
    ])
    assert np.array_equal(schema.a_mask, expected_mask)
    labels = (
        ["A0"]
        + ["AB[{}]".format(i) for i in range(schema.n_parameters)]
        + ["BA[{}]".format(i) for i in range(schema.n_parameters)]
        + ["B0"]
    )
    assert labels == ["A0", "AB[0]", "AB[1]", "BA[0]", "BA[1]", "B0"]

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
    requested_sizes = []

    def matrix_generator(n_rows, n_parameters):
        requested_sizes.append((n_rows, n_parameters))
        return blocks.pop(0)

    simulation = SaltelliSchemaSimulation(
        forward_simulation=ForwardModelSimulation(),
        matrix_generator=matrix_generator,
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
    assert requested_sizes == [(1, 2), (1, 2)]


def test_saltelli_simulation_propagates_inputs_through_local_sampler():
    blocks = [
        np.array([[0.1, 0.2], [0.5, 0.6]]),
        np.array([[0.3, 0.4], [0.7, 0.8]]),
    ]
    requested_sizes = []

    def matrix_generator(n_rows, n_parameters):
        requested_sizes.append((n_rows, n_parameters))
        return blocks.pop(0)[:n_rows, :n_parameters]

    simulation = SaltelliSchemaSimulation(
        forward_simulation=ForwardModelSimulation(),
        matrix_generator=matrix_generator,
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

    assert requested_sizes == [(2, 2), (2, 2)]
    scheduled_samples = storage.load_scheduled_samples()[0]
    first_sample_id, first_sample_input = scheduled_samples[0]
    second_sample_id, second_sample_input = scheduled_samples[1]
    assert first_sample_id == "L00_S0000000"
    assert second_sample_id == "L00_S0000001"
    assert np.allclose(first_sample_input, np.array([
        [0.1, 0.2],
        [0.3, 0.2],
        [0.1, 0.4],
        [0.1, 0.4],
        [0.3, 0.2],
        [0.3, 0.4],
    ]))
    assert np.allclose(second_sample_input, np.array([
        [0.5, 0.6],
        [0.7, 0.6],
        [0.5, 0.8],
        [0.5, 0.8],
        [0.7, 0.6],
        [0.7, 0.8],
    ]))
    assert len(storage._results[0]) == 2
    first_fine = storage._results[0][0, 0, :]
    second_fine = storage._results[0][1, 0, :]
    assert np.allclose(first_fine, np.array([1.3, 3.3, 1.5, 1.5, 3.3, 3.5]))
    assert np.allclose(second_fine, np.array([5.7, 7.7, 5.9, 5.9, 7.7, 7.9]))


def test_renew_failed_samples_uses_stored_sample_input(tmp_path):
    sample_input = np.array([[0.1, 0.2], [0.3, 0.4]])
    storage = SampleStorageHDF(file_path=str(tmp_path / "mlmc.hdf5"))
    pool = RecordingPool()
    sampler = Sampler(
        sample_storage=storage,
        sampling_pool=pool,
        sim_factory=ForwardModelSimulation(),
        level_parameters=[[0.1]],
    )
    level_sim = sampler._level_sim_objects[0]

    def prepare_samples(_sample_ids):
        raise AssertionError("renew_failed_samples must not regenerate sample inputs")

    level_sim.prepare_samples = prepare_samples
    storage.save_scheduled_samples(0, [("L00_S0000000", sample_input)])
    storage.save_samples({}, {0: [("L00_S0000000", "failed")]})

    sampler.renew_failed_samples()

    assert len(pool.scheduled) == 1
    renewed_sample, renewed_level_sim = pool.scheduled[0]
    renewed_sample_id, renewed_input = renewed_sample
    assert renewed_sample_id == "L00_S0000000"
    assert np.allclose(renewed_input, sample_input)
    assert renewed_level_sim is level_sim
    assert storage.failed_samples() == {"0": []}
