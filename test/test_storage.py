import os
import shutil
import h5py
import numpy as np
import random
import pytest
from mlmc.sample_storage import Memory
from mlmc.sample_storage_hdf import SampleStorageHDF
from mlmc.sim.simulation import QuantitySpec


def add_samples(storage, n_levels, n_successful=5, n_failed=4, res_lenght=3):
    """
    Create samples for sample storage save
    :param storage: SampleStorage
    :param n_levels: number of levels
    :param n_successful: number of successful samples
    :param n_failed: number of failed samples
    :param res_lenght: result length
    :return: None
    """
    successful_samples = {}
    failed_samples = {}
    n_ops = {}
    # It is possible to save arbitrary format (there is no dependency on data)
    format_quant = [QuantitySpec(name="length", unit="m", shape=(2, res_lenght - 2), times=[1, 2, 3], locations=['10', '20']),
              QuantitySpec(name="width", unit="mm", shape=(2, res_lenght - 2), times=[1, 2, 3], locations=['30', '40'])]

    storage.save_global_data(level_parameters=np.ones(n_levels), result_format=format_quant)

    for l_id in range(n_levels):
        # Dict[level_id, List[Tuple[sample_id:str, Tuple[fine_result: ndarray, coarse_result: ndarray]]]]
        successful_samples[l_id] = [(str(sample_id), (np.ones((res_lenght,)), np.zeros((res_lenght,))))
                                    for sample_id in range(n_successful)]

        # Dict[level_id, List[Tuple[sample_id: str, error message: str]]]
        failed_samples[l_id] = [(str(sample_id), "error message") for sample_id in range(n_failed)]
        # Dict[level_id: int, List[total time: float, number of success samples: int]]
        n_ops[l_id] = [random.random(), n_successful]

        storage.save_scheduled_samples(l_id, samples=["S{:07d}".format(i) for i in range(n_successful + n_failed)])

    storage.save_samples(successful_samples, failed_samples)
    storage.save_n_ops(list(n_ops.items()))

    return format_quant

@pytest.mark.parametrize("storage", ['memory', 'hdf'])
@pytest.mark.parametrize("n_levels", [1, 2, 5])
def test_storage(storage, n_levels):
    if storage == 'memory':
        storage = Memory()
    elif storage == 'hdf':
        work_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '_test_tmp')
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)
        os.makedirs(work_dir)
        storage = SampleStorageHDF(file_path=os.path.join(work_dir, "mlmc.hdf5".format()))

    n_successful = 5
    n_failed = 4
    res_length = 3
    format_quant = add_samples(storage, n_levels, n_successful=n_successful, n_failed=n_failed, res_lenght=res_length)

    scheduled = storage.load_scheduled_samples()

    assert len(scheduled) == n_levels
    for _, l_sch in scheduled.items():
        assert len(l_sch) == n_successful + n_failed

    results = storage.sample_pairs()

    assert len(results) == n_levels
    for level_res in results:
        assert level_res.shape[1] == n_successful
        assert level_res.shape[0] == res_length
        assert np.allclose(level_res[:, :, 0], 1)

    n_ops = storage.get_n_ops()
    assert len(n_ops) == n_levels

    loaded_format = storage.load_result_format()

    assert len(format_quant) == len(loaded_format)
    for f1, f2 in zip(format_quant, loaded_format):
        assert f1.name == f2.name
        assert f1.unit == f2.unit

    n_finished = storage.n_finished()

    assert len(n_finished) == n_levels
    assert np.allclose(n_finished, n_successful + n_failed)


@pytest.mark.parametrize("storage_kind", ['memory', 'hdf'])
def test_storage_keeps_scheduled_sample_inputs(storage_kind, tmp_path):
    if storage_kind == 'memory':
        storage = Memory()
    elif storage_kind == 'hdf':
        storage = SampleStorageHDF(file_path=os.path.join(tmp_path, "mlmc.hdf5"))
        storage.save_global_data(
            level_parameters=np.ones(2),
            result_format=[QuantitySpec(name="value", unit="1", shape=(1,), times=[0], locations=["0"])],
        )

    scalar_scheduled_samples = [
        ("L00_S0000000", 123),
    ]
    array_scheduled_samples = [
        ("L01_S0000000", np.array([[1.0, 2.0], [3.0, 4.0]])),
    ]
    storage.save_scheduled_samples(0, scalar_scheduled_samples)
    storage.save_scheduled_samples(1, array_scheduled_samples)
    storage.save_samples({}, {0: [("L00_S0000000", "failed")]})

    loaded_scheduled_samples = storage.load_scheduled_samples()
    loaded_scalar_id, loaded_scalar_input = loaded_scheduled_samples[0][0]
    loaded_array_id, loaded_array_input = loaded_scheduled_samples[1][0]

    assert loaded_scalar_id == "L00_S0000000"
    assert np.isscalar(loaded_scalar_input)
    assert loaded_scalar_input == 123
    assert loaded_array_id == "L01_S0000000"
    assert np.allclose(loaded_array_input, np.array([[1.0, 2.0], [3.0, 4.0]]))

    if storage_kind == 'hdf':
        failed_sample_id, failed_sample_input = storage.failed_samples()["0"][0]
        assert failed_sample_id == "L00_S0000000"
        assert np.isscalar(failed_sample_input)
        assert failed_sample_input == 123
    else:
        assert storage.failed_samples() == {}


def test_hdf_append():
    work_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '_test_tmp')
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir)
    storage = SampleStorageHDF(file_path=os.path.join(work_dir, "mlmc.hdf5"))
    n_levels = 4
    format_quant = add_samples(storage, n_levels)
    results = storage.sample_pairs()

    storage = SampleStorageHDF(file_path=os.path.join(work_dir, "mlmc.hdf5"))
    loaded_results = storage.sample_pairs()

    assert len(results) == n_levels
    for l_id, (level_res, level_res_loaded) in enumerate(zip(results, loaded_results)):
        assert np.allclose(np.array(level_res)[:, :, 0], 1)
        assert np.allclose(np.array(level_res)[:, :, 0], np.array(level_res_loaded)[:, :, 0])
        if l_id > 0:
            assert np.allclose(np.array(level_res)[:, :, 1], 0)
            assert np.allclose(np.array(level_res)[:, :, 1], np.array(level_res_loaded)[:, :, 1])

    n_ops = storage.get_n_ops()
    assert len(n_ops) == n_levels

    loaded_format = storage.load_result_format()
    assert len(format_quant) == len(loaded_format)
    for f1, f2 in zip(format_quant, loaded_format):
        assert f1.name == f2.name
        assert f1.unit == f2.unit

    n_finished = storage.n_finished()
    assert len(n_finished) == n_levels


def test_hdf_heterogeneous_result_format():
    work_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '_test_tmp')
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir)
    file_path = os.path.join(work_dir, "mlmc.hdf5")

    format_quant = [
        QuantitySpec(name="length", unit="m", shape=(2, 1), times=[1], locations=['10']),
        QuantitySpec(name="width", unit="mm", shape=(2, 2), times=[1, 2, 3], locations=['30', '40']),
    ]

    storage = SampleStorageHDF(file_path=file_path)
    storage.save_global_data(level_parameters=np.ones(1), result_format=format_quant)

    with h5py.File(file_path, "r") as hdf_file:
        format_group = hdf_file["result_format"]
        assert sorted(format_group.keys()) == ["0000", "0001"]
        assert format_group["0000"].dtype != format_group["0001"].dtype

    loaded_format = storage.load_result_format()
    assert loaded_format == format_quant


#test_hdf_append()
