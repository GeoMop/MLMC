import os
import shutil
import numpy as np
import random
import pytest
from mlmc.sample_storage import Memory
from mlmc.sample_storage_hdf import SampleStorageHDF
from mlmc.simulation import QuantitySpec


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
    res_lenght = 3

    successful_samples = {}
    failed_samples = {}
    n_ops = {}
    # It is possible to save arbitrary format (there is no dependency on data)
    format = [QuantitySpec(name="length", unit="m", shape=(2, res_lenght - 2), times=[1, 2, 3], locations=['10', '20']),
              QuantitySpec(name="width", unit="mm", shape=(2, res_lenght - 2), times=[1, 2, 3], locations=['30', '40'])]

    storage.save_global_data(step_range=np.ones(n_levels), result_format=format)

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
    storage.save_n_ops(n_ops)

    scheduled = storage.load_scheduled_samples()

    assert len(scheduled) == n_levels
    for _, l_sch in scheduled.items():
        assert len(l_sch) == n_successful + n_failed

    results = np.array(storage.sample_pairs())

    assert len(results) == n_levels
    for level_res in results:
        assert level_res.shape[1] == n_successful
        assert level_res.shape[0] == res_lenght
        assert np.allclose(level_res[:, :, 0], 1)
        assert np.allclose(level_res[:, :, 1], 0)

    n_ops = storage.get_n_ops()
    assert len(n_ops) == n_levels

    loaded_format = storage.load_result_format()

    assert len(format) == len(loaded_format)
    for f1, f2 in zip(format, loaded_format):
        assert f1.name == f2.name
        assert f1.unit == f2.unit

    n_finished = storage.n_finished()

    assert len(n_finished) == n_levels
    assert np.allclose(n_finished, n_successful + n_failed)
