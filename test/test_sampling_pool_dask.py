import time

import numpy as np
import pytest

distributed = pytest.importorskip("distributed")
from distributed import Client, LocalCluster

from mlmc.level_simulation import LevelSimulation
from mlmc.quantity.quantity_spec import QuantitySpec
from mlmc.sample_storage import Memory
from mlmc.sampler import Sampler
from mlmc.sampling_pool_dask import SamplingPoolDask


class DaskSimulation:
    need_workspace = False

    def __init__(self, sleep=0.0, fail=False, need_workspace=False):
        self.sleep = sleep
        self.fail = fail
        self.need_workspace = need_workspace

    def level_instance(self, fine_level_params, coarse_level_params):
        config = {
            "fine": {"step": fine_level_params[0]},
            "coarse": {"step": coarse_level_params[0]},
            "sleep": self.sleep,
            "fail": self.fail,
            "res_format": self.result_format(),
        }
        return LevelSimulation(config_dict=config, task_size=0.0, need_sample_workspace=self.need_workspace)

    @staticmethod
    def calculate(config, seed):
        if config["sleep"]:
            time.sleep(config["sleep"])
        if config["fail"]:
            raise RuntimeError("sample failure")

        fine = np.array([float(seed % 1000) + config["fine"]["step"]])
        coarse = np.array([0.0 if config["coarse"]["step"] == 0 else float(seed % 1000)])
        return fine, coarse

    @staticmethod
    def result_format():
        return [QuantitySpec(name="value", unit="1", shape=(1,), times=[0], locations=["0"])]


class RestartMemory(Memory):
    def unfinished_ids(self):
        return [
            sample_id
            for sample_ids in self.load_scheduled_samples().values()
            for sample_id in sample_ids
        ]


@pytest.fixture
def dask_client():
    cluster = LocalCluster(n_workers=2, threads_per_worker=1, processes=False, dashboard_address=None)
    client = Client(cluster)
    try:
        yield client
    finally:
        client.close()
        cluster.close()


def make_level_simulation(simulation, level_id=0):
    level_sim = simulation.level_instance([0.1], [0])
    level_sim._calculate = simulation.calculate
    level_sim._result_format = simulation.result_format
    level_sim._level_id = level_id
    return level_sim


def collect_until_finished(sampler, timeout=5.0):
    deadline = time.monotonic() + timeout
    n_running = 1
    while n_running > 0 and time.monotonic() < deadline:
        n_running = sampler.ask_sampling_pool_for_samples(timeout=0.05)
        time.sleep(0.01)
    assert n_running == 0


def test_dask_pool_get_finished_is_non_blocking(dask_client):
    pool = SamplingPoolDask(client=dask_client)
    level_sim = make_level_simulation(DaskSimulation(sleep=0.5))

    pool.schedule_sample("L00_S0000000", level_sim)
    pool.schedule_sample("L00_S0000001", level_sim)

    start = time.monotonic()
    successful, failed, n_running, n_ops = pool.get_finished()
    elapsed = time.monotonic() - start

    assert elapsed < 0.3
    assert successful == {}
    assert failed == {}
    assert n_running == 2
    assert n_ops == []

    deadline = time.monotonic() + 5
    while n_running > 0 and time.monotonic() < deadline:
        successful, failed, n_running, n_ops = pool.get_finished()
        time.sleep(0.01)

    assert n_running == 0
    assert len(successful[0]) == 2
    assert failed == {}


def test_dask_pool_collects_sampler_samples(dask_client):
    storage = Memory()
    pool = SamplingPoolDask(client=dask_client)
    sampler = Sampler(
        sample_storage=storage,
        sampling_pool=pool,
        sim_factory=DaskSimulation(sleep=0.01),
        level_parameters=[[0.1], [0.01]],
    )

    sampler.set_initial_n_samples([4, 2])
    sampler.schedule_samples()
    collect_until_finished(sampler)

    assert np.all(storage.n_finished() == np.array([4, 2]))


def test_dask_pool_supports_adaptive_sample_addition(dask_client):
    storage = Memory()
    pool = SamplingPoolDask(client=dask_client)
    sampler = Sampler(
        sample_storage=storage,
        sampling_pool=pool,
        sim_factory=DaskSimulation(sleep=0.01),
        level_parameters=[[0.1], [0.01]],
    )

    sampler.set_initial_n_samples([2, 2])
    sampler.schedule_samples()

    sampler.process_adding_samples(np.array([6, 4]), sleep=0.01, add_coeff=0.5, timeout=0.05)

    assert np.all(np.array(sampler.l_scheduled_samples()) >= np.array([4, 3]))
    collect_until_finished(sampler)


def test_dask_pool_resubmits_unfinished_workspace_samples_on_sampler_restart(dask_client, tmp_path):
    storage = RestartMemory()
    storage.save_scheduled_samples(0, ["L00_S0000000", "L00_S0000001"])
    first_pool = SamplingPoolDask(client=dask_client, work_dir=str(tmp_path))
    first_pool._save_level_sim(make_level_simulation(DaskSimulation(sleep=0.01, need_workspace=True)))
    restarted_pool = SamplingPoolDask(client=dask_client, work_dir=str(tmp_path), clean=False)

    sampler = Sampler(
        sample_storage=storage,
        sampling_pool=restarted_pool,
        sim_factory=DaskSimulation(sleep=0.01, need_workspace=True),
        level_parameters=[[0.1]],
    )

    assert restarted_pool._n_running == 2
    collect_until_finished(sampler)

    assert np.all(storage.n_finished() == np.array([2]))


def test_dask_pool_reports_failed_samples(dask_client):
    storage = Memory()
    pool = SamplingPoolDask(client=dask_client)
    sampler = Sampler(
        sample_storage=storage,
        sampling_pool=pool,
        sim_factory=DaskSimulation(fail=True),
        level_parameters=[[0.1]],
    )

    sampler.set_initial_n_samples([3])
    sampler.schedule_samples()
    collect_until_finished(sampler)

    assert len(storage._failed[0]) == 3
    assert all("sample failure" in err_msg for _, err_msg in storage._failed[0])
