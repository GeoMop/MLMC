import os
import pickle
import traceback

from mlmc.sampling_pool import OneProcessPool, SamplingPool

try:
    from distributed import fire_and_forget
except ImportError as exc:
    raise ImportError(
        "SamplingPoolDask requires the optional Dask dependency. "
        "Install MLMC with the 'dask' extra or install 'dask' and 'distributed'."
    ) from exc


class SamplingPoolDask(OneProcessPool):
    """
    Dask-backed sampling pool.

    The caller owns the Dask client and passes it to the constructor. Samples are
    submitted one by one so the existing MLMC Sampler can adapt target sample
    counts while older futures are still running.
    """

    FUTURE_KEY_PREFIX = "mlmc-sample"
    LEVEL_SIM_CONFIG = "level_{}_simulation_config"

    def __init__(self, client, work_dir=None, debug=False, clean=True, submit_kwargs=None):
        """
        Initialize the pool with an existing Dask client.

        Parameters
        ----------
        client
            dask.distributed.Client instance.
        work_dir : str, optional
            Working directory for sample output.
        debug : bool, default=False
            If True, keeps sample directories.
        clean : bool, default=True
            If False, preserves an existing output directory on construction.
            Use this when restarting unfinished workspace samples.
        submit_kwargs : dict, optional
            Extra keyword arguments passed to client.submit().
        """
        super().__init__(work_dir=work_dir, debug=debug or not clean)
        self._debug = debug
        self._client = client
        self._submit_kwargs = {} if submit_kwargs is None else dict(submit_kwargs)
        self._future_to_task = {}
        self._sample_to_future = {}

    def schedule_sample(self, sample_id, level_sim):
        """
        Submit one sample to Dask.

        Dask task keys are deterministic in the MLMC sample id. This gives a
        restarted master a chance to reconnect to scheduler-known tasks; if that
        is not possible, submitting the same sample id recomputes the same result
        because seeding is deterministic.
        """
        if sample_id in self._sample_to_future:
            return

        if self._output_dir is None and level_sim.need_sample_workspace:
            self._output_dir = os.getcwd()

        self._save_level_sim(level_sim)
        seed = SamplingPool.compute_seed(sample_id)
        future = self._client.submit(
            SamplingPool.calculate_sample,
            sample_id,
            level_sim,
            self._output_dir,
            seed,
            key=self._future_key(sample_id),
            pure=True,
            **self._submit_kwargs
        )
        fire_and_forget(future)
        self._future_to_task[future] = (sample_id, level_sim)
        self._sample_to_future[sample_id] = future
        self._n_running += 1

    def have_permanent_samples(self, sample_ids):
        """
        Reconnect or resubmit samples scheduled before a master restart.

        Dask does not provide PBS-like durable result files. Recovery is therefore
        based on workspace simulations, persisted per-level simulation metadata,
        and deterministic sample seeds. The restarted worker task receives a
        sample id and first re-enters the existing sample workspace.
        """
        if not sample_ids:
            return False
        if self._output_dir is None:
            return False

        for sample_id in sample_ids:
            self._submit_permanent_sample(sample_id)
        return True

    def get_finished(self):
        """
        Collect only futures that have already completed.
        """
        completed_futures = [
            future for future in list(self._future_to_task)
            if self._future_done(future)
        ]

        for future in completed_futures:
            sample_id, level_sim = self._future_to_task.pop(future)
            self._sample_to_future.pop(sample_id, None)
            result = self._future_result(future, sample_id)
            self._process_result(*result, level_sim)
            future.release()

        return super().get_finished()

    @classmethod
    def _future_key(cls, sample_id):
        return "{}-{}".format(cls.FUTURE_KEY_PREFIX, sample_id)

    @staticmethod
    def _future_done(future):
        if hasattr(future, "done"):
            return future.done()
        return getattr(future, "status", None) in {"finished", "error"}

    @staticmethod
    def _future_result(future, sample_id):
        try:
            return future.result()
        except Exception:
            err_msg = traceback.format_exc()
            return sample_id, (None, None), err_msg, 0.0

    @staticmethod
    def _level_id_from_sample_id(sample_id):
        try:
            return int(str(sample_id).split("_", 1)[0][1:])
        except (IndexError, TypeError, ValueError) as exc:
            raise ValueError("Cannot determine level id from sample id {!r}".format(sample_id)) from exc

    def _save_level_sim(self, level_sim):
        if self._output_dir is None or not level_sim.need_sample_workspace:
            return

        file_path = self._level_sim_file(level_sim._level_id)
        if os.path.exists(file_path):
            return

        with open(file_path, "wb") as level_sim_file:
            pickle.dump(level_sim, level_sim_file)

    def _submit_permanent_sample(self, sample_id):
        if sample_id in self._sample_to_future:
            return

        seed = SamplingPool.compute_seed(sample_id)
        level_sim = self._load_level_sim(sample_id)
        future = self._client.submit(
            _calculate_permanent_sample,
            sample_id,
            self._output_dir,
            seed,
            key=self._future_key(sample_id),
            pure=True,
            **self._submit_kwargs
        )
        fire_and_forget(future)
        self._future_to_task[future] = (sample_id, level_sim)
        self._sample_to_future[sample_id] = future
        self._n_running += 1

    def _load_level_sim(self, sample_id):
        level_id = self._level_id_from_sample_id(sample_id)
        file_path = self._level_sim_file(level_id)
        with open(file_path, "rb") as level_sim_file:
            return pickle.load(level_sim_file)

    def _level_sim_file(self, level_id):
        return os.path.join(self._output_dir, self.LEVEL_SIM_CONFIG.format(level_id))


def _calculate_permanent_sample(sample_id, output_dir, seed):
    level_id = SamplingPoolDask._level_id_from_sample_id(sample_id)
    level_sim_file = os.path.join(output_dir, SamplingPoolDask.LEVEL_SIM_CONFIG.format(level_id))
    with open(level_sim_file, "rb") as level_sim_config:
        level_sim = pickle.load(level_sim_config)

    return SamplingPool.calculate_sample(sample_id, level_sim, output_dir, seed)
