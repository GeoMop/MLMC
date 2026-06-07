import traceback
from typing import Any, Dict, Iterable, Optional, Tuple

from mlmc.sampling_pool import OneProcessPool, SamplingPool
from mlmc.level_simulation import LevelSimulation


SampleInput = Tuple[str, Any]


def import_distributed() -> Any:
    """
    Import and return the optional ``distributed`` module.
    """
    try:
        import distributed
    except ImportError as exc:
        raise ImportError(
            "SamplingPoolDask requires the optional Dask dependency. "
            "Install MLMC with the 'dask' extra or install 'dask' and 'distributed'."
        ) from exc
    return distributed


class SamplingPoolDask(OneProcessPool):
    """
    Dask-backed sampling pool.

    The caller owns the Dask client and passes it to the constructor. Samples are
    submitted one by one so the existing MLMC Sampler can adapt target sample
    counts while older futures are still running. Scheduled samples are
    propagated as ``(sample_id, sample_input)`` tuples produced by
    ``LevelSimulation.prepare_samples()``.
    """

    FUTURE_KEY_PREFIX = "mlmc-sample"

    def __init__(self, client: Any, work_dir: Optional[str] = None, debug: bool = False,
                 clean: bool = True, submit_kwargs: Optional[Dict[str, Any]] = None):
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
            If False, preserves an existing output directory on construction;
            kept for constructor compatibility with previous Dask pool versions.
        submit_kwargs : dict, optional
            Extra keyword arguments passed to client.submit().
        """
        super().__init__(work_dir=work_dir, debug=debug or not clean)
        self._distributed = import_distributed()
        self._client = client
        self._submit_kwargs = {} if submit_kwargs is None else dict(submit_kwargs)
        self._sample_to_future: Dict[str, Tuple[Any, LevelSimulation]] = {}

    def schedule_sample(self, sample_input: SampleInput, level_sim: LevelSimulation) -> None:
        """
        Submit one sample to Dask.

        Parameters
        ----------
        sample_input
            Tuple ``(sample_id, input_value)`` prepared on the master.
        level_sim
            Level simulation containing config, result format, and calculate
            callable.

        Notes
        -----
        Dask task keys are deterministic in ``sample_id``. The submitted worker
        task receives ``sample_input`` unchanged.
        """
        sample_id, _input_value = sample_input
        if sample_id in self._sample_to_future:
            return

        future = self._client.submit(
            SamplingPool.calculate_sample,
            sample_input,
            level_sim,
            self._output_dir,
            key=self._future_key(sample_id),
            pure=True,
            **self._submit_kwargs
        )
        self._distributed.fire_and_forget(future)
        self._sample_to_future[sample_id] = (future, level_sim)
        self._n_running += 1

    def have_permanent_samples(self, sample_ids: Iterable[Any]) -> bool:
        """
        Return whether the Dask pool has permanent samples to reconnect.

        Dask tasks are not PBS-like durable jobs in this implementation. If the
        Dask scheduler/workers are stopped with the master, unfinished sample
        ids remain in storage and must be scheduled again by a new sampler run.
        """
        return False

    def get_finished(self) -> Tuple[Dict[int, list], Dict[int, list], int, list]:
        """
        Collect completed Dask futures without blocking.

        Returns the same tuple shape as ``OneProcessPool.get_finished()``:
        successful samples, failed samples, running count, and runtime stats.
        """
        completed_items = [
            (sample_id, future, level_sim)
            for sample_id, (future, level_sim) in list(self._sample_to_future.items())
            if self._future_done(future)
        ]

        for sample_id, future, level_sim in completed_items:
            self._sample_to_future.pop(sample_id, None)
            result = self._future_result(future, sample_id)
            self._process_result(*result, level_sim)
            future.release()

        return super().get_finished()

    @classmethod
    def _future_key(cls, sample_id: str) -> str:
        """
        Return a deterministic Dask task key for one MLMC sample id.
        """
        return "{}-{}".format(cls.FUTURE_KEY_PREFIX, sample_id)

    @staticmethod
    def _future_done(future: Any) -> bool:
        """
        Return whether a Dask future is finished or failed.
        """
        if hasattr(future, "done"):
            return future.done()
        return getattr(future, "status", None) in {"finished", "error"}

    @staticmethod
    def _future_result(future: Any, sample_id: str) -> Tuple[str, Any, str, float]:
        """
        Return a worker result tuple, converting escaped Dask errors to failures.
        """
        try:
            return future.result()
        except Exception:
            err_msg = traceback.format_exc()
            return sample_id, (None, None), err_msg, 0.0
