import attr
import hashlib
import numpy as np
from typing import List, Dict, Any, Optional, Callable, Tuple
from mlmc.quantity.quantity_spec import QuantitySpec

SampleInput = Tuple[str, np.ndarray | float | int]



@attr.s(auto_attribs=True)
class LevelSimulation:
    """
    Class for passing simulation configuration and metadata for a given level between
    a Sampler and a SamplingPool.

    User shouldn't modify this class manually.
    """

    config_dict: Dict[Any, Any]
    # Level-specific simulation configuration dictionary.

    common_files: Optional[List[str]] = None
    # List of files in the level workspace to copy or symlink to the sample workspace.

    need_sample_workspace: bool = False
    # Whether the simulation requires an individual workspace for each sample.

    task_size: float = 0.0
    # Relative size (or computational cost) of the simulation task at this level.
    # When using PBS or SLURM, note that the job size is the sum of task_sizes.
    # If this sum exceeds 1.0, the job is queued and scheduled by the system.

    ### Internal attributes — users should not modify these ###
    _calculate: Optional[Callable] = None
    # Calculation method used internally by the sampler.
    # Signature: calculate(config_dict: dict, sample_input: Any) -> tuple[fine, coarse].
    # For ordinary simulations sample_input is the deterministic seed.

    _level_id: Optional[int] = None
    # Level identifier, set automatically by mlmc.sampler.Sampler.

    _result_format: Optional[List[QuantitySpec]] = None
    # Format specification for simulation results (defined by QuantitySpec instances).

    def prepare_samples(self, sample_ids):
        """
        Prepare scheduled work items on the master before pool submission.

        Signature: ``prepare_samples(sample_ids: list[str]) -> list``.
        The default keeps backward-compatible seed-based sampling by returning
        ``(sample_id, seed)`` tuples. Simulations with externally planned inputs
        can return ``(sample_id, input_vector)`` tuples; storage receives the
        complete scheduled work items so restarts can reuse planned inputs.
        """
        return [
            (sample_id, self.compute_seed(sample_id))
            for sample_id in sample_ids
        ]

    @staticmethod
    def compute_seed(sample_id: str) -> int:
        """
        Compute the deterministic default sample input for a sample id.
        """
        hash_val = hashlib.md5(sample_id.encode('ascii'))
        seed = np.frombuffer(hash_val.digest(), dtype='uint32')[0]
        return int(seed)
