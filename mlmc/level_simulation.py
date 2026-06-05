import attr
from typing import List, Dict, Any, Optional, Callable
from mlmc.quantity.quantity_spec import QuantitySpec


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

    _level_id: Optional[int] = None
    # Level identifier, set automatically by mlmc.sampler.Sampler.

    _result_format: Optional[List[QuantitySpec]] = None
    # Format specification for simulation results (defined by QuantitySpec instances).
