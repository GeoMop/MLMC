import attr
from typing import List, Dict, Any


@attr.s(auto_attribs=True)
class LevelSimulation:
    """
    This class is used to pass simulation data at a given level between a Sampler and a SamplingPool
    """
    config_dict: Dict[Any, Any]
    # Calculate configuration.

    calculate: Any = None
    # Calculate method

    common_files: List[str] = None
    # List of files in the level workspace to copy/symlink to the sample workspace.

    level_id: int = None
    # Level id is set by mlmc.sampler.Sampler. It is internal variable and user shouldn't change it.

    need_sample_workspace: bool = False
    # If the simulation needs sample workspace at all.

    task_size: int = 0
    # Relative size of the simulation at this level.

    sample_workspace: str = None

