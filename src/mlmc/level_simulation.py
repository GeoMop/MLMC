import attr
from typing import List, Dict, Any


@attr.s(auto_attribs=True)
class LevelSimulation:
    # Calculate configuration.
    config_dict: Dict[Any, Any]

    # Calculate method
    calculate: Any = None

    # List of files in the level workspace to copy/symlink to the sample workspace.
    common_files: List[str] = None

    # Set out of the Simulation by the Sampler. Necessary in the sampling pool.
    level_id: int = None

    # If the simulation needs sample workspace at all.
    need_sample_workspace: bool = False

    sample_workspace: str = None

    # Relative size of the simulation at this level.
    task_size: int = 0
