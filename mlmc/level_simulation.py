import attr
from typing import List, Dict, Any
from mlmc.quantity.quantity_spec import QuantitySpec


@attr.s(auto_attribs=True)
class LevelSimulation:
    """
    This class is used to pass simulation data at a given level between a Sampler and a SamplingPool
    User shouldn't change this class
    """
    config_dict: Dict[Any, Any]
    # Calculate configuration.

    common_files: List[str] = None
    # List of files in the level workspace to copy/symlink to the sample workspace.

    need_sample_workspace: bool = False
    # If the simulation needs sample workspace at all.

    task_size: int = 0
    # Relative size of the simulation at this level.
    # When using PBS, keep in mind that the pbs job size is the sum of task_sizes, and if this sum is above 1,
    # the job is scheduled and PBS scheduler manages it

    ### User shouldn't modify the following attributes ###
    _calculate: Any = None
    # Calculate method

    _level_id: int = None
    # Level id is set by mlmc.sampler.Sampler. It is internal variable and user shouldn't change it.

    _result_format: List[QuantitySpec] = None
    # Simulation result format
