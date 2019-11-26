from dataclasses import dataclass
from src.mlmc.new_simulation import Simulation


@dataclass
class MLView:
    n_levels: int
    sim_factory: Simulation
    step_range: [float, float]
