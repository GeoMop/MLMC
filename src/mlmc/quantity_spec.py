import attr
import numpy as np
from typing import List, Tuple, Union


@attr.s(auto_attribs=True)
class QuantitySpec:
    name: str
    unit: str
    shape: Tuple[int, int]
    times: List[float]
    locations: Union[List[str], List[Tuple[float, float, float]]]

    def __eq__(self, other):
        if (self.name, self.unit) == (other.name, other.unit) and np.allclose(self.shape, other.shape) and np.allclose(self.times, other.times)\
                and not (set(self.locations) - set(other.locations)):
            return True
        return False


@attr.s(auto_attribs=True, frozen=True)
class ChunkSpec:
    level_id: int
    # Level identifier
    chunk_id: int = 0
    # Chunk identifier
    n_samples: int = None
    # Number of samples which we want to retrieve
    chunk_size: int = 512000000
    # Chunk size in bytes in decimal, determines number of samples in chunk

