import attr
import numpy as np
from typing import List, Tuple, Union


@attr.s(auto_attribs=True, eq=False)
class QuantitySpec:
    name: str
    unit: str
    shape: Tuple[int, int]
    times: List[float]
    locations: Union[List[str], List[Tuple[float, float, float]]]

    # Note: auto generated eq raises ValueError
    def __eq__(self, other):
        if (self.name, self.unit) == (other.name, other.unit) \
                and np.array_equal(self.shape, other.shape)\
                and np.array_equal(self.times, other.times)\
                and not (set(self.locations) - set(other.locations)):
            return True
        return False


@attr.s(auto_attribs=True)
class ChunkSpec:
    chunk_id: int = None
    chunk_slice: slice = None
    level_id: int = None
