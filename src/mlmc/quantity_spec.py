import attr
from typing import List, Tuple, Union


@attr.s(auto_attribs=True)
class QuantitySpec:
    name: str
    unit: str
    shape: Tuple[int, int]
    times: List[float]
    locations: Union[List[str], List[Tuple[float, float, float]]]


@attr.s(auto_attribs=True, eq=False)  # eq=False allows custom __hash__ and __eq__
class ChunkSpec:
    level_id: int
    # Level identifier
    chunk_id: int = 0
    # Chunk identifier
    n_samples: int = None
    # Number of samples which we want to retrieve
    chunk_size: int = 512000000
    # Chunk size in bytes in decimal, determines number of samples in chunk

    def __hash__(self):
        return hash((self.level_id, self.chunk_id, self.n_samples, self.chunk_size))

    def __eq__(self, other):
        return (self.level_id, self.chunk_id, self.n_samples, self.chunk_size) == \
               (other.level_id, other.chunk_id, other.n_samples, other.chunk_size)


