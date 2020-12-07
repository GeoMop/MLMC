import attr
from typing import List, Tuple, Union


@attr.s(auto_attribs=True)
class QuantitySpec:
    name: str
    unit: str
    shape: Tuple[int, int]
    times: List[float]
    locations: Union[List[str], List[Tuple[float, float, float]]]


@attr.s(auto_attribs=True)
class ChunkSpec:
    level_id: int
    # Level identifier
    chunk_id: int = 0
    # Chunk identifier
    n_samples: int = None
    # Number of samples which we want to retrieve
    chunk_size: int = 512000000
    # Chunk size in bytes in decimal, determines number of samples in chunk
