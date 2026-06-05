import attr
import numpy as np
from typing import List, Tuple, Union


@attr.s(auto_attribs=True, eq=False)
class QuantitySpec:
    """
    Specification of a physical quantity for simulation or data storage.

    :param name: Name of the quantity (e.g. 'pressure', 'velocity').
    :param unit: Unit of the quantity (e.g. 'm/s', 'Pa').
    :param shape: Tuple describing the shape of the data (e.g. (64, 64)).
    :param times: List of time points associated with this quantity.
    :param locations: List of either string-based identifiers or 3D coordinates
                      (x, y, z) where the quantity is defined.
    """

    name: str
    unit: str
    shape: Tuple[int, int]
    times: List[float]
    locations: Union[List[str], List[Tuple[float, float, float]]]

    def __eq__(self, other):
        """
        Compare two QuantitySpec instances for equality.

        :param other: Another QuantitySpec instance to compare with.
        :return: True if both instances describe the same quantity, False otherwise.
        """
        if not isinstance(other, QuantitySpec):
            return False

        # Compare name, unit, shape, and times
        same_basic_attrs = (
            (self.name, self.unit) == (other.name, other.unit)
            and np.array_equal(self.shape, other.shape)
            and np.array_equal(self.times, other.times)
        )

        # Compare locations (set difference = ∅ → same)
        same_locations = not (set(self.locations) - set(other.locations))

        return same_basic_attrs and same_locations


@attr.s(auto_attribs=True)
class ChunkSpec:
    """
    Specification of a simulation or dataset chunk.

    :param chunk_id: Integer identifier of the chunk.
    :param chunk_slice: Slice object defining the range of data indices in the chunk.
    :param level_id: Identifier of the refinement or simulation level.
    """

    chunk_id: int = None
    chunk_slice: slice = None
    level_id: int = None
