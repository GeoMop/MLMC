import abc
import copy
import numpy as np
from scipy import interpolate
from typing import List, Tuple
import mlmc.quantity.quantity


class QType(metaclass=abc.ABCMeta):
    """
    Base class for quantity types.

    :param qtype: inner/contained QType or Python type
    """

    def __init__(self, qtype):
        self._qtype = qtype

    def size(self) -> int:
        """
        Size of the type in flattened units.

        :return: int
        """
        raise NotImplementedError

    def base_qtype(self):
        """
        Return the base scalar/bool type for nested types.

        :return: QType
        """
        return self._qtype.base_qtype()

    def replace_scalar(self, substitute_qtype):
        """
        Find ScalarType and replace it with substitute_qtype.

        :param substitute_qtype: QType that replaces ScalarType
        :return: QType (new instance with scalar replaced)
        """
        inner_qtype = self._qtype.replace_scalar(substitute_qtype)
        new_qtype = copy.deepcopy(self)
        new_qtype._qtype = inner_qtype
        return new_qtype

    @staticmethod
    def keep_dims(chunk: np.ndarray) -> np.ndarray:
        """
        Ensure chunk has shape [M, chunk size, 2].

        For scalar quantities the input block can have shape (chunk size, 2).
        Sometimes we need to 'flatten' first few dimensions to achieve desired chunk shape.

        :param chunk: numpy array
        :return: numpy array with shape [M, chunk size, 2]
        :raises ValueError: if chunk.ndim < 2
        """
        # Keep dims [M, chunk size, 2]
        if len(chunk.shape) == 2:
            chunk = chunk[np.newaxis, :]
        elif len(chunk.shape) > 2:
            chunk = chunk.reshape((int(np.prod(chunk.shape[:-2])), chunk.shape[-2], chunk.shape[-1]))
        else:
            raise ValueError("Chunk shape not supported: need ndim >= 2")
        return chunk

    def _make_getitem_op(self, chunk: np.ndarray, key):
        """
        Extract a slice from chunk while preserving chunk dims.

        :param chunk: level chunk, numpy array with shape [M, chunk size, 2]
        :param key: index/slice used by parent QType
        :return: numpy array with shape [M', chunk size', 2]
        """
        return QType.keep_dims(chunk[key])

    def reshape(self, data: np.ndarray) -> np.ndarray:
        """
        Default reshape (identity).

        :param data: numpy array
        :return: numpy array
        """
        return data


class ScalarType(QType):
    """
    Scalar quantity type (leaf type).
    """

    def __init__(self, qtype=float):
        """
        :param qtype: Python type or nested type used as underlying scalar type
        """
        self._qtype = qtype

    def base_qtype(self):
        """
        :return: base scalar QType (self or underlying BoolType base)
        """
        if isinstance(self._qtype, BoolType):
            return self._qtype.base_qtype()
        return self

    def size(self) -> int:
        """
        :return: int size of the scalar (defaults to 1 or uses `_qtype.size()` if present)
        """
        if hasattr(self._qtype, "size"):
            return self._qtype.size()
        return 1

    def replace_scalar(self, substitute_qtype):
        """
        Replace ScalarType with substitute type.

        :param substitute_qtype: QType that replaces ScalarType
        :return: substitute_qtype
        """
        return substitute_qtype


class BoolType(ScalarType):
    """
    Boolean scalar type (inherits ScalarType).
    """
    pass


class ArrayType(QType):
    """
    Array quantity type.

    :param shape: int or tuple describing array shape
    :param qtype: contained QType for array elements
    """

    def __init__(self, shape, qtype: QType):
        if isinstance(shape, int):
            shape = (shape,)
        self._shape = shape
        self._qtype = qtype

    def size(self) -> int:
        """
        :return: total flattened size (product of shape * inner qtype size)
        """
        return int(np.prod(self._shape)) * int(self._qtype.size())

    def get_key(self, key):
        """
        ArrayType indexing.

        :param key: int, tuple of ints or slice objects
        :return: Tuple (QuantityType, offset) where offset is 0 for this implementation
        """
        # Get new shape by applying indexing on an empty array of the target shape
        new_shape = np.empty(self._shape)[key].shape

        # If one selected item is considered to be a scalar QType
        if len(new_shape) == 1 and new_shape[0] == 1:
            new_shape = ()

        # Result is also array
        if len(new_shape) > 0:
            q_type = ArrayType(new_shape, qtype=self._qtype)
        # Result is single array item
        else:
            q_type = self._qtype
        return q_type, 0

    def _make_getitem_op(self, chunk: np.ndarray, key):
        """
        Slice operation for ArrayType while restoring original shape.

        :param chunk: numpy array [M, chunk size, 2]
        :param key: slice or index to apply on the array-shaped leading dims
        :return: numpy array with preserved dims via QType.keep_dims
        """
        assert self._shape is not None
        chunk = chunk.reshape((*self._shape, chunk.shape[-2], chunk.shape[-1]))
        return QType.keep_dims(chunk[key])

    def reshape(self, data: np.ndarray) -> np.ndarray:
        """
        Reshape flattened data to array shape.

        :param data: numpy array
        :return: reshaped numpy array
        """
        if isinstance(self._qtype, ScalarType):
            return data.reshape(self._shape)
        else:
            # assume trailing dimension belongs to inner types
            total = np.prod(data.shape)
            leading = int(np.prod(self._shape))
            return data.reshape((*self._shape, int(total // leading)))


class TimeSeriesType(QType):
    """
    Time-series quantity type.

    :param times: iterable of time points
    :param qtype: QType for each time slice
    """

    def __init__(self, times, qtype):
        if isinstance(times, np.ndarray):
            times = times.tolist()
        self._times = times
        self._qtype = qtype

    def size(self) -> int:
        """
        :return: total size = number of time points * inner qtype.size()
        """
        return len(self._times) * int(self._qtype.size())

    def get_key(self, key):
        """
        Get a qtype and offset corresponding to a given time key.

        :param key: time value to locate
        :return: Tuple (q_type, offset)
        """
        q_type = self._qtype
        try:
            position = self._times.index(key)
        except ValueError:
            # keep behavior similar to original: print available items
            print(
                "Item "
                + str(key)
                + " was not found in TimeSeries"
                + ". Available items: "
                + str(list(self._times))
            )
            # raise to make the error explicit
            raise
        return q_type, position * q_type.size()

    @staticmethod
    def time_interpolation(quantity, value):
        """
        Interpolate a time-series quantity to a single time value.

        :param quantity: Quantity instance with qtype being a TimeSeriesType
        :param value: float time value where to interpolate
        :return: Quantity object representing interpolated value
        """
        def interp(y):
            split_indices = np.arange(1, len(quantity.qtype._times)) * quantity.qtype._qtype.size()
            y = np.split(y, split_indices, axis=-3)
            f = interpolate.interp1d(quantity.qtype._times, y, axis=0)
            return f(value)

        return mlmc.quantity.quantity.Quantity(
            quantity_type=quantity.qtype._qtype,
            input_quantities=[quantity],
            operation=interp
        )


class FieldType(QType):
    """
    Field type composed of named entries each having the same base qtype.

    :param args: List of (name, QType) pairs
    """

    def __init__(self, args: List[Tuple[str, QType]]):
        self._dict = dict(args)
        self._qtype = args[0][1]
        assert all(q_type.size() == self._qtype.size() for _, q_type in args)

    def size(self) -> int:
        """
        :return: total size = number of fields * inner qtype size
        """
        return len(self._dict.keys()) * int(self._qtype.size())

    def get_key(self, key):
        """
        Access sub-field by name.

        :param key: field name
        :return: Tuple (q_type, offset)
        """
        q_type = self._qtype
        try:
            position = list(self._dict.keys()).index(key)
        except ValueError:
            print(
                "Key "
                + str(key)
                + " was not found in FieldType"
                + ". Available keys: "
                + str(list(self._dict.keys())[:5])
                + "..."
            )
            raise
        return q_type, position * q_type.size()


class DictType(QType):
    """
    Dictionary-like type of named QTypes which may differ in size.

    :param args: List of (name, QType) pairs
    """

    def __init__(self, args: List[Tuple[str, QType]]):
        self._dict = dict(args)  # keep ordered mapping semantics
        self._check_base_type()

    def _check_base_type(self):
        """
        Ensure all contained qtypes share the same base_qtype.

        :raises TypeError: if base_qtypes differ
        """
        qtypes = list(self._dict.values())
        qtype_0_base_type = qtypes[0].base_qtype()
        for qtype in qtypes[1:]:
            if not isinstance(qtype.base_qtype(), type(qtype_0_base_type)):
                raise TypeError(
                    "qtype {} has base QType {}, expecting {}. "
                    "All QTypes must have same base QType, either ScalarType or BoolType".format(
                        qtype, qtype.base_qtype(), qtype_0_base_type
                    )
                )

    def base_qtype(self):
        """
        :return: base_qtype of the first element
        """
        return next(iter(self._dict.values())).base_qtype()

    def size(self) -> int:
        """
        :return: total flattened size (sum of sizes of contained qtypes)
        """
        return int(sum(q_type.size() for _, q_type in self._dict.items()))

    def get_qtypes(self):
        """
        :return: iterable of contained qtypes
        """
        return self._dict.values()

    def replace_scalar(self, substitute_qtype):
        """
        Replace scalar types recursively inside dict entries.

        :param substitute_qtype: QType that replaces ScalarType
        :return: new DictType instance
        """
        dict_items = []
        for key, qtype in self._dict.items():
            new_qtype = qtype.replace_scalar(substitute_qtype)
            dict_items.append((key, new_qtype))
        return DictType(dict_items)

    def get_key(self, key):
        """
        Return the QType and starting offset for a named key.

        :param key: name of entry
        :return: Tuple (q_type, start_offset)
        """
        try:
            q_type = self._dict[key]
        except KeyError:
            print(
                "Key "
                + str(key)
                + " was not found in DictType"
                + ". Available keys: "
                + str(list(self._dict.keys())[:5])
                + "..."
            )
            raise

        start = 0
        for k, qt in self._dict.items():
            if k == key:
                break
            start += qt.size()
        return q_type, start
