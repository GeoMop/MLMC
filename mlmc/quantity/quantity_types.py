import abc
import copy
import numpy as np
from scipy import interpolate
from typing import List, Tuple
import mlmc.quantity.quantity


class QType(metaclass=abc.ABCMeta):
    def __init__(self, qtype):
        self._qtype = qtype

    def size(self) -> int:
        """
        Size of type
        :return: int
        """

    def base_qtype(self):
        return self._qtype.base_qtype()

    def replace_scalar(self, substitute_qtype):
        """
        Find ScalarType and replace it with substitute_qtype
        :param substitute_qtype: QType, replaces ScalarType
        :return: QType
        """
        inner_qtype = self._qtype.replace_scalar(substitute_qtype)
        new_qtype = copy.deepcopy(self)
        new_qtype._qtype = inner_qtype
        return new_qtype

    @staticmethod
    def keep_dims(chunk):
        """
        Always keep chunk shape to be [M, chunk size, 2]!
        For scalar quantities, the input block can have the shape (chunk size, 2)
        Sometimes we need to 'flatten' first few shape to have desired chunk shape
        :param chunk: list
        :return: list
        """
        # Keep dims [M, chunk size, 2]
        if len(chunk.shape) == 2:
            chunk = chunk[np.newaxis, :]
        elif len(chunk.shape) > 2:
            chunk = chunk.reshape((np.prod(chunk.shape[:-2]), chunk.shape[-2], chunk.shape[-1]))
        else:
            raise ValueError("Chunk shape not supported")
        return chunk

    def _make_getitem_op(self, chunk, key):
        """
        Operation
        :param chunk: level chunk, list with shape [M, chunk size, 2]
        :param key: parent QType's key, needed for ArrayType
        :return: list
        """
        return QType.keep_dims(chunk[key])

    def reshape(self, data):
        return data


class ScalarType(QType):
    def __init__(self, qtype=float):
        self._qtype = qtype

    def base_qtype(self):
        if isinstance(self._qtype, BoolType):
            return self._qtype.base_qtype()
        return self

    def size(self) -> int:
        if hasattr(self._qtype, 'size'):
            return self._qtype.size()
        return 1

    def replace_scalar(self, substitute_qtype):
        """
        Find ScalarType and replace it with substitute_qtype
        :param substitute_qtype: QType, replaces ScalarType
        :return: QType
        """
        return substitute_qtype


class BoolType(ScalarType):
    pass


class ArrayType(QType):
    def __init__(self, shape, qtype: QType):

        if isinstance(shape, int):
            shape = (shape,)

        self._shape = shape
        self._qtype = qtype

    def size(self) -> int:
        return np.prod(self._shape) * self._qtype.size()

    def get_key(self, key):
        """
        ArrayType indexing
        :param key: int, tuple of ints or slice objects
        :return: QuantityType - ArrayType or self._qtype
        """
        # Get new shape
        new_shape = np.empty(self._shape)[key].shape

        # One selected item is considered to be a scalar QType
        if len(new_shape) == 1 and new_shape[0] == 1:
            new_shape = ()

        # Result is also array
        if len(new_shape) > 0:
            q_type = ArrayType(new_shape, qtype=self._qtype)
        # Result is single array item
        else:
            q_type = self._qtype
        return q_type, 0

    def _make_getitem_op(self, chunk, key):
        """
        Operation
        :param chunk: list [M, chunk size, 2]
        :param key: slice
        :return:
        """
        # Reshape M to original shape to allow access
        assert self._shape is not None
        chunk = chunk.reshape((*self._shape, chunk.shape[-2], chunk.shape[-1]))
        return QType.keep_dims(chunk[key])

    def reshape(self, data):
        if isinstance(self._qtype, ScalarType):
            return data.reshape(self._shape)
        else:
            return data.reshape((*self._shape, np.prod(data.shape) // np.prod(self._shape)))


class TimeSeriesType(QType):
    def __init__(self, times, qtype):
        if isinstance(times, np.ndarray):
            times = times.tolist()
        self._times = times
        self._qtype = qtype

    def size(self) -> int:
        return len(self._times) * self._qtype.size()

    def get_key(self, key):
        q_type = self._qtype
        try:
            position = self._times.index(key)
        except KeyError:
            print("Item " + str(key) + " was not found in TimeSeries" + ". Available items: " + str(list(self._times)))
        return q_type, position * q_type.size()

    @staticmethod
    def time_interpolation(quantity, value):
        """
        Interpolation in time
        :param quantity: Quantity instance
        :param value: point where to interpolate
        :return: Quantity
        """
        def interp(y):
            split_indeces = np.arange(1, len(quantity.qtype._times)) * quantity.qtype._qtype.size()
            y = np.split(y, split_indeces, axis=-3)
            f = interpolate.interp1d(quantity.qtype._times, y, axis=0)
            return f(value)
        return mlmc.quantity.quantity.Quantity(quantity_type=quantity.qtype._qtype, input_quantities=[quantity], operation=interp)


class FieldType(QType):
    def __init__(self, args: List[Tuple[str, QType]]):
        """
        QType must have same structure
        :param args:
        """
        self._dict = dict(args)
        self._qtype = args[0][1]
        assert all(q_type.size() == self._qtype.size() for _, q_type in args)

    def size(self) -> int:
        return len(self._dict.keys()) * self._qtype.size()

    def get_key(self, key):
        q_type = self._qtype
        try:
            position = list(self._dict.keys()).index(key)
        except KeyError:
            print("Key " + str(key) + " was not found in FieldType" +
                  ". Available keys: " + str(list(self._dict.keys())[:5]) + "...")
        return q_type, position * q_type.size()


class DictType(QType):
    def __init__(self, args: List[Tuple[str, QType]]):
        self._dict = dict(args)  # Be aware we it is ordered dictionary
        self._check_base_type()

    def _check_base_type(self):
        qtypes = list(self._dict.values())
        qtype_0_base_type = qtypes[0].base_qtype()
        for qtype in qtypes[1:]:
            if not isinstance(qtype.base_qtype(), type(qtype_0_base_type)):
                raise TypeError("qtype {} has base QType {}, expecting {}. "
                                "All QTypes must have same base QType, either SacalarType or BoolType".
                                format(qtype, qtype.base_qtype(), qtype_0_base_type))

    def base_qtype(self):
        return next(iter(self._dict.values())).base_qtype()

    def size(self) -> int:
        return int(sum(q_type.size() for _, q_type in self._dict.items()))

    def get_qtypes(self):
        return self._dict.values()

    def replace_scalar(self, substitute_qtype):
        """
        Find ScalarType and replace it with substitute_qtype
        :param substitute_qtype: QType, replaces ScalarType
        :return: DictType
        """
        dict_items = []
        for key, qtype in self._dict.items():
            new_qtype = qtype.replace_scalar(substitute_qtype)
            dict_items.append((key,  new_qtype))
        return DictType(dict_items)

    def get_key(self, key):
        try:
            q_type = self._dict[key]
        except KeyError:
            print("Key " + str(key) + " was not found in DictType" +
                  ". Available keys: " + str(list(self._dict.keys())[:5]) + "...")
        start = 0
        for k, qt in self._dict.items():
            if k == key:
                break
            start += qt.size()
        return q_type, start
