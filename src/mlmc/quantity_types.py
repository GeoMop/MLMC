import abc
import numpy as np
import copy
from typing import List, Tuple


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

    def __eq__(self, other):
        if isinstance(other, QType):
            return self.size() == other.size()
        return False

    @staticmethod
    def replace_scalar(original_qtype, substitute_qtype):
        """
        Find ScalarType and replace it with new_qtype
        :param substitute_qtype: QType, replace ScalarType
        :return: None
        """
        qtypes = []
        current_qtype = original_qtype
        while True:
            if isinstance(current_qtype, DictType):
                qtypes.append(DictType.replace_scalar(current_qtype, substitute_qtype))
                break

            if isinstance(current_qtype, (ScalarType, BoolType)):
                if isinstance(current_qtype, (ScalarType, BoolType)):
                    qtypes.append(substitute_qtype)
                    break

            qtypes.append(current_qtype)
            current_qtype = current_qtype._qtype

        first_qtype = qtypes[0]
        new_qtype = first_qtype

        for i in range(1, len(qtypes)):
            new_qtype._qtype = qtypes[i]
            new_qtype = new_qtype._qtype
        return first_qtype

    def _keep_dims(self, chunk):
        """
        Always keep chunk dimensions to be [M, chunk size, 2]
        :param chunk: list
        :return: list
        """
        # Keep dims [M, chunk size, 2]
        if len(chunk.shape) == 2:
            chunk = chunk[np.newaxis, :]
        elif len(chunk.shape) > 2:
            chunk = chunk.reshape((np.prod(chunk.shape[:-2]), chunk.shape[-2], chunk.shape[-1]))

        return chunk

    def _make_getitem_op(self, chunk, new_qtype, key=None):
        """
        Operation
        :param chunk: level chunk, list with shape [M, chunk size, 2]
        :param new_qtype: QType
        :param key: parent QType's key, needed for ArrayType
        :return: list
        """
        start = new_qtype.start
        end = new_qtype.start + new_qtype.size()
        slice_key = slice(start, end)
        return self._keep_dims(chunk[slice_key])


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


class BoolType(ScalarType):
    pass


class ArrayType(QType):
    def __init__(self, shape, qtype: QType, start=0):
        self._shape = shape
        self._qtype = qtype
        self.start = start

    def size(self) -> int:
        return np.prod(self._shape) * self._qtype.size()

    def __getitem__(self, key):
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
            q_type = ArrayType(new_shape, qtype=copy.deepcopy(self._qtype))
        # Result is single array item
        else:
            q_type = copy.deepcopy(self._qtype)

        return q_type

    def _make_getitem_op(self, chunk, new_qtype, key=None):
        """
        Operation
        :param chunk: list [M, chunk size, 2]
        :param new_qtype: QType
        :param key: Qtype key
        :return:
        """
        # Reshape M to original shape to allow access
        if self._shape is not None:
            chunk = chunk.reshape((*self._shape, chunk.shape[-2], chunk.shape[-1]))
        return self._keep_dims(chunk[key])


class TimeSeriesType(QType):
    def __init__(self, times, qtype, start=0):
        if isinstance(times, np.ndarray):
            times = times.tolist()
        self._times = times
        self._qtype = qtype
        self.start = start

    def size(self) -> int:
        return len(self._times) * self._qtype.size()

    def __getitem__(self, key):
        if key not in self._times:
            raise KeyError("Item " + str(key) + " was not found in TimeSeries" +
                           ". Available items: " + str(list(self._times)))

        q_type = copy.deepcopy(self._qtype)
        position = self._times.index(key)
        q_type.start = position * q_type.size()
        return q_type


class FieldType(QType):
    def __init__(self, args: List[Tuple[str, QType]], start=0):
        """
        QType must have same structure
        :param args:
        """
        self._dict = dict(args)
        self._qtype = args[0][1]
        self.start = start
        assert all(q_type == self._qtype for _, q_type in args)

    def size(self) -> int:
        return len(self._dict.keys()) * self._qtype.size()

    def __getitem__(self, key):
        if key not in self._dict:
            raise KeyError("Key " + str(key) + " was not found in FieldType" +
                           ". Available keys: " + str(list(self._dict.keys())))

        q_type = copy.deepcopy(self._qtype)
        position = list(self._dict.keys()).index(key)
        q_type.start = position * q_type.size()
        return q_type

    def __copy__(self):
        new = type(self)([(k, v) for k, v in self._dict.items()])
        new.__dict__.update(self.__dict__)
        return new


class DictType(QType):
    def __init__(self, args: List[Tuple[str, QType]]):
        self._dict = dict(args)  # Be aware we it is ordered dictionary
        self.start = 0

        self._check_base_type()

    def _check_base_type(self):
        qtypes = list(self._dict.values())
        for qtype in qtypes[1:]:
            if not isinstance(qtype.base_qtype(), type(qtypes[0].base_qtype())):
                raise TypeError("qtype {} has base QType {}, expecting {}. "
                                "All QTypes must have same base QType, either SacalarType or BoolType".
                                format(qtype, qtype.base_qtype(), qtypes[0].base_qtype()))

    def base_qtype(self):
        return list(self._dict.values())[0].base_qtype()

    def size(self) -> int:
        return int(sum(q_type.size() for _, q_type in self._dict.items()))

    def get_qtypes(self):
        return self._dict.values()

    @staticmethod
    def replace_scalar(original_qtype, substitute_qtype):
        dict_items = []
        for key, qtype in original_qtype._dict.items():
            if isinstance(qtype, ScalarType):
                dict_items.append((key, substitute_qtype))
            else:
                dict_items.append((key,  QType.replace_scalar(qtype, substitute_qtype)))
        return DictType(dict_items)

    def __getitem__(self, key):
        if key not in self._dict:
            raise KeyError("Key " + str(key) + " was not found in DictType" +
                           ". Available keys: " + str(list(self._dict.keys())))

        q_type = self._dict[key]

        size = 0
        for k, qt in self._dict.items():
            if k == key:
                break
            size += qt.size()

        q_type.start = size
        return q_type
