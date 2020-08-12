import abc
import numpy as np
from typing import List, Tuple
from collections import OrderedDict
from mlmc.sample_storage import SampleStorage
from mlmc.sim.simulation import QuantitySpec


def make_root_quantity(storage: SampleStorage, q_specs: List[QuantitySpec]):
    """
    :param storage: sample storage
    :param q_specs: same as result format in simulation class
    :return: dict
    """
    dict_types = []
    for q_spec in q_specs:
        scalar_type = ScalarType(float)
        array_type = ArrayType(q_spec.shape, scalar_type)
        field_type = FieldType([(tuple(q_spec.locations), array_type)])
        ts_type = TimeSeriesType(q_spec.times, field_type)
        dict_types.append((q_spec.name, ts_type))
    dict_type = DictType(dict_types)
    return QuantityStorage(storage, dict_type)


def estimate_mean(quantity):#, selection_quantity - se):
    """
    Concept of the MLMC mean estimator.
    TODO:
    - test
    - calculate variance estimates as well
    :param quantity:
    :return:
    """
    quantity_vec_size = quantity.size()
    n_samples = None
    sums = None
    i_chunk = 0

    while True:
        level_ids = quantity.n_levels()

        if i_chunk == 0:
            # initialization
            n_levels = len(level_ids)
            n_samples = [0] * n_levels

        for level_id in level_ids:
            # Chunk of samples for given level id
            chunk = quantity.samples(level_id, i_chunk)

            if chunk is not None:
                if level_id == 0:
                    sums = [np.zeros(chunk.shape[0]) for _ in range(n_levels)]

                # level_chunk is Numpy Array with shape [M, chunk_size, 2]
                n_samples[level_id] += len(chunk)
                assert(chunk.shape[0] == quantity_vec_size)
                sums[level_id] += np.sum(chunk[:, :, 0] - chunk[:, :, 1], axis=1)

        if chunk is None:
            break
        i_chunk += 1

    mean = np.zeros_like(sums[0])
    for s, n in zip(sums, n_samples):
        mean += s / n

    return quantity._make_value(mean)


# Just for type hints, change to protocol
class Quantity:
    def __init__(self, quantity_type, input_quantities, operation=None):
        #assert (len(input_quantities) > 0)
        self._qtype = quantity_type
        # List of quantities on which the 'self' dependes. their number have to match number of arguments to the operation.
        self._operation = operation
        self._input_quantities = input_quantities
        # function  lambda(*args : List[array[M, N, 2]]) -> List[array[M, N, 2])]
        # It takes list of chunks for individual levels as a single argument, with number of arguments matching the
        # number of input qunatities. Operation performs actual quantity operation on the sample chunks.
        # One chunk is a np.array with shape [sample_vector_size, n_samples_in_chunk, 2], 2 = (coarse, fine) pair.
        #self._size = sum((q._size() for q in self._input_quantities))
        # Number of values allocated by this quantity in the sample vector.

        # TODO: design a separate class and mechanisms to avoid repetitive evaluation of quantities
        # that are parents to more then one quantity
        # can possibly omit memoization of quantities that are just subsets of their own parent
        #
        # can e.g. be done through a decorator

        # self._dof_vec_subset = None
        # # list of indices to get form the parent quantity
        # # None for the computing quantities, that memoize their chinks
        # self._chunk = None
        # #  memoized  resulting chunk

    def size(self):
        return self._qtype.size()

    def samples(self, level_id, i_chunk):
        """
        Yields list of sample chunks for individual levels.
        Possibly calls underlaying quantities.
        Try to write into the chunk array.
        TODO: possibly pass down the full table of composed quantities and store comuted temporaries directly into the composed table
        TODO: possible problem when more quantities dependes on a single one, this probabely leads to
        repetitive evaluation of the shared quantity
        This prevents some coies in concatenation.
        """
        chunks_quantity_level = [q.samples(level_id, i_chunk) for q in self._input_quantities]
        is_valid = (ch is not None for ch in chunks_quantity_level)
        if any(is_valid):
            assert (all(is_valid))
            return self._operation(*chunks_quantity_level)
        else:
            return None

    def _make_value(self, data_row: np.array):
        """
        Crate a new quantity with the same structure but containing fixed data vector.
        Primary usage is to organise computed means and variances.
        Can possibly be used also to organise single sample row.
        :param data_row:
        :return:
        """
        return data_row  # @TODO: expand possibilities of use

    def _reduction_op(self, quantities, operation):
        """
        Check if the quantities have the same structure and possibly return copy of the common quantity
        structure depending on the other quantities with given operation.
        TODO: Not so simple, but similar problem to constraction of the initial structure over storage.
        :param quantities:
        :param operation: function which is run with given quantitiees
        :return:
        @TODO: quantity same structure test: check unit, c
        """
        # @TODO: check if all items in quantitites are Quantity
        assert all(x.size() == quantities[0].size() for x in quantities), "Quantity must have same structure"

        return Quantity(quantities[0]._qtype, operation=operation, input_quantities=quantities)

    def __add__(self, other):
        def add_op(x, y):
            return x + y
        return self._reduction_op([self, other], add_op)

    def __mul__(self, other):
        if isinstance(other, (float, int)):
            return self.__const_mult(other)

        def mult_op(x, y):
            return x * y
        return self._reduction_op([self, other], mult_op)

    def __rmul__(self, other):
        if isinstance(other, (float, int)):
            return self.__const_mult(other)
        assert False

    def __const_mult(self, other):
        def cmult_op(x, c=other):
            return c * x
        return self._reduction_op([self], cmult_op)

    def __getitem__(self, key):
        try:
            return Quantity(quantity_type=self._qtype[key], input_quantities=self._input_quantities, operation=self._operation)
        except KeyError:
            return key

    def __iter__(self):
        raise Exception("This class is not iterable")

    def n_levels(self):
        return self._input_quantities[0].n_levels()


class QuantityStorage(Quantity):
    def __init__(self, storage, qtype):
        self._storage = storage
        self._qtype = qtype
        self._input_quantities = None
        self._operation = None

    def n_levels(self):
        return self._storage.get_level_ids()

    def samples(self, level_id, i_chunk):
        """
        Get results for given level id and chunk id
        :param level_id: int
        :param i_chunk: int
        :return: Array[M, chunk size, 2]
        """
        level_chunk = self._storage.sample_pairs_level(level_id, i_chunk)  # Array[M, chunk size, 2]
        if level_chunk is not None:
            assert self._qtype.size() == level_chunk.shape[0]
        return level_chunk


class QType(metaclass=abc.ABCMeta):
    def size(self):
        """
        Size of type
        :return:
        """

    def __eq__(self, other):
        if isinstance(other, QType):
            return self.size == other.size
        return False


class ScalarType(QType):
    def __init__(self, qtype=float):
        self._qtype = qtype

    def size(self):
        return 1


class ArrayType(QType):
    def __init__(self, shape, qtype: QType):
        self._shape = shape
        self._qtype = qtype

    def size(self):
        return np.prod(self._shape) * self._qtype.size()


class TimeSeriesType(QType):
    def __init__(self, object, qtype):
        self._object = object
        self._qtype = qtype

    def size(self):
        return len(self._object) * self._qtype.size()


class FieldType(QType):
    def __init__(self, args: List[Tuple[Tuple, QType]]):
        """
        QType must have same structure
        :param args:
        """
        self._dict = OrderedDict(args)
        self._locations = self._dict.keys()
        self._qtype = args[0][1]
        assert all(q_type == self._qtype for _, q_type in args)

    def size(self):
        return np.sum(len(loc) * self._qtype.size() for loc in self._locations)


class DictType(QType):
    def __init__(self, args: List[Tuple[str, QType]]):
        self._dict = OrderedDict(args)

    def size(self):
        return np.sum(q_type.size() for _, q_type in self._dict.items())

    def __getitem__(self, key):
        return self._dict[key]
