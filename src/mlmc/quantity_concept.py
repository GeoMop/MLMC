import abc
import numpy as np
import copy
from functools import lru_cache
from memoization import cached
from scipy import interpolate
from typing import List, Tuple
from mlmc.sample_storage import SampleStorage
from mlmc.sim.simulation import QuantitySpec

cached_functions = []


def clearable_cache(*args, **kwargs):
    def decorator(func):
        func = cached(*args, **kwargs)(func)
        cached_functions.append(func)
        return func

    return decorator


def clear_all_cached_functions():
    for func in cached_functions:
        func.cache_clear()


def make_root_quantity(storage: SampleStorage, q_specs: List[QuantitySpec]):
    """
    :param storage: sample storage
    :param q_specs: same as result format in simulation class
    :return: dict
    """
    clear_all_cached_functions()
    dict_types = []
    for q_spec in q_specs:
        scalar_type = ScalarType(float)
        array_type = ArrayType(q_spec.shape, scalar_type)
        field_type = FieldType([(loc, array_type) for loc in q_spec.locations])
        ts_type = TimeSeriesType(q_spec.times, field_type)
        dict_types.append((q_spec.name, ts_type))
    dict_type = DictType(dict_types)

    return Quantity(quantity_type=dict_type, input_quantities=[QuantityStorage(storage, dict_type)])


def estimate_mean(quantity):
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
        level_ids = quantity.level_ids()
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
                n_samples[level_id] += chunk.shape[1]
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
    def __init__(self, quantity_type, input_quantities=None, operation=None):

        # @TODO: check if variable
        self.qtype = quantity_type
        # List of quantities on which the 'self' dependens. their number have to match number of arguments to the operation.
        self._operation = operation
        self._input_quantities = input_quantities

        # Cache mechanism

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

    def set_range(self, start, size):
        for quantity in self._input_quantities:
            quantity.set_range(start, size)

    def size(self) -> int:
        return self.qtype.size()

    # def get_cache_key(self, level_id, i_chunk):
    #     return str(hash(str(level_id) + str(i_chunk) + str(self._input_quantities) + str(id(self))))

    #@clearable_cache(custom_key_maker=get_cache_key)
    #@lru_cache
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
            # Operation not set return first quantity samples - used in make_root_quantity
            if self._operation is None:
                return chunks_quantity_level[0]

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
        if np.isnan(data_row).all():
            return []
        return data_row

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

        return Quantity(quantities[0].qtype, operation=operation, input_quantities=quantities)

    def select(self, quantity):
        def op(x):
            return x[:, quantity._operation(x)]

        return Quantity(quantity_type=self.qtype, input_quantities=[self], operation=op)

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

    def _process_mask(self, mask):
        # all values for sample must meet given condition,
        # if any value doesn't meet the condition, whole sample is eliminated
        return mask.all(axis=0).all(axis=1)

    def _create_quantity(self, value, op):
        if isinstance(value, float) or isinstance(value, int):
            return Quantity(quantity_type=self.qtype, input_quantities=[self], operation=op)
        return self

    def __lt__(self, value):
        def lt_op(x):
            return self._process_mask(x < value)
        return self._create_quantity(value, lt_op)

    def __le__(self, value):
        def le_op(x):
            return self._process_mask(x <= value)
        return self._create_quantity(value, le_op)

    def __gt__(self, value):
        def gt_op(x):
            return self._process_mask(x > value)
        return self._create_quantity(value, gt_op)

    def __ge__(self, value):
        def ge_op(x):
            return self._process_mask(x >= value)
        return self._create_quantity(value, ge_op)

    def __eq__(self, value):
        def eq_op(x):
            return self._process_mask(x == value)
        return self._create_quantity(value, eq_op)

    def __ne__(self, value):
        def ne_op(x):
            return self._process_mask(x != value)
        return self._create_quantity(value, ne_op)

    def sampling(self, size):
        def mask_gen(x):
            indices = np.random.choice(x.shape[1], size=size)
            mask = np.zeros(x.shape[1], bool)
            mask[indices] = True
            return mask

        return self._create_quantity(size, mask_gen)

    def __getitem__(self, key):
        """
        Get value from dictionary, it allows access item of quantities which have FieldType or DictType
        :param key: supported dict key
        :return: Quantity
        """
        try:
            new_qtype = self.qtype[key]  # New quantity type
            input_quantities = custom_copy(self._input_quantities)  # 'deep' copy of input quantities
            for i, quantity in enumerate(input_quantities):
                # Change quantity range
                # e.g. allows select Field from time interpolation quantity
                # it generally allows to determine interval of 'selected' values from results (it is used in QuantityStorage.samples())
                quantity.set_range(quantity.qtype.start + new_qtype.start, new_qtype.size())

            new_quantity = Quantity(quantity_type=new_qtype, input_quantities=input_quantities,
                                    operation=self._operation)
            return new_quantity
        except KeyError:
            return key

    def __iter__(self):
        raise Exception("This class is not iterable")

    def __copy__(self):
        new = type(self)(quantity_type=self.qtype, input_quantities=custom_copy(self._input_quantities),
                         operation=self._operation)
        return new

    def __hash__(self):
        return hash(str(hash(self.qtype)) +
                    str(hash(id(self._operation))) +
                    str(hash(self._operation([0]))) if self._operation is not None else '0' +
                    ' '.join([str(hash(input_quantity)) for input_quantity in self._input_quantities]))

    def time_interpolation(self, value):
        """
        Interpolation in time
        :param value:
        :return:
        """
        def interp(*y):
            f = interpolate.interp1d(self.qtype._times, y, axis=0)
            return f(value)

        quantities_in_time = []
        # Split TimeSeries to FieldTypes, create corresponding Quantities
        for i in range(len(self.qtype._times)):
            new_qtype = copy.copy(self.qtype._qtype)
            new_qtype.start = i * new_qtype.size()

            quantity_t = Quantity(quantity_type=new_qtype, input_quantities=custom_copy([self]))
            quantity_t.set_range(new_qtype.start, new_qtype.size())

            quantities_in_time.append(quantity_t)

        return Quantity(quantity_type=self.qtype._qtype, input_quantities=quantities_in_time, operation=interp)

    def level_ids(self):
        return self._input_quantities[0].level_ids()


class QuantityStorage(Quantity):
    def __init__(self, storage, qtype):
        self._storage = storage
        self.qtype = qtype
        self.start = None
        self.end = None

    def set_range(self, range_start: int, size: int):
        """
        Range of selected values from results
        :param range_start: int
        :param size: int
        :return: None
        """
        self.start = range_start
        self.end = range_start + size

    def level_ids(self):
        """
        Number of levels
        :return: list
        """
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
            assert self.qtype.size() == level_chunk.shape[0]
            # Select values from given interval self.start:self.end
            if self.start is not None and self.end is not None:
                return level_chunk[self.start:self.end, :, :]
        return level_chunk

    def __copy__(self):
        new = type(self)(self._storage, self.qtype)
        new.__dict__.update(self.__dict__)
        return new

    def __hash__(self):
        return hash(str(self.start) + str(self.end))


class QType(metaclass=abc.ABCMeta):
    def size(self) -> int:
        """
        Size of type
        :return: int
        """

    def __eq__(self, other):
        if isinstance(other, QType):
            return self.size() == other.size()
        return False


class ScalarType(QType):
    def __init__(self, qtype=float):
        self._qtype = qtype

    def size(self) -> int:
        return 1

    def __hash__(self):
        return hash(self._qtype)


class ArrayType(QType):
    def __init__(self, shape, qtype: QType, start=0):
        self._shape = shape
        self._qtype = qtype
        self.start = start

    def size(self) -> int:
        return np.prod(self._shape) * self._qtype.size()

    def __hash__(self):
        return hash(str(self._shape) + str(self.start) + str(hash(self._qtype)))


class TimeSeriesType(QType):
    def __init__(self, times, qtype, start=0):
        self._times = times
        self._qtype = qtype
        self.start = start

    def size(self) -> int:
        return len(self._times) * self._qtype.size()

    def __hash__(self):
        return hash(str(self._times) + str(self.start) + str(hash(self._qtype)))


class FieldType(QType):
    def __init__(self, args: List[Tuple[Tuple, QType]], start=0):
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
        q_type = self._qtype  # @TODO: deep copy
        position = list(self._dict.keys()).index(key)
        q_type.start = position * self._dict[key].size()
        return q_type

    def __copy__(self):
        new = type(self)([(k, v) for k, v in self._dict.items()])
        new.__dict__.update(self.__dict__)
        return new

    def __hash__(self):
        return hash(str(self._dict) + str(self.start) + str(hash(self._qtype)))


class DictType(QType):
    def __init__(self, args: List[Tuple[str, QType]]):
        self._dict = dict(args)
        self.start = 0

    def size(self) -> int:
        return int(np.sum(q_type.size() for _, q_type in self._dict.items()))

    def __getitem__(self, key):
        q_type = self._dict[key]
        position = list(self._dict.keys()).index(key)
        size = 0
        for k, qt in self._dict.items():
            if k == key:
                break
            size += qt.size()

        q_type.start = size

        return q_type

    def __hash__(self):
        return hash(str(self._dict) + str(self.start))


def custom_copy(quantities):
    return[copy.copy(quantity) for quantity in quantities]
