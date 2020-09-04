import sys
import abc
import numpy as np
import copy
import operator
from memoization import cached
from scipy import interpolate
from typing import List, Tuple
from mlmc.sample_storage import SampleStorage
from mlmc.sim.simulation import QuantitySpec

cached_functions = []

this_module = sys.modules[__name__]


def make_method(method):
    def _method(quantities):
        if type(quantities) == list:
            assert all(isinstance(quantity, Quantity) for quantity in quantities),\
                "Quantity must be instance of Quantity"
            assert all(quantity.size() == quantities[0].size() for quantity in quantities),\
                "Quantity must have same structure"
        elif isinstance(quantities, Quantity):
            quantities = [quantities]
        return Quantity(quantity_type=quantities[0].qtype, input_quantities=custom_copy(quantities), operation=method)
    return _method


for method in [np.sin, np.cos, np.add, np.maximum, np.logical_and, np.logical_or]:
    _method = make_method(method)
    setattr(this_module, method.__name__, _method)


def apply(quantities, function):
    """
    Works for functions which results have same shape like input quantities
    :param quantities: List[Quantity]
    :param function: numpy function
    :return: Quantity
    """
    assert all(isinstance(quantity, Quantity) for quantity in quantities), "Quantity must be instance of Quantity"
    assert all(quantity.size() == quantities[0].size() for quantity in quantities), "Quantity must have same structure"

    return Quantity(quantity_type=quantities[0].qtype, input_quantities=custom_copy(quantities), operation=function)


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
    MLMC mean estimator.
    :param quantity: Quantity
    :return: QuantityMean which holds both mean and variance
    """
    quantity_vec_size = quantity.size()
    n_samples = None
    sums = None
    sums_power = None
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
                    sums_power = [np.zeros(chunk.shape[0]) for _ in range(n_levels)]

                    # Coarse result for level 0, there is issue for moments processing (not know about level)
                    chunk[..., 1] = 0

                # level_chunk is Numpy Array with shape [M, chunk_size, 2]
                n_samples[level_id] += chunk.shape[1]
                assert(chunk.shape[0] == quantity_vec_size)
                sums[level_id] += np.sum(chunk[..., 0] - chunk[..., 1], axis=1)
                sums_power[level_id] += np.sum((chunk[..., 0] - chunk[..., 1])**2, axis=1)

        if chunk is None:
            break
        i_chunk += 1

    mean = np.zeros_like(sums[0])
    mean_power = np.zeros_like(sums[0])
    for s, sp, n in zip(sums, sums_power, n_samples):
        mean += s / n
        mean_power += sp / n

    return quantity._make_value(mean=mean, var=mean_power - mean**2)


def moment(quantity, moments_fn, i=0):
    """
    Estimate moment
    :param quantity: Quantity instance
    :param moments_fn: mlmc.moments.Moments child
    :param i: index of moment
    :return: Quantity
    """
    def eval_moment(x):
        return moments_fn.eval_fine_coarse(i, value=x)
    return Quantity(quantity_type=quantity.qtype, input_quantities=[quantity], operation=eval_moment)


def moments(quantity, moments_fn, mom_at_bottom=True):
    """
    Estimates moments
    :param quantity: Quantity
    :param moments_fn: mlmc.moments.Moments child
    :param mom_at_bottom: bool, if True moments are underneath
    :return: Quantity
    """
    def eval_moments(x):
        if mom_at_bottom:
            mom = moments_fn.eval_all(x).transpose((0, 3, 1, 2))  # [M, R, N, 2]
        else:
            mom = moments_fn.eval_all(x).transpose((3, 0, 1, 2))  # [R, M, N, 2]
        return mom.reshape((np.prod(mom.shape[:-2]), mom.shape[-2], mom.shape[-1]))  # [M, N, 2]

    if mom_at_bottom:
        moments_array_type = ArrayType(shape=(moments_fn.size,), qtype=ScalarType())
        moments_qtype = copy.deepcopy(quantity.qtype)
        moments_qtype.replace_scalar(moments_array_type)
    else:
        moments_qtype = ArrayType(shape=(moments_fn.size,), qtype=quantity.qtype)
    return Quantity(quantity_type=moments_qtype, input_quantities=[quantity], operation=eval_moments)


def covariance(quantity, moments_fn, cov_at_bottom=True):
    """
    Estimate covariance matrix
    :param quantity: Quantity
    :param moments_fn: mlmc.moments.Moments child
    :param cov_at_bottom: bool, if True moments are underneath
    :return: Quantity
    """
    def eval_cov(x):
        moments = moments_fn.eval_all(x)
        mom_fine = moments[..., 0, :]
        mom_coarse = moments[..., 1, :]
        cov_fine = np.einsum('...i,...j', mom_fine, mom_fine)
        cov_coarse = np.einsum('...i,...j', mom_coarse, mom_coarse)

        if cov_at_bottom:
            cov = np.array([cov_fine, cov_coarse]).transpose((1, 3, 4, 2, 0))   # [M, R, R, N, 2]
        else:
            cov = np.array([cov_fine, cov_coarse]).transpose((3, 4, 1, 2, 0))   # [R, R, M, N, 2]
        return cov.reshape((np.prod(cov.shape[:-2]), cov.shape[-2], cov.shape[-1]))

    if cov_at_bottom:
        moments_array_type = ArrayType(shape=(moments_fn.size, moments_fn.size, ), qtype=ScalarType())
        moments_qtype = copy.deepcopy(quantity.qtype)
        moments_qtype.replace_scalar(moments_array_type)
    else:
        moments_qtype = ArrayType(shape=(moments_fn.size, moments_fn.size, ), qtype=quantity.qtype)

    return Quantity(quantity_type=moments_qtype, input_quantities=[quantity], operation=eval_cov)


# def numpy_matmul(quantity_1, quantity_2):
#     """
#     @TODO: think matrix multiplication over
#     :param x:
#     :param y:
#     :return:
#     """
#     assert quantity_1.size() == quantity_2.size(), "Quantity must have same structure"
#
#     def multiply(x, y):
#         return np.matmul(x, y)
#
#     new_qtype = ArrayType(shape=)  # how to specify shape
#
#     return Quantity(quantity_type=new_qtype, input_quantities=[quantity_1, quantity_2], operation=multiply)


class Quantity:
    def __init__(self, quantity_type, input_quantities=None, operation=None):
        self.qtype = quantity_type
        # List of quantities on which the 'self' dependens. their number have to match number of arguments to the operation.
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

    def storage_id(self):
        st_ids = []
        for input_quantity in self._input_quantities:
            st_id = input_quantity.storage_id()
            st_ids.extend(st_id if type(st_id) == list else [st_id])
        return st_ids

    def size(self) -> int:
        return self.qtype.size()

    def get_cache_key(self, level_id, i_chunk):
        return str(hash((level_id, i_chunk, id(self), *[id(q) for q in self._input_quantities])))

    @clearable_cache(custom_key_maker=get_cache_key)
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

    def _make_value(self, mean: np.array, var: np.array):
        """
        Crate a new quantity with the same structure but containing fixed data vector.
        Primary usage is to organise computed means and variances.
        Can possibly be used also to organise single sample row.
        :param data_row:
        :return:
        """
        if np.isnan(mean).all():
            mean = []
            var = []
        return QuantityMean(self.qtype, mean, var)

    def _reduction_op(self, quantities, operation):
        """
        Check if the quantities have the same structure and same storage possibly return copy of the common quantity
        structure depending on the other quantities with given operation.
        :param quantities: List[Quantity]
        :param operation: function which is run with given quantities
        :return: Quantity
        """
        assert all(np.allclose(q.storage_id(), quantities[0].storage_id()) for q in quantities), \
            "All quantities must be from same storage"

        assert all(q.size() == quantities[0].size() for q in quantities), "Quantity must have same structure"
        return Quantity(quantities[0].qtype, operation=operation, input_quantities=quantities)

    @staticmethod
    def get_mask(quantity, x):
        """
        Get all masks, it performs logical and of masks
        :param quantity: Quantity instance
        :param x: int or double
        :return: np.ndarray
        """
        masks = [q.get_mask(q, x) for q in quantity._input_quantities]

        mask = None
        if len(masks) > 0:
            mask = masks[0]

            if len(masks) > 1:
                for m in masks:
                    if m is not None:
                        mask = np.logical_and(mask, m)

        if quantity._operation is not None:
            try:
                current_mask = quantity._operation(x)
            except Exception:
                return None
            if not np.array_equal(current_mask, current_mask.astype(bool)):
                return None
        else:
            return None

        if mask is None:
            mask = np.ones((len(current_mask)))

        return np.logical_and(current_mask, mask)

    def select(self, quantity):
        def op(x):
            mask = Quantity.get_mask(quantity, x)
            return x[..., mask, :]  # [...sample size, cut number of samples, 2]

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

    def _process_mask(self, x, value, operator):
        # all values for sample must meet given condition,
        # if any value doesn't meet the condition, whole sample is eliminated

        # It is most likely zero level
        if np.all((x[..., 1] == 0)):
            mask = operator(x[..., 0], value)  # use just fine samples
            return mask.all(axis=tuple(range(mask.ndim - 1)))

        mask = operator(x, value)
        return mask.all(axis=tuple(range(mask.ndim - 2))).all(axis=1)

    def _create_quantity(self, value, op):
        if isinstance(value, float) or isinstance(value, int):
            return Quantity(quantity_type=self.qtype, input_quantities=[self], operation=op)
        return self

    def __lt__(self, value):
        def lt_op(x):
            return self._process_mask(x, value, operator.lt)
        return self._create_quantity(value, lt_op)  # vraci masku

    def __le__(self, value):
        def le_op(x):
            return self._process_mask(x, value, operator.le)
        return self._create_quantity(value, le_op)

    def __gt__(self, value):
        def gt_op(x):
            return self._process_mask(x, value, operator.gt)
        return self._create_quantity(value, gt_op)

    def __ge__(self, value):
        def ge_op(x):
            return self._process_mask(x, value, operator.ge)
        return self._create_quantity(value, ge_op)

    def __eq__(self, value):
        def eq_op(x):
            return self._process_mask(x, value, operator.eq)
        return self._create_quantity(value, eq_op)

    def __ne__(self, value):
        def ne_op(x):
            return self._process_mask(x, value, operator.ne)
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
        Get items from Quantity, quantity type must support brackets access
        :param key: str, int, tuple
        :return: Quantity
        """
        new_qtype = self.qtype[key]  # New quantity type

        reshape_shape = None
        # ArrayType might be accessed directly regardless of qtype start and size
        if isinstance(self.qtype, ArrayType):
            slice_key = key
            reshape_shape = self.qtype._shape
        # Other accessible quantity types use start and size
        else:
            start = new_qtype.start
            end = new_qtype.start + new_qtype.size()
            slice_key = slice(start, end)

        def getitem_op(y):
            # Reshape M to original shape to allow access
            if reshape_shape is not None:
                y = y.reshape((*reshape_shape, y.shape[-2], y.shape[-1]))
            y_get_item = y[slice_key] # indexing

            # Keep dims [M, N, 2]
            if len(y_get_item.shape) == 2:
                y_get_item = y_get_item[np.newaxis, :]
            elif len(y_get_item.shape) > 2:
                y_get_item = y_get_item.reshape((np.prod(y_get_item.shape[:-2]), y_get_item.shape[-2],
                                                 y_get_item.shape[-1]))

            return y_get_item

        return Quantity(quantity_type=new_qtype, input_quantities=[self], operation=getitem_op)

    def __iter__(self):
        raise Exception("This class is not iterable")

    def __copy__(self):
        return Quantity(quantity_type=self.qtype, input_quantities=custom_copy(self._input_quantities),
                        operation=self._operation)

    def time_interpolation(self, value):
        """
        Interpolation in time
        :param value:
        :return:
        """
        def interp(y):
            split_indeces = np.arange(1, len(self.qtype._times)) * self.qtype._qtype.size()
            y = np.split(y, split_indeces, axis=-3)
            f = interpolate.interp1d(self.qtype._times, y, axis=0)
            return f(value)

        return Quantity(quantity_type=self.qtype._qtype, input_quantities=custom_copy([self]), operation=interp)

    def level_ids(self):
        return self._input_quantities[0].level_ids()


class QuantityMean:

    def __init__(self, quantity_type, mean, var):
        self.qtype = quantity_type
        self._mean = mean
        self._var = var

    def __call__(self):
        return self._mean

    def __getitem__(self, key):
        """
        Get items from Quantity, quantity type must support brackets access
        :param key: str, int, tuple
        :return: np.ndarray
        """
        new_qtype = self.qtype[key]  # New quantity type
        reshape_shape = None
        newshape = None
        # ArrayType might be accessed directly regardless of qtype start and size
        if isinstance(self.qtype, ArrayType):
            slice_key = key
            reshape_shape = self.qtype._shape

            # If QType inside array is also array
            # set newshape which holds shape of inner array - good for reshape process
            if isinstance(new_qtype, ArrayType):
                 newshape = new_qtype._shape
        # Other accessible quantity types uses start and size
        else:
            start = new_qtype.start
            end = new_qtype.start + new_qtype.size()
            slice_key = slice(start, end)

        mean = self._mean
        var = self._var

        if reshape_shape is not None:
            if newshape is not None:  # reshape [Mr] to e.g. [..., R, R, M]
                mean = mean.reshape((*reshape_shape, *newshape))
                var = var.reshape((*reshape_shape, *newshape))
            elif (np.prod(mean.shape) // np.prod(reshape_shape)) > 1:
                mean = mean.reshape(*reshape_shape, np.prod(mean.shape) // np.prod(reshape_shape))
                var = var.reshape(*reshape_shape, np.prod(mean.shape) // np.prod(reshape_shape))
            else:
                mean = mean.reshape(*reshape_shape)
                var = var.reshape(*reshape_shape)

        mean_get_item = mean[slice_key]
        var_get_item = var[slice_key]
        return QuantityMean(quantity_type=new_qtype, mean=mean_get_item, var=var_get_item)


class QuantityStorage(Quantity):
    def __init__(self, storage, qtype):
        self._storage = storage
        self.qtype = qtype
        self._input_quantities = []
        self._operation = None
        self.start = None
        self.end = None

    def level_ids(self):
        """
        Number of levels
        :return: list
        """
        return self._storage.get_level_ids()

    def storage_id(self):
        return id(self)

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

    def replace_scalar(self, new_qtype):
        """
        Find ScalarType and replace him with new_qtype
        :param new_qtype: QType
        :return: None
        """
        if isinstance(self._qtype, ScalarType):
            self._qtype = new_qtype
        else:
            self._qtype.replace_scalar(new_qtype)


class ScalarType(QType):
    def __init__(self, qtype=float):
        self._qtype = qtype

    def size(self) -> int:
        return 1


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
        # int key to tuple
        if isinstance(key, int):
            key = (key,)

        if len(key) > len(self._shape):
            raise KeyError("Key {} does not match array shape {}".format(key, self._shape))

        # Create new shape
        new_shape = []
        for k, s in zip(key, self._shape):
            # handle slice objects
            if isinstance(k, slice):
                start, stop, step = k.indices(s)
                new_shape.append(int((stop - start) / step))

        new_shape = tuple(new_shape)

        # Result is also array
        if len(new_shape) > 0:
            q_type = ArrayType(new_shape, qtype=copy.deepcopy(self._qtype))
        # Result is single array item
        else:
            q_type = copy.deepcopy(self._qtype)
        return q_type


class TimeSeriesType(QType):
    def __init__(self, times, qtype, start=0):
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
        self._dict = dict(args)
        self.start = 0

    def size(self) -> int:
        return int(np.sum(q_type.size() for _, q_type in self._dict.items()))

    def get_qtypes(self):
        return self._dict.values()

    def replace_scalar(self, new_qtype):
        for key, qtype in self._dict.items():
            if isinstance(qtype, ScalarType):
                self._dict[key] = new_qtype
            else:
                qtype.replace_scalar(new_qtype)

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


def custom_copy(quantities):
    return[copy.copy(quantity) for quantity in quantities]
