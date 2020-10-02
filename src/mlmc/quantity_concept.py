import abc
import numpy as np
import copy
import operator
from inspect import signature
from memoization import cached
from scipy import interpolate
from typing import List, Tuple
from mlmc.sample_storage import SampleStorage
from mlmc.sim.simulation import QuantitySpec


def _method(ufunc, method, *args, **kwargs):
    """
    To process input quantities to perform numpy method
    :param args: args
    :return: Quantity
    """
    def _aux_method(*input_quantities_chunks):
        if len(input_quantities_chunks) == 1:
            return getattr(ufunc, method)(input_quantities_chunks[0], **kwargs)
        else:
            return getattr(ufunc, method)(*input_quantities_chunks, **kwargs)

    if all(isinstance(quantity, Quantity) for quantity in args):
        assert all(isinstance(quantity.qtype.base_qtype(), BoolType) for quantity in args) or\
               all(isinstance(quantity.qtype.base_qtype(), ScalarType) for quantity in args), \
            "All quantities must have same base type either ScalarType or BoolType"

        return Quantity(quantity_type=args[0].qtype, input_quantities=list(args), operation=_aux_method)
    else:
        raise ValueError("All args parameters must be quantities")


def make_root_quantity(storage: SampleStorage, q_specs: List[QuantitySpec]):
    """
    Create a root quantity that has QuantityStorage as the input quantity,
    QuantityStorage is the only class that directly accesses the stored data.
    Quantity type is created based on the q_spec parameter
    :param storage: SampleStorage
    :param q_specs: same as result format in simulation class
    :return: QuantityStorage
    """
    # Set chunk size as the case may be
    if storage.chunk_size is None:
        storage.chunk_size = 512000  # bytes in decimal

    dict_types = []
    for q_spec in q_specs:
        scalar_type = ScalarType(float)
        array_type = ArrayType(q_spec.shape, scalar_type)
        field_type = FieldType([(loc, array_type) for loc in q_spec.locations])
        ts_type = TimeSeriesType(q_spec.times, field_type)
        dict_types.append((q_spec.name, ts_type))
    dict_type = DictType(dict_types)

    return QuantityStorage(storage, dict_type)


def remove_nan_samples(chunk):
    """
    Mask out samples that contain NaN in either fine or coarse part of the result
    :param chunk: np.ndarray [M, chunk_size, 2]
    :return: np.ndarray
    """
    # Fine and coarse moments mask
    mask = np.any(np.isnan(chunk), axis=0)
    m = ~mask.any(axis=1)
    return chunk[..., m, :]


def estimate_mean(quantity):
    """
    MLMC mean estimator.
    The MLMC method is used to compute the mean estimate to the Quantity dependent on the collected samples.
    The squared error of the estimate (the estimator variance) is estimated using the central limit theorem.
    Data is processed by chunks, so that it also supports big data processing
    :param quantity: Quantity
    :return: QuantityMean which holds both mean and variance
    """
    Quantity.samples.cache_clear()
    quantity_vec_size = quantity.size()
    n_samples = None
    sums = None
    sums_of_squares = None
    i_chunk = 0
    level_chunks_none = np.zeros(1)  # if ones than all level chunks are empty (None)

    while not np.alltrue(level_chunks_none):
        level_ids = quantity.level_ids()
        if i_chunk == 0:
            # initialization
            n_levels = len(level_ids)
            n_samples = [0] * n_levels

        level_chunks_none = np.zeros(n_levels)
        for level_id in level_ids:
            # Chunk of samples for given level id
            chunk = quantity.samples(level_id, i_chunk)

            if chunk is not None:
                if level_id == 0:
                    # Set variables for level sums and sums of powers
                    if i_chunk == 0:
                        sums = [np.zeros(chunk.shape[0]) for _ in range(n_levels)]
                        sums_of_squares = [np.zeros(chunk.shape[0]) for _ in range(n_levels)]

                    # Coarse result for level 0, there is issue for moments processing (not know about level)
                    chunk[..., 1] = 0

                chunk = remove_nan_samples(chunk)
                # level_chunk is Numpy Array with shape [M, chunk_size, 2]
                n_samples[level_id] += chunk.shape[1]

                assert(chunk.shape[0] == quantity_vec_size)
                sums[level_id] += np.sum(chunk[:, :, 0] - chunk[:, :, 1], axis=1)
                sums_of_squares[level_id] += np.sum((chunk[:, :, 0] - chunk[:, :, 1])**2, axis=1)
            else:
                level_chunks_none[level_id] = True

        i_chunk += 1

    mean = np.zeros_like(sums[0])
    var = np.zeros_like(sums[0])

    for s, sp, n in zip(sums, sums_of_squares, n_samples):
        mean += s / n
        var += (sp - (s**2/n)) / ((n-1)*n)

    return quantity.create_quantity_mean(mean=mean, var=var)


def moment(quantity, moments_fn, i=0):
    """
    Create quantity with operation that evaluates particular moment
    :param quantity: Quantity instance
    :param moments_fn: mlmc.moments.Moments child
    :param i: index of moment
    :return: Quantity
    """
    def eval_moment(x):
        return moments_fn.eval_single_moment(i, value=x)
    return Quantity(quantity_type=quantity.qtype, input_quantities=[quantity], operation=eval_moment)


def moments(quantity, moments_fn, mom_at_bottom=True):
    """
    Create quantity with operation that evaluates moments
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

    # Create quantity type which has moments at the bottom
    if mom_at_bottom:
        moments_array_type = ArrayType(shape=(moments_fn.size,), qtype=ScalarType())
        moments_qtype = copy.deepcopy(quantity.qtype)
        moments_qtype.replace_scalar(moments_array_type)
    # Create quantity type that has moments on the surface
    else:
        moments_qtype = ArrayType(shape=(moments_fn.size,), qtype=quantity.qtype)
    return Quantity(quantity_type=moments_qtype, input_quantities=[quantity], operation=eval_moments)


def covariance(quantity, moments_fn, cov_at_bottom=True):
    """
    Create quantity with operation that evaluates covariance matrix
    :param quantity: Quantity
    :param moments_fn: mlmc.moments.Moments child
    :param cov_at_bottom: bool, if True cov matrices are underneath
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

    # Create quantity type which has covariance matrices at the bottom
    if cov_at_bottom:
        moments_array_type = ArrayType(shape=(moments_fn.size, moments_fn.size, ), qtype=ScalarType())
        moments_qtype = copy.deepcopy(quantity.qtype)
        moments_qtype.replace_scalar(moments_array_type)
    # Create quantity type that has covariance matrices on the surface
    else:
        moments_qtype = ArrayType(shape=(moments_fn.size, moments_fn.size, ), qtype=quantity.qtype)

    return Quantity(quantity_type=moments_qtype, input_quantities=[quantity], operation=eval_cov)


class Quantity:

    def __init__(self, quantity_type, input_quantities=None, operation=None):
        """
        Quantity class represents real quantity and also provides operation that can be performed with stored values.
        Each Quantity has Qtype which describes its structure.
        :param quantity_type: QType instance
        :param input_quantities: List[Quantity]
        :param operation: function
        """
        self.qtype = quantity_type
        self._operation = operation
        self._input_quantities = input_quantities
        # List of quantities on which the 'self' depends, their number have to match number of arguments
        # to the operation.

    def storage_id(self):
        """
        Get storage ids of all input quantities
        :return: List[int]
        """
        st_ids = []
        for input_quantity in self._input_quantities:
            st_id = input_quantity.storage_id()
            st_ids.extend(st_id if type(st_id) == list else [st_id])
        return st_ids

    def size(self) -> int:
        """
        Quantity size from qtype
        :return: int
        """
        return self.qtype.size()

    def get_cache_key(self, level_id, i_chunk):
        """
        Create cache key
        :param level_id: int
        :param i_chunk: int
        :return: tuple
        """
        return (level_id, i_chunk, id(self))  # redundant parentheses needed due to py36, py37

    @cached(custom_key_maker=get_cache_key)
    def samples(self, level_id, i_chunk):
        """
        Yields list of sample chunks for individual levels.
        Possibly calls underlying quantities.
        :param level_id: int
        :param i_chunk: int
        :return: np.ndarray
        """
        if not all(np.allclose(q.storage_id(), self._input_quantities[0].storage_id()) for q in self._input_quantities):
            raise Exception("Not all input quantities come from the same quantity storage")

        chunks_quantity_level = [q.samples(level_id, i_chunk) for q in self._input_quantities]

        is_valid = (ch is not None for ch in chunks_quantity_level)
        if any(is_valid):
            assert (all(is_valid))
            # Operation not set return first quantity samples - used in make_root_quantity
            if self._operation is None:
                return chunks_quantity_level[0]

            additional_params = {}
            try:
                if self._operation is not None:
                    sig_params = signature(self._operation).parameters
                    if 'level_id' in sig_params:
                        additional_params['level_id'] = level_id
                    if 'i_chunk' in sig_params:
                        additional_params['i_chunk'] = i_chunk
            except:
                pass
            return self._operation(*chunks_quantity_level, **additional_params)
        else:
            return None

    def create_quantity_mean(self, mean: np.ndarray, var: np.ndarray):
        """
        Crate a new quantity with the same structure but containing fixed data vector.
        Primary usage is to organise computed means and variances.
        Can possibly be used also to organise single sample row.
        :param mean: np.ndarray
        :param var: np.ndarray
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
        assert all(q.size() == quantities[0].size() for q in quantities), "Quantities don't have have the same size"
        return Quantity(quantities[0].qtype, operation=operation, input_quantities=quantities)

    def select(self, *args):
        """
        Performs sample selection based on conditions
        :param args: Quantity
        :return: Quantity
        """
        # args always has len() at least 1
        masks = args[0]

        for quantity in args:
            if not isinstance(quantity.qtype.base_qtype(), BoolType):
                raise Exception("Quantity: {} doesn't have BoolType, instead it has QType: {}"
                                .format(quantity, quantity.qtype.base_qtype()))

        # More conditions leads to default AND
        if len(args) > 1:
            for m in args[1:]:
                masks = np.logical_and(masks, m)  # method from this module

        def op(x, level_id, i_chunk):
            mask = masks.samples(level_id, i_chunk)
            if mask is not None:
                return x[..., mask, :]  # [...sample size, cut number of samples, 2]
            return x

        return Quantity(quantity_type=self.qtype, input_quantities=[self], operation=op)

    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        return _method(ufunc, method, *args, **kwargs)

    def __add__(self, other):
        def add_op(x, y):
            return x + y
        return self._reduction_op([self, other], add_op)

    def __sub__(self, other):
        def sub_op(x, y):
            return x + y
        return self._reduction_op([self, other], sub_op)

    def __mul__(self, other):
        if isinstance(other, (float, int)):
            return self.__const_mult(other)

        def mult_op(x, y):
            return x * y
        return self._reduction_op([self, other], mult_op)

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

    @staticmethod
    def _process_mask(x, y, operator, level_id):
        """
        Create samples mask
        All values for sample must meet given condition, if any value doesn't meet the condition,
        whole sample is eliminated
        :param x: Quantity
        :param y: Quantity or int, float
        :param operator: operator module function
        :param level_id: int, level identifier
        :return: np.ndarray of bools
        """
        # Zero level - use just fine samples
        if level_id == 0:
            if isinstance(y, int) or isinstance(y, float):
                mask = operator(x[..., 0], y)  # y is int or float
            else:
                mask = operator(x[..., 0], y[..., 0])  # y is from other quantity
            return mask.all(axis=tuple(range(mask.ndim - 1)))

        mask = operator(x, y)
        return mask.all(axis=tuple(range(mask.ndim - 2))).all(axis=1)

    def _mask_quantity(self, other, op):
        """
        Create quantity that represent bool mask
        :param other: number or Quantity
        :param op: operation
        :return: Quantity
        """
        bool_type = BoolType()
        new_qtype = copy.deepcopy(self.qtype)
        new_qtype.replace_scalar(bool_type)

        if isinstance(other, (float, int)):
            if not isinstance(self.qtype.base_qtype(), ScalarType):
                raise TypeError("Quantity has base qtype {}. "
                                "Quantities with base qtype ScalarType are the only ones that support comparison".
                                format(self.qtype.base_qtype()))

            return Quantity(quantity_type=new_qtype, input_quantities=[self], operation=op)
        elif isinstance(other, Quantity):
            if self.qtype.size() != other.qtype.size():
                raise Exception("Quantities don't have have the same size")
            if not isinstance(self.qtype.base_qtype(), ScalarType) or not isinstance(other.qtype.base_qtype(), ScalarType):
                raise TypeError("Quantity has base qtype {}. "
                                "Quantities with base qtype ScalarType are the only ones that support comparison".
                                format(self.qtype.base_qtype()))
            return Quantity(quantity_type=new_qtype, input_quantities=[self, other], operation=op)

    def __lt__(self, other):
        def lt_op(x, y=other, level_id=0):
            return Quantity._process_mask(x, y, operator.lt, level_id)
        return self._mask_quantity(other, lt_op)

    def __le__(self, other):
        def le_op(x, y=other, level_id=0):
            return self._process_mask(x, y, operator.le, level_id)
        return self._mask_quantity(other, le_op)

    def __gt__(self, other):
        def gt_op(x, y=other, level_id=0):
            return self._process_mask(x, y, operator.gt, level_id)
        return self._mask_quantity(other, gt_op)

    def __ge__(self, other):
        def ge_op(x, y=other, level_id=0):
            return self._process_mask(x, y, operator.ge, level_id)
        return self._mask_quantity(other, ge_op)

    def __eq__(self, other):
        def eq_op(x, y=other, level_id=0):
            return self._process_mask(x, y, operator.eq, level_id)
        return self._mask_quantity(other, eq_op)

    def __ne__(self, other):
        def ne_op(x, y=other, level_id=0):
            return self._process_mask(x, y, operator.ne, level_id)
        return self._mask_quantity(other, ne_op)

    def sampling(self, size):
        """
        Random sampling
        :param size: number of samples
        :return: np.ndarray
        """
        def mask_gen(x, *args):
            indices = np.random.choice(x.shape[1], size=size)
            mask = np.zeros(x.shape[1], bool)
            mask[indices] = True
            return mask
        return self._mask_quantity(size, mask_gen)

    def __getitem__(self, key):
        """
        Get items from Quantity, quantity type must support brackets access
        :param key: str, int, tuple
        :return: Quantity
        """
        new_qtype = self.qtype[key]  # New quantity type

        def _make_getitem_op(y):
            return self.qtype._make_getitem_op(y, new_qtype=new_qtype, key=key)

        return Quantity(quantity_type=new_qtype, input_quantities=[self], operation=_make_getitem_op)

    def __iter__(self):
        raise Exception("This class is not iterable")

    def __copy__(self):
        return Quantity(quantity_type=self.qtype, input_quantities=self._input_quantities, operation=self._operation)

    def time_interpolation(self, value):
        """
        Interpolation in time
        :param value: point where to interpolate
        :return: Quantity
        """
        def interp(y):
            split_indeces = np.arange(1, len(self.qtype._times)) * self.qtype._qtype.size()
            y = np.split(y, split_indeces, axis=-3)
            f = interpolate.interp1d(self.qtype._times, y, axis=0)
            return f(value)

        return Quantity(quantity_type=self.qtype._qtype, input_quantities=[self], operation=interp)

    def level_ids(self):
        """
        List of level ids, all input quantities must be from the same storage,
        so getting level IDs from one of them should be completely fine
        :return: List[int]
        """
        return self._input_quantities[0].level_ids()

    @staticmethod
    def concatenate(quantities, qtype, axis=0):
        """
        Concatenate level_chunks
        :param quantities: list of quantities
        :param qtype: QType
        :param axis: int
        :return: Quantity
        """
        def op_concatenate(*chunks):
            y = np.concatenate(tuple(chunks), axis=axis)
            return y
        return Quantity(qtype, input_quantities=[*quantities], operation=op_concatenate)


class QuantityMean:

    def __init__(self, quantity_type, mean, var):
        """
        QuantityMean represents result of estimate_mean method
        :param quantity_type: QType
        :param mean: np.ndarray
        :param var: np.ndarray
        """
        self.qtype = quantity_type
        self._mean = mean
        self._var = var

    def __call__(self):
        """
        Return mean
        :return:
        """
        return self.mean()

    def var(self):
        return self._var

    def mean(self):
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
        """
        Special Quantity for direct access to SampleStorage
        :param storage: mlmc.sample_storage.SampleStorage child
        :param qtype: QType
        """
        self._storage = storage
        self.qtype = qtype
        self._input_quantities = []
        self._operation = None
        self.start = None
        self.end = None

    def level_ids(self):
        """
        Number of levels
        :return: List[int]
        """
        return self._storage.get_level_ids()

    def storage_id(self):
        """
        Identity of QuantityStorage instance
        :return: int
        """
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

    def base_qtype(self):
        return self._qtype.base_qtype()

    def __eq__(self, other):
        if isinstance(other, QType):
            return self.size() == other.size()
        return False

    def replace_scalar(self, new_qtype):
        """
        Find ScalarType and replace it with new_qtype
        :param new_qtype: QType
        :return: None
        """
        if isinstance(self._qtype, ScalarType):
            self._qtype = new_qtype
        else:
            self._qtype.replace_scalar(new_qtype)

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
        return self

    def size(self) -> int:
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
