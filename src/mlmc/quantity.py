import numpy as np
import operator
from inspect import signature
from memoization import cached
from typing import List
from mlmc.sample_storage import SampleStorage
from mlmc.quantity_spec import QuantitySpec
import mlmc.quantity_types as qt

CHUNK_SIZE = 512000  # bytes in decimal


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
        storage.chunk_size = CHUNK_SIZE

    dict_types = []
    for q_spec in q_specs:
        scalar_type = qt.ScalarType(float)
        array_type = qt.ArrayType(q_spec.shape, scalar_type)
        field_type = qt.FieldType([(loc, array_type) for loc in q_spec.locations])
        ts_type = qt.TimeSeriesType(q_spec.times, field_type)
        dict_types.append((q_spec.name, ts_type))
    dict_type = qt.DictType(dict_types)

    return QuantityStorage(storage, dict_type)


class Quantity:
    def __init__(self, quantity_type, operation, input_quantities=[]):
        """
        Quantity class represents real quantity and also provides operation that can be performed with stored values.
        Each Quantity has Qtype which describes its structure.
        :param quantity_type: QType instance
        :param operation: function
        :param input_quantities: List[Quantity]
        """
        self.qtype = quantity_type
        self._operation = operation
        self._input_quantities = input_quantities
        # List of quantities on which the 'self' depends, their number have to match number of arguments
        # to the operation.
        self._storage = self.get_quantity_storage()
        # QuantityStorage instance
        self._selection_id = None
        # Identifier of selection, should be set in select() method
        self._check_selection_ids()
        self._op_additional_params()

    def get_quantity_storage(self):
        """
        Get QuantityStorage instance
        :return: None, QuantityStorage
        """
        if len(self._input_quantities) == 0:
            return None
        for in_quantity in self._input_quantities:
            storage = in_quantity.get_quantity_storage()
            if storage is not None:
                self._storage = storage
                return storage
        return None

    def _check_selection_ids(self):
        """
        Make sure the all input quantities come from the same QuantityStorage
        """
        # All input quantities are QuantityConst instances
        if self._storage is None:
            return
        # Check selection ids otherwise
        for input_quantity in self._input_quantities:
            sel_id = input_quantity.selection_id()
            if sel_id is None:
                continue
            if sel_id != self.selection_id():
                raise AssertionError("Not all input quantities come from the same quantity storage")

    def _op_additional_params(self):
        """
        Handle operation additional params
        There are level_id a i_chunk params used in the sampling method and during the selection procedure
        """
        self._additional_params = {}
        sig_params = signature(self._operation).parameters
        if 'level_id' in sig_params:
            self._additional_params['level_id'] = 0
        if 'i_chunk' in sig_params:
            self._additional_params['i_chunk'] = 0

    def selection_id(self):
        """
        Get storage ids of all input quantities
        :return: List[int]
        """
        if self._selection_id is not None:
            return id(self)
        else:
            if self._storage is None:
                self._storage = self.get_quantity_storage()
            return id(self._storage)

    def size(self) -> int:
        """
        Quantity size from qtype
        :return: int
        """
        return self.qtype.size()

    def get_cache_key(self, level_id, i_chunk=0, n_samples=np.inf):
        """
        Create cache key
        :param level_id: int
        :param i_chunk: int
        :return: tuple
        """
        return (level_id, i_chunk, id(self), n_samples)  # redundant parentheses needed due to py36, py37

    @cached(custom_key_maker=get_cache_key)
    def samples(self, level_id, i_chunk=0, n_samples=np.inf):
        """
        Yields list of sample chunks for individual levels.
        Possibly calls underlying quantities.
        :param level_id: int
        :param i_chunk: int
        :return: np.ndarray or None
        """
        chunks_quantity_level = [q.samples(level_id, i_chunk) for q in self._input_quantities]
        if not self._additional_params:  # dictionary is empty
            if 'level_id' in self._additional_params:
                self._additional_params['level_id'] = level_id
            if 'i_chunk' in self._additional_params:
                self._additional_params['i_chunk'] = i_chunk
        return self._operation(*chunks_quantity_level, **self._additional_params)

    def _reduction_op(self, quantities, operation):
        """
        Check if the quantities have the same structure and same storage possibly return copy of the common quantity
        structure depending on the other quantities with given operation.
        :param quantities: List[Quantity]
        :param operation: function which is run with given quantities
        :return: Quantity
        """
        for quantity in quantities:
            if not isinstance(quantity, QuantityConst):
                return Quantity(quantity.qtype, operation=operation, input_quantities=quantities)
        # Quantity from QuantityConst instances
        return QuantityConst(quantities[0].qtype, value=operation(*[q._value for q in quantities]))

    def select(self, *args):
        """
        Performs sample selection based on conditions
        :param args: Quantity
        :return: Quantity
        """
        # args always has len() at least 1
        masks = args[0]

        for quantity in args:
            if not isinstance(quantity.qtype.base_qtype(), qt.BoolType):
                raise Exception("Quantity: {} doesn't have BoolType, instead it has QType: {}"
                                .format(quantity, quantity.qtype.base_qtype()))

        # More conditions leads to default AND
        if len(args) > 1:
            for m in args[1:]:
                masks = np.logical_and(masks, m)  # method from this module

        def op(x, mask):
            return x[..., mask, :]  # [...sample size, cut number of samples, 2]
        q = Quantity(quantity_type=self.qtype, input_quantities=[self, masks], operation=op)
        q._selection_id = id(q)
        return q

    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        return Quantity._method(ufunc, method, *args, **kwargs)

    def __add__(self, other):
        return self._reduction_op([self, Quantity.wrap(other)], Quantity.add_op)

    def __sub__(self, other):
        return self._reduction_op([self, Quantity.wrap(other)], Quantity.sub_op)

    def __mul__(self, other):
        return self._reduction_op([self, Quantity.wrap(other)], Quantity.mult_op)

    def __truediv__(self, other):
        return self._reduction_op([self, Quantity.wrap(other)], Quantity.truediv_op)

    def __mod__(self, other):
        return self._reduction_op([self, Quantity.wrap(other)], Quantity.mod_op)

    def __radd__(self, other):
        return self._reduction_op([Quantity.wrap(other), self], Quantity.add_op)

    def __rsub__(self, other):
        return self._reduction_op([Quantity.wrap(other), self], Quantity.sub_op)

    def __rmul__(self, other):
        return self._reduction_op([Quantity.wrap(other), self], Quantity.mult_op)

    def __rtruediv__(self, other):
        return self._reduction_op([Quantity.wrap(other), self], Quantity.truediv_op)

    def __rmod__(self, other):
        return self._reduction_op([Quantity.wrap(other), self], Quantity.mod_op)

    @staticmethod
    def add_op(x, y):
        return x + y

    @staticmethod
    def sub_op(x, y):
        return x - y

    @staticmethod
    def mult_op(x, y):
        return x * y

    @staticmethod
    def truediv_op(x, y):
        return x / y

    @staticmethod
    def mod_op(x, y):
        return x % y

    @staticmethod
    def _process_mask(x, y, operator, level_id):
        """
        Create samples mask
        All values for sample must meet given condition, if any value doesn't meet the condition,
        whole sample is eliminated
        :param x: Quantity chunk
        :param y: Quantity chunk or int, float
        :param operator: operator module function
        :param level_id: int, level identifier
        :return: np.ndarray of bools
        """
        # Zero level - use just fine samples
        if level_id == 0:
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
        bool_type = qt.BoolType()
        new_qtype = self.qtype
        new_qtype = qt.QType.replace_scalar(new_qtype, bool_type)
        other = Quantity.wrap(other)

        if not isinstance(self.qtype.base_qtype(), qt.ScalarType) or not isinstance(other.qtype.base_qtype(), qt.ScalarType):
            raise TypeError("Quantity has base qtype {}. "
                            "Quantities with base qtype ScalarType are the only ones that support comparison".
                            format(self.qtype.base_qtype()))
        return Quantity(quantity_type=new_qtype, input_quantities=[self, other], operation=op)

    def __lt__(self, other):
        def lt_op(x, y, level_id=0):
            return Quantity._process_mask(x, y, operator.lt, level_id)
        return self._mask_quantity(other, lt_op)

    def __le__(self, other):
        def le_op(x, y, level_id=0):
            return self._process_mask(x, y, operator.le, level_id)
        return self._mask_quantity(other, le_op)

    def __gt__(self, other):
        def gt_op(x, y, level_id=0):
            return self._process_mask(x, y, operator.gt, level_id)
        return self._mask_quantity(other, gt_op)

    def __ge__(self, other):
        def ge_op(x, y, level_id=0):
            return self._process_mask(x, y, operator.ge, level_id)
        return self._mask_quantity(other, ge_op)

    def __eq__(self, other):
        def eq_op(x, y, level_id=0):
            return self._process_mask(x, y, operator.eq, level_id)
        return self._mask_quantity(other, eq_op)

    def __ne__(self, other):
        def ne_op(x, y, level_id=0):
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

    def subsample(self, sample_vec):
        """
        Random subsampling
        :param sample_vec: list of number of samples at each level
        :return: np.ndarray
        """
        quantity_storage = self.get_quantity_storage()
        n_collected = quantity_storage.n_collected()

        rnd_indices = {}
        for level_id in self.get_quantity_storage().level_ids():
            rnd_indices[level_id] = np.sort(np.random.choice(n_collected[level_id], size=sample_vec[level_id]))

        def mask_gen(x, level_id, i_chunk, *args):
            chunks_info = quantity_storage.get_chunks_info(level_id, i_chunk)  # start and end index in collected values
            chunk_indices = list(range(*chunks_info))
            indices = np.intersect1d(rnd_indices[level_id], chunk_indices)
            final_indices = np.where(np.isin(chunk_indices, indices))[0]
            mask = np.zeros(x.shape[1], bool)

            mask[final_indices] = True
            return mask
        return self._mask_quantity(0, mask_gen)

    def __getitem__(self, key):
        """
        Get items from Quantity, quantity type must support brackets access
        :param key: str, int, tuple
        :return: Quantity
        """
        new_qtype, start = self.qtype[key]  # New quantity type

        if not isinstance(self.qtype, qt.ArrayType):
            key = slice(start, start + new_qtype.size())

        def _make_getitem_op(y):
            return self.qtype._make_getitem_op(y, key=key)

        return Quantity(quantity_type=new_qtype, input_quantities=[self], operation=_make_getitem_op)

    def __getattr__(self, name):
        static_fun = getattr(self.qtype, name)  # We support only static function call forwarding

        def apply_on_quantity(*attr, **d_attr):
            return static_fun(self, *attr, **d_attr)
        return apply_on_quantity

    @staticmethod
    def _concatenate(quantities, qtype, axis=0):
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

    @staticmethod
    def _get_base_qtype(args_quantities):
        """
        Get quantities base Qtype
        :param args_quantities: list of quantities and other passed arguments,
         we expect at least one of the arguments is Quantity
        :return: base QType, ScalarType if any quantity has that base type, otherwise BoolType
        """
        # Either all quantities are BoolType or it is considered to be ScalarType
        for quantity in args_quantities:
            if isinstance(quantity, Quantity):
                if type(quantity.qtype.base_qtype()) == qt.ScalarType:
                    return qt.ScalarType()
        return qt.BoolType()

    @staticmethod
    def _method(ufunc, method, *args, **kwargs):
        """
        Process input parameters to perform numpy ufunc.
        Get base QType of passed quantities, QuantityStorage instance, ...
        Determine the resulting QType from the first few samples
        :param ufunc: ufunc object that was called
        :param method: string, indicating which Ufunc method was called
        :param args: tuple of the input arguments to the ufunc
        :param kwargs: dictionary containing the optional input arguments of the ufunc
        :return: Quantity
        """
        def _ufunc_call(*input_quantities_chunks):
            return getattr(ufunc, method)(*input_quantities_chunks, **kwargs)

        quantities = []
        for arg in args:
            quantities.append(Quantity.wrap(arg))

        result_qtype = Quantity._result_qtype(_ufunc_call, quantities)
        return Quantity(quantity_type=result_qtype, input_quantities=list(quantities), operation=_ufunc_call)

    @staticmethod
    def wrap(value):
        """
        Convert flat, bool or array (list) to Quantity
        :param value: flat, bool, array (list) or Quantity
        :return: Quantity
        """
        if isinstance(value, Quantity):
            return value
        elif isinstance(value, (int, float)):
            quantity = QuantityConst(quantity_type=qt.ScalarType(), value=value)
        elif isinstance(value, bool):
            quantity = QuantityConst(quantity_type=qt.BoolType(), value=value)
        elif isinstance(value, (list, np.ndarray)):
            value = np.array(value)
            qtype = qt.ArrayType(shape=value.shape, qtype=qt.ScalarType())
            quantity = QuantityConst(quantity_type=qtype, value=value)
        else:
            raise ValueError("Values {} are not flat, bool or array (list)".format(value))
        return quantity

    @staticmethod
    def _result_qtype(method, quantities):
        """
        Determine QType from evaluation with given method and first few samples from storage
        :param quantities: list of Quantities
        :param method: ufunc function
        :return: QType
        """
        chunks_quantity_level = [q.samples(level_id=0, i_chunk=0, n_samples=10) for q in quantities]
        result = np.array(method(*chunks_quantity_level))  # numpy array of [M, <=10, 2]
        qtype = qt.ArrayType(shape=result.shape[0], qtype=Quantity._get_base_qtype(quantities))
        return qtype

    @staticmethod
    def QArray(quantities):
        flat_quantities = np.array(quantities).flatten()
        qtype = Quantity._check_same_qtype(flat_quantities)
        array_type = qt.ArrayType(np.array(quantities).shape, qtype)
        return Quantity._concatenate(flat_quantities, qtype=array_type)

    @staticmethod
    def QDict(key_quantity):
        dict_type = qt.DictType([(key, quantity.qtype) for key, quantity in key_quantity])
        return Quantity._concatenate(np.array(key_quantity)[:, 1], qtype=dict_type)

    @staticmethod
    def QTimeSeries(time_quantity):
        qtype = Quantity._check_same_qtype(np.array(time_quantity)[:, 1])
        times = np.array(time_quantity)[:, 0]
        return Quantity._concatenate(np.array(time_quantity)[:, 1], qtype=qt.TimeSeriesType(times=times, qtype=qtype))

    @staticmethod
    def QField(key_quantity):
        Quantity._check_same_qtype(np.array(key_quantity)[:, 1])
        field_type = qt.FieldType([(key, quantity.qtype) for key, quantity in key_quantity])
        return Quantity._concatenate(np.array(key_quantity)[:, 1], qtype=field_type)

    @staticmethod
    def _check_same_qtype(quantities):
        qtype = quantities[0].qtype
        for quantity in quantities[1:]:
            if qtype != quantity.qtype:
                raise ValueError("Quantities don't have same QType")
        return qtype


class QuantityConst(Quantity):
    def __init__(self, quantity_type, value):
        """
        QuantityConst class represents constant quantity and also provides operation
        that can be performed with quantity values.
        The quantity is constant, meaning that this class stores the data itself
        :param quantity_type: QType instance
        :param value: quantity value
        """
        self.qtype = quantity_type
        self._value = self._process_value(value)
        self._input_quantities = []
        self._selection_id = None
        # List of input quantities should be empty,
        # but we still need this attribute due to storage_id() and level_ids() method

    def _process_value(self, value):
        """
        Reshape value if array, otherwise create array first
        :param value: quantity value
        :return: value with shape [M, 1, 1] which suitable for further broadcasting
        """
        if isinstance(value, (int, float, bool)):
            value = np.array([value])
        return value[:, np.newaxis, np.newaxis]

    def get_cache_key(self, level_id, i_chunk, n_samples=np.inf):
        """
        Create cache key
        :param level_id: int
        :param i_chunk: int
        :return: tuple
        """
        return id(self)

    def selection_id(self):
        """
        Get storage ids of all input quantities
        :return: List[int]
        """
        return self._selection_id

    @cached(custom_key_maker=get_cache_key)
    def samples(self, level_id, i_chunk, n_samples=np.inf):
        """
        Get constant values with an enlarged number of axes
        :param level_id: int
        :param i_chunk: int
        :return: np.ndarray
        """
        return self._value


class QuantityMean:

    def __init__(self, quantity_type, mean, var, l_means=[], l_vars=[], n_samples=None, n_rm_samples=0):
        """
        QuantityMean represents result of estimate_mean method
        :param quantity_type: QType
        :param mean: np.ndarray
        :param var: np.ndarray
        """
        self.qtype = quantity_type
        self._mean = mean
        self._var = var
        self._l_means = np.array(l_means)
        self._l_vars = np.array(l_vars)
        self._n_samples = n_samples
        self._n_rm_samples = n_rm_samples

    def __call__(self):
        """
        Return mean
        :return:
        """
        return self._reshape(self.mean)

    @property
    def mean(self):
        return self._reshape(self._mean)

    @property
    def var(self):
        return self._reshape(self._var)

    @property
    def l_means(self):
        return self._reshape(self._l_means, levels=True)

    @property
    def l_vars(self):
        return self._reshape(self._l_vars, levels=True)

    @property
    def n_samples(self):
        return self._n_samples

    def _reshape(self, data, levels=False):
        if isinstance(self.qtype, qt.ArrayType):
            reshape_shape = self.qtype._shape
            size = self.qtype._qtype.size()
            if isinstance(reshape_shape, int):
                reshape_shape = [reshape_shape]
            if levels:
                return data.reshape((data.shape[0], *reshape_shape))
            if size > 1:
                return data.reshape(*reshape_shape, size)
            else:
                return data.reshape(*reshape_shape)
        return data

    def __getitem__(self, key):
        """
        Get items from Quantity, quantity type must support brackets access
        :param key: str, int, tuple
        :return: np.ndarray
        """
        new_qtype, start = self.qtype[key]  # New quantity type
        reshape_shape = None
        newshape = None
        # ArrayType might be accessed directly regardless of qtype start and size
        if isinstance(self.qtype, qt.ArrayType):
            slice_key = key
            reshape_shape = self.qtype._shape
            if isinstance(reshape_shape, int):
                reshape_shape = [reshape_shape]
        # Other accessible quantity types uses start and size
        else:
            end = start + new_qtype.size()
            slice_key = slice(start, end)

        mean = self._mean
        var = self._var
        l_means = self._l_means
        l_vars = self._l_vars

        if reshape_shape is not None:
            if (np.prod(mean.shape) // np.prod(reshape_shape)) > 1:
                mean = mean.reshape(*reshape_shape, np.prod(mean.shape) // np.prod(reshape_shape))
                var = var.reshape(*reshape_shape, np.prod(mean.shape) // np.prod(reshape_shape))
                l_means = l_means.reshape((l_means.shape[0], *reshape_shape, np.prod(mean.shape[1:]) // np.prod(reshape_shape)))
                l_vars = l_vars.reshape((l_vars.shape[0], *reshape_shape, np.prod(mean.shape) // np.prod(reshape_shape)))
            else:
                mean = mean.reshape(*reshape_shape)
                var = var.reshape(*reshape_shape)
                l_means = l_means.reshape((l_means.shape[0], *reshape_shape))
                l_vars = l_vars.reshape((l_vars.shape[0], *reshape_shape))

        mean_get_item = mean[slice_key].flatten()
        var_get_item = var[slice_key].flatten()

        # Handle level means and variances
        if len(l_means) > 0:
            if isinstance(slice_key, slice):
                l_means = l_means[:, slice_key]
                l_vars = l_vars[:, slice_key]
            else:
                if isinstance(slice_key, int):
                    slice_key = [slice_key]
                if len(l_means.shape) - (len(slice_key) +1) > 0:
                    l_means = l_means[(slice(0, l_means.shape[0]), *slice_key, slice(0, l_means.shape[-1]))]
                    l_vars = l_vars[(slice(0, l_vars.shape[0]), *slice_key, slice(0, l_vars.shape[-1]))]
                else:
                    l_means = l_means[(slice(0, l_means.shape[0]), *slice_key)]
                    l_vars = l_vars[(slice(0, l_vars.shape[0]), *slice_key)]

        if len(l_means.shape) > 1:
            l_means = l_means.reshape(l_means.shape[0], np.prod(l_means.shape[1:]))
            l_vars = l_vars.reshape(l_vars.shape[0], np.prod(l_vars.shape[1:]))

        return QuantityMean(quantity_type=new_qtype, mean=mean_get_item, var=var_get_item,
                            l_means=l_means, l_vars=l_vars, n_samples=self._n_samples, n_rm_samples=self._n_rm_samples)


class QuantityStorage(Quantity):
    def __init__(self, storage, qtype):
        """
        Special Quantity for direct access to SampleStorage
        :param storage: mlmc._sample_storage.SampleStorage child
        :param qtype: QType
        """
        self._storage = storage
        self.qtype = qtype
        self._input_quantities = []
        self._operation = None

    def level_ids(self):
        """
        Number of levels
        :return: List[int]
        """
        return self._storage.get_level_ids()

    def selection_id(self):
        """
        Identity of QuantityStorage instance
        :return: int
        """
        return id(self)

    def get_quantity_storage(self):
        return self

    def samples(self, level_id, i_chunk=0, n_samples=np.inf):
        """
        Get results for given level id and chunk id
        :param level_id: int
        :param i_chunk: int
        :param n_samples: int, number of retrieved samples
        :return: Array[M, chunk size, 2]
        """
        level_chunk = self._storage.sample_pairs_level(level_id, i_chunk, n_samples=n_samples)  # Array[M, chunk size, 2]
        assert self.qtype.size() == level_chunk.shape[0]
        return level_chunk

    def get_chunks_info(self, level_id, i_chunk):
        return self._storage.get_chunks_info(level_id, i_chunk)

    def n_collected(self):
        return self._storage.get_n_collected()



