import operator
import numpy as np
import scipy.stats
from memoization import cached
from typing import List
from mlmc.sample_storage import SampleStorage
from mlmc.quantity.quantity_spec import QuantitySpec, ChunkSpec
import mlmc.quantity.quantity_types as qt


RNG = np.random.default_rng()


def make_root_quantity(storage: SampleStorage, q_specs: List[QuantitySpec]):
    """
    Create a root quantity that has QuantityStorage as the input quantity,
    QuantityStorage is the only class that directly accesses the stored data.
    Quantity type is created based on the q_spec parameter
    :param storage: SampleStorage
    :param q_specs: same as result format in simulation class
    :return: QuantityStorage
    """
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
        self._selection_id = self.set_selection_id()
        # Identifier of selection, should be set in select() method
        self._check_selection_ids()

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

    def set_selection_id(self):
        """
        Set selection id
        selection id is None by default,
         but if we create new quantity from quantities that are result of selection we need to pass selection id
        """
        selection_id = None
        for input_quantity in self._input_quantities:
            if selection_id is None:
                selection_id = input_quantity.selection_id()
            elif input_quantity.selection_id() is not None and selection_id != input_quantity.selection_id():
                raise Exception("Different selection IDs among input quantities")
        return selection_id

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

    def selection_id(self):
        """
        Get storage ids of all input quantities
        :return: List[int]
        """
        if self._selection_id is not None:
            return self._selection_id
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

    def get_cache_key(self, chunk_spec):
        """
        Create cache key
        """
        chunk_size = None
        if chunk_spec.chunk_slice is not None:
            chunk_size = chunk_spec.chunk_slice.stop - chunk_spec.chunk_slice.start
        return (chunk_spec.level_id, chunk_spec.chunk_id, chunk_size, id(self))  # redundant parentheses needed due to py36, py37

    @cached(custom_key_maker=get_cache_key)
    def samples(self, chunk_spec):
        """
        Return list of sample chunks for individual levels.
        Possibly calls underlying quantities.
        :param chunk_spec: object containing chunk identifier level identifier and chunk_slice - slice() object
        :return: np.ndarray or None
        """
        chunks_quantity_level = [q.samples(chunk_spec) for q in self._input_quantities]
        return self._operation(*chunks_quantity_level)

    def _reduction_op(self, quantities, operation):
        """
        :param quantities: List[Quantity]
        :param operation: function which is run with given quantities
        :return: Quantity or QuantityConst
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
        return Quantity.create_quantity([self, Quantity.wrap(other)], Quantity.add_op)

    def __sub__(self, other):
        return Quantity.create_quantity([self, Quantity.wrap(other)], Quantity.sub_op)

    def __mul__(self, other):
        return Quantity.create_quantity([self, Quantity.wrap(other)], Quantity.mult_op)

    def __truediv__(self, other):
        return Quantity.create_quantity([self, Quantity.wrap(other)], Quantity.truediv_op)

    def __mod__(self, other):
        return Quantity.create_quantity([self, Quantity.wrap(other)], Quantity.mod_op)

    def __radd__(self, other):
        return Quantity.create_quantity([Quantity.wrap(other), self], Quantity.add_op)

    def __rsub__(self, other):
        return Quantity.create_quantity([Quantity.wrap(other), self], Quantity.sub_op)

    def __rmul__(self, other):
        return Quantity.create_quantity([Quantity.wrap(other), self], Quantity.mult_op)

    def __rtruediv__(self, other):
        return Quantity.create_quantity([Quantity.wrap(other), self], Quantity.truediv_op)

    def __rmod__(self, other):
        return Quantity.create_quantity([Quantity.wrap(other), self], Quantity.mod_op)

    @staticmethod
    def create_quantity(quantities, operation):
        """
        Create new quantity (Quantity or QuantityConst) based on given quantities and operation.
        There are two scenarios:
        1. At least one of quantities is Quantity instance then all quantities are considered to be input_quantities
         of new Quantity
        2. All of quantities are QuantityConst instances then new QuantityConst is created
        :param quantities: List[Quantity]
        :param operation: function which is run with given quantities
        :return: Quantity
        """
        for quantity in quantities:
            if not isinstance(quantity, QuantityConst):
                return Quantity(quantity.qtype, operation=operation, input_quantities=quantities)
        # Quantity from QuantityConst instances
        return QuantityConst(quantities[0].qtype, value=operation(*[q._value for q in quantities]))

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
    def _process_mask(x, y, operator):
        """
        Create samples mask
        All values for sample must meet given condition, if any value doesn't meet the condition,
        whole sample is eliminated
        :param x: Quantity chunk
        :param y: Quantity chunk or int, float
        :param operator: operator module function
        :return: np.ndarray of bools
        """
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
        new_qtype = new_qtype.replace_scalar(bool_type)
        other = Quantity.wrap(other)

        if not isinstance(self.qtype.base_qtype(), qt.ScalarType) or not isinstance(other.qtype.base_qtype(), qt.ScalarType):
            raise TypeError("Quantity has base qtype {}. "
                            "Quantities with base qtype ScalarType are the only ones that support comparison".
                            format(self.qtype.base_qtype()))
        return Quantity(quantity_type=new_qtype, input_quantities=[self, other], operation=op)

    def __lt__(self, other):
        def lt_op(x, y):
            return Quantity._process_mask(x, y, operator.lt)
        return self._mask_quantity(other, lt_op)

    def __le__(self, other):
        def le_op(x, y):
            return self._process_mask(x, y, operator.le)
        return self._mask_quantity(other, le_op)

    def __gt__(self, other):
        def gt_op(x, y):
            return self._process_mask(x, y, operator.gt)
        return self._mask_quantity(other, gt_op)

    def __ge__(self, other):
        def ge_op(x, y):
            return self._process_mask(x, y, operator.ge)
        return self._mask_quantity(other, ge_op)

    def __eq__(self, other):
        def eq_op(x, y):
            return self._process_mask(x, y, operator.eq)
        return self._mask_quantity(other, eq_op)

    def __ne__(self, other):
        def ne_op(x, y):
            return self._process_mask(x, y, operator.ne)
        return self._mask_quantity(other, ne_op)

    @staticmethod
    def pick_samples(chunk, subsample_params):
        """
        Pick samples some samples from chunk in order to have 'k' samples from 'n' after all chunks are processed
        Inspired by https://dl.acm.org/doi/10.1145/23002.23003 method S

        :param chunk: np.ndarray, shape M, N, 2, where N denotes number of samples in chunk
        :param subsample_params: instance of SubsampleParams class, it has two parameters:
                        k: number of samples which we want to get from all chunks
                        n: number of all samples among all chunks
        :return: np.ndarray
        """
        size = scipy.stats.hypergeom(subsample_params.n, subsample_params.k, chunk.shape[1]).rvs(size=1)
        out = RNG.choice(chunk, size=size, axis=1)
        subsample_params.k -= out.shape[1]
        subsample_params.n -= chunk.shape[1]
        return out

    def subsample(self, sample_vec):
        """
        Subsampling
        :param sample_vec: list of number of samples at each level
        :return: Quantity
        """
        class SubsampleParams:
            def __init__(self, num_subsample, num_collected):
                """
                Auxiliary object for subsampling
                :param num_subsample: the number of samples we want to obtain from all samples
                :param num_collected: total number of samples
                """
                self._orig_k = num_subsample
                self._orig_n = num_collected
                self._orig_total_n = num_collected
                self.k = num_subsample
                self.n = num_collected
                self.total_n = num_collected

        # SubsampleParams for each level
        subsample_level_params = {key: SubsampleParams(sample_vec[key], value)
                                  for key, value in enumerate(self.get_quantity_storage().n_collected())}
        # Create a QuantityConst of dictionary in the sense of hashing dictionary items
        quantity_subsample_params = Quantity.wrap(hash(frozenset(subsample_level_params.items())))

        def adjust_value(values, level_id):
            """
            Custom implementation of QuantityConst.adjust_value()
            It allows us to get different parameters for different levels
            """
            subsample_l_params_obj = subsample_level_params[level_id]
            subsample_l_params_obj.k = subsample_l_params_obj._orig_k
            subsample_l_params_obj.n = subsample_l_params_obj._orig_n
            subsample_l_params_obj.total_n = subsample_l_params_obj._orig_total_n
            return subsample_l_params_obj
        quantity_subsample_params._adjust_value = adjust_value

        return Quantity(quantity_type=self.qtype.replace_scalar(qt.BoolType()),
                        input_quantities=[self, quantity_subsample_params], operation=Quantity.pick_samples)

    def __getitem__(self, key):
        """
        Get items from Quantity, quantity type must support brackets access
        :param key: str, int, tuple
        :return: Quantity
        """
        new_qtype, start = self.qtype.get_key(key)  # New quantity type

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
        chunks_quantity_level = []
        for q in quantities:
            quantity_storage = q.get_quantity_storage()
            # QuantityConst doesn't have QuantityStorage
            if quantity_storage is None:
                chunk_spec = ChunkSpec()
            else:
                chunk_spec = next(quantity_storage.chunks())
            chunks_quantity_level.append(q.samples(chunk_spec))

        result = method(*chunks_quantity_level)  # numpy array of [M, <=10, 2]
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
        # List of input quantities should be empty,
        # but we still need this attribute due to storage_id() and level_ids() method
        self._selection_id = None

    def _process_value(self, value):
        """
        Reshape value if array, otherwise create array first
        :param value: quantity value
        :return: value with shape [M, 1, 1] which suitable for further broadcasting
        """
        if isinstance(value, (int, float, bool)):
            value = np.array([value])
        return value[:, np.newaxis, np.newaxis]

    def selection_id(self):
        """
        Get storage ids of all input quantities
        :return: List[int]
        """
        return self._selection_id

    def _adjust_value(self, value, level_id=None):
        """
        Allows process value based on chunk_epc params (such as level_id, ...).
        The custom implementation is used in Qunatity.subsample method
        :param value: np.ndarray
        :param level_id: int
        :return: np.ndarray, particular type depends on implementation
        """
        return value

    @cached(custom_key_maker=Quantity.get_cache_key)
    def samples(self, chunk_spec):
        """
        Get constant values with an enlarged number of axes
        :param chunk_spec: object containing chunk identifier level identifier and chunk_slice - slice() object
        :return: np.ndarray
        """
        return self._adjust_value(self._value, chunk_spec.level_id)


class QuantityMean:

    def __init__(self, quantity_type, l_means, l_vars, n_samples, n_rm_samples):
        """
        QuantityMean represents results of mlmc.quantity_estimate.estimate_mean method
        :param quantity_type: QType
        :param l_means: np.ndarray, shape: L, M
        :param l_vars: np.ndarray, shape: L, M
        :param n_samples: List, number of samples that were used for means at each level
        :param n_rm_samples: List, number of removed samples at each level,
                             n_samples + n_rm_samples = all successfully collected samples
        """
        self.qtype = quantity_type
        self._mean = None
        self._var = None
        self._l_means = np.array(l_means)
        self._l_vars = np.array(l_vars)
        self._n_samples = np.array(n_samples)
        self._n_rm_samples = np.array(n_rm_samples)

    def _calculate_mean_var(self):
        """
        Calculates the overall estimates of the mean and the variance from the means and variances at each level
        """
        self._mean = np.sum(self._l_means, axis=0)
        self._var = np.sum(self._l_vars / self._n_samples[:, None], axis=0)

    @property
    def mean(self):
        if self._mean is None:
            self._calculate_mean_var()
        return self._reshape(self._mean)

    @property
    def var(self):
        if self._var is None:
            self._calculate_mean_var()
        return self._reshape(self._var)

    @property
    def l_means(self):
        return np.array([self._reshape(means) for means in self._l_means])

    @property
    def l_vars(self):
        return np.array([self._reshape(vars) for vars in self._l_vars])

    @property
    def n_samples(self):
        return self._n_samples

    @property
    def n_rm_samples(self):
        return self._n_rm_samples

    def _reshape(self, data):
        """
        Reshape passed data, expected means or vars
        :param data: flatten np.ndarray
        :return: np.ndarray, reshaped data, the final data shape depends on the particular QType
                             there is currently a reshape for ArrayType only
        """
        return self.qtype.reshape(data)

    def __getitem__(self, key):
        """
        Get item from current QuantityMean, quantity type must support brackets access
        All levels means and vars are reshaped to their QType shape and then the item is gotten,
        ath the end, new QuantityMean instance is created with flatten selected means and vars
        :param key: str, int, tuple
        :return: QuantityMean
        """
        new_qtype, start = self.qtype.get_key(key)  # New quantity type

        if not isinstance(self.qtype, qt.ArrayType):
            key = slice(start, start + new_qtype.size())

        # Getting items, it performs reshape inside
        l_means = self.l_means[:, key]
        l_vars = self.l_vars[:, key]

        return QuantityMean(quantity_type=new_qtype, l_means=l_means.reshape((l_means.shape[0], -1)),
                            l_vars=l_vars.reshape((l_vars.shape[0], -1)), n_samples=self._n_samples,
                            n_rm_samples=self._n_rm_samples)


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

    def chunks(self, level_id=None):
        return self._storage.chunks(level_id)

    def samples(self, chunk_spec):
        """
        Get results for given level id and chunk id
        :param chunk_spec: object containing chunk identifier level identifier and chunk_slice - slice() object
        :return: Array[M, chunk size, 2]
        """
        return self._storage.sample_pairs_level(chunk_spec)  # Array[M, chunk size, 2]

    def n_collected(self):
        return self._storage.get_n_collected()
