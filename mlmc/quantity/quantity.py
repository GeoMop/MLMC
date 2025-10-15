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
    Create a root quantity that has QuantityStorage as the input quantity.
    QuantityStorage is the only class that directly accesses the stored data.
    The returned QuantityStorage uses a QType built from provided QuantitySpec objects.

    :param storage: SampleStorage instance that provides stored samples
    :param q_specs: list of QuantitySpec describing the simulation result format
    :return: QuantityStorage that wraps the provided SampleStorage with a matching QType
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
    """
    Represents a quantity (a measurable value or expression) constructed from a QType,
    an operation (callable) and zero-or-more input quantities. Quantities are lazy:
    their actual data are returned by calling `samples(chunk_spec)`.

    - qtype: structure description (QType)
    - _operation: callable that takes sample-chunks from input_quantities and returns result chunks
    - _input_quantities: dependencies (other Quantity instances)
    """

    def __init__(self, quantity_type, operation, input_quantities=[]):
        """
        :param quantity_type: QType instance describing the shape/structure
        :param operation: callable implementing the transform on input chunks
        :param input_quantities: List[Quantity] dependencies (may be empty for constants)
        """
        self.qtype = quantity_type
        self._operation = operation
        self._input_quantities = input_quantities
        # Underlying QuantityStorage (inherited from one of the inputs, if present)
        self._storage = self.get_quantity_storage()
        # Selection identifier - used to tie selections together (set by select)
        self._selection_id = self.set_selection_id()
        # Validate that input quantities use consistent selection/storage
        self._check_selection_ids()

    def get_quantity_storage(self):
        """
        Find the first QuantityStorage among inputs (if any) and return it.

        :return: QuantityStorage instance or None if not found
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
        Determine the selection id for this Quantity. If inputs have a selection id
        (created by select), inherit it; if multiple different selection ids are
        present among inputs, raise an exception.

        :return: selection id or None
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
        Ensure that all input quantities that have selection ids share the same one.
        If no QuantityStorage is present, nothing to check.
        """
        if self._storage is None:
            return
        for input_quantity in self._input_quantities:
            sel_id = input_quantity.selection_id()
            if sel_id is None:
                continue
            if sel_id != self.selection_id():
                raise AssertionError("Not all input quantities come from the same quantity storage")

    def selection_id(self):
        """
        Return this Quantity's selection id. If not set, use id(self._storage) to
        identify the underlying storage instance.

        :return: selection identifier (int or None)
        """
        if self._selection_id is not None:
            return self._selection_id
        else:
            if self._storage is None:
                self._storage = self.get_quantity_storage()
            return id(self._storage)

    def size(self) -> int:
        """
        Return the number of scalar components described by the QType.

        :return: int
        """
        return self.qtype.size()

    def get_cache_key(self, chunk_spec):
        """
        Create a cache key used by memoization for samples. We include:
          - level id
          - chunk id
          - chunk size (derived from slice)
          - id(self) to distinguish different quantity instances

        :param chunk_spec: ChunkSpec
        :return: tuple key
        """
        chunk_size = None
        if chunk_spec.chunk_slice is not None:
            chunk_size = chunk_spec.chunk_slice.stop - chunk_spec.chunk_slice.start
        return (chunk_spec.level_id, chunk_spec.chunk_id, chunk_size, id(self))  # py36/37 compatibility

    @cached(custom_key_maker=get_cache_key)
    def samples(self, chunk_spec):
        """
        Evaluate and return the data chunk for this quantity at the specified chunk_spec.
        Calls samples(chunk_spec) recursively on inputs and passes the results to _operation.

        :param chunk_spec: ChunkSpec object with level_id, chunk_id, and optional slice
        :return: np.ndarray (M, chunk_size, 2) or None
        """
        chunks_quantity_level = [q.samples(chunk_spec) for q in self._input_quantities]
        return self._operation(*chunks_quantity_level)

    def _reduction_op(self, quantities, operation):
        """
        Helper for building a reduction Quantity from many inputs.

        If any input is a non-constant Quantity, return a Quantity with the operation and inputs.
        If all inputs are QuantityConst, evaluate the operation immediately and return QuantityConst.

        :param quantities: List[Quantity]
        :param operation: Callable to apply
        :return: Quantity or QuantityConst
        """
        for quantity in quantities:
            if not isinstance(quantity, QuantityConst):
                return Quantity(quantity.qtype, operation=operation, input_quantities=quantities)
        # All constant -> precompute value
        return QuantityConst(quantities[0].qtype, value=operation(*[q._value for q in quantities]))

    def select(self, *args):
        """
        Apply boolean selection masks to this Quantity's samples.

        :param args: One or more Quantity instances with BoolType that act as masks.
        :return: Quantity representing the selected samples (mask applied on sample axis)
        """
        # First mask
        masks = args[0]

        # Validate masks are BoolType
        for quantity in args:
            if not isinstance(quantity.qtype.base_qtype(), qt.BoolType):
                raise Exception("Quantity: {} doesn't have BoolType, instead it has QType: {}"
                                .format(quantity, quantity.qtype.base_qtype()))

        # Combine multiple masks with logical AND
        if len(args) > 1:
            for m in args[1:]:
                masks = np.logical_and(masks, m)

        def op(x, mask):
            # Mask samples (reduce number of sample columns)
            return x[..., mask, :]  # [..., selected_samples, 2]

        q = Quantity(quantity_type=self.qtype, input_quantities=[self, masks], operation=op)
        q._selection_id = id(q)  # mark selection id to ensure consistency
        return q

    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        """
        Support numpy ufuncs by routing them through _method which constructs a new Quantity.
        """
        return Quantity._method(ufunc, method, *args, **kwargs)

    # Arithmetic operator wrappers - build new Quantities or constants as needed.
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
        Create a new Quantity or QuantityConst. If any input is non-constant, return
        a Quantity that will evaluate lazily. If all are constant, return QuantityConst.

        :param quantities: list-like of Quantity / QuantityConst
        :param operation: callable to combine inputs
        :return: Quantity or QuantityConst
        """
        for quantity in quantities:
            if not isinstance(quantity, QuantityConst):
                return Quantity(quantity.qtype, operation=operation, input_quantities=quantities)
        # all constant -> precompute
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
        Create a boolean mask that marks full samples passing the given per-element condition.

        The operator is applied elementwise; then we require that *every* element within the sample
        passes to keep that sample. This collapses non-sample axes and returns a 1-D boolean array.

        :param x: Quantity chunk (ndarray)
        :param y: Quantity chunk or scalar
        :param operator: operator module function like operator.lt
        :return: 1-D boolean numpy array indexing samples
        """
        mask = operator(x, y)
        # collapse over spatial/time axes and per-sample axis, keep sample index axis
        return mask.all(axis=tuple(range(mask.ndim - 2))).all(axis=1)

    def _mask_quantity(self, other, op):
        """
        Helper to build a BoolType Quantity representing comparisons (>, <, ==, etc.)

        :param other: Quantity or scalar to compare with
        :param op: operation callable that builds the boolean mask from chunked arrays
        :return: Quantity producing a boolean mask per sample
        """
        bool_type = qt.BoolType()
        new_qtype = self.qtype.replace_scalar(bool_type)
        other = Quantity.wrap(other)

        # Only scalar base types support comparison
        if not isinstance(self.qtype.base_qtype(), qt.ScalarType) or not isinstance(other.qtype.base_qtype(), qt.ScalarType):
            raise TypeError("Quantity has base qtype {}. "
                            "Quantities with base qtype ScalarType are the only ones that support comparison".
                            format(self.qtype.base_qtype()))
        return Quantity(quantity_type=new_qtype, input_quantities=[self, other], operation=op)

    # Comparison operators returning boolean mask Quantities
    def __lt__(self, other):
        def lt_op(x, y):
            return Quantity._process_mask(x, y, operator.lt)
        return self._mask_quantity(other, lt_op)

    def __le__(self, other):
        def le_op(x, y):
            return Quantity._process_mask(x, y, operator.le)
        return self._mask_quantity(other, le_op)

    def __gt__(self, other):
        def gt_op(x, y):
            return Quantity._process_mask(x, y, operator.gt)
        return self._mask_quantity(other, gt_op)

    def __ge__(self, other):
        def ge_op(x, y):
            return Quantity._process_mask(x, y, operator.ge)
        return self._mask_quantity(other, ge_op)

    def __eq__(self, other):
        def eq_op(x, y):
            return Quantity._process_mask(x, y, operator.eq)
        return self._mask_quantity(other, eq_op)

    def __ne__(self, other):
        def ne_op(x, y):
            return Quantity._process_mask(x, y, operator.ne)
        return self._mask_quantity(other, ne_op)

    @staticmethod
    def pick_samples(chunk, subsample_params):
        """
        Subsample a chunk using Method S (hypergeometric sampling) so that across chunks
        we end up with k samples from n total.

        :param chunk: ndarray of shape (M, N, 2) where N is number of samples in this chunk
        :param subsample_params: object with attributes k (remaining desired) and n (remaining available)
        :return: selected sub-chunk array with shape (M, m, 2), where m is chosen by hypergeometric draw
        """
        # Draw how many to pick from this chunk using hypergeometric distribution
        size = scipy.stats.hypergeom(subsample_params.n, subsample_params.k, chunk.shape[1]).rvs(size=1)
        out = RNG.choice(chunk, size=size, axis=1)
        subsample_params.k -= out.shape[1]  # reduce remaining desired
        subsample_params.n -= chunk.shape[1]  # reduce remaining available
        return out

    def subsample(self, sample_vec):
        """
        Build a Quantity that implements subsampling across levels to obtain a specified
        number of samples per level (sample_vec).

        Returns a Quantity whose operation will pick samples according to subsample params
        stored per-level. Uses QuantityConst with a level-aware _adjust_value to pass
        different parameters to each level chunk.

        :param sample_vec: list-like of desired numbers of samples per level
        :return: Quantity producing subsampled chunks
        """
        class SubsampleParams:
            """
            Small helper to carry per-level parameters while subsampling across chunks.
            """
            def __init__(self, num_subsample, num_collected):
                """
                :param num_subsample: desired number of samples to pick from this level
                :param num_collected: total available samples on this level
                """
                self._orig_k = num_subsample
                self._orig_n = num_collected
                self._orig_total_n = num_collected
                self.k = num_subsample
                self.n = num_collected
                self.total_n = num_collected

        # Build params per level using level collected counts from the storage
        subsample_level_params = {key: SubsampleParams(sample_vec[key], value)
                                  for key, value in enumerate(self.get_quantity_storage().n_collected())}

        # Wrap a hashed version of this parameters dict in a QuantityConst to feed into operation
        quantity_subsample_params = Quantity.wrap(hash(frozenset(subsample_level_params.items())))

        def adjust_value(values, level_id):
            """
            Method assigned to QuantityConst._adjust_value so each level receives its own SubsampleParams.
            Re-initializes k/n for repeated calls.
            """
            subsample_l_params_obj = subsample_level_params[level_id]
            subsample_l_params_obj.k = subsample_l_params_obj._orig_k
            subsample_l_params_obj.n = subsample_l_params_obj._orig_n
            subsample_l_params_obj.total_n = subsample_l_params_obj._orig_total_n
            return subsample_l_params_obj

        quantity_subsample_params._adjust_value = adjust_value

        # Build resulting Quantity that uses pick_samples as its operation
        return Quantity(quantity_type=self.qtype.replace_scalar(qt.BoolType()),
                        input_quantities=[self, quantity_subsample_params], operation=Quantity.pick_samples)

    def __getitem__(self, key):
        """
        Create a Quantity representing indexed/ sliced access into this quantity (similar to numpy slicing).

        :param key: index or slice or tuple interpreted by qtype.get_key
        :return: Quantity restricted to the requested key
        """
        new_qtype, start = self.qtype.get_key(key)  # New quantity type for selection

        if not isinstance(self.qtype, qt.ArrayType):
            # Convert key to a slice covering the sub-array if base is not ArrayType
            key = slice(start, start + new_qtype.size())

        def _make_getitem_op(y):
            return self.qtype._make_getitem_op(y, key=key)

        return Quantity(quantity_type=new_qtype, input_quantities=[self], operation=_make_getitem_op)

    def __getattr__(self, name):
        """
        Forward static QType methods as Quantity methods so that QType-level helpers are available
        as operations on quantities (e.g., aggregation helpers).
        """
        static_fun = getattr(self.qtype, name)

        def apply_on_quantity(*attr, **d_attr):
            return static_fun(self, *attr, **d_attr)
        return apply_on_quantity

    @staticmethod
    def _concatenate(quantities, qtype, axis=0):
        """
        Construct a Quantity that concatenates multiple quantities along a given axis.

        :param quantities: sequence of Quantity instances
        :param qtype: QType describing result shape
        :param axis: axis along which concatenation happens
        :return: Quantity that when evaluated concatenates input chunks
        """
        def op_concatenate(*chunks):
            y = np.concatenate(tuple(chunks), axis=axis)
            return y
        return Quantity(qtype, input_quantities=[*quantities], operation=op_concatenate)

    @staticmethod
    def _get_base_qtype(args_quantities):
        """
        Determine base QType for arithmetic/ufunc results: if any argument has a ScalarType base
        return ScalarType(), otherwise BoolType().

        :param args_quantities: iterable containing Quantity instances and possibly other values
        :return: base QType instance
        """
        for quantity in args_quantities:
            if isinstance(quantity, Quantity):
                if type(quantity.qtype.base_qtype()) == qt.ScalarType:
                    return qt.ScalarType()
        return qt.BoolType()

    @staticmethod
    def _method(ufunc, method, *args, **kwargs):
        """
        Generic handler for numpy ufunc operations mapped to Quantities.

        1) Wrap inputs as Quantities.
        2) Determine the result QType by calling the ufunc on a small sample.
        3) Return a new Quantity that performs the ufunc at evaluation time.

        :param ufunc: numpy ufunc object
        :param method: method name to call on ufunc (e.g., '__call__' or 'reduce')
        :param args: positional arguments passed to ufunc (may include Quantities)
        :param kwargs: optional ufunc kwargs
        :return: Quantity representing ufunc applied to inputs
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
        Convert a primitive (int, float, bool), a numpy/list array, or an existing Quantity into a Quantity.

        :param value: scalar, bool, list/ndarray, or Quantity
        :return: Quantity or QuantityConst wrapping the value
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
        Infer the resulting QType for an operation by evaluating the operation on the first
        available chunk from each input quantity.

        :param method: callable that takes input chunks and returns sample chunk result
        :param quantities: list of Quantity instances
        :return: inferred QType (ArrayType)
        """
        chunks_quantity_level = []
        for q in quantities:
            quantity_storage = q.get_quantity_storage()
            # If QuantityConst (no storage), use an empty default ChunkSpec
            if quantity_storage is None:
                chunk_spec = ChunkSpec()
            else:
                chunk_spec = next(quantity_storage.chunks())
            chunks_quantity_level.append(q.samples(chunk_spec))

        result = method(*chunks_quantity_level)  # expect shape [M, <=10, 2]
        qtype = qt.ArrayType(shape=result.shape[0], qtype=Quantity._get_base_qtype(quantities))
        return qtype

    @staticmethod
    def QArray(quantities):
        """
        Build a Quantity representing an array-of-quantities aggregated into a single QType.
        """
        flat_quantities = np.array(quantities).flatten()
        qtype = Quantity._check_same_qtype(flat_quantities)
        array_type = qt.ArrayType(np.array(quantities).shape, qtype)
        return Quantity._concatenate(flat_quantities, qtype=array_type)

    @staticmethod
    def QDict(key_quantity):
        """
        Build a Quantity representing a dictionary of quantities.
        :param key_quantity: iterable of (key, Quantity)
        """
        dict_type = qt.DictType([(key, quantity.qtype) for key, quantity in key_quantity])
        return Quantity._concatenate(np.array(key_quantity)[:, 1], qtype=dict_type)

    @staticmethod
    def QTimeSeries(time_quantity):
        """
        Build a Quantity representing a time series constructed from (time, Quantity) pairs.
        """
        qtype = Quantity._check_same_qtype(np.array(time_quantity)[:, 1])
        times = np.array(time_quantity)[:, 0]
        return Quantity._concatenate(np.array(time_quantity)[:, 1], qtype=qt.TimeSeriesType(times=times, qtype=qtype))

    @staticmethod
    def QField(key_quantity):
        """
        Build a Quantity representing a field (mapping of locations to quantities).
        """
        Quantity._check_same_qtype(np.array(key_quantity)[:, 1])
        field_type = qt.FieldType([(key, quantity.qtype) for key, quantity in key_quantity])
        return Quantity._concatenate(np.array(key_quantity)[:, 1], qtype=field_type)

    @staticmethod
    def _check_same_qtype(quantities):
        """
        Validate that all provided quantities share the same QType.

        :param quantities: sequence of Quantity instances
        :return: the shared QType
        :raise ValueError: if a mismatch is found
        """
        qtype = quantities[0].qtype
        for quantity in quantities[1:]:
            if qtype != quantity.qtype:
                raise ValueError("Quantities don't have same QType")
        return qtype


class QuantityConst(Quantity):
    """
    Represents a constant quantity whose value is stored directly in the instance.
    The samples() method returns the constant value broadcasted to the requested chunk shape.
    """

    def __init__(self, quantity_type, value):
        """
        :param quantity_type: QType describing the const
        :param value: scalar or array-like value
        """
        self.qtype = quantity_type
        self._value = self._process_value(value)
        # No input dependencies for a constant
        self._input_quantities = []
        self._selection_id = None

    def _process_value(self, value):
        """
        Ensure the constant is stored as an array with axes [M, 1, 1] suitable for broadcasting.

        :param value: scalar or array-like
        :return: ndarray shaped for broadcasting into (M, chunk_size, 2)
        """
        if isinstance(value, (int, float, bool)):
            value = np.array([value])
        return value[:, np.newaxis, np.newaxis]

    def selection_id(self):
        """
        Constants have no selection id (they are independent of storage).
        """
        return self._selection_id

    def _adjust_value(self, value, level_id=None):
        """
        Hook to adjust constant value per-level. By default returns the stored value unchanged.
        This method gets overridden by consumers (e.g., subsample) to provide level-specific params.

        :param value: constant value array
        :param level_id: int, level index (optional)
        :return: possibly adjusted value
        """
        return value

    @cached(custom_key_maker=Quantity.get_cache_key)
    def samples(self, chunk_spec):
        """
        Return the constant value, optionally adjusted for the given level via _adjust_value.

        :param chunk_spec: ChunkSpec with level_id
        :return: ndarray representing the constant for this chunk
        """
        return self._adjust_value(self._value, chunk_spec.level_id)


class QuantityMean:
    """
    Container for aggregated mean/variance results computed by mlmc.quantity.quantity_estimate.estimate_mean.

    - qtype: QType of the quantity
    - _l_means: per-level mean contributions (L x M flattened)
    - _l_vars: per-level variance contributions (L x M flattened)
    - _n_samples: number of samples used per level
    - _n_rm_samples: number of removed samples per level
    """

    def __init__(self, quantity_type, l_means, l_vars, n_samples, n_rm_samples):
        """
        :param quantity_type: QType
        :param l_means: ndarray shape (L, M_flat) of level-wise mean contributions
        :param l_vars: ndarray shape (L, M_flat) of level-wise variance contributions
        :param n_samples: list/ndarray length L with number of samples used per level
        :param n_rm_samples: list/ndarray length L with removed samples count per level
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
        Compute overall mean and variance from per-level contributions:
          mean = sum_l l_means[l]
          var = sum_l (l_vars[l] / n_samples[l])
        """
        self._mean = np.sum(self._l_means, axis=0)
        self._var = np.sum(self._l_vars / self._n_samples[:, None], axis=0)

    @property
    def mean(self):
        """
        Reshaped overall mean according to QType.
        """
        if self._mean is None:
            self._calculate_mean_var()
        return self._reshape(self._mean)

    @property
    def var(self):
        """
        Reshaped overall variance according to QType.
        """
        if self._var is None:
            self._calculate_mean_var()
        return self._reshape(self._var)

    @property
    def l_means(self):
        """
        Level means reshaped according to QType for each level.
        """
        return np.array([self._reshape(means) for means in self._l_means])

    @property
    def l_vars(self):
        """
        Level variances reshaped according to QType for each level.
        """
        return np.array([self._reshape(vars) for vars in self._l_vars])

    @property
    def n_samples(self):
        return self._n_samples

    @property
    def n_rm_samples(self):
        return self._n_rm_samples

    def _reshape(self, data):
        """
        Reshape a flat data vector (flattened M) into the structure determined by qtype.

        :param data: flattened ndarray
        :return: reshaped ndarray according to qtype
        """
        return self.qtype.reshape(data)

    def __getitem__(self, key):
        """
        Index into QuantityMean similarly to Quantity.__getitem__:
        reshape level-wise means/vars and select the requested key, then return a new QuantityMean.

        :param key: indexing key (int, slice, str, etc.)
        :return: QuantityMean restricted to the requested key
        """
        new_qtype, start = self.qtype.get_key(key)  # New quantity type

        if not isinstance(self.qtype, qt.ArrayType):
            key = slice(start, start + new_qtype.size())

        # Selecting and reshaping level arrays
        l_means = self.l_means[:, key]
        l_vars = self.l_vars[:, key]

        return QuantityMean(quantity_type=new_qtype,
                            l_means=l_means.reshape((l_means.shape[0], -1)),
                            l_vars=l_vars.reshape((l_vars.shape[0], -1)),
                            n_samples=self._n_samples,
                            n_rm_samples=self._n_rm_samples)


class QuantityStorage(Quantity):
    """
    Special Quantity that provides direct access to SampleStorage.
    It implements the bridge between storage and the Quantity abstraction.
    """

    def __init__(self, storage, qtype):
        """
        :param storage: SampleStorage instance (in-memory or HDF5, etc.)
        :param qtype: QType describing stored data structure
        """
        # Store underlying storage reference and QType
        self._storage = storage
        self.qtype = qtype
        # No operation or inputs required for storage root
        self._input_quantities = []
        self._operation = None

    def level_ids(self):
        """
        Return list of available level ids from the SampleStorage.
        :return: List[int]
        """
        return self._storage.get_level_ids()

    def selection_id(self):
        """
        Identity of this QuantityStorage (unique by object id).
        :return: int
        """
        return id(self)

    def get_quantity_storage(self):
        """
        For QuantityStorage the storage is itself.
        :return: self
        """
        return self

    def chunks(self, level_id=None):
        """
        Proxy to SampleStorage.chunks which yields ChunkSpec instances describing available chunks.
        :param level_id: optional level id to restrict chunks
        :return: generator of ChunkSpec
        """
        return self._storage.chunks(level_id)

    def samples(self, chunk_spec):
        """
        Retrieve stored sample pairs for the requested level/chunk.

        :param chunk_spec: ChunkSpec describing (level, chunk slice)
        :return: ndarray shaped [M, chunk_size, 2] where M is number of result quantities
        """
        return self._storage.sample_pairs_level(chunk_spec)  # Array[M, chunk size, 2]

    def n_collected(self):
        """
        Return number of collected results per level from the underlying SampleStorage.
        :return: list of ints
        """
        return self._storage.get_n_collected()
