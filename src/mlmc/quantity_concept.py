import numpy as np


def estimate_mean(quantity):
    """
    Concept of the MLMC mean estimator.
    TODO:
    - test
    - calculate variance estimates as well
    :param quantity:
    :return:
    """
    quantity_vec_size = quantity._size()
    n_levels = 1
    n_samples = None
    sums = None
    i_chunk = 0
    while True:
        chunk = quantity._samples(i_chunk)
        if chunk is None:
            break;
        if i_chunk == 0:
            # initialization
            n_levels = len(chunk)
            n_samples = [0] * n_levels
            sums = [np.zeros(chunk[0].shape[1]) for level in chunk]
        for i_level, level_chunk in enumerate(chunk):
            # level_chunk is Numpy Array with shape [M, chunk_size, 2]
            n_samples[i_level] += len(level_chunk)
            assert(level_chunk.shape[1] == quantity_vec_size)
            sums[i_level] += np.sum(level_chunk[:, :, 0] - level_chunk[:, :, 1], axis=1)
    mean = np.zeros_like(sums[0])
    for s, n in zip(sums, n_samples):
        mean += s / n
    return quantity.make_value(mean)



# Just for type hints, change to protocol
class QBase:
    def __init__(self, input_quantities, operation = None):
        assert (len(input_quantities) > 0)
        self._input_quantities = input_quantities
        # List of quantities on which the 'self' dependes. their number have to match number of arguments to the operation.
        self._operation = operation
        # function  lambda(*args : List[array[M, N, 2]]) -> List[array[M, N, 2])]
        # It takes list of chunks for individual levels as a single argument, with number of arguments matching the
        # number of input qunatities. Operation performs actual quantity operation on the sample chunks.
        # One chunk is a np.array with shape [sample_vector_size, n_samples_in_chunk, 2], 2 = (coarse, fine) pair.
        self._size = sum( (q._size for q in self._input_quantities) )
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


    def _samples(self, i_chunk):
        """
        Yields list of sample chunks for individual levels.
        Possibly calls underlaying quantities.
        Try to write into the chunk array.
        TODO: possibly pass down the full table of composed quantities and store comuted temporaries directly into the composed table

        TODO: possible problem when more quantities dependes on a single one, this probabely leads to
        repetitive evaluation of the shared quantity
        This prevents some coies in concatenation.
        """
        chunks_quantity_level = [q._samples(i_chunk) for  q in self._input_quantities]
        is_valid = (ch is not None for ch in chunks_quantity_level)
        if any(is_valid):
            assert (all(is_valid))
            chunks_level_quantity = zip(*chunks_quantity_level)
            return [self.operation(*one_level_chunks) for one_level_chunks in chunks_level_quantity]
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
        pass

    def _reduction_op(self, *quantities, operation):
        """
        Check if the quantities are the same and possibly return copy of the common quantity
        structure depending on the other quantities with given operation.
        TODO: Not so simple, but similar problem to constraction of the initial structure over storage.
        :param quantities:
        :return:
        """



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
        assert(False)

    def __const_mult(self, other):
        def cmult_op(x, c=other):
            return c * x
        return self._reduction_op([self], cmult_op)

def make_root_quantity(storage: Storage, q_specs : List[QSpec])
    i_range = 0
    dict_items = []
    for q_spec in q_specs:
        time_frames = []
        for t in q_spec.times:
            field_values = []
            for loc in q_spec.locations:
                q_range = QRange(storage)
                array = Array.from_range(q_spec.shape, q_range)
                q_range.set(i_range, array._size)
                i_range += array._size
                field_values.append( (loc, array) )
            time_frames.append( (t, Field(field_values)) )
        dict_items = (q_spec.name, TimeSeries(time_frames))
    return Dict(dict_items)



class QRange:
    def __init__(self, storage):
        self._storage = storage

    def set(self, range_begin, size):
        self.begin = range_begin
        self._size = size
        self.end = range_begin + size

    def _samples(self, i_chunk):
        level_chunks = self.storage.sample_pairs(i_chunk)
        return [chunk[self.begin:self.end, :, :]  for chunk in level_chunks]




class TimeSeries(QBase):
    """
    We should assume that all quantities have same structure (check only same size)
    but can be of any type. Meanigful are only onther types then TimeSeries.
    """
    def __init__(self, time_frames : List[Tuple[float, QBase]]):
        super().__init__(time_frames)
        self.frame0 = time_frames[0][1]
        assert( all( (q._size == self.frame0._size for t,q in time_frames) ) )
        assert( all( (q.__class__ == self.frame0.__class__ for t,q in time_frames) ))


        self.time_frames = time_frames
        # list of pairs (time, value)
        self.block_starts = []
        # starts of blocks of the time values on the single sample row
        self.operation = lambda lch : np.concatenate(lch, axis=0)

    def max(self):
        """
        Create quantity for the maximum over the time series.
        :return:
        """
        def max_op(*input_chunks):
            table = np.stack(input_chunks)
            return np.max(table, axis=0)

        return self.frame0._copy(*[q for t, q in self.time_frames], operation = max_op)

    def interpolate(self):
        """
        Create quantity for the maximum over the time series.
        :return:
        """

        def max_op(*input_chunks):
            table = np.stack(input_chunks)
            return np.max(table, axis=0)

        return self.frame0._copy(*[q for t, q in self.time_frames], operation=max_op)



class Dict(QBase):
    def __init__(self, *args: Tuple[str, QBase]):
        self.quantity_dict = OrderedDict(*args)

    def __getattr__(self, key):
        # JB: TODO
        if key in self.quantity_dict:
            return self.quantity_dict[key]
        return getattr(key)

    def max(self):
        """
        Create quantity for the maximum over the time series.
        :return:
        """
        return _QMax(*[q for q in self.quantity_dict.values()])



class Array(QBase):
    def __init__(self, *args):
        assert( all( (a.is_array() for a in args) ) )
        assert( all( (a.shape == args[0].shape for a in args) ) )
        self.q_list = args
        self.shape = [len(self.q_list)] + args[0].shape



class Field(QBase):
    def __init__(self, *args: Tuple[str, QBase]):
        self.quantity_dict = OrderedDict(*args)

