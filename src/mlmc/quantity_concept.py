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
    quantity_vec_size = quantity.get_size()
    n_levels = 1
    n_samples = None
    sums = None
    for chunk in quantity.samples():
        if n_samples is None:
            # initialization
            n_levels = len(chunk)
            n_samples = [0] * n_levels
            sums = [np.zeros(chunk[0].shape[1]) for level in chunk]
        for i_level, level_chunk in enumerate(chunk):
            # level_chunk is Numpy Array with shape [chunk_size, M, 2]
            n_samples[i_level] += len(level_chunk)
            assert(level_chunk.shape[1] == quantity_vec_size)
            sums[i_level] += np.sum(level_chunk[:, :, 0] - level_chunk[:, :, 1], axis=0)
    mean = np.zeros_like(sums[0])
    for s, n in zip(sums, n_samples):
        mean += s / n
    return quantity.make_value(mean)

# Just for type hints, change to protocol
class Quantity:
    def _get_size(self):
        """
        M - for the quantity
        :return:
        """
        pass

    # def set_range(self, total_dofs, start):
    #     """
    #     Set sub range of the quantity in the composed vector, eliminates copies.
    #     computing quantities write to this range but they se
    #     :return:
    #     """
    #     pass

    def _samples(self):
        """
        Yields list of sample chunks for individual levels.
        Possibly calls underlaying quantities.
        Try to write into the chunk array.
        TODO: possibly pass down the full table of composed quantities and store comuted temporaries directly into the composed table
        This prevents some coies in concatenation.
        """
        pass

    def make_value(self, data_row: np.array):
        """
        Crate a new quantity with the same structure but containing fixed data vector.
        Primary usage is to organise computed means and variances.
        Can possibly be used also to organise single sample row.
        :param data_row:
        :return:
        """
        pass

class TimeSeries(Quantity):
    def __init__(self, time_frames : List[Tuple[float, Quantity]]):
        self.time_frames = time_frames
        # list of pairs (time, value)
        self.block_starts = []
        # starts of blocks of the time values on the single sample row

    def _get_size(self):
        sizes = [value.size() for t, value in self.time_frames]
        self.block_starts = [0]
        for s in sizes:
            self.block_starts.append(self.block_starts[-1] + s)
        return self.block_starts[-1]


    def _samples(self):
        """
        Glue tables for individual times.
        TODO: how we then know which part of the vector is which time, similar for other, one solution is to allow
        arbitrary shape after (2, chunk_size, ...) not solution for the dict.
        Other solution is own structure that supports basic operations necessary in the estimate.

        Note: this should be primarily class to support operations on time series not for creating the composition.

        :return:
        """
        generators = [q.samples() for t, q in self.time_frames]
        for frames in itertools.zip_longest(*generators):
            is_valid = (f is not None for f in frames)
            assert(all(is_valid))
            yield np.concatenate(frames, axis=1)

    def max(self):
        """
        Create quantity for the maximum over the time series.
        :return:
        """
        return _QMax(*[q for t, q in self.time_frames])


class Dict(Quantity):
    def __init__(self, *args: Tuple[str, Quantity]):
        self.quantity_dict = OrderedDict(*args)

    def _get_size(self):
        sizes = [value.size() for t, value in self.time_frames]
        self.block_starts = [0]
        for s in sizes:
            self.block_starts.append(self.block_starts[-1] + s)
        return self.block_starts[-1]

    def _samples(self):
        """
        Glue tables for individual times.
        TODO: how we then know which part of the vector is which time, similar for other, one solution is to allow
        arbitrary shape after (2, chunk_size, ...) not solution for the dict.
        Other solution is own structure that supports basic operations necessary in the estimate.

        Note: this should be primarily class to support operations on time series not for creating the composition.

        :return:
        """
        # TODO: similar as in TimeSeries

    def __getattr__(self, key):
        if key in self.quantity_dict:
            return self.quantity_dict[key]
        return getattr(key)

    def max(self):
        """
        Create quantity for the maximum over the time series.
        :return:
        """
        return _QMax(*[q for q in self.quantity_dict.values()])



class Array(Quantity):
    def __init__(self, *args):
        assert( all( (a.is_array() for a in args) ) )
        assert( all( (a.shape == args[0].shape for a in args) ) )
        self.q_list = args
        self.shape = [len(self.q_list)] + args[0].shape

    def _get_size(self):
        n = 1
        for ax in self.shape:
            n *= ax
        return n

    def _samples(self):
        """
        Glue tables for individual times.
        TODO: how we then know which part of the vector is which time, similar for other, one solution is to allow
        arbitrary shape after (2, chunk_size, ...) not solution for the dict.
        Other solution is own structure that supports basic operations necessary in the estimate.

        Note: this should be primarily class to support operations on time series not for creating the composition.

        :return:
        """
        generators = [q.samples() for q in self.q_list]
        for frames in itertools.zip_longest(*generators):
            is_valid = (f is not None for f in frames)
            assert(all(is_valid))
            table = np.stack(frames)
            yield table.reshape(-1, *table.shape[-3:])


class _QMax(Array):
    """
    Common recepie for various reduce operations like: max, min, average, sum, ...
    Can possibly be implemented in generic way, the only difference is the np.method in _samples.
    TODO: possibly use Array first to concatenate and then call the reduction op, however Array performs reshape
    """
    def __init__(self, *args: Quantity):
        assert( all( (a.is_array() for a in args) ) )
        assert( all( (a.shape == args[0].shape for a in args) ) )
        self.q_list = args

    def _get_size(self):
        return self.q_list[0].get_size()

    def _samples(self):
        """
        Glue tables for individual times.
        TODO: how we then know which part of the vector is which time, similar for other, one solution is to allow
        arbitrary shape after (2, chunk_size, ...) not solution for the dict.
        Other solution is own structure that supports basic operations necessary in the estimate.

        Note: this should be primarily class to support operations on time series not for creating the composition.

        :return:
        """
        generators = [q.samples() for q in self.q_list]
        for frames in itertools.zip_longest(*generators):
            is_valid = (f is not None for f in frames)
            assert(all(is_valid))
            table = np.stack(frames)
            yield np.max(table, axis=0)


