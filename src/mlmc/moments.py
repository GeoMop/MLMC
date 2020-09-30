import numpy as np
import numpy.ma as ma


class Moments:
    """
    Class for moments of random distribution
    """
    def __init__(self, size, domain, log=False, safe_eval=True, mean=0):
        assert size > 0
        self.size = size
        self.domain = domain
        self._is_log = log
        self._is_clip = safe_eval
        self.mean = mean

        if log:
            lin_domain = (np.log(domain[0]), np.log(domain[1]))
        else:
            lin_domain = domain

        diff = lin_domain[1] - lin_domain[0]
        assert diff > 0
        diff = max(diff, 1e-15)
        self._linear_scale = (self.ref_domain[1] - self.ref_domain[0]) / diff
        self._linear_shift = lin_domain[0]

        if safe_eval and log:
            self.transform = lambda val: self.clip(self.linear(np.log(val)))
            self.inv_transform = lambda ref: np.exp(self.inv_linear(ref))
        elif safe_eval and not log:
            self.transform = lambda val: self.clip(self.linear(val))
            self.inv_transform = lambda ref: self.inv_linear(ref)
        elif not safe_eval and log:
            self.transform = lambda val: self.linear(np.log(val))
            self.inv_transform = lambda ref: np.exp(self.inv_linear(ref))
        elif not safe_eval and not log:
            self.transform = lambda val: self.linear(val)
            self.inv_transform = lambda ref: self.inv_linear(ref)

    def __eq__(self, other):
        """
        Compare two moment functions. Equal if they returns same values.
        """
        return type(self) is type(other) \
                and self.size == other.size \
                and np.all(self.domain == other.domain) \
                and self._is_log == other._is_log \
                and self._is_clip == other._is_clip

    def change_size(self, size):
        """
        Return moment object with different size.
        :param size: int, new number of moments
        """
        return self.__class__(size, self.domain, self._is_log, self._is_clip)

    def clip(self, value):
        """
        Remove outliers and replace them with NaN
        :param value: array of numbers
        :return: masked_array, out
        """
        # Masked array
        out = ma.masked_outside(value, self.ref_domain[0], self.ref_domain[1])
        # Replace outliers with NaN
        return ma.filled(out, np.nan)

    def linear(self, value):
        return (value - self._linear_shift) * self._linear_scale + self.ref_domain[0]

    def inv_linear(self, value):
        return (value - self.ref_domain[0]) / self._linear_scale + self._linear_shift

    def __call__(self, value):
        return self._eval_all(value, self.size)

    def eval(self, i, value):
        value = self._center(value)
        return self._eval_all(value, i+1)[:, -1]

    def eval_single_moment(self, i, value):
        """
        Be aware this implementation is inefficient for large i
        :param i: int, order of moment
        :param value: float
        :return: np.ndarray
        """
        value = self._center(value)
        return self._eval_all(value, i+1)[..., i]

    def eval_all(self, value, size=None):
        if size is None:
            size = self.size

        value = self._center(value)

        return self._eval_all(value, size)

    def _center(self, value):
        if not isinstance(self.mean, int):
            if np.all(value[..., 1]) == 0:
                value[..., 0] = value[..., 0] - self.mean[:, None]
            else:
                value[...] = value[...] - self.mean[:, None, None]
        else:
            if np.all(value[..., 1]) == 0:
                value[..., 0] = value[..., 0] - self.mean
            else:
                value[...] = value[...] - self.mean

        return value


class Monomial(Moments):
    def __init__(self, size, domain=(0, 1), ref_domain=None, log=False, safe_eval=True, mean=0):
        if ref_domain is not None:
            self.ref_domain = ref_domain
        else:
            self.ref_domain = (0, 1)
        super().__init__(size, domain, log=log, safe_eval=safe_eval, mean=mean)

    def _eval_all(self, value, size):
        # Create array from values and transform values outside the ref domain
        t = self.transform(np.atleast_1d(value))
        # Vandermonde matrix
        return np.polynomial.polynomial.polyvander(t, deg=size - 1)

    def eval(self, i, value):
        t = self.transform(np.atleast_1d(value))
        return t**i


class Fourier(Moments):
    def __init__(self, size, domain=(0, 2*np.pi), ref_domain=None, log=False, safe_eval=True, mean=0):
        if ref_domain is not None:
            self.ref_domain = ref_domain
        else:
            self.ref_domain = (0, 2*np.pi)

        super().__init__(size, domain, log=log, safe_eval=safe_eval, mean=mean)

    def _eval_all(self, value, size):
        # Transform values
        t = self.transform(np.atleast_1d(value))

        # Half the number of moments
        R = int(size / 2)
        shorter_sin = 1 - int(size % 2)
        k = np.arange(1, R + 1)
        kx = np.outer(t, k)

        res = np.empty((len(t), size))
        res[:, 0] = 1

        # Odd column index
        res[:, 1::2] = np.cos(kx[:, :])
        # Even column index
        res[:, 2::2] = np.sin(kx[:, : R - shorter_sin])
        return res

    def eval(self, i, value):
        t = self.transform(np.atleast_1d(value))
        if i == 0:
            return 1
        elif i % 2 == 1:
            return np.sin( (i - 1) / 2 * t)
        else:
            return np.cos(i / 2 * t)


class Legendre(Moments):

    def __init__(self, size, domain, ref_domain=None, log=False, safe_eval=True, mean=0):
        if ref_domain is not None:
            self.ref_domain = ref_domain
        else:
            self.ref_domain = (-1, 1)

        self.mean = mean
        super().__init__(size, domain, log, safe_eval, mean)

    def _eval_all(self, value, size):
        t = self.transform(np.atleast_1d(value))
        return np.polynomial.legendre.legvander(t, deg=size - 1)


class TransformedMoments(Moments):
    def __init__(self, other_moments, matrix):
        """
        Set a new moment functions as linear combination of the previous.
        new_moments = matrix . old_moments

        We assume that new_moments[0] is still == 1. That means
        first row of the matrix must be (1, 0 , ...).
        :param other_moments: Original moments.
        :param matrix: Linear combinations of the original moments.
        """
        n, m = matrix.shape
        assert m == other_moments.size

        self.size = n
        self.domain = other_moments.domain

        self._origin = other_moments
        self._transform = matrix
        #self._inv = inv
        #assert np.isclose(matrix[0, 0], 1) and np.allclose(matrix[0, 1:], 0)
        # TODO: find last nonzero for every row to compute which origianl moments needs to be evaluated for differrent sizes.

    def __eq__(self, other):
        return type(self) is type(other) \
                and self.size == other.size \
                and self._origin == other._origin \
                and np.all(self._transform == other._transform)

    def _eval_all(self, value, size):
        orig_moments = self._origin._eval_all(value, self._origin.size)
        x1 = np.matmul(orig_moments, self._transform.T)
        #x2 = np.linalg.solve(self._inv, orig_moments.T).T
        return x1[:, :size]
