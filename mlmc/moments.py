import numpy as np
import numpy.ma as ma
from scipy.interpolate import BSpline


class Moments:
    """
    Class for calculating moments of a random variable
    """
    def __init__(self, size, domain, log=False, safe_eval=True):
        assert size > 0
        self.size = size
        self.domain = domain
        self._is_log = log
        self._is_clip = safe_eval

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
        :param size: int, new number of _moments_fn
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
        return self._eval_all(value, i+1)[:, -1]

    def eval_single_moment(self, i, value):
        """
        Be aware this implementation is inefficient for large i
        :param i: int, order of moment
        :param value: float
        :return: np.ndarray
        """
        return self._eval_all(value, i+1)[..., i]

    def eval_all(self, value, size=None):
        if size is None:
            size = self.size
        return self._eval_all(value, size)

    def eval_all_der(self, value, size=None, degree=1):
        if size is None:
            size = self.size
        return self._eval_all_der(value, size, degree)

    def eval_diff(self, value, size=None):
        if size is None:
            size = self.size
        return self._eval_diff(value, size)

    def eval_diff2(self, value, size=None):
        if size is None:
            size = self.size
        return self._eval_diff2(value, size)


class Monomial(Moments):
    """
    Monomials generalized moments
    """
    def __init__(self, size, domain=(0, 1), ref_domain=None, log=False, safe_eval=True):
        if ref_domain is not None:
            self.ref_domain = ref_domain
        else:
            self.ref_domain = (0, 1)
        super().__init__(size, domain, log=log, safe_eval=safe_eval)

    def _eval_all(self, value, size):
        # Create array from values and transform values outside the ref domain
        t = self.transform(np.atleast_1d(value))
        # Vandermonde matrix
        return np.polynomial.polynomial.polyvander(t, deg=size - 1)

    def eval(self, i, value):
        t = self.transform(np.atleast_1d(value))
        return t**i


class Fourier(Moments):
    """
    Fourier functions generalized moments
    """
    def __init__(self, size, domain=(0, 2*np.pi), ref_domain=None, log=False, safe_eval=True):
        if ref_domain is not None:
            self.ref_domain = ref_domain
        else:
            self.ref_domain = (0, 2*np.pi)

        super().__init__(size, domain, log=log, safe_eval=safe_eval)

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
    """
    Legendre polynomials generalized moments
    """

    def __init__(self, size, domain, ref_domain=None, log=False, safe_eval=True):
        if ref_domain is not None:
            self.ref_domain = ref_domain
        else:
            self.ref_domain = (-1, 1)

        self.diff_mat = np.zeros((size, size))
        for n in range(size - 1):
            self.diff_mat[n, n + 1::2] = 2 * n + 1
        self.diff2_mat = self.diff_mat @ self.diff_mat

        super().__init__(size, domain, log, safe_eval)

    def _eval_value(self, x, size):
        return np.polynomial.legendre.legvander(x, deg=size-1)

    def _eval_all(self, value, size):
        value = self.transform(np.atleast_1d(value))
        return np.polynomial.legendre.legvander(value, deg=size - 1)

    def _eval_all_der(self, value, size, degree=1):
        """
        Derivative of Legendre polynomials
        :param value: values to evaluate
        :param size: number of moments
        :param degree: degree of derivative
        :return:
        """
        value = self.transform(np.atleast_1d(value))
        eval_values = np.empty((value.shape + (size,)))

        for s in range(size):
            if s == 0:
                coef = [1]
            else:
                coef = np.zeros(s+1)
                coef[-1] = 1

            coef = np.polynomial.legendre.legder(coef, degree)
            eval_values[:, s] = np.polynomial.legendre.legval(value, coef)
        return eval_values

    def _eval_diff(self, value, size):
        t = self.transform(np.atleast_1d(value))
        P_n = np.polynomial.legendre.legvander(t, deg=size - 1)
        return P_n @ self.diff_mat

    def _eval_diff2(self, value, size):
        t = self.transform(np.atleast_1d(value))
        P_n = np.polynomial.legendre.legvander(t, deg=size - 1)
        return P_n @ self.diff2_mat


class TransformedMoments(Moments):
    def __init__(self, other_moments, matrix):
        """
        Set a new moment functions as linear combination of the previous.
        new_moments = matrix . old_moments

        We assume that new_moments[0] is still == 1. That means
        first row of the matrix must be (1, 0 , ...).
        :param other_moments: Original _moments_fn.
        :param matrix: Linear combinations of the original _moments_fn.
        """
        n, m = matrix.shape
        assert m == other_moments.size
        self.size = n
        self.domain = other_moments.domain
        self._origin = other_moments
        self._transform = matrix

    def __eq__(self, other):
        return type(self) is type(other) \
                and self.size == other.size \
                and self._origin == other._origin \
                and np.all(self._transform == other._transform)

    def _eval_all(self, value, size):
        orig_moments = self._origin._eval_all(value, self._origin.size)
        x1 = np.matmul(orig_moments, self._transform.T)
        return x1[..., :size]

    def _eval_all_der(self, value, size, degree=1):
        orig_moments = self._origin._eval_all_der(value, self._origin.size, degree=degree)
        x1 = np.matmul(orig_moments, self._transform.T)
        return x1[..., :size]

    def _eval_diff(self, value, size):
        orig_moments = self._origin.eval_diff(value, self._origin.size)
        x1 = np.matmul(orig_moments, self._transform.T)
        return x1[..., :size]

    def _eval_diff2(self, value, size):
        orig_moments = self._origin.eval_diff2(value, self._origin.size)
        x1 = np.matmul(orig_moments, self._transform.T)
        return x1[..., :size]
