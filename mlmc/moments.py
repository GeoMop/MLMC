import numpy as np
import numpy.ma as ma
from scipy.interpolate import BSpline


class Moments:
    """
    Base class for computing moment functions of a random variable.

    Provides transformation, scaling, and evaluation utilities common
    to various types of generalized moment bases (monomial, Fourier, Legendre, etc.).
    """

    def __init__(self, size, domain, log=False, safe_eval=True):
        """
        Initialize the moment function set.

        :param size: int
            Number of moment functions.
        :param domain: tuple(float, float)
            Domain of the input variable (min, max).
        :param log: bool
            If True, use logarithmic transformation of the domain.
        :param safe_eval: bool
            If True, clip transformed values outside the reference domain
            and replace them with NaN.
        """
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

        # Define transformation and inverse transformation functions
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
        Compare two Moments objects for equality.

        :param other: Moments
            Another Moments instance.
        :return: bool
            True if both instances have the same parameters and configuration.
        """
        return (
            type(self) is type(other)
            and self.size == other.size
            and np.all(self.domain == other.domain)
            and self._is_log == other._is_log
            and self._is_clip == other._is_clip
        )

    def change_size(self, size):
        """
        Return a new moment object with a different number of basis functions.

        :param size: int
            New number of moment functions.
        :return: Moments
            New instance of the same class with updated size.
        """
        return self.__class__(size, self.domain, self._is_log, self._is_clip)

    def clip(self, value):
        """
        Clip values to the reference domain, replacing outliers with NaN.

        :param value: array-like
            Input data to be clipped.
        :return: ndarray
            Array with out-of-bound values replaced by NaN.
        """
        out = ma.masked_outside(value, self.ref_domain[0], self.ref_domain[1])
        return ma.filled(out, np.nan)

    def linear(self, value):
        """Apply linear transformation to reference domain."""
        return (value - self._linear_shift) * self._linear_scale + self.ref_domain[0]

    def inv_linear(self, value):
        """Inverse linear transformation back to the original domain."""
        return (value - self.ref_domain[0]) / self._linear_scale + self._linear_shift

    def __call__(self, value):
        """Evaluate all moment functions for the given value(s)."""
        return self._eval_all(value, self.size)

    def eval(self, i, value):
        """
        Evaluate the i-th moment function.

        :param i: int
            Index of the moment function to evaluate (0-based).
        :param value: float or array-like
            Input value(s).
        :return: ndarray
            Values of the i-th moment function.
        """
        return self._eval_all(value, i + 1)[:, -1]

    def eval_single_moment(self, i, value):
        """
        Evaluate a single moment function (less efficient for large i).

        :param i: int
            Order of the moment.
        :param value: float or array-like
            Input value(s).
        :return: ndarray
            Evaluated moment values.
        """
        return self._eval_all(value, i + 1)[..., i]

    def eval_all(self, value, size=None):
        """
        Evaluate all moments up to the specified size.

        :param value: float or array-like
            Input value(s).
        :param size: int or None
            Number of moments to evaluate. If None, use self.size.
        :return: ndarray
            Matrix of evaluated moments.
        """
        if size is None:
            size = self.size
        return self._eval_all(value, size)

    def eval_all_der(self, value, size=None, degree=1):
        """
        Evaluate derivatives of all moment functions.

        :param value: float or array-like
            Input value(s).
        :param size: int or None
            Number of moments to evaluate.
        :param degree: int
            Derivative degree (1 for first derivative, etc.).
        :return: ndarray
            Matrix of evaluated derivatives.
        """
        if size is None:
            size = self.size
        return self._eval_all_der(value, size, degree)

    def eval_diff(self, value, size=None):
        """
        Evaluate first derivatives of all moment functions.

        :param value: float or array-like
            Input value(s).
        :param size: int or None
            Number of moments to evaluate.
        :return: ndarray
            Matrix of first derivatives.
        """
        if size is None:
            size = self.size
        return self._eval_diff(value, size)

    def eval_diff2(self, value, size=None):
        """
        Evaluate second derivatives of all moment functions.

        :param value: float or array-like
            Input value(s).
        :param size: int or None
            Number of moments to evaluate.
        :return: ndarray
            Matrix of second derivatives.
        """
        if size is None:
            size = self.size
        return self._eval_diff2(value, size)


# -------------------------------------------------------------------------
# Specific moment types
# -------------------------------------------------------------------------
class Monomial(Moments):
    """
    Monomial basis functions for generalized moment evaluation.
    """

    def __init__(self, size, domain=(0, 1), ref_domain=None, log=False, safe_eval=True):
        if ref_domain is not None:
            self.ref_domain = ref_domain
        else:
            self.ref_domain = (0, 1)
        super().__init__(size, domain, log=log, safe_eval=safe_eval)

    def _eval_all(self, value, size):
        """
        Evaluate monomial basis (Vandermonde matrix).

        :param value: array-like
            Input values.
        :param size: int
            Number of moments to compute.
        :return: ndarray
            Vandermonde matrix of monomials.
        """
        t = self.transform(np.atleast_1d(value))
        return np.polynomial.polynomial.polyvander(t, deg=size - 1)

    def eval(self, i, value):
        """Evaluate the i-th monomial t^i."""
        t = self.transform(np.atleast_1d(value))
        return t ** i


class Fourier(Moments):
    """
    Fourier basis functions for generalized moment evaluation.
    """

    def __init__(self, size, domain=(0, 2 * np.pi), ref_domain=None, log=False, safe_eval=True):
        if ref_domain is not None:
            self.ref_domain = ref_domain
        else:
            self.ref_domain = (0, 2 * np.pi)
        super().__init__(size, domain, log=log, safe_eval=safe_eval)

    def _eval_all(self, value, size):
        """
        Evaluate Fourier moment basis (cosine/sine terms).

        :param value: array-like
            Input values.
        :param size: int
            Number of moments to compute.
        :return: ndarray
            Matrix of evaluated Fourier functions.
        """
        t = self.transform(np.atleast_1d(value))
        R = int(size / 2)
        shorter_sin = 1 - int(size % 2)
        k = np.arange(1, R + 1)
        kx = np.outer(t, k)

        res = np.empty((len(t), size))
        res[:, 0] = 1
        res[:, 1::2] = np.cos(kx[:, :])
        res[:, 2::2] = np.sin(kx[:, : R - shorter_sin])
        return res

    def eval(self, i, value):
        """
        Evaluate a single Fourier basis function.

        :param i: int
            Index of the moment function.
        :param value: float or array-like
            Input values.
        :return: ndarray
            Evaluated function values.
        """
        t = self.transform(np.atleast_1d(value))
        if i == 0:
            return 1
        elif i % 2 == 1:
            return np.sin((i - 1) / 2 * t)
        else:
            return np.cos(i / 2 * t)


class Legendre(Moments):
    """
    Legendre polynomial basis functions for generalized moments.
    """

    def __init__(self, size, domain, ref_domain=None, log=False, safe_eval=True):
        if ref_domain is not None:
            self.ref_domain = ref_domain
        else:
            self.ref_domain = (-1, 1)

        # Precompute derivative matrices
        self.diff_mat = np.zeros((size, size))
        for n in range(size - 1):
            self.diff_mat[n, n + 1::2] = 2 * n + 1
        self.diff2_mat = self.diff_mat @ self.diff_mat

        super().__init__(size, domain, log, safe_eval)

    def _eval_value(self, x, size):
        """Evaluate Legendre polynomials up to the given order."""
        return np.polynomial.legendre.legvander(x, deg=size - 1)

    def _eval_all(self, value, size):
        """Evaluate all Legendre polynomials."""
        value = self.transform(np.atleast_1d(value))
        return np.polynomial.legendre.legvander(value, deg=size - 1)

    def _eval_all_der(self, value, size, degree=1):
        """
        Evaluate derivatives of Legendre polynomials.

        :param value: array-like
            Points at which to evaluate.
        :param size: int
            Number of moment functions.
        :param degree: int
            Derivative order.
        :return: ndarray
            Matrix of derivative values.
        """
        value = self.transform(np.atleast_1d(value))
        eval_values = np.empty((value.shape + (size,)))

        for s in range(size):
            if s == 0:
                coef = [1]
            else:
                coef = np.zeros(s + 1)
                coef[-1] = 1

            coef = np.polynomial.legendre.legder(coef, degree)
            eval_values[:, s] = np.polynomial.legendre.legval(value, coef)
        return eval_values

    def _eval_diff(self, value, size):
        """Evaluate first derivatives using precomputed differentiation matrix."""
        t = self.transform(np.atleast_1d(value))
        P_n = np.polynomial.legendre.legvander(t, deg=size - 1)
        return P_n @ self.diff_mat

    def _eval_diff2(self, value, size):
        """Evaluate second derivatives using precomputed differentiation matrix."""
        t = self.transform(np.atleast_1d(value))
        P_n = np.polynomial.legendre.legvander(t, deg=size - 1)
        return P_n @ self.diff2_mat


class TransformedMoments(Moments):
    """
    Linearly transformed moment basis.

    Creates a new set of moment functions as linear combinations
    of another existing set of basis functions.
    """

    def __init__(self, other_moments, matrix):
        """
        Initialize transformed moment functions.

        :param other_moments: Moments
            Original set of moment functions.
        :param matrix: ndarray
            Linear transformation matrix where:
            new_moments = matrix @ old_moments

            The first row must correspond to (1, 0, 0, ...),
            ensuring that new_moments[0] = 1.
        """
        n, m = matrix.shape
        assert m == other_moments.size
        self.size = n
        self.domain = other_moments.domain
        self._origin = other_moments
        self._transform = matrix

    def __eq__(self, other):
        """Check equality with another TransformedMoments object."""
        return (
            type(self) is type(other)
            and self.size == other.size
            and self._origin == other._origin
            and np.all(self._transform == other._transform)
        )

    def _eval_all(self, value, size):
        """Evaluate all transformed moment functions."""
        orig_moments = self._origin._eval_all(value, self._origin.size)
        x1 = np.matmul(orig_moments, self._transform.T)
        return x1[..., :size]

    def _eval_all_der(self, value, size, degree=1):
        """Evaluate derivatives of transformed moment functions."""
        orig_moments = self._origin._eval_all_der(value, self._origin.size, degree=degree)
        x1 = np.matmul(orig_moments, self._transform.T)
        return x1[..., :size]

    def _eval_diff(self, value, size):
        """Evaluate first derivatives of transformed moment functions."""
        orig_moments = self._origin.eval_diff(value, self._origin.size)
        x1 = np.matmul(orig_moments, self._transform.T)
        return x1[..., :size]

    def _eval_diff2(self, value, size):
        """Evaluate second derivatives of transformed moment functions."""
        orig_moments = self._origin.eval_diff2(value, self._origin.size)
        x1 = np.matmul(orig_moments, self._transform.T)
        return x1[..., :size]
