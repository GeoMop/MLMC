import numpy as np
import numpy.ma as ma
from scipy.interpolate import BSpline
import numpy.polynomial.polynomial as P


class Moments:
    """
    Class for _moments_fn of random distribution
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

        self.diff_mat = np.zeros((size, size))
        for n in range(size - 1):
            self.diff_mat[n, n + 1::2] = 2 * n + 1
        self.diff2_mat = self.diff_mat @ self.diff_mat

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
        if isinstance(value, (int, float)):
            return value - self.mean

        if not isinstance(self.mean, int):
            value[...] = value[...] - self.mean[:, None, None]
        else:
            if isinstance(value, (float, int)):
                value = value - self.mean
            else:
                value[...] = value[...] - self.mean

        return value

    def eval_all_der(self, value, size=None, degree=1):
        value = self._center(value)
        if size is None:
            size = self.size
        return self._eval_all_der(value, size, degree)

    def eval_diff(self, value, size=None):
        value = self._center(value)
        if size is None:
            size = self.size
        return self._eval_diff(value, size)

    def eval_diff2(self, value, size=None):
        value = self._center(value)
        if size is None:
            size = self.size
        return self._eval_diff2(value, size)


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

    def _eval_all_der(self, value, size, degree=1):
        """
        Derivative of Legendre polynomials
        :param value: values to evaluate
        :param size: number of _moments_fn
        :param degree: degree of derivative
        :return:
        """
        t = self.transform(np.atleast_1d(value))
        # Vandermonde matrix
        poly_matrix = np.polynomial.polynomial.polyvander(t, deg=size-1+degree)
        return P.polyder(poly_matrix, m=degree, axis=1)

    def eval(self, i, value):
        t = self.transform(np.atleast_1d(value))
        return t**i

    def _eval_diff(self, value, size):
        t = self.transform(np.atleast_1d(value))
        P_n = np.polynomial.polynomial.polyvander(t, deg=size - 1)
        return P_n @ self.diff_mat

    def _eval_diff2(self, value, size):
        t = self.transform(np.atleast_1d(value))
        P_n = np.polynomial.polynomial.polyvander(t, deg=size - 1)
        return P_n @ self.diff2_mat


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

        # Half the number of _moments_fn
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

    def _eval_value(self, x, size):
        return np.polynomial.legendre.legvander(x, deg=size-1)

    def _eval_all(self, value, size):
        value = self.transform(np.atleast_1d(value))
        return np.polynomial.legendre.legvander(value, deg=size - 1)

    def _eval_all_der(self, value, size, degree=1):
        """
        Derivative of Legendre polynomials
        :param value: values to evaluate
        :param size: number of _moments_fn
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


class BivariateMoments:
    def __init__(self, moment_x, moment_y):
        self.moment_x = moment_x
        self.moment_y = moment_y
        assert self.moment_y.size == self.moment_x.size

        self.size = self.moment_x.size
        self.domain = [self.moment_x.domain, self.moment_y.domain]

    def eval_value(self, value):
        x, y = value
        results = np.empty((self.size, self.size))
        for i in range(self.size):
            for j in range(self.size):
                results[i, j] = np.squeeze(self.moment_x(x))[i] * np.squeeze(self.moment_y(y))[j]

        return results

    def eval_all(self, value):
        results, x, y = self._preprocess_value(value)
        for i in range(self.size):
            for j in range(self.size):
                results[:, i, j] = np.squeeze(self.moment_x(x))[:, i] * np.squeeze(self.moment_y(y))[:, j]
        return results

    def eval_all_der(self, value, degree=1):
        results, x, y = self._preprocess_value(value)

        for i in range(self.size):
            for j in range(self.size):
                results[:, i, j] = np.squeeze(self.moment_x.eval_all_der(x, degree=degree))[:, i] *\
                                   np.squeeze(self.moment_y.eval_all_der(y, degree=degree))[:, j]
        return results

    def _preprocess_value(self, value):
        if not isinstance(value[0], (list, tuple, np.ndarray)):
            return self.eval_value(value)
        value = np.array(value)
        x = value[0, :]
        y = value[1, :]
        return np.empty((len(value[0]), self.size, self.size)), x, y


class Spline(Moments):

    def __init__(self, size, domain, log=False, safe_eval=True):
        self.ref_domain = domain
        self.poly_degree = 3
        self.polynomial = None

        super().__init__(size, domain, log, safe_eval)

        self._generate_knots(size)
        self._generate_splines()

    def _generate_knots(self, size=2):
        """
        Code from bgem
        """
        knot_range = self.ref_domain
        degree = self.poly_degree
        n_intervals = size
        n = n_intervals + 2 * degree + 1
        knots = np.array((knot_range[0],) * n)
        diff = (knot_range[1] - knot_range[0]) / n_intervals
        for i in range(degree + 1, n - degree):
            knots[i] = (i - degree) * diff + knot_range[0]
        knots[-degree - 1:] = knot_range[1]
        self.knots = knots

    def _generate_splines(self):
        self.splines = []
        if len(self.knots) <= self.size:
            self._generate_knots(self.size)
        for i in range(self.size-1):
            c = np.zeros(len(self.knots))
            c[i] = 1
            self.splines.append(BSpline(self.knots, c, self.poly_degree))

    def _eval_value(self, x, size):
        values = np.zeros(size)
        index = 0
        values[index] = 1
        for spline in self.splines:
            index += 1
            if index >= size:
                break
            values[index] = spline(x)
        return values

    def _eval_all(self, x, size):
        x = self.transform(np.atleast_1d(x))

        if len(x.shape) == 1:
            values = np.zeros((size, len(x)))
            transpose_tuple = (1, 0)
            values[0] = np.ones(len(x))
            index = 0

        elif len(x.shape) == 2:
            values = np.zeros((size, x.shape[0], x.shape[1]))
            transpose_tuple = (1, 2, 0)
            values[0] = np.ones((x.shape[0], x.shape[1]))
            index = 0

        x = np.array(x, copy=False, ndmin=1) + 0.0

        for spline in self.splines:
            index += 1
            if index >= size:
                break
            values[index] = spline(x)

        return values.transpose(transpose_tuple)

    def _eval_all_der(self, x, size, degree=1):
        """
        Derivative of Legendre polynomials
        :param x: values to evaluate
        :param size: number of _moments_fn
        :param degree: degree of derivative
        :return:
        """
        x = self.transform(np.atleast_1d(x))

        if len(x.shape) == 1:
            values = np.zeros((size, len(x)))
            transpose_tuple = (1, 0)
            values[0] = np.zeros(len(x))
            index = 0

        elif len(x.shape) == 2:
            values = np.zeros((size, x.shape[0], x.shape[1]))
            transpose_tuple = (1, 2, 0)
            values[0] = np.zeros((x.shape[0], x.shape[1]))
            index = 0

        x = np.array(x, copy=False, ndmin=1) + 0.0

        for spline in self.splines:
            index += 1
            if index >= size:
                break
            values[index] = (spline.derivative(degree))(x)
        return values.transpose(transpose_tuple)


class TransformedMoments(Moments):
    def __init__(self, other_moments, matrix, mean=0):
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
        self.mean = 0
        self.size = n
        self.domain = other_moments.domain
        self.mean = mean
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


class TransformedMomentsDerivative(Moments):
    def __init__(self, other_moments, matrix, degree=2, mean=0):
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
        self.mean = mean
        self._origin = other_moments
        self._transform = matrix
        self._degree = degree

    def __eq__(self, other):
        return type(self) is type(other) \
                and self.size == other.size \
                and self._origin == other._origin \
                and np.all(self._transform == other._transform)

    def _eval_all(self, value, size):
        value = np.squeeze(value)
        orig_moments = self._origin._eval_all_der(value, self._origin.size, degree=self._degree)
        x1 = np.matmul(orig_moments, self._transform.T)
        return x1[..., :size]
