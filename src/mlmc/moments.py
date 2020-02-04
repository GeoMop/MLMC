import autograd.numpy as np
import numpy
import numpy.ma as ma


class Moments:
    """
    Class for moments of random distribution
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
        return self._eval_all(value, i+1)[:, -1]

    def eval_all(self, value, size=None):
        if size is None:
            size = self.size
        return self._eval_all(value, size)

    def eval_all_der(self, value, size=None, degree=1):
        if size is None:
            size = self.size
        return self._eval_all_der(value, size, degree)


class Monomial(Moments):
    def __init__(self, size, domain=(0, 1), log=False, safe_eval=True):
        self.ref_domain = (0, 1)
        super().__init__(size, domain, log=log, safe_eval=safe_eval)

    def _eval_all(self, value, size):
        # Create array from values and transform values outside the ref inter_points_domain
        t = self.transform(np.atleast_1d(value))
        # Vandermonde matrix
        return numpy.polynomial.polynomial.polyvander(t, deg = size - 1)

    def eval(self, i, value):
        t = self.transform(np.atleast_1d(value))
        return t**i


class Fourier(Moments):
    def __init__(self, size, domain=(0, 2*np.pi), log=False, safe_eval=True):
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

    def __init__(self, size, domain, log=False, safe_eval=True):
        self.ref_domain = (-1, 1)
        super().__init__(size, domain, log, safe_eval)

    def _eval_value(self, x, size):
        return numpy.polynomial.legendre.legvander(x, deg=size-1)

    def _eval_all(self, value, size):
        value = self.transform(np.atleast_1d(value))
        return numpy.polynomial.legendre.legvander(value, deg=size - 1)

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

            coef = numpy.polynomial.legendre.legder(coef, degree)
            eval_values[:, s] = numpy.polynomial.legendre.legval(value, coef)#COEF[s])
            #eval_values[:, 0] = 1

        return eval_values


class Spline(Moments):

    def __init__(self, size, domain, log=False, safe_eval=True, smoothing_factor=1, interpolation_points=None):
        self.ref_domain = (-1, 1)
        self.poly_degree = 3
        self.smothing_factor = smoothing_factor
        self.polynomial = None

        ################################
        #accuracy = 1e-3

        #self.smothing_factor = accuracy *(1/(1+self.poly_degree))

        if interpolation_points is None:
            self.interpolation_points = np.linspace(self.ref_domain[0], self.ref_domain[1], size)
        else:
            self.interpolation_points = interpolation_points

        self._create_polynomial()
        super().__init__(size, domain, log, safe_eval)

    def _create_polynomial(self):
        coeficients_matrix = np.empty((self.poly_degree + 1, self.poly_degree + 1))
        constants_matrix = np.empty(self.poly_degree + 1)

        # g(1) = 0, g(-1) = 1
        coeficients_matrix[0] = np.ones(self.poly_degree + 1)
        coeficients_matrix[1] = [1 if i % 2 != 0 or i == self.poly_degree else -1 for i in range(self.poly_degree + 1)]
        constants_matrix[0] = 0
        constants_matrix[1] = 1

        for j in range(self.poly_degree - 1):
            coeficients_matrix[j + 2] = np.flip(np.array([(1 ** (i + j + 1) - (-1) ** (i + j + 1)) / (i + j + 1) for i
                                                          in range(self.poly_degree + 1)]))
            constants_matrix[j + 2] = (-1) ** j / (j + 1)

        poly_coefs = np.linalg.solve(coeficients_matrix, constants_matrix)
        self.polynomial = np.poly1d(poly_coefs)

    def _eval_value(self, x, size):
        values = np.zeros(size)
        values[0] = 1
        for index in range(self.interpolation_points-1):
            values[index+1] = self.polynomial(x - self.interpolation_points[index+1]) - self.polynomial(x - self.interpolation_points[index])
        return values

    def _eval_all(self, x, size):
        x = self.transform(np.atleast_1d(x))
        values = np.zeros((len(x), size))
        values[:, 0] = 1
        index = 0

        poly_1 = self.polynomial((x - self.interpolation_points[index + 1])/self.smothing_factor)
        poly_2 = self.polynomial((x - self.interpolation_points[index])/self.smothing_factor)


        pom_values = []

        pom_values.append(np.ones(x.shape))
        for index in range(len(self.interpolation_points) - 1):
            # values[:, index + 1] = self.polynomial((x - self.interpolation_points[index + 1])/self.smothing_factor) - \
            #                     self.polynomial((x - self.interpolation_points[index])/self.smothing_factor)

            pom_values.append((self.polynomial((x - self.interpolation_points[index + 1]) / self.smothing_factor) - \
                                   self.polynomial((x - self.interpolation_points[index]) / self.smothing_factor)))

        pom_values = np.array(pom_values)

        if len(pom_values.shape) == 3:
            return pom_values.transpose((1, 2, 0))
        return pom_values.T

    def _eval_all_der(self, x, size, degree=1):
        """
        Derivative of Legendre polynomials
        :param x: values to evaluate
        :param size: number of moments
        :param degree: degree of derivative
        :return:
        """
        x = self.transform(np.atleast_1d(x))
        polynomial = self.polynomial.deriv(degree)

        values = np.zeros((len(x), size))
        values[:, 0] = 1

        # poly_1 = polynomial((x - self.interpolation_points[index + 1]) / self.smothing_factor)
        # poly_2 = polynomial((x - self.interpolation_points[index]) / self.smothing_factor)

        pom_values = []

        pom_values.append(np.ones(x.shape))
        for index in range(len(self.interpolation_points) - 1):
            # values[:, index + 1] = self.polynomial((x - self.interpolation_points[index + 1])/self.smothing_factor) - \
            #                     self.polynomial((x - self.interpolation_points[index])/self.smothing_factor)

            pom_values.append((polynomial((x - self.interpolation_points[index + 1]) / self.smothing_factor) - \
                               polynomial((x - self.interpolation_points[index]) / self.smothing_factor)))


        pom_values = np.array(pom_values)

        if len(pom_values.shape) == 3:
            return pom_values.transpose((1, 2, 0))

        return pom_values.T


    # def _eval_all_der(self, value, size, degree=1):
    #     """
    #     Derivative of Legendre polynomials
    #     :param value: values to evaluate
    #     :param size: number of moments
    #     :param degree: degree of derivative
    #     :return:
    #     """
    #     value = self.transform(np.atleast_1d(value))
    #     eval_values = np.empty((value.shape + (size,)))
    #
    #     for s in range(size):
    #         if s == 0:
    #             coef = [1]
    #         else:
    #             coef = np.zeros(s+1)
    #             coef[-1] = 1
    #
    #         coef = numpy.polynomial.legendre.legder(coef, degree)
    #         eval_values[:, s] = numpy.polynomial.legendre.legval(value, coef)#COEF[s])
    #
    #     return eval_values


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


        # if not isinstance(value, (float,)) and len(value) > 100:
        #
        #     self.mom_max = np.zeros(size)
        #     self.mom_min = np.zeros(size)
        #
        #     for i in range(size):
        #         self.mom_max[i] = np.max(x1[:, i])
        #         self.mom_min[i] = np.min(x1[:, i])
        #
        #     # print("mom_max - mom_min ", self.mom_max[1:] - self.mom_min[1:])
        #     #
        #     #
        #     # print("x1[:, 1:size] / (mom_max - mom_min) ", (x1[:, 1:size]  - self.mom_min[1:]) /
        #     (self.mom_max[1:] - self.mom_min[1:]))
        #     #
        #     # print("mom max ", self.mom_max)
        #     # print("mom min ", self.mom_min)
        #
        # x1[:, 1:size] = (x1[:, 1:size] - self.mom_min[1:]) / (self.mom_max[1:] - self.mom_min[1:])

        return x1[:, :size]

    def _eval_all_der(self, value, size, degree=1):
        import numpy

        if type(value).__name__ == 'ArrayBox':
            value = value._value

        orig_moments = self._origin._eval_all_der(value, self._origin.size, degree=degree)
        x1 = numpy.matmul(orig_moments, self._transform.T)

        return x1[:, :size]


class TransformedMomentsDerivative(Moments):
    def __init__(self, other_moments, matrix, degree=2):
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
        self._degree = degree
        #self._inv = inv
        #assert np.isclose(matrix[0, 0], 1) and np.allclose(matrix[0, 1:], 0)
        # TODO: find last nonzero for every row to compute which origianl moments needs to be evaluated for differrent sizes.

    def __eq__(self, other):
        return type(self) is type(other) \
                and self.size == other.size \
                and self._origin == other._origin \
                and np.all(self._transform == other._transform)

    def _eval_all(self, value, size):
        if type(value).__name__ == 'ArrayBox':
            value = value._value

        value = numpy.squeeze(value)

        orig_moments = self._origin._eval_all_der(value, self._origin.size, degree=self._degree)
        x1 = numpy.matmul(orig_moments, self._transform.T)

        return x1[:, :size]
