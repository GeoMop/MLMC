import numpy as np


# class Moments:
#     """
#     Class for moments of random distribution
#     """
#     def __init__(self, n_moments):
#         self.mean = 0.0
#         self._bounds = None
#         self._moments_function = None
#         self.n_moments = n_moments
#         self.fixed_quad_n = n_moments
#         self.eps = 0
#
#     @property
#     def bounds(self):
#         """
#         Bounds of random variable
#         :return: array
#         """
#         return self._bounds
#
#     @bounds.setter
#     def bounds(self, bounds):
#         if len(bounds) != 2:
#             raise TypeError("Bounds should have two items")
#         if bounds[1] < bounds[0]:
#             raise ValueError("Second bound must be greater than the first one")
#         self._bounds = bounds


def moment_argument_scale(value, a=0.0, b=None, target = (0.0, 1.0)):
    """
    Transform from interval (a,b) to (0,1).
    :param value: np.array of values to scale.
    :param a, b: interval end points, for None b, jsut shift by 'a' is performed.
    :param target: target interval
    :return: transformed values
    """
    if b is None:
        b = a + 1.0
    assert target[1] - target[0] > 1e-16
    coef = (target[1] - target[0]) / max(b - a, 1e-16)
    shift = target[0]
    return (value - a) * coef + shift


def monomial_moments(value, size=1, a=0.0, b=None):
    """
    Evaluate monomial basis on (a,b) interval in given points.
    :param value: value or np.array of length N
    :param size; basis up to degree size -1
    :param a,b, interval end points, for None b, jsut shift by 'a' is performed.
    :return: moments matrix N x size;
    """
    transformed = moment_argument_scale(value, a, b, target = (0, 1.0))
    moment_mat = np.polynomial.polynomial.polyvander(transformed, deg = size - 1)
    return moment_mat


def fourier_moments(value, size=1, a=0.0, b=None):
    """
    Evaluate Fourier basis on (a,b) interval in given points.
    :param value: value or np.array of length N
    :param size; number of basis functions from sequence: (1.0, cos(x), sin(x), cos(2*x), sin(2*x), ...)
                 reasonable choice is 1 + 2*R
    :param a,b, interval end points, for None b, jsut shift by 'a' is performed.
    :return: moments matrix N x size;
    """
    value = np.atleast_1d(value)
    transformed = moment_argument_scale(value, a, b, target = (0.0, 2*np.pi) )

    R = int(size / 2)
    shorter_sin = 1 - int(size % 2)
    k = np.arange(1, R + 1)
    kx = np.outer(transformed, k)


    res = np.empty((len(value), size))
    res[:, 0] = 1
    res[:, 1::2] = np.cos(kx[:, :])
    res[:, 2::2] = np.sin(kx[:, : R -shorter_sin])
    return res


def legendre_moments(value, size=1, a=0.0, b=None, safe_eval=True):
    """
    Evaluate Legendere polynomials basis on (a,b) interval in given points.
    :param value: value or np.array of length N
    :param size; polynomials up to degree size -1
    :param a,b, interval end points, for None b, jsut shift by 'a' is performed.
    :param safe_eval, project values to the domain (a,b)
    :return: moments matrix N x size;
    """
    transformed = moment_argument_scale(value, a, b, target = (-1.0, 1.0))
    if safe_eval:
        transformed = np.atleast_1d(transformed)
        mask = transformed < -1.0
        n_out = np.sum(mask)
        transformed[mask] = -1.0
        mask = transformed > 1.0
        n_out += np.sum(mask)
        transformed[mask] = 1.0
        #transformed = np.minimum(1.0, np.maximum(-1.0, transformed))
        if n_out > 0.1 * len(transformed):
            print("N outlayers: ", n_out)
    return np.polynomial.legendre.legvander(transformed, deg = size - 1)
