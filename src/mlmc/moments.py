import numpy as np
import numpy.ma as ma


class Moments:
    """
    Class for moments of random distribution
    """
    def __init__(self, size, domain, log=False, safe_eval=True):
        assert size > 0
        self.size = size
        self.domain = domain

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


class Monomial(Moments):
    def __init__(self, size, domain=(0, 1), log=False, safe_eval=True):
        self.ref_domain = (0, 1)
        super().__init__(size, domain, log=log, safe_eval=safe_eval)

    def __call__(self, value):
        # Create array from values and transform values outside the ref domain
        t = self.transform(np.atleast_1d(value))
        # Vandermonde matrix
        return np.polynomial.polynomial.polyvander(t, deg = self.size - 1)


class Fourier(Moments):
    def __init__(self, size, domain=(0, 2*np.pi), log=False, safe_eval=True):
        self.ref_domain = (0, 2*np.pi)
        super().__init__(size, domain, log=log, safe_eval=safe_eval)

    def __call__(self, value):
        # Transform values
        t = self.transform(np.atleast_1d(value))

        # Half the number of moments
        R = int(self.size / 2)
        shorter_sin = 1 - int(self.size % 2)
        k = np.arange(1, R + 1)
        kx = np.outer(t, k)

        res = np.empty((len(t), self.size))
        res[:, 0] = 1

        # Odd column index
        res[:, 1::2] = np.cos(kx[:, :])
        # Even column index
        res[:, 2::2] = np.sin(kx[:, : R - shorter_sin])
        return res


class Legendre(Moments):

    def __init__(self, size, domain, log=False, safe_eval=True):
        self.ref_domain = (-1, 1)
        super().__init__(size, domain, log, safe_eval)

    def __call__(self, value):
        t = self.transform(np.atleast_1d(value))
        return np.polynomial.legendre.legvander(t, deg=(self.size - 1))
