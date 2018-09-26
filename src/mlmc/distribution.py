import numpy as np
import scipy as sc
import scipy.integrate as integrate


class Distribution:
    """
    Calculation of the distribution
    """

    def __init__(self, moments_obj, moment_data, is_positive=False, domain=None):
        """
        :param moments_fn: Function for calculating moments
        :param moment_data: Array  of moments
        :param positive_distr: Indication of distribution for a positive variable.
        """
        # Family of moments basis functions.
        self.moments_basis = moments_obj

        # Moment evaluation function with bounded number of moments and their domain.
        self.moments_fn = None

        # Domain of the density approximation (and moment functions).
        self.domain = domain

        # Approximation of moment values.
        self.moment_means = moment_data[:, 0]
        self.moment_vars = moment_data[:, 1]

        # Force density with positive support.
        self.is_positive = is_positive

        # Approximation paramters. Lagrange multipliers for moment equations.
        self.multipliers = None
        # Number of basis functions to approximate the density.
        # In future can be smaller then number of provided approximative moments.
        self.approx_size = len(self.moment_means)
        assert moments_obj.size == self.approx_size
        self.moments_fn = moments_obj

    def choose_parameters_from_samples(self, samples):
        """
        Determine model hyperparameters, in particular domain of the density function,
        from given samples.
        :param samples: np array of samples from the distribution or its approximation.
        :return: None
        """
        self.domain = (np.min(samples), np.max(samples))

    @staticmethod
    def choose_parameters_from_moments(mean, variance, quantile=0.9999, log=False):
        """
        Determine model hyperparameters, in particular domain of the density function,
        from given samples.
        :param samples: np array of samples from the distribution or its approximation.
        :return: None
        """
        if log:
            # approximate by log normal
            # compute mu, sigma parameters from observed mean and variance
            sigma_sq = np.log(np.exp(np.log(variance) - 2.0 * np.log(mean)) + 1.0)
            mu = np.log(mean) - sigma_sq / 2.0
            sigma = np.sqrt(sigma_sq)
            domain = tuple(sc.stats.lognorm.ppf([1.0 - quantile, quantile], s=sigma, scale=np.exp(mu)))
            assert np.isclose(mean, sc.stats.lognorm.mean(s=sigma, scale=np.exp(mu)))
            assert np.isclose(variance, sc.stats.lognorm.var(s=sigma, scale=np.exp(mu)))
        else:
            domain = tuple(sc.stats.norm.ppf([1.0 - quantile, quantile], loc=mean, scale=np.sqrt(variance)))
        return domain

    def choose_parameters_from_approximation(self):
        pass


    def estimate_density_minimize(self, tol=1e-5):
        """
        Optimize density estimation
        :return: None
        """
        # Initialize domain, multipliers, ...
        self._initialize_params(tol)

        result = sc.optimize.minimize(self.functional, self.multipliers, method='trust-exact',
                                      jac=self._calculate_moments_approximation,
                                      hess=self._calculate_jacobian_matrix,
                                      options={'gtol': tol, 'disp': False, 'maxiter': 1000})

        jac_norm = np.linalg.norm(result.jac)
        if result.success or jac_norm < tol:
            result.success = True
        self.multipliers = result.x
        result.fun_norm = jac_norm
        return result

    def estimate_density(self, tol=None):
        """
        Run nonlinear iterative solver to estimate density, use previous solution as initial guess.
        :return: None
        """
        # Initialize domain, multipliers, ...
        self._initialize_params(tol)

        result = sc.optimize.root(
            fun=self._calculate_moments_approximation,
            x0=self.multipliers,
            jac=self._calculate_jacobian_matrix,
            tol=tol
        )

        fun_norm = np.linalg.norm(result.fun)
        if result.success or fun_norm < tol:
            result.success = True

        self.multipliers = result.x
        result.fun_norm = fun_norm
        return result

    def _initialize_params(self, tol=None):
        """
        Initialize parameters for density estimation
        :return: None
        """
        assert self.domain is not None
        if self.is_positive:
            self.domain = (max(0.0, self.domain[0]), self.domain[1])

        self._n_quad_points = 20 * self.approx_size
        self._end_point_diff = self.end_point_derivatives()
        self._penalty_coef = 1e5

        assert tol is not None
        if self.multipliers is None:
            self.multipliers = np.zeros(self.approx_size)
        self.multipliers[1:3] = 1.0

    def _iteration_monitor(self, x, f):
        print("Norm: {} x: {}".format(np.linalg.norm(f), x))

    def density(self, value, moments_fn=None):
        """
        :param value: float or np.array
        :param moments_fn: counting moments function
        :return: density for passed value
        """
        if moments_fn is None:
            moments = self.moments_fn(value)
        else:
            moments = moments_fn(value)
        return np.exp(-np.sum(moments * self.multipliers, axis=1))

    def cdf(self, values):
        values = np.atleast_1d(values)
        np.sort(values)
        last_x = self.domain[0]
        last_y = 0
        cdf_y = np.empty(len(values))

        for i, val in enumerate(values):
            if val <= self.domain[0]:
                last_y = 0
            elif val >= self.domain[1]:
                last_y = 1
            else:
                dy = integrate.fixed_quad(self.density, last_x, val, n=10)[0]
                last_x = val
                last_y = last_y + dy
            cdf_y[i] = last_y
        return cdf_y

    def functional(self, multipliers):
        """
        Maximized functional
        :param multipliers: current multipliers
        :return: float
        """

        def integrand(x):
            return np.exp(-np.sum(self.moments_fn(x) * multipliers, axis=1))

        integral = sc.integrate.fixed_quad(integrand, self.domain[0], self.domain[1], n=self._n_quad_points)[0]
        sum = np.sum(self.moment_means * multipliers)

        end_diff = np.dot(self._end_point_diff, multipliers)
        penalty = np.sum(np.maximum(end_diff, 0)**2)
        fun =  sum + integral + self._penalty_coef * penalty
        return fun

    def end_point_derivatives(self):
        """
        Compute approximation of moment derivatives at endpoints of the domain.
        :return: array (2, n_moments)
        """
        eps = 1e-10
        left_diff  = self.moments_fn(self.domain[0] + eps) - self.moments_fn(self.domain[0])
        right_diff = -self.moments_fn(self.domain[1]) - self.moments_fn(self.domain[1] - eps)
        return np.stack((left_diff[0,:], right_diff[0,:]), axis=0)/eps


    def _calculate_moments_approximation(self, multipliers):
        """
        :param lagrangians: array, lagrangians parameters
        :return: array, moments approximation
        """

        def integrand(value, lg=multipliers):
            moments = self.moments_fn(value)
            density = np.exp(- np.sum(moments * lg, axis=1))
            return moments.T * density

        integral = sc.integrate.fixed_quad(integrand, self.domain[0], self.domain[1], n=self._n_quad_points)

        end_diff = np.dot(self._end_point_diff, multipliers)
        penalty = 2 * np.dot( np.maximum(end_diff, 0), self._end_point_diff)
        return self.moment_means - integral[0] + self._penalty_coef * penalty

    def _calculate_jacobian_matrix(self, multipliers):
        """
        :param multipliers: np.array, lambda
        :return: jacobian matrix, symmetric,
        """
        triu_idx = np.triu_indices(self.approx_size)

        def integrand(value, lg=multipliers, triu_idx=triu_idx):
            """
            Upper triangle of the matrix, flatten.
            """
            moments = self.moments_fn(value)
            density = np.exp(- np.sum(moments * lg, axis=1))
            moment_outer = np.einsum('ki,kj->ijk', moments, moments)
            triu_outer = moment_outer[triu_idx[0], triu_idx[1], :]
            return triu_outer * density

        # Initialization of matrix
        integral = sc.integrate.fixed_quad(integrand, self.domain[0], self.domain[1],
                                           n=self._n_quad_points)


        jacobian_matrix = np.empty(shape=(self.approx_size, self.approx_size))
        jacobian_matrix[triu_idx[0], triu_idx[1]] = integral[0]
        jacobian_matrix[triu_idx[1], triu_idx[0]] = integral[0]

        end_diff = np.dot(self._end_point_diff, multipliers)
        for side in [0,1]:
            if end_diff[side] > 0:
                penalty = 2 * np.outer(self._end_point_diff[side], self._end_point_diff[side])
                jacobian_matrix += self._penalty_coef * penalty

        return jacobian_matrix

    def _calculate_gradient(self, multipliers):
        """
        Estimate gradient for current multipliers
        :param multipliers: array
        :return: array
        """
        epsilon = np.ones(len(multipliers)) * 1e-5
        approx_gradient = sc.optimize.approx_fprime(multipliers, self.functional, epsilon)

        return approx_gradient


def compute_exact_moments(moments_fn, density, tol=1e-4):
    """
    Compute approximation of moments using exact density.
    :param moments_fn: Moments function.
    :param n_moments: Number of mements to compute.
    :param density: Density function (must accept np vectors).
    :param a, b: Integral bounds, approximate integration over R.
    :param tol: Tolerance of integration.
    :return: np.array, moment values
    """

    def integrand(x):
        return moments_fn(x).T * density(x)

    a, b = moments_fn.domain
    last_integral = integrate.fixed_quad(integrand, a, b, n=moments_fn.size)[0]

    n_points = 2 * moments_fn.size
    integral = integrate.fixed_quad(integrand, a, b, n=n_points)[0]

    while np.linalg.norm(integral - last_integral) > tol:
        last_integral = integral
        n_points *= 2
        integral = integrate.fixed_quad(integrand, a, b, n=n_points)[0]
    return integral


def KL_divergence(prior_density, posterior_density, a, b):
    """
    \int_R P(x) \log( P(X)/Q(x)) \dx
    :param prior_density: P
    :param posterior_density: Q
    :return: KL divergence value
    """
    integrand = lambda x: prior_density(x) * np.log(prior_density(x) / posterior_density(x))
    return integrate.quad(integrand, a, b)[0]


def L2_distance(prior_density, posterior_density, a, b):
    integrand = lambda x: (posterior_density(x) - prior_density(x)) ** 2
    return np.sqrt(integrate.quad(integrand, a, b))[0]
