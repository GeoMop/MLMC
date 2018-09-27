import numpy as np
import scipy as sc
import scipy.integrate as integrate


class Distribution:
    """
    Calculation of the distribution
    """

    def __init__(self, moments_obj, moment_data, domain=None, force_decay=(True, True)):
        """
        :param moments_obj: Function for calculating moments
        :param moment_data: Array  of moments and their vars; (n_moments, 2)
        :param domain: Explicit domain fo reconstruction. None = use domain of moments.
        :param force_decay: Flag for each domain side to enforce decay of the PDF approximation.
        """
        # Family of moments basis functions.
        self.moments_basis = moments_obj

        # Moment evaluation function with bounded number of moments and their domain.
        self.moments_fn = None

        # Domain of the density approximation (and moment functions).
        if domain is None:
            domain = moments_obj.domain
        self.domain = domain
        # Indicates whether force decay of PDF at domain endpoints.
        self.decay_penalty = force_decay

        # Approximation of moment values.
        self.moment_means = moment_data[:, 0]
        self.moment_vars = moment_data[:, 1]

        # Approximation paramters. Lagrange multipliers for moment equations.
        self.multipliers = None
        # Number of basis functions to approximate the density.
        # In future can be smaller then number of provided approximative moments.
        self.approx_size = len(self.moment_means)
        assert moments_obj.size == self.approx_size
        self.moments_fn = moments_obj

    # def choose_parameters_from_samples(self, samples):
    #     """
    #     Determine model hyperparameters, in particular domain of the density function,
    #     from given samples.
    #     :param samples: np array of samples from the distribution or its approximation.
    #     :return: None
    #     """
    #     self.domain = (np.min(samples), np.max(samples))
    #
    # @staticmethod
    # def choose_parameters_from_moments(mean, variance, quantile=0.9999, log=False):
    #     """
    #     Determine model hyperparameters, in particular domain of the density function,
    #     from given samples.
    #     :param samples: np array of samples from the distribution or its approximation.
    #     :return: None
    #     """
    #     if log:
    #         # approximate by log normal
    #         # compute mu, sigma parameters from observed mean and variance
    #         sigma_sq = np.log(np.exp(np.log(variance) - 2.0 * np.log(mean)) + 1.0)
    #         mu = np.log(mean) - sigma_sq / 2.0
    #         sigma = np.sqrt(sigma_sq)
    #         domain = tuple(sc.stats.lognorm.ppf([1.0 - quantile, quantile], s=sigma, scale=np.exp(mu)))
    #         assert np.isclose(mean, sc.stats.lognorm.mean(s=sigma, scale=np.exp(mu)))
    #         assert np.isclose(variance, sc.stats.lognorm.var(s=sigma, scale=np.exp(mu)))
    #     else:
    #         domain = tuple(sc.stats.norm.ppf([1.0 - quantile, quantile], loc=mean, scale=np.sqrt(variance)))
    #     return domain
    #
    # def choose_parameters_from_approximation(self):
    #     pass


    def estimate_density_minimize(self, tol=1e-5):
        """
        Optimize density estimation
        :return: None
        """
        # Initialize domain, multipliers, ...
        self._initialize_params(tol)
        
        result = sc.optimize.minimize(self._calculate_functional, self.multipliers, method='trust-exact',
                                      jac=self._calculate_gradient,
                                      hess=self._calculate_jacobian_matrix,
                                      options={'gtol': tol, 'disp': False, 'maxiter': 200})

        # result = sc.optimize.minimize(self._calculate_functional, self.multipliers, method='BFGS',
        #                               jac=self._calculate_gradient,
        #                               options={'gtol': tol, 'disp': False, 'maxiter': 100})

        jac_norm = np.linalg.norm(result.jac)
        if result.success or jac_norm < tol:
            result.success = True
        if not result.success and result.message[:5] == 'A bad':
            result.success = True
        self.multipliers = result.x
        result.fun_norm = jac_norm
        return result

    def estimate_density(self, tol=None):
        """
        Run nonlinear iterative solver to estimate density, use previous solution as initial guess.
        Faster, but worse stability.
        :return: None
        """
        # Initialize domain, multipliers, ...
        self._initialize_params(tol)

        result = sc.optimize.root(
            fun=self._calculate_gradient,
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


    def _initialize_params(self, tol=None):
        """
        Initialize parameters for density estimation
        :return: None
        """
        assert self.domain is not None

        assert tol is not None
        self._quad_tolerance = tol / 16

        # initial point
        if self.multipliers is None:
            self.multipliers = np.zeros(self.approx_size)
        self.multipliers[1:3] = 1.0

        # Degree of Gauss quad to use on every subinterval determined by adaptive quad.
        self._gauss_degree = 21
        # Last multipliers and corresponding gradient.
        self._last_multipliers = np.zeros_like(self.multipliers)
        self._last_gradient = np.ones_like(self.multipliers) * tol
        # Evaluate endpoint derivatives of the moments.
        self._end_point_diff = self.end_point_derivatives()
        # Panalty coef for endpoint derivatives
        self._penalty_coef = 100
        # Log to store error messages from quad, report only on conv. problem.
        self._quad_log = []

    def _update_quadrature(self, multipliers, force=False):
        """
        Update quadrature points and their moments and weights based on integration of the density.
        """
        mult_norm = np.linalg.norm(multipliers - self._last_multipliers)
        grad_norm = np.linalg.norm(self._last_gradient)
        if not force and grad_norm * mult_norm < self._quad_tolerance:
            print("OPT")
            return

        print(grad_norm * mult_norm, self._quad_tolerance)
        # More precise but depends on actual gradient which may not be available
        # quad_err_estimate = np.abs(np.dot(self._last_gradient, (multipliers - self._last_multipliers)))
        # quad_err_estimate > self._quad_tolerance

        def integrand(x):
            return np.exp(-np.sum(self.moments_fn(x) * multipliers, axis=1))

        result = sc.integrate.quad(integrand, self.domain[0], self.domain[1], full_output = 1)
        if len(result) > 3:
            y, abserr, info, message = result
            self._quad_log.append(result)
        else:
            y, abserr, info = result
        pt, w = np.polynomial.legendre.leggauss(self._gauss_degree)
        K = info['last']
        a = info['alist'][:K, None]
        b = info['blist'][:K, None]
        points = (pt[None, :] + 1) / 2 * (b - a) + a
        weights = w[None, :] * (b - a) / 2
        self._quad_points = points.flatten()
        self._quad_weights = weights.flatten()
        self._quad_moments = self.moments_fn(self._quad_points)

    def end_point_derivatives(self):
        """
        Compute approximation of moment derivatives at endpoints of the domain.
        :return: array (2, n_moments)
        """
        eps = 1e-10
        left_diff = right_diff = np.zeros((1, self.moments_fn.size))
        if self.decay_penalty[0]:
            left_diff  = self.moments_fn(self.domain[0] + eps) - self.moments_fn(self.domain[0])
        if self.decay_penalty[1]:
            right_diff = -self.moments_fn(self.domain[1]) + self.moments_fn(self.domain[1] - eps)

        return np.stack((left_diff[0,:], right_diff[0,:]), axis=0)/eps


    def _calculate_functional(self, multipliers):
        """
        Minimized functional.
        :param multipliers: current multipliers
        :return: float
        """
        self._update_quadrature(multipliers)
        q_density = np.exp(-np.dot(self._quad_moments, multipliers))
        integral = np.dot(q_density, self._quad_weights)
        sum = np.sum(self.moment_means * multipliers)

        end_diff = np.dot(self._end_point_diff, multipliers)
        penalty = np.sum(np.maximum(end_diff, 0)**2)
        fun =  sum + integral
        fun = fun + np.abs(fun) * self._penalty_coef * penalty
        return fun


    def _calculate_gradient(self, multipliers):
        """
        Gradient of th functional
        :return: array, shape (n_moments,)
        """
        self._update_quadrature(multipliers)
        q_gradient = self._quad_moments.T * np.exp(-np.dot(self._quad_moments, multipliers))
        integral = np.dot(q_gradient, self._quad_weights)

        end_diff = np.dot(self._end_point_diff, multipliers)
        penalty = 2 * np.dot( np.maximum(end_diff, 0), self._end_point_diff)
        fun = np.sum(self.moment_means * multipliers) + integral[0]
        gradient =  self.moment_means - integral + np.abs(fun) * self._penalty_coef * penalty

        self._last_gradient = gradient
        self._last_multipliers = multipliers
        return gradient

    def _calculate_jacobian_matrix(self, multipliers):
        """
        :return: jacobian matrix, symmetric, (n_moments, n_moments)
        """
        self._update_quadrature(multipliers)

        q_density = np.exp(-np.dot(self._quad_moments, multipliers))
        moment_outer = np.einsum('ki,kj->ijk', self._quad_moments, self._quad_moments)
        triu_idx = np.triu_indices(self.approx_size)
        triu_outer = moment_outer[triu_idx[0], triu_idx[1], :]
        q_jac = triu_outer * q_density
        integral = np.dot(q_jac, self._quad_weights)

        jacobian_matrix = np.empty(shape=(self.approx_size, self.approx_size))
        jacobian_matrix[triu_idx[0], triu_idx[1]] = integral
        jacobian_matrix[triu_idx[1], triu_idx[0]] = integral

        end_diff = np.dot(self._end_point_diff, multipliers)
        fun = np.sum(self.moment_means * multipliers) + jacobian_matrix[0,0]
        for side in [0,1]:
            if end_diff[side] > 0:
                penalty = 2 * np.outer(self._end_point_diff[side], self._end_point_diff[side])
                jacobian_matrix += np.abs(fun) * self._penalty_coef * penalty

        self._last_gradient = jacobian_matrix[0,:]
        self._last_multipliers = multipliers
        return jacobian_matrix


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

    if np.linalg.norm(integral - last_integral) > tol:
        for i in range(moments_fn.size):
            fn = lambda x, m = i: moments_fn(x)[0,m] * density(x)
            integral[i] = integrate.quad(fn, a, b, epsabs = tol)[0]
    return integral


def KL_divergence(prior_density, posterior_density, a, b):
    """
    Compute D_KL(P | Q) = \int_R P(x) \log( P(X)/Q(x)) \dx
    :param prior_density: P
    :param posterior_density: Q
    :return: KL divergence value
    """
    integrand = lambda x: prior_density(x) * max(np.log(prior_density(x) / posterior_density(x)), -1e300)
    value = integrate.quad(integrand, a, b, epsabs=1e-10)[0]
    return max(value, 1e-10)


def L2_distance(prior_density, posterior_density, a, b):
    integrand = lambda x: (posterior_density(x) - prior_density(x)) ** 2
    return np.sqrt(integrate.quad(integrand, a, b))[0]
