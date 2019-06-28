import numpy as np
import scipy as sc
import scipy.integrate as integrate


class Distribution:
    """
    Calculation of the distribution
    """

    def __init__(self, moments_obj, moment_data, domain=None, force_decay=(True, True),
                 monitor=False):
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
        self.moment_errs = np.sqrt(moment_data[:, 1])

        # Approximation paramters. Lagrange multipliers for moment equations.
        self.multipliers = None
        # Number of basis functions to approximate the density.
        # In future can be smaller then number of provided approximative moments.
        self.approx_size = len(self.moment_means)
        assert moments_obj.size >= self.approx_size
        self.moments_fn = moments_obj

        # Degree of Gauss quad to use on every subinterval determined by adaptive quad.
        self._gauss_degree = 21
        # Panalty coef for endpoint derivatives
        self._penalty_coef = 10

        # Flag for monitoring convergence on stdout.
        self.monitor = monitor
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


    def estimate_density_minimize(self, tol=1e-5, reg_param =0.01):
        """
        Optimize density estimation
        :param tol: Tolerance for the nonlinear system residual, after division by std errors for
        individual moment means, i.e.
        res = || (F_i - \mu_i) / \sigma_i ||_2
        :return: None
        """
        # Initialize domain, multipliers, ...

        # Geometrical series for sizes with base 1.2.
        # Using just odd numbers.
        self._reg_param = reg_param
        base = 1.2
        if self.approx_size <= 5:
            sizes = [self.approx_size]
        else:
            size = self.approx_size
            sizes = [size]
            while size > 4:
                size /= base
                odd_size = 2*round((size-1)/2)+1
                if odd_size != sizes[-1]:
                    sizes.append(odd_size)
            sizes.reverse()
        #print("sizes: ", sizes)

        self.approx_size = sizes[0]
        self._initialize_params(self.approx_size, tol)
        self.extend_size(self.approx_size)
        init_error = np.linalg.norm(self._calculate_gradient(self.multipliers))

        if len(sizes) == 1:
            tolerances = [ tol ]
        else:
            t1 = tol
            t0 = max(tol, init_error / 10)
            t = (np.array(sizes) - sizes[0]) / ( sizes[-1] - sizes[0])
            tolerances = np.exp(np.log(t1) * t + np.log(t0) * (1-t))
        #print("tolerances: ", tolerances)

        total_nit = 0
        for approx_size, approx_tol in  zip(sizes, tolerances):
            self._quad_tolerance = approx_tol / 16
            self.extend_size(approx_size)
            #if approx_size == self.moments_fn.size or approx_size == sizes[0]:
            #    max_it = 200
            #else:
            #    max_it = 20
            max_it = 200
            method = 'trust-exact'
            #method ='Newton-CG'
            result = sc.optimize.minimize(self._calculate_functional, self.multipliers, method=method,
                                          jac=self._calculate_gradient,
                                          hess=self._calculate_jacobian_matrix,
                                          options={'tol': approx_tol, 'xtol': approx_tol,
                                                   'gtol': approx_tol, 'disp': False,  'maxiter': max_it})
            self.multipliers = result.x
            jac_norm = np.linalg.norm(result.jac)
            total_nit += result.nit
            if self.monitor:
                print("Iteration: size: {} nits: {} tol: {:5.3g} res: {:5.3g} msg: {}".format(
                   self.approx_size, result.nit, approx_tol, jac_norm, result.message))

        # Fix normalization
        gradient, _ = self._calculate_exact_moment(self.multipliers, m=0, full_output=0)
        self.multipliers /= gradient

        if result.success or jac_norm < tol:
            result.success = True
        result.nit = total_nit
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
        size = self.approx_size
        if moments_fn is None:
            moments_fn = self.moments_fn
        moments = moments_fn.eval_all(value, size=size)
        return np.exp(-np.sum(moments * self.multipliers / self._moment_errs, axis=1))

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


    def _initialize_params(self, size, tol=None):
        """
        Initialize parameters for density estimation
        :return: None
        """
        assert self.domain is not None

        assert tol is not None
        self._quad_tolerance = tol / 16

        self.moment_errs[0] = np.min(self.moment_errs[1:]) / 8
        # Start with uniform distribution
        self.multipliers = np.zeros(size)
        self.multipliers[0] = -np.log(1/(self.domain[1] - self.domain[0])) * self.moment_errs[0]
        # Log to store error messages from quad, report only on conv. problem.
        self._quad_log = []


    def extend_size(self, new_size):
        self._last_solved_multipliers = self.multipliers
        self._stab_penalty = self._reg_param / np.linalg.norm(self.multipliers)
        #self._stab_penalty = 0.0

        self.approx_size = new_size
        multipliers = self.multipliers
        self.multipliers = np.zeros(new_size)
        self.multipliers[:len(multipliers)] = multipliers
        self._moment_means = self.moment_means[:self.approx_size]
        self._moment_errs = self.moment_errs[:self.approx_size]


        # Evaluate endpoint derivatives of the moments.
        self._end_point_diff = self.end_point_derivatives()
        self._update_quadrature(self.multipliers, force=True)
        #self._calculate_gradient(self.multipliers)

    def eval_moments(self, x):
        return self.moments_fn.eval_all(x, self.approx_size)

    def _calculate_exact_moment(self, multipliers, m=0, full_output=0):
        def integrand(x):
            moms = self.eval_moments(x)
            power = -np.sum(moms * multipliers / self._moment_errs, axis=1)
            power = np.minimum(np.maximum(power, -200), 200)
            return np.exp(power) * moms[:,m]

        result = sc.integrate.quad(integrand, self.domain[0], self.domain[1],
                                   epsabs=self._quad_tolerance, full_output=full_output)
        return result[0], result

    def _calculate_exact_hessian(self, i, j, multipliers=None):
        if multipliers is None:
            multipliers = self.multipliers

        def integrand(x):
            moms = self.eval_moments(x)
            power = -np.sum(moms * multipliers / self._moment_errs, axis=1)
            power = np.minimum(np.maximum(power, -200), 200)
            return np.exp(power) * moms[:,i] * moms[:,j]

        result = sc.integrate.quad(integrand, self.domain[0], self.domain[1],
                                   epsabs=self._quad_tolerance, full_output=False)
        return result[0], result


    def _update_quadrature(self, multipliers, force=False):
        """
        Update quadrature points and their moments and weights based on integration of the density.
        return: True if update of gradient is necessary
        """
        if not force:
            mult_norm = np.linalg.norm(multipliers - self._last_multipliers)
            grad_norm = np.linalg.norm(self._last_gradient)
            if grad_norm * mult_norm < self._quad_tolerance:
                #print("OPT")
                return

            #print(grad_norm * mult_norm, self._quad_tolerance)
            # More precise but depends on actual gradient which may not be available
            quad_err_estimate = np.abs(np.dot(self._last_gradient, (multipliers - self._last_multipliers)))
            if quad_err_estimate < self._quad_tolerance:
                return

        val, result = self._calculate_exact_moment(multipliers, m=0, full_output=1)

        if len(result) > 3:
            y, abserr, info, message = result
            self._quad_log.append(result)
        else:
            y, abserr, info = result
            message =""
        pt, w = np.polynomial.legendre.leggauss(self._gauss_degree)
        K = info['last']
        #print("Update Quad: {} {} {} {}".format(K, y, abserr, message))
        a = info['alist'][:K, None]
        b = info['blist'][:K, None]
        points = (pt[None, :] + 1) / 2 * (b - a) + a
        weights = w[None, :] * (b - a) / 2
        self._quad_points = points.flatten()
        self._quad_weights = weights.flatten()
        self._quad_moments = self.eval_moments(self._quad_points)

        power = -np.dot(self._quad_moments, multipliers/self._moment_errs)
        power = np.minimum(np.maximum(power, -200), 200)
        q_gradient = self._quad_moments.T * np.exp(power)
        integral = np.dot(q_gradient, self._quad_weights) / self._moment_errs
        self._last_multipliers = multipliers
        self._last_gradient = integral

    def end_point_derivatives(self):
        """
        Compute approximation of moment derivatives at endpoints of the domain.
        :return: array (2, n_moments)
        """
        eps = 1e-10
        left_diff = right_diff = np.zeros((1, self.approx_size))
        if self.decay_penalty[0]:
            left_diff  = self.eval_moments(self.domain[0] + eps) - self.eval_moments(self.domain[0])
        if self.decay_penalty[1]:
            right_diff = -self.eval_moments(self.domain[1]) + self.eval_moments(self.domain[1] - eps)

        return np.stack((left_diff[0,:], right_diff[0,:]), axis=0)/eps/ self._moment_errs[None, :]


    def _calculate_functional(self, multipliers):
        """
        Minimized functional.
        :param multipliers: current multipliers
        :return: float
        """
        update_grad = self._update_quadrature(multipliers)

        power = -np.dot(self._quad_moments, multipliers / self._moment_errs)
        power = np.minimum(np.maximum(power, -200), 200)
        q_density = np.exp(power)
        integral = np.dot(q_density, self._quad_weights)
        sum = np.sum(self._moment_means * multipliers / self._moment_errs)

        end_diff = np.dot(self._end_point_diff, multipliers)
        penalty = np.sum(np.maximum(end_diff, 0)**2)
        fun =  sum + integral
        fun = fun + np.abs(fun) * self._penalty_coef * penalty
        last_size = len(self._last_solved_multipliers)
        fun += 0.5 * self._stab_penalty * np.sum((self._last_solved_multipliers - multipliers[:last_size])**2)
        return fun


    def _calculate_gradient(self, multipliers):
        """
        Gradient of th functional
        :return: array, shape (n_moments,)
        """
        self._update_quadrature(multipliers)
        power = -np.dot(self._quad_moments, multipliers / self._moment_errs)
        #power = np.minimum(np.maximum(power, -200), 200)
        q_density = np.exp(power)
        q_gradient = self._quad_moments.T * q_density
        integral = np.dot(q_gradient, self._quad_weights) / self._moment_errs

        end_diff = np.dot(self._end_point_diff, multipliers)
        penalty = 2 * np.dot( np.maximum(end_diff, 0), self._end_point_diff)
        fun = np.sum(self._moment_means * multipliers / self._moment_errs) + integral[0] * self._moment_errs[0]
        gradient =  self._moment_means / self._moment_errs - integral + np.abs(fun) * self._penalty_coef * penalty
        last_size = len(self._last_solved_multipliers)
        gradient[:last_size] += self._stab_penalty * (multipliers[:last_size] - self._last_solved_multipliers)
        return gradient

    def _calculate_jacobian_matrix(self, multipliers):
        """
        :return: jacobian matrix, symmetric, (n_moments, n_moments)
        """
        self._update_quadrature(multipliers)

        power = -np.dot(self._quad_moments, multipliers / self._moment_errs)
        power = np.minimum(np.maximum(power, -200), 200)
        q_density = np.exp(power)
        moment_outer = np.einsum('ki,kj->ijk', self._quad_moments, self._quad_moments)
        triu_idx = np.triu_indices(self.approx_size)
        triu_outer = moment_outer[triu_idx[0], triu_idx[1], :]
        q_jac = triu_outer * q_density
        integral = np.dot(q_jac, self._quad_weights)
        integral /= self._moment_errs[triu_idx[0]]
        integral /= self._moment_errs[triu_idx[1]]

        jacobian_matrix = np.empty(shape=(self.approx_size, self.approx_size))
        jacobian_matrix[triu_idx[0], triu_idx[1]] = integral
        jacobian_matrix[triu_idx[1], triu_idx[0]] = integral

        end_diff = np.dot(self._end_point_diff, multipliers)
        fun = np.sum(self._moment_means * multipliers / self._moment_errs) \
              + jacobian_matrix[0,0] * self._moment_errs[0]**2
        for side in [0,1]:
            if end_diff[side] > 0:
                penalty = 2 * np.outer(self._end_point_diff[side], self._end_point_diff[side])
                jacobian_matrix += np.abs(fun) * self._penalty_coef * penalty

        jacobian_matrix[np.diag_indices_from(jacobian_matrix)] += self._stab_penalty


        #e_vals = np.linalg.eigvalsh(jacobian_matrix)

        #print(multipliers)
        #print("jac spectra: ", e_vals[0], e_vals[-1], e_vals[-1]/e_vals[0])
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
    a, b = moments_fn.domain
    integral = np.zeros(moments_fn.size)
    for i in range(moments_fn.size):
        fn = lambda x, m = i: moments_fn.eval(m, x) * density(x)
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
