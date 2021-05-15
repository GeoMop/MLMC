import numpy as np
import scipy as sc
import scipy.integrate as integrate
import mlmc.moments
import mlmc.tool.plot

EXACT_QUAD_LIMIT = 1000
GAUSS_DEGREE = 70
HUBER_MU = 0.001


class SimpleDistribution:
    """
    Calculation of the distribution
    """

    def __init__(self, moments_obj, moment_data, domain=None, force_decay=(True, True), max_iter=30):
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

        self.functional_value = None

        # Approximation of moment values.
        if moment_data is not None:
            self.moment_means = moment_data[:, 0]
            self.moment_errs = np.sqrt(moment_data[:, 1])
        self.moment_errs[:] = 1

        # Approximation parameters. Lagrange multipliers for moment equations.
        self._multipliers = None
        # Number of basis functions to approximate the density.
        # In future can be smaller then number of provided approximative moments.
        self.approx_size = len(self.moment_means)

        assert moments_obj.size >= self.approx_size
        self.moments_fn = moments_obj

        # Degree of Gauss quad to use on every subinterval determined by adaptive quad.
        self._gauss_degree = GAUSS_DEGREE
        # Panalty coef for endpoint derivatives
        self._penalty_coef = 0#.001

        self._iter_moments = []
        self.max_iter = max_iter

        self.gradients = []
        self.cond_number = 0

    @property
    def multipliers(self):
        return self._multipliers

    @multipliers.setter
    def multipliers(self, multipliers):
        self._multipliers = multipliers

    def estimate_density_minimize(self, tol=1e-7, multipliers=None):
        """
        Optimize density estimation
        :param tol: Tolerance for the nonlinear system residual, after division by std errors for
        individual moment means, i.e.
        res = || (F_i - \mu_i) / \sigma_i ||_2
        :return: None
        """
        # Initialize domain, multipliers, ...
        self._initialize_params(self.approx_size, tol)
        max_it = self.max_iter

        if multipliers is not None:
            self.multipliers = multipliers

        #print("sefl multipliers ", self.multipliers)
        method = 'trust-exact'
        #method = 'L-BFGS-B'
        #method ='Newton-CG'
        #method = 'trust-ncg'

        result = sc.optimize.minimize(self._calculate_functional, self.multipliers, method=method,
                                      jac=self._calculate_gradient,
                                      hess=self._calculate_jacobian_matrix,
                                      options={'tol': tol, 'xtol': tol,
                                               'gtol': tol, 'disp': True, 'maxiter': max_it}
                                      # options={'disp': True, 'maxiter': max_it}
                                      )


        self.multipliers = result.x
        jac_norm = np.linalg.norm(result.jac)
        print("size: {} nits: {} tol: {:5.3g} res: {:5.3g} msg: {}".format(
           self.approx_size, result.nit, tol, jac_norm, result.message))

        jac = self._calculate_jacobian_matrix(self.multipliers)
        self.final_jac = jac

        result.eigvals = np.linalg.eigvalsh(jac)
        kappa = np.max(result.eigvals) / np.min(result.eigvals)
        self.cond_number = kappa
        result.solver_res = result.jac
        # Fix normalization
        moment_0, _ = self._calculate_exact_moment(self.multipliers, m=0, full_output=0)
        m0 = sc.integrate.quad(self.density, self.domain[0], self.domain[1], epsabs=self._quad_tolerance)[0]
        print("moment[0]: {} m0: {}".format(moment_0, m0))

        self.multipliers[0] += np.log(moment_0)

        #print("final multipliers ", self.multipliers)

        #m0 = sc.integrate.quad(self.density, self.domain[0], self.domain[1])[0]
        #moment_0, _ = self._calculate_exact_moment(self.multipliers, m=0, full_output=0)
        #print("moment[0]: {} m0: {}".format(moment_0, m0))

        if result.success or jac_norm < tol:
            result.success = True
        # Number of iterations
        result.nit = max(result.nit, 1)
        result.fun_norm = jac_norm

        return result

    def jacobian_spectrum(self):
        self._regularity_coef = 0.0
        jac = self._calculate_jacobian_matrix(self.multipliers)
        return np.linalg.eigvalsh(jac)

    def density(self, value):
        """
        :param value: float or np.array
        :param moments_fn: counting moments function
        :return: density for passed value
        """
        moms = self.eval_moments(value)
        power = -np.sum(moms * self.multipliers / self._moment_errs, axis=1)
        power = np.minimum(np.maximum(power, -200), 200)
        return np.exp(power)

    def density_log(self, value):
        return np.log(self.density(value))

    def density_exp(self, value):
        return self.density(np.log(value))

    def mult_mom_der(self, value, degree=1):
        moms = self.eval_moments_der(value, degree)
        return -np.sum(moms * self.multipliers, axis=1)

    def density_derivation(self, value):
        moms = self.eval_moments(value)
        power = -np.sum(moms * self.multipliers / self._moment_errs, axis=1)
        power = np.minimum(np.maximum(power, -200), 200)
        return np.exp(power) * np.sum(-self.multipliers * self.eval_moments_der(value))

    def density_second_derivation(self, value):
        moms = self.eval_moments(value)

        power = -np.sum(moms * self.multipliers / self._moment_errs, axis=1)
        power = np.minimum(np.maximum(power, -200), 200)
        return (np.exp(power) * np.sum(-self.multipliers * self.eval_moments_der(value, degree=2))) +\
               (np.exp(power) * np.sum(self.multipliers * moms)**2)


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

    def _initialize_params(self, size, tol=1e-10):
        """
        Initialize parameters for density estimation
        :return: None
        """
        assert self.domain is not None
        assert tol is not None
        self._quad_tolerance = 1e-10

        self._moment_errs = self.moment_errs

        # Start with uniform distribution
        self.multipliers = np.zeros(size)
        self.multipliers[0] = -np.log(1/(self.domain[1] - self.domain[0]))
        # Log to store error messages from quad, report only on conv. problem.
        self._quad_log = []

        # Evaluate endpoint derivatives of the moments.
        self._end_point_diff = self.end_point_derivatives()
        self._update_quadrature(self.multipliers, force=True)

    def set_quadrature(self, other_distr):
        self._quad_points = other_distr._quad_points
        self._quad_weights = other_distr._quad_weights
        self._quad_moments = other_distr._quad_moments
        self._quad_moments_diffs = other_distr._quad_moments_diffs
        self._quad_moments_2nd_der = other_distr._quad_moments_2nd_der
        self._fixed_quad = True

    def eval_moments(self, x):
        return self.moments_fn.eval_all(x, self.approx_size)

    def eval_moments_der(self, x, degree=1):
        return self.moments_fn.eval_all_der(x, self.approx_size, degree)

    def _calculate_exact_moment(self, multipliers, m=0, full_output=0):
        """
        Compute moment 'm' using adaptive quadrature to machine precision.
        :param multipliers:
        :param m:
        :param full_output:
        :return:
        """
        def integrand(x):
            moms = self.eval_moments(x)
            power = -np.sum(moms * multipliers / self._moment_errs, axis=1)
            power = np.minimum(np.maximum(power, -200), 200)

            if type(power).__name__ == 'ArrayBox':
                power = power._value
                if type(power).__name__ == 'ArrayBox':
                    power = power._value

            return np.exp(power) * moms[:, m]

        result = sc.integrate.quad(integrand, self.domain[0], self.domain[1],
                                   epsabs=self._quad_tolerance, full_output=full_output)

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
                return

            # More precise but depends on actual gradient which may not be available
            quad_err_estimate = np.abs(np.dot(self._last_gradient, (multipliers - self._last_multipliers)))
            if quad_err_estimate < self._quad_tolerance:
                return

        val, result = self._calculate_exact_moment(multipliers, m=self.approx_size-1, full_output=1)

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
        self._quad_moments_diffs = self.moments_fn.eval_diff(self._quad_points)
        self._quad_moments_2nd_der = self.eval_moments_der(self._quad_points, degree=2)
        self._quad_moments_3rd_der = self.eval_moments_der(self._quad_points, degree=3)

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
            left_diff = self.eval_moments(self.domain[0] + eps) - self.eval_moments(self.domain[0])
        if self.decay_penalty[1]:
            right_diff = -self.eval_moments(self.domain[1]) + self.eval_moments(self.domain[1] - eps)

        return np.stack((left_diff[0,:], right_diff[0,:]), axis=0)/eps/self._moment_errs[None, :]

    def _density_in_quads(self, multipliers):
        power = -np.dot(self._quad_moments, multipliers / self._moment_errs)
        power = np.minimum(np.maximum(power, -200), 200)
        return np.exp(power)

    def _calculate_functional(self, multipliers):
        """
        Minimized functional.
        :param multipliers: current multipliers
        :return: float
        """
        self.multipliers = multipliers
        self._update_quadrature(multipliers, True)

        self._iter_moments.append(self.moments_by_quadrature())

        q_density = self._density_in_quads(multipliers)
        integral = np.dot(q_density, self._quad_weights)
        sum = np.sum(self.moment_means * multipliers / self._moment_errs)
        fun = sum + integral

        if self._penalty_coef != 0:
            end_diff = np.dot(self._end_point_diff, multipliers)
            penalty = np.sum(np.maximum(end_diff, 0) ** 2)
            fun = fun + np.abs(fun) * self._penalty_coef * penalty

        self.functional_value = fun
        return fun

    def moments_by_quadrature(self, der=1):
        q_density = self._density_in_quads(self.multipliers)
        if der == 2:
            q_gradient = self._quad_moments_2nd_der.T * q_density
        else:
            q_gradient = self._quad_moments.T * q_density
        return np.dot(q_gradient, self._quad_weights) / self._moment_errs

    def _calculate_gradient(self, multipliers):
        """
        Gradient of th functional
        :return: array, shape (n_moments,)
        """
        self._update_quadrature(multipliers)
        q_density = self._density_in_quads(multipliers)
        q_gradient = self._quad_moments.T * q_density
        integral = np.dot(q_gradient, self._quad_weights) / self._moment_errs

        if self._penalty_coef != 0:
            end_diff = np.dot(self._end_point_diff, multipliers)
            penalty = 2 * np.dot(np.maximum(end_diff, 0), self._end_point_diff)
            fun = np.sum(self.moment_means * multipliers / self._moment_errs) + integral[0] * self._moment_errs[0]

            gradient = self.moment_means / self._moment_errs - integral + np.abs(fun) * self._penalty_coef * penalty
        else:

            gradient = self.moment_means / self._moment_errs - integral# + np.abs(fun) * self._penalty_coef * penalty
        self.gradients.append(gradient)

        return gradient

    def _calc_jac(self):
        q_density = self.density(self._quad_points)
        q_density_w = q_density * self._quad_weights

        jacobian_matrix = (self._quad_moments.T * q_density_w) @ self._quad_moments
        return jacobian_matrix

    def _calculate_jacobian_matrix(self, multipliers):
        """
        :return: jacobian matrix, symmetric, (n_moments, n_moments)
        """
        # jacobian_matrix_hess = hessian(self._calculate_functional)(multipliers)
        # print(pd.DataFrame(jacobian_matrix_hess))
        jacobian_matrix = self._calc_jac()

        if self._penalty_coef != 0:
            end_diff = np.dot(self._end_point_diff, multipliers)
            fun = np.sum(self.moment_means * multipliers / self._moment_errs) + jacobian_matrix[0, 0] * self._moment_errs[0] ** 2
            for side in [0, 1]:
                if end_diff[side] > 0:
                    penalty = 2 * np.outer(self._end_point_diff[side], self._end_point_diff[side])
                    jacobian_matrix += np.abs(fun) * self._penalty_coef * penalty

        return jacobian_matrix


def compute_exact_moments(moments_fn, density, tol=1e-10):
    """
    Compute approximation of moments using exact density.
    :param moments_fn: Moments function.
    :param density: Density function (must accept np vectors).
    :param tol: Tolerance of integration.
    :return: np.array, moment values
    """
    a, b = moments_fn.domain
    integral = np.zeros(moments_fn.size)

    for i in range(moments_fn.size):
        def fn(x):
             return moments_fn.eval(i, x) * density(x)

        integral[i] = integrate.quad(fn, a, b, epsabs=tol)[0]

    return integral


def compute_semiexact_moments(moments_fn, density, tol=1e-10):
    a, b = moments_fn.domain
    m = moments_fn.size - 1

    def integrand(x):
        moms = moments_fn.eval_all(x)[0, :]
        return density(x) * moms[m]

    result = sc.integrate.quad(integrand, a, b,
                               epsabs=tol, full_output=True)

    if len(result) > 3:
        y, abserr, info, message = result
    else:
        y, abserr, info = result
    pt, w = np.polynomial.legendre.leggauss(GAUSS_DEGREE)
    K = info['last']
    # print("Update Quad: {} {} {} {}".format(K, y, abserr, message))
    a = info['alist'][:K, None]
    b = info['blist'][:K, None]
    points = (pt[None, :] + 1) / 2 * (b - a) + a
    weights = w[None, :] * (b - a) / 2
    quad_points = points.flatten()
    quad_weights = weights.flatten()
    quad_moments = moments_fn.eval_all(quad_points)
    q_density = density(quad_points)
    q_density_w = q_density * quad_weights

    moments = q_density_w @ quad_moments
    return moments


def compute_exact_cov(moments_fn, density, tol=1e-10, domain=None):
    """
    Compute approximation of covariance matrix using exact density.
    :param moments_fn: Moments function.
    :param density: Density function (must accept np vectors).
    :param tol: Tolerance of integration.
    :return: np.array, moment values
    """
    a, b = moments_fn.domain
    if domain is not None:
        a_2, b_2 = domain
    else:
        a_2, b_2 = a, b

    integral = np.zeros((moments_fn.size, moments_fn.size))
    print("a_2: {}, b_2: {}".format(a_2, b_2))

    for i in range(moments_fn.size):
        for j in range(i+1):
            def fn(x):
                moments = moments_fn.eval_all(x)[0, :]
                density_value = density(x)
                return moments[i] * moments[j] * density_value # * density(x)

            integral[j][i] = integral[i][j] = integrate.quad(fn, a, b, epsabs=tol)[0]
    return integral


def compute_semiexact_cov(moments_fn, density, tol=1e-10):
    """
    Compute approximation of covariance matrix using exact density.
    :param moments_fn: Moments function.
    :param density: Density function (must accept np vectors).
    :param tol: Tolerance of integration.
    :return: np.array, moment values
    """
    a, b = moments_fn.domain
    m = moments_fn.size - 1

    def integrand(x):
        moms = moments_fn.eval_all(x)[0, :]
        return density(x) * moms[m] * moms[m]

    result = sc.integrate.quad(integrand, a, b, epsabs=tol, full_output=True)

    if len(result) > 3:
        y, abserr, info, message = result
    else:
        y, abserr, info = result
    # Computes the sample points and weights for Gauss-Legendre quadrature
    pt, w = np.polynomial.legendre.leggauss(GAUSS_DEGREE)
    K = info['last']
    # print("Update Quad: {} {} {} {}".format(K, y, abserr, message))
    a = info['alist'][:K, None]
    b = info['blist'][:K, None]

    points = (pt[None, :] + 1) / 2 * (b - a) + a
    weights = w[None, :] * (b - a) / 2

    quad_points = points.flatten()
    quad_weights = weights.flatten()
    quad_moments = moments_fn.eval_all(quad_points)
    q_density = density(quad_points)
    q_density_w = q_density * quad_weights
    jacobian_matrix = (quad_moments.T * q_density_w) @ quad_moments

    return jacobian_matrix


def KL_divergence_2(prior_density, posterior_density, a, b):
    def integrand(x):
        # prior
        p = prior_density(x)
        # posterior
        q = max(posterior_density(x), 1e-300)
        # modified integrand to provide positive value even in the case of imperfect normalization
        return p * np.log(p / q)

    value = integrate.quad(integrand, a, b)#, epsabs=1e-10)

    return value[0]


def KL_divergence(prior_density, posterior_density, a, b):
    """
    Compute D_KL(P | Q) = \int_R P(x) \log( P(X)/Q(x)) \dx
    :param prior_density: P
    :param posterior_density: Q
    :return: KL divergence value
    """
    def integrand(x):
        # prior
        p = max(prior_density(x), 1e-300)
        # posterior
        q = max(posterior_density(x), 1e-300)
        # modified integrand to provide positive value even in the case of imperfect normalization
        return p * np.log(p / q) - p + q

    value = integrate.quad(integrand, a, b)#, epsabs=1e-10)

    return value[0]


def KL_divergence_log(prior_density, posterior_density, a, b):
    """
    Compute D_KL(P | Q) = \int_R P(x) \log( P(X)/Q(x)) \dx
    :param prior_density: P
    :param posterior_density: Q
    :return: KL divergence value
    """
    lim = 0.01
    def integrand(x):
        x = np.exp(x)
        x_div = x
        if x_div < lim:
            x_div = 1/x_div

        # prior
        p = prior_density(x)/x_div
        # posterior
        q = max(posterior_density(x)/x_div, 1e-300)
        # modified integrand to provide positive value even in the case of imperfect normalization
        return p * np.log(p / q) - p + q

    value = integrate.quad(integrand, a, b)#, epsabs=1e-10)

    return value[0]


def L2_distance(prior_density, posterior_density, a, b):
    """
    L2 norm
    :param prior_density:
    :param posterior_density:
    :param a:
    :param b:
    :return:
    """
    integrand = lambda x: (posterior_density(x) - prior_density(x)) ** 2
    return np.sqrt(integrate.quad(integrand, a, b))[0]


def total_variation_int(func, a, b):
    def integrand(x):
        return huber_l1_norm(func, x)
    return integrate.quad(integrand, a, b)[0]


def huber_l1_norm(func, x):
    r = func(x)

    mu = HUBER_MU
    y = mu * (np.sqrt(1+(r**2/mu**2)) - 1)

    return y

def best_fit_all(values, range_a, range_b):
    best_fit = None
    best_fit_value = np.inf
    for a in range_a:
        for b in range_b:
            if 0 <= a and  a + 2 < b < len(values):

                Y = values[a:b]

                X = np.arange(a, b)
                assert len(X) == len(Y), "a:{}  b:{}".format(a,b)
                fit, res, _, _, _ = np.polyfit(X, Y, deg=1, full=1)

                fit_value = res / ((b - a)**2)
                if fit_value < best_fit_value:
                    best_fit = (a, b, fit)
                    best_fit_value = fit_value
    return best_fit


def best_p1_fit(values):
    """
    Find indices a < b such that linear fit for values[a:b]
    have smallest residual / (b - a)** alpha
    alpha is fixed parameter.
    This should find longest fit with reasonably small residual.
    :return: (a, b)
    """
    if len(values) > 12:
        # downscale
        end = len(values)  - len(values) % 2    # even size of result
        avg_vals = np.mean(values[:end].reshape((-1, 2)), axis=1)
        a, b, fit = best_p1_fit(avg_vals)
        # upscale
        a, b = 2*a, 2*b

        return best_fit_all(values, [a-1, a, a+1], [b-1, b, b+1])
    else:
        v_range = range(len(values))
        return best_fit_all(values, v_range, v_range)


def detect_treshold_slope_change(values, log=True):
    """
    Find a longest subsequence with linear fit residual X% higher then the best
    at least 4 point fit. Extrapolate this fit to the left.

    :param values: Increassing sequence.
    :param log: Use logarithm of the sequence.
    :return: Index K for which K: should have same slope.
    """
    values = np.array(values)
    i_first_positive = 0
    if log:
        i_first_positive = np.argmax(values > 0)
        values[i_first_positive:] = np.log(values[i_first_positive:])

    a, b, fit = best_p1_fit(values[i_first_positive:])
    p = np.poly1d(fit)

    i_treshold = a + i_first_positive
    mod_vals = values.copy()
    mod_vals[:i_treshold] = p(np.arange(-i_first_positive, a))
    #self.plot_values(values, val2=mod_vals, treshold=i_treshold)
    if log:
        mod_vals = np.exp(mod_vals)
    return i_treshold, mod_vals

def _add_to_eigenvalues(cov_center, tol, moments):
    eval, evec = np.linalg.eigh(cov_center)

    # we need highest eigenvalues first
    eval = np.flip(eval, axis=0)
    evec = np.flip(evec, axis=1)

    original_eval = eval
    diag_value = tol - np.min([np.min(eval), 0])
    diagonal = np.zeros(moments.size)
    diagonal[1:] += diag_value
    eval += diagonal

    return eval, evec, original_eval


def _cut_eigenvalues(cov_center, tol):
    eval, evec = np.linalg.eigh(cov_center)
    original_eval = eval
    threshold = 0

    if tol is None:
        if len(eval) > 1:
            aux_eval = np.flip(eval)
            # treshold by statistical test of same slopes of linear models
            threshold, fixed_eval = detect_treshold_slope_change(aux_eval, log=True)
            threshold = np.argmax(aux_eval - fixed_eval[0] > 0)
            threshold = len(eval) - threshold
    else:
        # threshold given by eigenvalue magnitude
        threshold = np.argmax(eval > tol)

    # cut eigen values under treshold
    new_eval = eval[threshold:]
    new_evec = evec[:, threshold:]

    eval = np.flip(new_eval, axis=0)
    evec = np.flip(new_evec, axis=1)

    return eval, evec, threshold, original_eval


def construct_orthogonal_moments(moments, cov, tol=None, orth_method=2, exact_cov=None):
    """
    For given moments find the basis orthogonal with respect to the covariance matrix, estimated from samples.
    :param moments: moments object
    :return: orthogonal moments object of the same size.
    """
    # centered covariance
    M = np.eye(moments.size)
    M[:, 0] = -cov[:, 0]
    cov_center = M @ cov @ M.T

    # Add const to eigenvalues
    if orth_method == 1:
        threshold = 0
        eval_flipped, evec_flipped, original_eval = _add_to_eigenvalues(cov_center, tol=tol, moments=moments)
    # Cut eigenvalues below threshold
    elif orth_method == 2:
        eval_flipped, evec_flipped, threshold, original_eval = _cut_eigenvalues(cov_center, tol=tol)

    icov_sqrt_t = M.T @ (evec_flipped * (1 / np.sqrt(eval_flipped))[None, :])
    R_nm, Q_mm = sc.linalg.rq(icov_sqrt_t, mode='full')

    # check
    L_mn = R_nm.T
    if L_mn[0, 0] < 0:
        L_mn = -L_mn

    ortogonal_moments = mlmc.moments.TransformedMoments(moments, L_mn)
    info = (original_eval, eval_flipped, threshold, L_mn)
    return ortogonal_moments, info, cov_center


