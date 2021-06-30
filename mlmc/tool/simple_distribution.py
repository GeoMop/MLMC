import numpy as np
import scipy as sc
import scipy.integrate as integrate
import mlmc.moments
import mlmc.plot.plots

EXACT_QUAD_LIMIT = 1000

class SimpleDistribution:
    """
    Calculation of the distribution
    """

    def __init__(self, moments_obj, moment_data, domain=None, force_decay=(True, True), verbose=False):
        """
        :param moments_obj: Function for calculating moments
        :param moment_data: Array  of moments and their vars; (n_moments, 2)
        :param domain: Explicit domain fo reconstruction. None = use domain of moments.
        :param force_decay: Flag for each domain side to enforce decay of the PDF approximation.
        """
        # Moment evaluation function with bounded number of moments and their domain.
        self.moments_fn = None

        # Domain of the density approximation (and moment functions).
        if domain is None:
            domain = moments_obj.domain
        self.domain = domain
        # Indicates whether force decay of PDF at domain endpoints.
        self.decay_penalty = force_decay
        self._verbose = verbose

        # Approximation of moment values.
        if moment_data is not None:
            self.moment_means = moment_data[:, 0]
            self.moment_errs = np.sqrt(moment_data[:, 1])

        # Approximation parameters. Lagrange multipliers for moment equations.
        self.multipliers = None
        # Number of basis functions to approximate the density.
        # In future can be smaller then number of provided approximative moments.
        self.approx_size = len(self.moment_means)
        assert moments_obj.size >= self.approx_size
        self.moments_fn = moments_obj

        # Degree of Gauss quad to use on every subinterval determined by adaptive quad.
        self._gauss_degree = 21
        # Panalty coef for endpoint derivatives
        self._penalty_coef = 0

    def estimate_density_minimize(self, tol=1e-5, reg_param =0.01):
        """
        Optimize density estimation
        :param tol: Tolerance for the nonlinear system residual, after division by std errors for
        individual moment means, i.e.
        res = || (F_i - \mu_i) / \sigma_i ||_2
        :return: None
        """
        # Initialize domain, multipliers, ...

        self._initialize_params(self.approx_size, tol)
        max_it = 20
        #method = 'trust-exact'
        #method ='Newton-CG'
        method = 'trust-ncg'
        result = sc.optimize.minimize(self._calculate_functional, self.multipliers, method=method,
                                      jac=self._calculate_gradient,
                                      hess=self._calculate_jacobian_matrix,
                                      options={'tol': tol, 'xtol': tol,
                                               'gtol': tol, 'disp': False,  'maxiter': max_it})
        self.multipliers = result.x
        jac_norm = np.linalg.norm(result.jac)
        if self._verbose:
            print("size: {} nits: {} tol: {:5.3g} res: {:5.3g} msg: {}".format(
               self.approx_size, result.nit, tol, jac_norm, result.message))

        jac = self._calculate_jacobian_matrix(self.multipliers)
        result.eigvals = np.linalg.eigvalsh(jac)
        #result.residual = jac[0] * self._moment_errs
        #result.residual[0] *= self._moment_errs[0]
        result.solver_res = result.jac
        # Fix normalization
        moment_0, _ = self._calculate_exact_moment(self.multipliers, m=0, full_output=0)
        m0 = sc.integrate.quad(self.density, self.domain[0], self.domain[1])[0]
        if self._verbose:
            print("moment[0]: {} m0: {}".format(moment_0, m0))
        self.multipliers[0] -= np.log(moment_0)

        if result.success or jac_norm < tol:
            result.success = True
        # Number of iterations
        result.nit = max(result.nit, 1)
        result.fun_norm = jac_norm

        return result

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
        #self._quad_tolerance = tol / 1024
        self._quad_tolerance = 1e-10

        #self.moment_errs[np.where(self.moment_errs == 0)] = np.min(self.moment_errs[np.where(self.moment_errs != 0)]/8)
        #self.moment_errs[0] = np.min(self.moment_errs[1:]) / 8

        self._moment_errs = self.moment_errs
        #self._moment_errs[0] = np.min(self.moment_errs[1:]) / 2

        # Start with uniform distribution
        self.multipliers = np.zeros(size)
        self.multipliers[0] = -np.log(1/(self.domain[1] - self.domain[0]))
        # Log to store error messages from quad, report only on conv. problem.
        self._quad_log = []

        # Evaluate endpoint derivatives of the moments.
        self._end_point_diff = self.end_point_derivatives()
        self._update_quadrature(self.multipliers, force=True)

    def eval_moments(self, x):
        return self.moments_fn.eval_all(x, self.approx_size)

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
            return np.exp(power) * moms[:, m]

        result = sc.integrate.quad(integrand, self.domain[0], self.domain[1],
                                   epsabs=self._quad_tolerance, full_output=full_output)

        return result[0], result

    # def _calculate_exact_hessian(self, i, j, multipliers=None):
    #     """
    #     Compute exact jacobian element (i,j).
    #     :param i:
    #     :param j:
    #     :param multipliers:
    #     :return:
    #     """
    #     if multipliers is None:
    #         multipliers = self.multipliers
    #
    #     def integrand(x):
    #         moms = self.eval_moments(x)
    #         power = -np.sum(moms * multipliers / self._moment_errs, axis=1)
    #         power = np.minimum(np.maximum(power, -200), 200)
    #         return np.exp(power) * moms[:,i] * moms[:,j]
    #
    #     result = sc.integrate.quad(integrand, self.domain[0], self.domain[1],
    #                                epsabs=self._quad_tolerance, full_output=False)
    #
    #     return result[0], result

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
        self._update_quadrature(multipliers)
        q_density = self._density_in_quads(multipliers)
        integral = np.dot(q_density, self._quad_weights)
        sum = np.sum(self.moment_means * multipliers / self._moment_errs)

        end_diff = np.dot(self._end_point_diff, multipliers)
        penalty = np.sum(np.maximum(end_diff, 0)**2)
        fun = sum + integral
        fun = fun + np.abs(fun) * self._penalty_coef * penalty

        return fun

    def _calculate_gradient(self, multipliers):
        """
        Gradient of th functional
        :return: array, shape (n_moments,)
        """
        self._update_quadrature(multipliers)
        q_density = self._density_in_quads(multipliers)
        q_gradient = self._quad_moments.T * q_density
        integral = np.dot(q_gradient, self._quad_weights) / self._moment_errs

        end_diff = np.dot(self._end_point_diff, multipliers)
        penalty = 2 * np.dot( np.maximum(end_diff, 0), self._end_point_diff)
        fun = np.sum(self.moment_means * multipliers / self._moment_errs) + integral[0] * self._moment_errs[0]
        gradient = self.moment_means / self._moment_errs - integral + np.abs(fun) * self._penalty_coef * penalty
        return gradient

    def _calculate_jacobian_matrix(self, multipliers):
        """
        :return: jacobian matrix, symmetric, (n_moments, n_moments)
        """
        self._update_quadrature(multipliers)
        q_density = self._density_in_quads(multipliers)
        q_density_w = q_density * self._quad_weights
        q_mom = self._quad_moments / self._moment_errs

        jacobian_matrix = (q_mom.T * q_density_w) @ q_mom

        # Compute just triangle use lot of memory (possibly faster)
        # moment_outer = np.einsum('ki,kj->ijk', q_mom, q_mom)
        # triu_idx = np.triu_indices(self.approx_size)
        # triu_outer = moment_outer[triu_idx[0], triu_idx[1], :]
        # integral = np.dot(triu_outer, q_density_w)
        # jacobian_matrix = np.empty(shape=(self.approx_size, self.approx_size))
        # jacobian_matrix[triu_idx[0], triu_idx[1]] = integral
        # jacobian_matrix[triu_idx[1], triu_idx[0]] = integral

        end_diff = np.dot(self._end_point_diff, multipliers)
        fun = np.sum(self.moment_means * multipliers / self._moment_errs) + jacobian_matrix[0,0] * self._moment_errs[0]**2
        for side in [0, 1]:
            if end_diff[side] > 0:
                penalty = 2 * np.outer(self._end_point_diff[side], self._end_point_diff[side])
                jacobian_matrix += np.abs(fun) * self._penalty_coef * penalty


        #e_vals = np.linalg.eigvalsh(jacobian_matrix)

        #print(multipliers)
        #print("jac spectra: ", e_vals)
        #print("means:", self.moment_means)
        #print("\n jac:", np.diag(jacobian_matrix))
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
    pt, w = np.polynomial.legendre.leggauss(21)
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


def compute_exact_cov(moments_fn, density, tol=1e-10):
    """
    Compute approximation of covariance matrix using exact density.
    :param moments_fn: Moments function.
    :param density: Density function (must accept np vectors).
    :param tol: Tolerance of integration.
    :return: np.array, moment values
    """
    a, b = moments_fn.domain
    integral = np.zeros((moments_fn.size, moments_fn.size))

    for i in range(moments_fn.size):
        for j in range(i+1):
            def fn(x):
                moments = moments_fn.eval_all(x)[0, :]
                return (moments[i] * moments[j]) * density(x)
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

    result = sc.integrate.quad(integrand, a, b,
                               epsabs=tol, full_output=True)

    if len(result) > 3:
        y, abserr, info, message = result
    else:
        y, abserr, info = result
    pt, w = np.polynomial.legendre.leggauss(21)
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

    return jacobian_matrix


def KL_divergence(prior_density, posterior_density, a, b):
    """
    Compute D_KL(P | Q) = \int_R P(x) \log( P(X)/Q(x)) \dx
    :param prior_density: P
    :param posterior_density: Q
    :return: KL divergence value
    """
    def integrand(x):
        # prior
        p = prior_density(x)
        # posterior
        q = max(posterior_density(x), 1e-300)
        # modified integrand to provide positive value even in the case of imperfect normalization
        return  p * np.log(p / q) - p + q

    value = integrate.quad(integrand, a, b, epsabs=1e-10)
    return max(value[0], 1e-10)


def L2_distance(prior_density, posterior_density, a, b):
    integrand = lambda x: (posterior_density(x) - prior_density(x)) ** 2
    return np.sqrt(integrate.quad(integrand, a, b))[0]









######################################



# def detect_treshold(self, values, log=True, window=4):
#     """
#     Detect most significant change of slope in the sorted sequence.
#     Negative values are omitted for log==True.
#
#     Notes: not work well since the slope difference is weighted by residuum so for
#     points nearly perfectly in line even small changes of slope can be detected.
#     :param values: Increassing sequence.
#     :param log: Use logarithm of the sequence.
#     :return: Index K for which K: should have same slope.
#     """
#     values = np.array(values)
#     orig_len = len(values)
#     if log:
#         min_positive = np.min(values[values>0])
#         values = np.maximum(values, min_positive)
#         values = np.log(values)
#
#     # fit model for all valid window positions
#     X = np.empty((window, 2))
#     X[:, 0] = np.ones(window)
#     X[:, 1] = np.flip(np.arange(window))
#     fit_matrix = np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T)
#     intercept = np.convolve(values, fit_matrix[0], mode='valid')
#     assert len(intercept) == len(values) - window + 1
#     slope = np.convolve(values, fit_matrix[1], mode='valid')
#     fits = np.stack( (intercept, slope) ).T
#
#     # We test hypothesis of equality of slopes from two non-overlapping windows.
#     # https://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/equalslo.htm
#     # https://ncss-wpengine.netdna-ssl.com/wp-content/themes/ncss/pdf/Procedures/PASS/Tests_for_the_Difference_Between_Two_Linear_Regression_Slopes.pdf
#     # Dupont and Plummer (1998)
#
#     df = 2 * window - 4
#     varX = np.var(np.arange(window)) * window
#     p_vals = np.ones_like(values)
#     for i, _ in enumerate(values):
#         ia = i - window + 1
#         ib = i
#         if ia < 0 or ib + window >= len(values):
#             p_vals[i] = 1.0
#             continue
#         res_a = values[ia:ia + window] - np.flip(np.dot(X, fits[ia]))
#         res_b = values[ib:ib + window] - np.flip(np.dot(X, fits[ib]))
#
#         varY = (np.sum(res_a**2) + np.sum(res_b**2)) / df
#         SS_r = varY * 2 / (window * varX)
#         T = (fits[ia, 1] -  fits[ib, 1]) / np.sqrt(SS_r)
#         # Single tail alternative: slope_a < slope_b
#         p_vals[i] = 1 - stats.t.cdf(T, df=df)
#         print(ia, ib, np.sqrt(SS_r), fits[ia, 1], fits[ib, 1], p_vals[i])
#
#
#     i_min = np.argmin(p_vals)
#     i_treshold = i_min + window + orig_len - len(values) - 1
#
#     self.plot_values(values, val2=p_vals, treshold=i_treshold)
#     return i_treshold, p_vals[i_min]


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
                #print("a b fit", a, b, fit_value)
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


# def detect_treshold_lm(self, values, log=True, window=4):
#     """
#     Detect most significant change of slope in the sorted sequence.
#     Negative values are omitted for log==True.
#
#     Just build a linear model for increasing number of values and find
#     the first one that do not fit significantly.
#
#     :param values: Increassing sequence.
#     :param log: Use logarithm of the sequence.
#     :return: Index K for which K: should have same slope.
#     """
#
#     values = np.array(values)
#     orig_len = len(values)
#     if log:
#         min_positive = np.min(values[values>0])
#         values = np.maximum(values, min_positive)
#         values = np.log(values)
#     values = np.flip(values)
#     i_break = 0
#     for i in range(2, len(values)):
#         # fit the mode
#         X = np.empty((i, 2))
#         X[:, 0] = np.ones(i)
#         X[:, 1] = np.arange(i)
#         fit_matrix = np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T)
#         Y = values[:i]
#         fit = np.dot(fit_matrix, Y)
#         i_val_model = fit[0] + fit[1]*i
#         diff =  i_val_model - values[i]
#         Y_model = np.matmul(X, fit)
#         if i > 3:
#             sigma = np.sqrt(np.sum((Y - Y_model)**2) / (i - 2))
#         else:
#             sigma = -fit[1]
#         #print(i, diff, fit[1], sigma)
#         if diff > 3*sigma and i_break == 0:
#             #print("break: ", i)
#             i_break = i
#     if i_break > 0:
#         i_break = len(values) - i_break
#     return i_break
#     #return i_treshold, p_vals[i_min]
#
# def optimal_n_moments(self):
#     """
#     Iteratively decrease number of used moments until no eigne values need to be removed.
#     :return:
#     """
#     reduced_moments = self.moments
#     i_eig_treshold = 1
#     while reduced_moments.size > 6 and i_eig_treshold > 0:
#
#         moments = reduced_moments
#         cov = self._covariance = self.mlmc.estimate_covariance(moments)
#
#         # centered covarince
#         M = np.eye(moments.size)
#         M[:, 0] = -cov[:, 0]
#         cov_center = M @ cov @ M.T
#         eval, evec = np.linalg.eigh(cov_center)
#         i_first_positive = np.argmax(eval > 0)
#         pos_eval = eval[i_first_positive:]
#         treshold = self.detect_treshold_lm(pos_eval)
#         i_eig_treshold = i_first_positive + treshold
#         #self.plot_values(pos_eval, log=True, treshold=treshold)
#
#         reduced_moments = moments.change_size(moments.size - i_eig_treshold)
#         print("mm: ", i_eig_treshold, " s: ", reduced_moments.size)
#
#     # Possibly cut remaining negative eigen values
#     i_first_positive = np.argmax(eval > 0)
#     eval = eval[i_first_positive:]
#     evec = evec[:, i_first_positive:]
#     eval = np.flip(eval)
#     evec = np.flip(evec, axis=1)
#     L = -(1/np.sqrt(eval))[:, None] * (evec.T @ M)
#     natural_moments = mlmc.moments.TransformedMoments(moments, L)
#
#     return natural_moments
#
#
# def detect_treshold_mse(self, eval, std_evals):
#     """
#     Detect treshold of eigen values by its estimation error:
#     1. eval, evec decomposition
#     2. rotated moments using just evec as the rotation matrix
#     3. compute covariance for rotated moments with errors, use errors of diagonal entries
#        as errors of eigenvalue estimate.
#     4. Set treshold to the last eigenvalue with relative error larger then 0.3
#
#     Notes: Significant errors occures also for correct eigen values, so this is not good treshold detection.
#
#     :param eval:
#     :param std_evals:
#     :return:
#     """
#     i_first_positive = np.argmax(eval > 0)
#     rel_err = std_evals[i_first_positive:] / eval[i_first_positive:]
#     rel_tol = 0.3
#     large_rel_err = np.nonzero(rel_err > rel_tol)[0]
#     treshold = large_rel_err[-1] if len(large_rel_err) > 0 else 0
#     return i_first_positive + treshold

# def eigenvalue_error(moments):
#     rot_cov, var_evals = self._covariance = self.mlmc.estimate_covariance(moments, mse=True)
#     var_evals = np.flip(var_evals)
#     var_evals[var_evals < 0] = np.max(var_evals)
#     std_evals = np.sqrt(var_evals)
#     return std_evals


def lsq_reconstruct(cov, eval, evec, treshold):
    #eval = np.flip(eval)
    #evec = np.flip(evec, axis=1)

    Q1 = evec[:, :treshold]
    Q20 = evec[:, treshold:]
    C = cov
    D = np.diag(eval)
    q_shape = Q20.shape
    I = np.eye(q_shape[0])

    def fun(x):
        alpha_orto = 2
        Q2 = x.reshape(q_shape)
        Q = np.concatenate( (Q1, Q2), axis=1)
        f = np.sum(np.abs(np.ravel(Q.T @ C @ Q - D))) + alpha_orto * np.sum(np.abs(np.ravel(Q @ Q.T - I)))
        return f

    result = sc.optimize.least_squares(fun, np.ravel(Q20))
    print("LSQ res: ", result.nfev, result.njev, result.cost)
    Q2 = result.x.reshape(q_shape)
    Q = np.concatenate((Q1, Q2), axis=1)

    print("D err", D - Q.T @ cov @ Q)
    print("D", D)
    print("QcovQT",  Q.T @ cov @ Q)
    print("I err:", I - Q @ Q.T)
    print("Q err:", Q20 - Q2)

    return Q

def construct_ortogonal_moments(moments, cov, tol=None):
    """
    For given moments find the basis orthogonal with respect to the covariance matrix, estimated from samples.
    :param moments: moments object
    :return: orthogonal moments object of the same size.
    """

    # centered covariance
    M = np.eye(moments.size)
    M[:, 0] = -cov[:, 0]
    cov_center = M @ cov @ M.T
    #cov_center = cov
    eval, evec = np.linalg.eigh(cov_center)
    # eval is in increasing order


    # Compute eigen value errors.
    #evec_flipped = np.flip(evec, axis=1)
    #L = (evec_flipped.T @ M)
    #rot_moments = mlmc.moments.TransformedMoments(moments, L)
    #std_evals = eigenvalue_error(rot_moments)


    if tol is None:
        # treshold by statistical test of same slopes of linear models
        threshold, fixed_eval = detect_treshold_slope_change(eval, log=True)
        threshold = np.argmax( eval - fixed_eval[0] > 0)
    else:
        # threshold given by eigenvalue magnitude
        threshold = np.argmax(eval > tol)

    #treshold, _ = self.detect_treshold(eval, log=True, window=8)

    # tresold by MSE of eigenvalues
    #treshold = self.detect_treshold_mse(eval, std_evals)

    # treshold


    #self.lsq_reconstruct(cov_center, fixed_eval, evec, treshold)

    #use fixed
    #eval[:treshold] = fixed_eval[:treshold]


    # set eig. values under the treshold to the treshold
    #eval[:treshold] = eval[treshold]

    # cut eigen values under treshold
    new_eval = eval[threshold:]
    new_evec = evec[:, threshold:]

    # we need highest eigenvalues first
    eval_flipped = np.flip(new_eval, axis=0)
    evec_flipped = np.flip(new_evec, axis=1)
    #conv_sqrt = -M.T @ evec_flipped * (1 / np.sqrt(eval_flipped))[:, None]
    #icov_sqrt_t = -M.T @ evec_flipped * (1/np.sqrt(eval_flipped))[None, :]
    icov_sqrt_t = M.T @ evec_flipped * (1 / np.sqrt(eval_flipped))[None, :]
    R_nm, Q_mm  = sc.linalg.rq(icov_sqrt_t, mode='full')
    # check
    L_mn = R_nm.T
    if L_mn[0, 0] < 0:
        L_mn = -L_mn


    ortogonal_moments = mlmc.moments.TransformedMoments(moments, L_mn)
    #ortogonal_moments = mlmc.moments.TransformedMoments(moments, cov_sqrt_t.T)

    #################################
    # cov = self.mlmc.estimate_covariance(ortogonal_moments)
    # M = np.eye(ortogonal_moments.size)
    # M[:, 0] = -cov[:, 0]
    # cov_center = M @ cov @ M.T
    # eval, evec = np.linalg.eigh(cov_center)
    #
    # # Compute eigen value errors.
    # evec_flipped = np.flip(evec, axis=1)
    # L = (evec_flipped.T @ M)
    # rot_moments = mlmc.moments.TransformedMoments(moments, L)
    # std_evals = self.eigenvalue_error(rot_moments)
    #
    # self.plot_values(eval, log=True, treshold=treshold)


    info = (eval, threshold, L_mn)
    return ortogonal_moments, info


# def construct_density(self, tol=1.95, reg_param=0.01):
#     """
#     Construct approximation of the density using given moment functions.
#     Args:
#         moments_fn: Moments object, determines also domain and n_moments.
#         tol: Tolerance of the fitting problem, with account for variances in moments.
#              Default value 1.95 corresponds to the two tail confidency 0.95.
#         reg_param: Regularization parameter.
#     """
#     moments_obj = self.construct_ortogonal_moments()
#     print("n levels: ", self.n_levels)
#     #est_moments, est_vars = self.mlmc.estimate_moments(moments)
#     est_moments = np.zeros(moments.size)
#     est_moments[0] = 1.0
#     est_vars = np.ones(moments.size)
#     min_var, max_var = np.min(est_vars[1:]), np.max(est_vars[1:])
#     print("min_err: {} max_err: {} ratio: {}".format(min_var, max_var, max_var / min_var))
#     moments_data = np.stack((est_moments, est_vars), axis=1)
#     distr_obj = SimpleDistribution(moments_obj, moments_data, domain=moments_obj.domain)
#     distr_obj.estimate_density_minimize(tol, reg_param)  # 0.95 two side quantile
#     self._distribution = distr_obj
#
#     # # [print("integral density ", integrate.simps(densities[index], x[index])) for index, density in
#     # # enumerate(densities)]
#     # moments_fn = self.moments
#     # domain = moments_fn.domain
#     #
#     # #self.mlmc.update_moments(moments_fn)
#     # cov = self._covariance = self.mlmc.estimate_covariance(moments_fn)
#     #
#     # # centered covarince
#     # M = np.eye(self.n_moments)
#     # M[:,0] = -cov[:,0]
#     # cov_center = M @ cov @ M.T
#     # #print(cov_center)
#     #
#     # eval, evec = np.linalg.eigh(cov_center)
#     # #self.plot_values(eval[:-1], log=False)
#     # #self.plot_values(np.maximum(np.abs(eval), 1e-30), log=True)
#     # #print("eval: ", eval)
#     # #min_pos = np.min(np.abs(eval))
#     # #assert min_pos > 0
#     # #eval = np.maximum(eval, 1e-30)
#     #
#     # i_first_positive = np.argmax(eval > 0)
#     # pos_eval = eval[i_first_positive:]
#     # pos_evec = evec[:, i_first_positive:]
#     #
#     # treshold = self.detect_treshold_lm(pos_eval)
#     # print("ipos: ", i_first_positive, "Treshold: ", treshold)
#     # self.plot_values(pos_eval, log=True, treshold=treshold)
#     # eval_reduced = pos_eval[treshold:]
#     # evec_reduced = pos_evec[:, treshold:]
#     # eval_reduced = np.flip(eval_reduced)
#     # evec_reduced = np.flip(evec_reduced, axis=1)
#     # print(eval_reduced)
#     # #eval[eval<0] = 0
#     # #print(eval)
#     #
#     #
#     # #opt_n_moments =
#     # #evec_reduced = evec
#     # # with reduced eigen vector matrix: P = n x m , n < m
#     # # \sqrt(Lambda) P^T = Q_1 R
#     # #SSV =  evec_reduced * (1/np.sqrt(eval_reduced))[None, :]
#     # #r, q = sc.linalg.rq(SSV)
#     # #Linv = r.T
#     # #Linv = Linv / Linv[0,0]
#     #
#     # #self.plot_values(np.maximum(eval, 1e-30), log=True)
#     # #print( np.matmul(evec, eval[:, None] * evec.T) - cov)
#     # #u,s,v = np.linalg.svd(cov, compute_uv=True)
#     # #print("S: ", s)
#     # #print(u - v.T)
#     # #L = np.linalg.cholesky(self._covariance)
#     # #L = sc.linalg.cholesky(cov, lower=True)
#     # #SSV = np.sqrt(s)[:, None] * v[:, :]
#     # #q, r = np.linalg.qr(SSV)
#     # #L = r.T
#     # #Linv = np.linalg.inv(L)
#     # #LCL = np.matmul(np.matmul(Linv, cov), Linv.T)
#     #
#     # L = -(1/np.sqrt(eval_reduced))[:, None] * (evec_reduced.T @ M)
#     # p_evec = evec.copy()
#     # #p_evec[:, :i_first_positive] = 0
#     # #L = evec.T @ M
#     # #L = M
#     # natural_moments = mlmc.moments.TransformedMoments(moments_fn, L)
#     # #self.plot_moment_functions(natural_moments, fig_file='natural_moments.pdf')
#     #
#     # # t_var = 1e-5
#     # # ref_diff_vars, _ = mlmc.estimate_diff_vars(moments_fn)
#     # # ref_moments, ref_vars = mc.estimate_moments(moments_fn)
#     # # ref_std = np.sqrt(ref_vars)
#     # # ref_diff_vars_max = np.max(ref_diff_vars, axis=1)
#     # # ref_n_samples = mc.set_target_variance(t_var, prescribe_vars=ref_diff_vars)
#     # # ref_n_samples = np.max(ref_n_samples, axis=1)
#     # # ref_cost = mc.estimate_cost(n_samples=ref_n_samples)
#     # # ref_total_std = np.sqrt(np.sum(ref_diff_vars / ref_n_samples[:, None]) / n_moments)
#     # # ref_total_std_x = np.sqrt(np.mean(ref_vars))
#     #
#     # #self.mlmc.update_moments(natural_moments)
#     # est_moments, est_vars = self.mlmc.estimate_moments(natural_moments)
#     # nat_cov_est = self.mlmc.estimate_covariance(natural_moments)
#     # nat_cov = L @ cov @ L.T
#     # nat_mom = L @ cov[:,0]
#     #
#     # print("nat_cov_est norm: ", np.linalg.norm(nat_cov_est - np.eye(natural_moments.size)))
#     # # def describe(arr):
#     # #     print("arr ", arr)
#     # #     q1, q3 = np.percentile(arr, [25, 75])
#     # #     print("q1 ", q1)
#     # #     print("q2 ", q3)
#     # #     return "{:f8.2} < {:f8.2} | {:f8.2} | {:f8.2} < {:f8.2}".format(
#     # #         np.min(arr), q1, np.mean(arr), q3, np.max(arr))
#     #
#     # print("n_levels: ", self.n_levels)
#     # print("moments: ", est_moments)
#     # est_moments[1:] = 0
#     # moments_data = np.stack((est_moments, est_vars), axis=1)
#     # distr_obj = Distribution(natural_moments, moments_data, domain=domain)
#     # distr_obj.estimate_density_minimize(tol, reg_param)  # 0.95 two side quantile
#     #
#     #
#     # F = [distr_obj._calculate_exact_moment(distr_obj.multipliers, m)[0] for m in range(natural_moments.size)]
#     # print("F norm: ", np.linalg.norm(np.array(F) - est_moments))
#     #
#     # H = [[distr_obj._calculate_exact_hessian(i,j)[0] for i in range(natural_moments.size)] \
#     #         for j in range(natural_moments.size)]
#     # print("H norm: ", np.linalg.norm(np.array(H) - np.eye(natural_moments.size)))
#     # # distr_obj.estimate_density_minimize(0.1)  # 0.95 two side quantile
#     # self._distribution = distr_obj
#
#
