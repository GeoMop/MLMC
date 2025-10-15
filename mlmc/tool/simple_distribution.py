import numpy as np
import scipy as sc
import scipy.integrate as integrate
import mlmc.moments
import mlmc.plot.plots

EXACT_QUAD_LIMIT = 1000


class SimpleDistribution:
    """
    Approximate a probability density function (PDF) from given moments.

    The class constructs a parametric PDF using Lagrange multipliers and
    fits those multipliers by minimizing a functional (or solving a root
    problem). Numerical integration and adaptive quadrature are used to
    compute moments, gradients and Jacobians required by the optimizer.
    """

    def __init__(self, moments_obj, moment_data, domain=None, force_decay=(True, True), verbose=False):
        """
        Initialize SimpleDistribution.

        :param moments_obj: Object providing moment functions and attributes:
                            - .domain: tuple (a, b) domain of the moment functions
                            - .size: number of available moment basis functions
                            - .eval_all(x, size) or .eval(i, x) for evaluating moments
        :param moment_data: numpy array of shape (n_moments, 2) or None.
                            If provided, column 0 = mean, column 1 = variance of moment estimates.
        :param domain: Optional (a, b) domain for PDF support. If None uses moments_obj.domain.
        :param force_decay: Tuple (bool, bool) controlling whether to penalize non-decay of the
                            PDF at left and right domain endpoints respectively.
        :param verbose: If True, print solver diagnostics.
        """
        # Moment evaluation function with bounded number of moments and their domain.
        self.moments_fn = None

        # Domain of the density approximation (and moment functions).
        if domain is None:
            domain = moments_obj.domain
        self.domain = domain

        # Indicates whether force decay of PDF at domain endpoints (left, right).
        self.decay_penalty = force_decay
        self._verbose = verbose

        # Approximation of moment values (means and standard errors).
        if moment_data is not None:
            self.moment_means = moment_data[:, 0]
            self.moment_errs = np.sqrt(moment_data[:, 1])

        # Lagrange multipliers for moment equations (to be estimated).
        self.multipliers = None

        # Number of basis functions to approximate the density.
        # In future can be smaller than number of provided approximate moments.
        self.approx_size = len(self.moment_means)
        assert moments_obj.size >= self.approx_size
        self.moments_fn = moments_obj

        # Degree of Gauss-Legendre quadrature to use on each adaptive subinterval.
        self._gauss_degree = 21

        # Penalty coefficient for endpoint derivative enforcement (decay penalty).
        # Set to 0 by default in SimpleDistribution (no penalty).
        self._penalty_coef = 0

    def estimate_density_minimize(self, tol=1e-5, reg_param=0.01):
        """
        Estimate multipliers by minimizing the dual functional.

        Uses scipy.optimize.minimize (trust-ncg by default) to minimize the
        functional _calculate_functional(multipliers). The function sets up
        quadrature and internal tolerances before solving, and updates the
        multipliers attribute with the optimizer result.

        :param tol: Optimization tolerance (used for jacobian/grad stopping).
        :param reg_param: Regularization parameter (not used directly here but kept for API parity).
        :return: scipy OptimizeResult with fields:
                 - x: optimized multipliers
                 - success: bool convergence flag (set to True if solver succeeded or residual < tol)
                 - nit: number of iterations (at least 1)
                 - fun_norm: norm of gradient at solution
                 - eigvals: eigenvalues of computed Jacobian (added by this method)
                 - solver_res: raw solver residual information (copy of result.jac)
        :notes:
        - After optimization the code enforces normalization by subtracting log(moment_0)
          from multipliers[0] so that the integral of the density is consistent with moment_0.
        """
        # Initialize multipliers, quadrature, etc.
        self._initialize_params(self.approx_size, tol)
        max_it = 20
        method = 'trust-ncg'  # solver selected for this simpler variant

        # Minimize functional using gradient and Hessian (Jacobian)
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

        # Compute Jacobian and its eigenvalues for diagnostics
        jac = self._calculate_jacobian_matrix(self.multipliers)
        result.eigvals = np.linalg.eigvalsh(jac)

        # Keep solver residual and diagnostics
        result.solver_res = result.jac

        # Fix normalization: ensure integral of density corresponds to moment_0
        moment_0, _ = self._calculate_exact_moment(self.multipliers, m=0, full_output=0)
        m0 = sc.integrate.quad(self.density, self.domain[0], self.domain[1])[0]
        if self._verbose:
            print("moment[0]: {} m0: {}".format(moment_0, m0))
        # Adjust the zeroth multiplier so that the integrated moment_0 matches
        self.multipliers[0] -= np.log(moment_0)

        # Mark solver as successful if solver thinks so or residual is small
        if result.success or jac_norm < tol:
            result.success = True

        # Ensure iteration count at least 1 for downstream code that expects it
        result.nit = max(result.nit, 1)
        result.fun_norm = jac_norm

        return result

    def density(self, value):
        """
        Evaluate the approximated density at the given point(s).

        :param value: scalar or numpy array of points
        :return: numpy array of density values (same shape as flattened input)
        :notes:
        - Uses self.multipliers, self._moment_errs and the basis moments returned
          by eval_moments() to build exponent power = sum_i multipliers_i * moment_i / err_i.
        - The result is clipped (power limited to [-200, 200]) to avoid overflow.
        """
        moms = self.eval_moments(value)
        power = -np.sum(moms * self.multipliers / self._moment_errs, axis=1)
        power = np.minimum(np.maximum(power, -200), 200)
        return np.exp(power)

    def cdf(self, values):
        """
        Evaluate the cumulative distribution function (CDF) at the given points.

        :param values: scalar or array-like points
        :return: numpy array of CDF values corresponding to input points
        :notes:
        - The method integrates the density piecewise between successive query points
          using fixed_quad with n=10 on subintervals determined by the adaptive quadrature info.
        - Values outside domain are mapped to 0 (left) or 1 (right).
        """
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
        Initialize multipliers, quadrature tolerance and related structures.

        :param size: number of multipliers to initialize (approximation order)
        :param tol: tolerance hint for integration/solver (not used directly here)
        :effects:
        - Sets self._quad_tolerance, self._moment_errs, self.multipliers,
          self._quad_log, evaluates endpoint derivatives and updates quadrature.
        """
        assert self.domain is not None
        assert tol is not None

        # Use a very tight quad tolerance for this simple class variant
        self._quad_tolerance = 1e-10

        # Keep a local copy of moment errors used in weighting
        self._moment_errs = self.moment_errs

        # Start multipliers from uniform (log of uniform density)
        self.multipliers = np.zeros(size)
        self.multipliers[0] = -np.log(1/(self.domain[1] - self.domain[0]))

        # Log storage for quadrature diagnostics
        self._quad_log = []

        # Evaluate endpoint derivatives and force quadrature update for initialization
        self._end_point_diff = self.end_point_derivatives()
        self._update_quadrature(self.multipliers, force=True)

    def eval_moments(self, x):
        """
        Evaluate all basis moment functions at x up to current approximation size.

        :param x: scalar or array-like points
        :return: numpy.ndarray of shape (n_points, approx_size) or similar depending on moments_fn.eval_all
        """
        return self.moments_fn.eval_all(x, self.approx_size)

    def _calculate_exact_moment(self, multipliers, m=0, full_output=0):
        """
        Compute exact integral of the m-th moment under the current parametric density.

        :param multipliers: array-like of multipliers used in exponent
        :param m: index of the moment to integrate (default 0)
        :param full_output: if set, pass full_output to scipy.integrate.quad for extra info
        :return: tuple (value, quad_result) where value is integral result and quad_result is
                 the raw return from scipy.integrate.quad (or a subset depending on full_output)
        :notes:
        - The integrand is exp(power) * moment_m. power is clipped to avoid overflow.
        - Uses self._quad_tolerance as epsabs for quad.
        """
        def integrand(x):
            moms = self.eval_moments(x)
            power = -np.sum(moms * multipliers / self._moment_errs, axis=1)
            power = np.minimum(np.maximum(power, -200), 200)
            return np.exp(power) * moms[:, m]

        result = sc.integrate.quad(integrand, self.domain[0], self.domain[1],
                                   epsabs=self._quad_tolerance, full_output=full_output)

        return result[0], result

    def _update_quadrature(self, multipliers, force=False):
        """
        Update quadrature points/weights and cached moments for the current multipliers.

        :param multipliers: current multipliers array
        :param force: if True, force a quadrature update even if error estimates are small
        :effects:
        - Computes adaptive quadrature using scipy.integrate.quad's 'full_output' info (alist/blist).
        - Stores flattened Gauss-Legendre nodes and weights across adaptive intervals in
          self._quad_points and self._quad_weights.
        - Evaluates self._quad_moments at quad points, computes current gradient-like integral
          and stores it as self._last_gradient for reuse.
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

        # Integrate the highest-order moment to get adaptive quadrature info
        val, result = self._calculate_exact_moment(multipliers, m=self.approx_size-1, full_output=1)

        if len(result) > 3:
            y, abserr, info, message = result
            self._quad_log.append(result)
        else:
            y, abserr, info = result
            message = ""

        # Build Gauss-Legendre nodes and weights on each subinterval returned by adaptive quad
        pt, w = np.polynomial.legendre.leggauss(self._gauss_degree)
        K = info['last']
        a = info['alist'][:K, None]
        b = info['blist'][:K, None]
        points = (pt[None, :] + 1) / 2 * (b - a) + a
        weights = w[None, :] * (b - a) / 2

        # Flatten into 1D arrays for convenience
        self._quad_points = points.flatten()
        self._quad_weights = weights.flatten()

        # Evaluate basis moments at quadrature nodes
        self._quad_moments = self.eval_moments(self._quad_points)

        # Compute density and weighted gradient integral used in gradient/Jacobian computations
        power = -np.dot(self._quad_moments, multipliers/self._moment_errs)
        power = np.minimum(np.maximum(power, -200), 200)
        q_gradient = self._quad_moments.T * np.exp(power)
        integral = np.dot(q_gradient, self._quad_weights) / self._moment_errs

        # Cache last multipliers and gradient
        self._last_multipliers = multipliers
        self._last_gradient = integral

    def end_point_derivatives(self):
        """
        Approximate derivatives of all moment basis functions at domain endpoints.

        :return: numpy array of shape (2, approx_size) where index 0 = left derivative,
                 index 1 = right derivative. The derivatives are scaled by moment errors.
        :notes:
        - Uses a tiny eps shift (1e-10) to compute forward/backward differences.
        - If corresponding decay_penalty is False for a side, that side's derivative is left as zeros.
        """
        eps = 1e-10
        left_diff = right_diff = np.zeros((1, self.approx_size))
        if self.decay_penalty[0]:
            left_diff  = self.eval_moments(self.domain[0] + eps) - self.eval_moments(self.domain[0])
        if self.decay_penalty[1]:
            right_diff = -self.eval_moments(self.domain[1]) + self.eval_moments(self.domain[1] - eps)

        return np.stack((left_diff[0, :], right_diff[0, :]), axis=0) / eps / self._moment_errs[None, :]

    def _density_in_quads(self, multipliers):
        """
        Evaluate the parameterized density at cached quadrature nodes.

        :param multipliers: multiplier vector used to compute density
        :return: 1D numpy array of density values at self._quad_points
        """
        power = -np.dot(self._quad_moments, multipliers / self._moment_errs)
        power = np.minimum(np.maximum(power, -200), 200)
        return np.exp(power)

    def _calculate_functional(self, multipliers):
        """
        The functional to be minimized with respect to multipliers.

        :param multipliers: array-like current multipliers
        :return: scalar functional value = sum(mean_i * lam_i / err_i) + integral(density)
        :notes:
        - Adds endpoint penalty if endpoint derivatives violate decay constraints.
        - This functional corresponds to the dual of moment-matching problem.
        """
        self._update_quadrature(multipliers)
        q_density = self._density_in_quads(multipliers)
        integral = np.dot(q_density, self._quad_weights)
        sum_ = np.sum(self.moment_means * multipliers / self._moment_errs)

        end_diff = np.dot(self._end_point_diff, multipliers)
        penalty = np.sum(np.maximum(end_diff, 0) ** 2)
        fun = sum_ + integral
        fun = fun + np.abs(fun) * self._penalty_coef * penalty

        return fun

    def _calculate_gradient(self, multipliers):
        """
        Gradient of the functional with respect to multipliers.

        :param multipliers: array-like current multipliers
        :return: numpy array gradient of shape (approx_size,)
        :notes:
        - Gradient = moment_means/err - integral(moments * density)/err + penalty_terms
        """
        self._update_quadrature(multipliers)
        q_density = self._density_in_quads(multipliers)
        q_gradient = self._quad_moments.T * q_density
        integral = np.dot(q_gradient, self._quad_weights) / self._moment_errs

        end_diff = np.dot(self._end_point_diff, multipliers)
        penalty = 2 * np.dot(np.maximum(end_diff, 0), self._end_point_diff)
        fun = np.sum(self.moment_means * multipliers / self._moment_errs) + integral[0] * self._moment_errs[0]
        gradient = self.moment_means / self._moment_errs - integral + np.abs(fun) * self._penalty_coef * penalty
        return gradient

    def _calculate_jacobian_matrix(self, multipliers):
        """
        Compute Jacobian (Hessian) matrix of the functional.

        :param multipliers: array-like current multipliers
        :return: square numpy array (approx_size, approx_size), symmetric
        :notes:
        - Uses matrix formulation (q_mom.T * diag(q_density * weights) * q_mom) for efficiency.
        - Adds endpoint-penalty contributions and diagonal stabilization if needed.
        """
        self._update_quadrature(multipliers)
        q_density = self._density_in_quads(multipliers)
        q_density_w = q_density * self._quad_weights
        q_mom = self._quad_moments / self._moment_errs

        # Efficient assembly: (Q^T * diag(w*density)) * Q
        jacobian_matrix = (q_mom.T * q_density_w) @ q_mom

        # Endpoint derivative penalty contribution
        end_diff = np.dot(self._end_point_diff, multipliers)
        fun = np.sum(self.moment_means * multipliers / self._moment_errs) + jacobian_matrix[0,0] * self._moment_errs[0]**2
        for side in [0, 1]:
            if end_diff[side] > 0:
                penalty = 2 * np.outer(self._end_point_diff[side], self._end_point_diff[side])
                jacobian_matrix += np.abs(fun) * self._penalty_coef * penalty

        return jacobian_matrix


def compute_exact_moments(moments_fn, density, tol=1e-10):
    """
    Compute moments by integrating moments_fn against provided density.

    :param moments_fn: object with .domain and .size and .eval(i, x)
    :param density: callable accepting numpy arrays (vectorized) returning density values
    :param tol: absolute tolerance for numerical integration
    :return: numpy array of length moments_fn.size containing integrated moments
    """
    a, b = moments_fn.domain
    integral = np.zeros(moments_fn.size)

    for i in range(moments_fn.size):
        def fn(x):
             return moments_fn.eval(i, x) * density(x)

        integral[i] = integrate.quad(fn, a, b, epsabs=tol)[0]
    return integral


def compute_semiexact_moments(moments_fn, density, tol=1e-10):
    """
    Compute moments using a hybrid approach: use adaptive quad to identify subintervals
    then apply Gauss-Legendre nodes inside those subintervals for an accurate quadrature.

    :param moments_fn: moments object with .domain and .size and .eval_all
    :param density: callable density(x)
    :param tol: quad tolerance
    :return: vector of integrated moments (length = moments_fn.size)
    """
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
    Compute covariance matrix of moment basis under the provided density.

    :param moments_fn: moments object
    :param density: callable density(x)
    :param tol: integration tolerance
    :return: symmetric matrix (size x size) containing E[m_i * m_j]
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
    Compute approximate covariance matrix using quadrature nodes determined by adaptive integration.

    :param moments_fn: moments object
    :param density: callable density(x)
    :param tol: integration tolerance
    :return: Jacobian-like matrix approximating covariance (moments weighted by density)
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


def KL_divergence(prior_density, posterior_density, a, b):
    """
    Compute Kullback-Leibler divergence between two densities over [a,b].

    Using numerically stable integrand:
        integrand = p * log(p/q) - p + q
    which equals D_KL(P||Q) when both integrate to 1 but remains finite
    even when Q is not perfectly normalized.

    :param prior_density: callable P(x)
    :param posterior_density: callable Q(x)
    :param a: left integration bound
    :param b: right integration bound
    :return: scalar KL divergence (floored at 1e-10)
    """
    def integrand(x):
        p = prior_density(x)
        q = max(posterior_density(x), 1e-300)
        return p * np.log(p / q) - p + q

    value = integrate.quad(integrand, a, b, epsabs=1e-10)
    return max(value[0], 1e-10)


def L2_distance(prior_density, posterior_density, a, b):
    """
    Compute L2 distance between two densities on [a, b].

    :param prior_density: callable P(x)
    :param posterior_density: callable Q(x)
    :param a: left bound
    :param b: right bound
    :return: scalar L2 norm: sqrt( integral (Q-P)^2 )
    """
    integrand = lambda x: (posterior_density(x) - prior_density(x)) ** 2
    return np.sqrt(integrate.quad(integrand, a, b))[0]


def best_fit_all(values, range_a, range_b):
    """
    Find the best linear fit across all given index ranges.

    The function searches all combinations of indices `a` and `b`
    within `range_a` and `range_b` such that `a < b` and fits a
    first-degree polynomial to the values between these indices.
    It then selects the fit with the smallest residual normalized
    by the square of the interval length.

    :param values: Array-like sequence of values to fit.
    :param range_a: Iterable of possible starting indices.
    :param range_b: Iterable of possible ending indices.
    :return: Tuple (a, b, fit) corresponding to the best fit,
             where `fit` is the array of polynomial coefficients.
    """
    best_fit = None
    best_fit_value = np.inf
    for a in range_a:
        for b in range_b:
            if 0 <= a and a + 2 < b < len(values):

                Y = values[a:b]
                X = np.arange(a, b)
                assert len(X) == len(Y), f"a:{a}  b:{b}"
                fit, res, _, _, _ = np.polyfit(X, Y, deg=1, full=1)

                fit_value = res / ((b - a) ** 2)
                if fit_value < best_fit_value:
                    best_fit = (a, b, fit)
                    best_fit_value = fit_value
    return best_fit


def best_p1_fit(values):
    """
    Recursively find the best linear (P1) fit segment of the sequence.

    The method finds indices `a < b` such that the segment
    `values[a:b]` has the smallest residual (least-squares error)
    normalized by `(b - a)**2`. If the array is large, it downsamples
    the data before recursively fitting.

    :param values: Sequence of numeric values to fit.
    :return: Tuple (a, b, fit) representing the best segment and
             corresponding linear coefficients.
    """
    if len(values) > 12:
        # downscale
        end = len(values) - len(values) % 2    # ensure even length
        avg_vals = np.mean(values[:end].reshape((-1, 2)), axis=1)
        a, b, fit = best_p1_fit(avg_vals)
        # upscale
        a, b = 2 * a, 2 * b
        return best_fit_all(values, [a - 1, a, a + 1], [b - 1, b, b + 1])
    else:
        v_range = range(len(values))
        return best_fit_all(values, v_range, v_range)


def detect_treshold_slope_change(values, log=True):
    """
    Detect the index where the slope of a sequence changes significantly.

    This function fits linear segments to the data (optionally in log scale)
    and detects where the slope begins to deviate, returning both the
    threshold index and a modified version of the input values where
    the slope change is extrapolated.

    :param values: Monotonically increasing numeric sequence.
    :param log: If True, the logarithm of the sequence is used for fitting.
    :return: Tuple (i_treshold, mod_vals)
             - i_treshold: Index where the slope change is detected.
             - mod_vals: Modified version of values with extrapolated segment.
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

    if log:
        mod_vals = np.exp(mod_vals)
    return i_treshold, mod_vals


def lsq_reconstruct(cov, eval, evec, treshold):
    """
    Perform least-squares reconstruction of the eigenvectors
    of a covariance matrix to restore orthogonality.

    This method adjusts the eigenvectors using nonlinear least-squares
    minimization so that the reconstructed eigenvectors are orthogonal
    and diagonalize the covariance matrix as closely as possible.

    :param cov: Covariance matrix (2D array).
    :param eval: Eigenvalues (1D array).
    :param evec: Eigenvectors (2D array).
    :param treshold: Number of eigenvectors to fix (use exact values up to this index).
    :return: Reconstructed orthogonal eigenvector matrix Q.
    """
    Q1 = evec[:, :treshold]
    Q20 = evec[:, treshold:]
    C = cov
    D = np.diag(eval)
    q_shape = Q20.shape
    I = np.eye(q_shape[0])

    def fun(x):
        alpha_orto = 2
        Q2 = x.reshape(q_shape)
        Q = np.concatenate((Q1, Q2), axis=1)
        f = np.sum(np.abs(np.ravel(Q.T @ C @ Q - D))) + alpha_orto * np.sum(np.abs(np.ravel(Q @ Q.T - I)))
        return f

    result = sc.optimize.least_squares(fun, np.ravel(Q20))
    print("LSQ res: ", result.nfev, result.njev, result.cost)
    Q2 = result.x.reshape(q_shape)
    Q = np.concatenate((Q1, Q2), axis=1)

    print("D err", D - Q.T @ cov @ Q)
    print("D", D)
    print("QcovQT", Q.T @ cov @ Q)
    print("I err:", I - Q @ Q.T)
    print("Q err:", Q20 - Q2)

    return Q


def construct_ortogonal_moments(moments, cov, tol=None):
    """
    Construct orthogonal statistical moments with respect to the covariance matrix.

    This function computes a transformation that makes the given moments
    orthogonal under the provided covariance matrix. It determines the
    threshold for significant eigenvalues (either via slope detection
    or tolerance) and constructs a transformation matrix accordingly.

    :param moments: Input moments object.
    :param cov: Covariance matrix estimated from samples.
    :param tol: Optional eigenvalue threshold. If None, an automatic
                slope-change detection is used.
    :return: Tuple (ortogonal_moments, info)
             - ortogonal_moments: Transformed (orthogonalized) moments.
             - info: Tuple containing (eval, threshold, transformation_matrix).
    """
    M = np.eye(moments.size)
    M[:, 0] = -cov[:, 0]
    cov_center = M @ cov @ M.T
    eval, evec = np.linalg.eigh(cov_center)

    if tol is None:
        # determine threshold using slope-change detection
        threshold, fixed_eval = detect_treshold_slope_change(eval, log=True)
        threshold = np.argmax(eval - fixed_eval[0] > 0)
    else:
        # threshold given by eigenvalue magnitude
        threshold = np.argmax(eval > tol)

    new_eval = eval[threshold:]
    new_evec = evec[:, threshold:]

    eval_flipped = np.flip(new_eval, axis=0)
    evec_flipped = np.flip(new_evec, axis=1)

    icov_sqrt_t = M.T @ evec_flipped * (1 / np.sqrt(eval_flipped))[None, :]
    R_nm, Q_mm = sc.linalg.rq(icov_sqrt_t, mode='full')

    L_mn = R_nm.T
    if L_mn[0, 0] < 0:
        L_mn = -L_mn

    ortogonal_moments = mlmc.moments.TransformedMoments(moments, L_mn)
    info = (eval, threshold, L_mn)
    return ortogonal_moments, info
