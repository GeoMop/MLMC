import autograd.numpy as np
import numpy
import scipy as sc
import scipy.integrate as integrate
import mlmc.moments
from autograd import elementwise_grad as egrad
from autograd import hessian
import mlmc.tool.plot
from abc import ABC, abstractmethod

from numpy import testing
import pandas as pd

import numdifftools as nd

EXACT_QUAD_LIMIT = 1000
GAUSS_DEGREE = 150
HUBER_MU = 0.001


class SimpleDistribution:
    """
    Calculation of the distribution
    """

    def __init__(self, moments_obj, moment_data, domain=None, force_decay=(True, True), reg_param=0, max_iter=30, regularization=None):
        """
        :param moments_obj: Function for calculating moments
        :param moment_data: Array  of moments and their vars; (n_moments, 2)
        :param domain: Explicit domain fo reconstruction. None = use domain of moments.
        :param force_decay: Flag for each domain side to enforce decay of the PDF approximation.
        """

        # Family of moments basis functions.
        self.moments_basis = moments_obj

        self.regularization = regularization

        self.reg_par = 1e-3

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
        self._penalty_coef = 0#1

        #self._reg_term_jacobian = None

        self.reg_param = reg_param
        self.max_iter = max_iter

        self.gradients = []
        self.reg_domain = domain
        self.cond_number = 0

    @property
    def multipliers(self):
        if type(self._multipliers).__name__ == 'ArrayBox':
            return self._multipliers._value
        return self._multipliers

    @multipliers.setter
    def multipliers(self, multipliers):
        if type(multipliers).__name__ == 'ArrayBox':
            self._multipliers = multipliers._value
        else:
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

        #print("init multipliers ", self.multipliers)
        # result = sc.optimize.minimize(self._calculate_functional, self.multipliers, method=method,
        #                               jac=self._calculate_gradient,
        #                               hess=self._calculate_jacobian_matrix,
        #                               options={'tol': tol, 'xtol': tol,
        #                                        'gtol': tol, 'disp': True, 'maxiter':max_it}
        #                               #options={'disp': True, 'maxiter': max_it}
        #                                 )
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
        # print("final jacobian")
        # with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        #     print(pd.DataFrame(jac))

        eval, evec = np.linalg.eigh(jac)

        #print("final jac eigen values ", eval)

        # exact_hessian = compute_exact_hessian(self.moments_fn, self.density,reg_param=self.reg_param, multipliers=self.multipliers)
        # print("exact hessian ")
        # print(pd.DataFrame(exact_hessian))

        # exact_cov_reg = compute_exact_cov_2(self.moments_fn, self.density, reg_param=self.reg_param)
        # print("exact cov with reg")
        # print(pd.DataFrame(exact_cov_reg))
        #
        # exact_cov = compute_exact_cov_2(self.moments_fn, self.density)
        # print("exact cov")
        # print(pd.DataFrame(exact_cov))

        result.eigvals = np.linalg.eigvalsh(jac)
        kappa = np.max(result.eigvals) / np.min(result.eigvals)
        self.cond_number = kappa
        #print("condition number ", kappa)
        #result.residual = jac[0] * self._moment_errs
        #result.residual[0] *= self._moment_errs[0]
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

    def density(self, value):
        """
        :param value: float or np.array
        :param moments_fn: counting moments function
        :return: density for passed value
        """
        moms = self.eval_moments(value)
        power = -np.sum(moms * self.multipliers / self._moment_errs, axis=1)
        power = np.minimum(np.maximum(power, -200), 200)

        if type(power).__name__ == 'ArrayBox':
            power = power._value
            if type(power).__name__ == 'ArrayBox':
                power = power._value

        return np.exp(power)

    def density_log(self, value):
        return np.log(self.density(value))

    # def mult_mom(self, value):
    #     moms = self.eval_moments(value)
    #     return -np.sum(moms * self.multipliers, axis=1)
    #
    def mult_mom_der(self, value, degree=1):
        moms = self.eval_moments_der(value, degree)
        return -np.sum(moms * self.multipliers, axis=1)

    # def _current_regularization(self):
    #     return np.sum(self._quad_weights * (np.dot(self._quad_moments_2nd_der, self.multipliers) ** 2))

    # def regularization(self, value):
    #     reg_term = np.dot(self.eval_moments_der(value, degree=2), self.multipliers)**2# self._current_regularization()
    #     reg_term = (np.dot(self._quad_moments_2nd_der, self.multipliers))
    #
    #     #print("np.sum(reg_term)", self.reg_param * np.sum(reg_term))
    #
    #     q_density = self._density_in_quads(self.multipliers)
    #     integral = np.dot(q_density, self._quad_weights)
    #
    #     beta_term = self._quad_weights * (softmax(np.dot(self._quad_moments, -self.multipliers)) ** 2) / (q_density**2)
    #
    #     reg_term_beta = self.reg_param_beta * beta_term#(softmax(np.dot(self.eval_moments(value), - self.multipliers)) **2 / self.density(value))
    #
    #
    #     return (self._quad_points, self.reg_param * (reg_term))

    # def beta_regularization(self, value):
    #     # def integrand(x):
    #     #     return softmax(-self.multipliers * self.eval_moments(x))**2 / self.density(x)
    #     #print("-self.multipliers * self.eval_moments(value) ", -self.multipliers * self.eval_moments(value))
    #
    #     q_density = self._density_in_quads(self.multipliers)
    #     beta_term = self._quad_weights * (softmax(np.dot(self._quad_moments, self.multipliers)))# / (q_density)
    #
    #     # reg_term = []
    #     # for x in value:
    #     #     pom = self.eval_moments_der(x, degree=2) * -self.multipliers
    #     #     # print("softmax(pom)**2 ", softmax(pom) ** 2)
    #     #     reg_term.append(np.sum(softmax(pom) ** 2))
    #     #
    #     # reg_term = np.array(reg_term)
    #
    #
    #     #print("self reg param beta" , self.reg_param_beta)
    #     return (self._quad_points, self.reg_param * (beta_term))
    #
    #     # print("self.eval_moments(value) SHAPE ", self.eval_moments(value).shape)
    #     # print("self multipleirs SHAPE ", self.multipliers.shape)
    #     #
    #     # print("-self.multipliers * self.eval_moments(value) ", -self.multipliers * self.eval_moments(value))
    #     #
    #     # print("-self.multipliers * self.eval_moments(value) ", np.dot(self.eval_moments(value), -self.multipliers))
    #
    #     return softmax(np.dot(self.eval_moments(value), -self.multipliers))
    #     return softmax(-self.multipliers * self.eval_moments(value))
    #
    #     multipliers = np.ones(self.multipliers.shape)
    #     multipliers = -self.multipliers
    #     return np.dot(self.eval_moments_der(value, degree=2), multipliers)
    #
    #     #return softmax(np.dot(self.eval_moments(value), -self.multipliers)) ** 2 / self.density(value)
    #     #return self.reg_param * self.reg_param_beta * softmax(np.dot(self.eval_moments(value), -self.multipliers))**2 / self.density(value)

    # def multipliers_dot_phi(self, value):
    #     return self.reg_param * np.dot(self.eval_moments(value), self.multipliers)
    #
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

    # def distr_den(self, values):
    #     distr = np.empty(len(values))
    #     density = np.empty(len(values))
    #     for index, val in enumerate(values):
    #         distr[index] = self.distr(val)
    #         density[index] = self.density(val)
    #
    #     return distr, density
    #
    # def distr(self, value):
    #     return integrate.quad(self.density, self.domain[0], value)[0]
    #
    # def density_from_distr(self, value):
    #     return egrad(self.distr)(value)

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

        self._moment_errs = self.moment_errs

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

    def eval_moments_der(self, x, degree=1):
        return self.moments_fn.eval_all_der(x, self.approx_size, degree)

    # def _calc_exact_moments(self):
    #     integral = np.zeros(self.moments_fn.size)
    #
    #     for i in range(self.moments_fn.size):
    #         def fn(x):
    #             return self.moments_fn.eval(i, x) * self.density(x)
    #         integral[i] = integrate.quad(fn, self.domain[0], self.domain[1], epsabs=self._quad_tolerance)[0]
    #
    #     return integral

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
        pt, w = numpy.polynomial.legendre.leggauss(self._gauss_degree)
        K = info['last']
        #print("Update Quad: {} {} {} {}".format(K, y, abserr, message))
        a = info['alist'][:K, None]
        b = info['blist'][:K, None]
        points = (pt[None, :] + 1) / 2 * (b - a) + a
        weights = w[None, :] * (b - a) / 2
        self._quad_points = points.flatten()
        self._quad_weights = weights.flatten()

        #print("quad points ", self._quad_points)
        self._quad_moments = self.eval_moments(self._quad_points)
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

    # def _regularization_term(self, tol=1e-10):
    #     """
    #     $\tilde{\rho} = exp^{-\vec{\lambda}\vec{\phi}(x)}$
    #
    #     $$\int_{\Omega} \alpha \exp^{\vec{\lambda}\vec{\phi}(x)} (\tilde{\rho}'')^2dx$$
    #     :param value:
    #     :param tol:
    #     :return:
    #     """
    #
    #     def integrand(x):
    #         moms = self.eval_moments(x)
    #
    #         power = -np.sum(moms * self.multipliers / self._moment_errs, axis=1)
    #         power = np.minimum(np.maximum(power, -200), 200)
    #         return self.reg_param * np.exp(power) * \
    #                (np.sum(-self.multipliers * self.eval_moments_der(x, degree=2)) + \
    #                 np.sum((self.multipliers * moms) ** 2)
    #                ) ** 2
    #
    #     return integrate.quad(integrand, self.domain[0], self.domain[1], epsabs=tol)[0]
    #
    # def plot_regularization(self, X):
    #     reg = []
    #     for x in X:
    #         reg.append(np.sum((self.multipliers * self.eval_moments(x)) ** 2))
    #
    #     return reg

    # def regularization(self, multipliers):
    #
    #     if type(multipliers).__name__ == 'ArrayBox':
    #         multipliers = multipliers._value
    #         if type(multipliers).__name__ == 'ArrayBox':
    #             multipliers = multipliers._value
    #
    #     self._update_quadrature(multipliers)
    #     quad_moments = self.eval_moments(self._quad_points)
    #     sum = np.sum((quad_moments * multipliers) ** 2)
    #
    #     return sum
    #
    #
    #     #return ((multipliers * self.eval_moments(x)) ** 4) / 12
    #     def integrand(x):
    #         #return np.sum(self.multipliers**2)
    #         return np.sum(((multipliers * self.eval_moments(x))**4)/12)
    #
    #     # reg_integrand = integrate.quad(integrand, self.domain[0], self.domain[1], epsabs=1e-5)[0]
    #     # self._update_quadrature(self.multipliers)
    #     #
    #     # reg_quad = np.sum((self.multipliers * self._quad_moments) ** 2)
    #     #
    #     # print("reg integrand ", reg_integrand)
    #     # print("reg_quad ", reg_quad)
    #     #
    #     # return np.sum((self.multipliers * self._quad_moments) ** 2)
    #
    #     return integrate.quad(integrand, self.domain[0], self.domain[1], epsabs=1e-5)[0]
    #     #
    #     # left = integrate.quad(integrand, self.domain[0], -10, epsabs=1e-5)[0]
    #     # right = integrate.quad(integrand, 10, self.domain[1], epsabs=1e-5)[0]
    #     return left + right

    # def _analyze_reg_term_jacobian(self, reg_params):
    #     self._calculate_reg_term_jacobian()
    #     print("self._reg term jacobian ")
    #     print(pd.DataFrame(self._reg_term_jacobian))
    #
    #     for reg_par in reg_params:
    #         print("reg param ", reg_par)
    #         reg_term_jacobian = 2 * reg_par * self._reg_term_jacobian
    #
    #         print("reg term jacobian")
    #         print(pd.DataFrame(reg_term_jacobian))
    #
    #         eigenvalues, eigenvectors = sc.linalg.eigh(reg_term_jacobian)
    #         print("eigen values ")
    #         print(pd.DataFrame(eigenvalues))
    #
    #         print("eigen vectors ")
    #         print(pd.DataFrame(eigenvectors))

    # def _functional(self):
    #     self._update_quadrature(self.multipliers, True)
    #     q_density = self._density_in_quads(self.multipliers)
    #     integral = np.dot(q_density, self._quad_weights)
    #     sum = np.sum(self.moment_means * self.multipliers / self._moment_errs)
    #     fun = sum + integral
    #
    #     return fun

    def _calculate_functional(self, multipliers):
        """
        Minimized functional.
        :param multipliers: current multipliers
        :return: float
        """
        print("CALCULATE FUNCTIONAL")
        self.multipliers = multipliers
        self._update_quadrature(multipliers, True)
        q_density = self._density_in_quads(multipliers)
        integral = np.dot(q_density, self._quad_weights)
        sum = np.sum(self.moment_means * multipliers / self._moment_errs)
        fun = sum + integral

        if self._penalty_coef != 0:
            end_diff = np.dot(self._end_point_diff, multipliers)
            penalty = np.sum(np.maximum(end_diff, 0) ** 2)
            fun = fun + np.abs(fun) * self._penalty_coef * penalty

        #reg_term = np.sum(self._quad_weights * (np.dot(self._quad_moments_2nd_der, self.multipliers) ** 2))
        if self.regularization is not None:
            print("regularization functional ", self.reg_param * self.regularization.functional_term(self))
            fun += self.reg_param * self.regularization.functional_term(self)
        # reg_term = np.sum(self._quad_weights * (np.dot(self._quad_moments_2nd_der, self.multipliers) ** 2))
        # fun += self.reg_param * reg_term
        self.functional_value = fun
        print("functional value ", fun)
        print("self multipliers ", self.multipliers)

        return fun

    # def derivative(self, f, a, method='central', h=0.01):
    #     '''Compute the difference formula for f'(a) with step size h.
    #
    #     Parameters
    #     ----------
    #     f : function
    #         Vectorized function of one variable
    #     a : number
    #         Compute derivative at x = a
    #     method : string
    #         Difference formula: 'forward', 'backward' or 'central'
    #     h : number
    #         Step size in difference formula
    #
    #     Returns
    #     -------
    #     float
    #         Difference formula:
    #             central: f(a+h) - f(a-h))/2h
    #             forward: f(a+h) - f(a))/h
    #             backward: f(a) - f(a-h))/h
    #     '''
    #     if method == 'central':
    #         return (f(a + h) - f(a - h)) / (2 * h)
    #     elif method == 'forward':
    #         return (f(a + h) - f(a)) / h
    #     elif method == 'backward':
    #         return (f(a) - f(a - h)) / h
    #     else:
    #         raise ValueError("Method must be 'central', 'forward' or 'backward'.")

    def _calculate_gradient(self, multipliers):
        """
        Gradient of th functional
        :return: array, shape (n_moments,)
        """
        print("CALCULATE GRADIENT")
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

        #print("gradient ", gradient)
        #########################
        # Numerical derivation

        # if self.reg_param != 0:
        #     # reg_term = np.empty(len(self.multipliers))
        #     # reg_term_quad = np.empty(len(self.multipliers))
        #     # for i in range(len(self.multipliers)):
        #     #     def integrand(x):
        #     #         moments = self.eval_moments_der(x, degree=2)[0, :]
        #     #         return np.dot(moments, self.multipliers) * moments[i]
        #     #
        #     #     reg_term[i] = (sc.integrate.quad(integrand, self.reg_domain[0], self.reg_domain[1])[0])
        #     #
        #     #     def integrand_2(x):
        #     #         moments = self.eval_moments_der(x, degree=2)
        #     #         print("moments ", moments)
        #     #         return np.dot(moments, self.multipliers) * moments[:, i]
        #     #
        #     #     [x, w] = numpy.polynomial.legendre.leggauss(GAUSS_DEGREE)
        #     #     a = self.reg_domain[0]
        #     #     b = self.reg_domain[1]
        #     #     x = (x[None, :] + 1) / 2 * (b - a) + a
        #     #     x = x.flatten()
        #     #     w = w.flatten()
        #     #     reg_term_quad[i] = (np.sum(w * integrand_2(x)) * 0.5 * (b - a))
        #     #
        #
        #     # def integrand(x):
        #     #     moments = self.eval_moments_der(x, degree=2)
        #     #     return np.dot(moments, self.multipliers) * moments.T
        #     #
        #     # [x, w] = numpy.polynomial.legendre.leggauss(GAUSS_DEGREE)
        #     # a = self.reg_domain[0]
        #     # b = self.reg_domain[1]
        #     # x = (x[None, :] + 1) / 2 * (b - a) + a
        #     # x = x.flacalc_tten()
        #     # w = w.flatten()
        #     # reg_term = (np.sum(w * integrand(x), axis=1) * 0.5 * (b - a))

        #reg_term = np.sum(self._quad_weights *
        #                  (np.dot(self._quad_moments_2nd_der, self.multipliers) * self._quad_moments_2nd_der.T), axis=1)

        if self.regularization is not None:
            #print("self.regularization gradient term ", self.regularization.gradient_term(self))
            gradient += self.reg_param * self.regularization.gradient_term(self)
        # quad_moments_2nd_der = self._quad_moments_2nd_der
        self.gradients.append(gradient)

        return gradient

    # def _calculate_reg_term_jacobian(self):
    #     self._reg_term_jacobian = (self._quad_moments_2nd_der.T * self._quad_weights) @ self._quad_moments_2nd_der

    def _calc_jac(self):
        q_density = self.density(self._quad_points)
        q_density_w = q_density * self._quad_weights

        jacobian_matrix = (self._quad_moments.T * q_density_w) @ self._quad_moments
        # if self.reg_param != 0:
        # if self._reg_term_jacobian is None:
        #     self._calculate_reg_term_jacobian()

        if self.reg_param != 0:
            if self.regularization is not None:
                # print("jacobian ")
                # print(pd.DataFrame(jacobian_matrix))
                #
                # print("regularization jacobian term")
                # print(self.regularization.jacobian_term(self))

                jacobian_matrix += self.reg_param * self.regularization.jacobian_term(self)

        return jacobian_matrix

    def _calculate_jacobian_matrix(self, multipliers):
        """
        :return: jacobian matrix, symmetric, (n_moments, n_moments)
        """
        print("CALCULATE JACOBIAN")
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

        # print("jacobian")
        # print(pd.DataFrame(jacobian_matrix))

        return jacobian_matrix


class Regularization(ABC):

    @abstractmethod
    def functional_term(self, simple_distr):
        """
        Regularization added to functional
        """

    @abstractmethod
    def gradient_term(self, simple_distr):
        """
        Regularization to gradient
        """

    @abstractmethod
    def jacobian_term(self, simple_distr):
        """
        Regularization to jacobian matrix
        """

    @abstractmethod
    def jacobian_precondition(self):
        """
        Jacobian matrix preconditioning
        :return:
        """


class Regularization2ndDerivation(Regularization):

    def functional_term(self, simple_distr):
        return np.sum(simple_distr._quad_weights * (np.dot(simple_distr._quad_moments_2nd_der,
                                                           simple_distr.multipliers) ** 2))

    def gradient_term(self, simple_distr):
        reg_term = np.sum(simple_distr._quad_weights *
                          (np.dot(simple_distr._quad_moments_2nd_der, simple_distr.multipliers) *
                           simple_distr._quad_moments_2nd_der.T), axis=1)

        return 2 * reg_term

    def jacobian_term(self, simple_distr):
        reg = 2 * (simple_distr._quad_moments_2nd_der.T * simple_distr._quad_weights) @\
               simple_distr._quad_moments_2nd_der

        return reg

    def jacobian_precondition(self, moments_fn, quad_points, quad_weights):
        """
        Jacobian matrix preconditioning
        :return:
        """
        quad_moments_2nd_der = moments_fn.eval_all_der(quad_points, degree=2)

        reg_term = (quad_moments_2nd_der.T * quad_weights) @ quad_moments_2nd_der

        return 2 * reg_term


class Regularization3rdDerivation(Regularization):

    def functional_term(self, simple_distr):
        return np.sum(simple_distr._quad_weights * (np.dot(simple_distr._quad_moments_3rd_der,
                                                           simple_distr.multipliers) ** 2))

    def gradient_term(self, simple_distr):
        reg_term = np.sum(simple_distr._quad_weights *
                          (np.dot(simple_distr._quad_moments_3rd_der, simple_distr.multipliers) *
                           simple_distr._quad_moments_3rd_der.T), axis=1)

        return 2 * reg_term

    def jacobian_term(self, simple_distr):
        reg = 2 * (simple_distr._quad_moments_3rd_der.T * simple_distr._quad_weights) @\
               simple_distr._quad_moments_3rd_der

        return reg

    def jacobian_precondition(self, moments_fn, quad_points, quad_weights):
        """
        Jacobian matrix preconditioning
        :return:
        """
        quad_moments_3rd_der = moments_fn.eval_all_der(quad_points, degree=3)

        reg_term = (quad_moments_3rd_der.T * quad_weights) @ quad_moments_3rd_der
        #print("reg term ", reg_term)

        return 2 * reg_term


class RegularizationInexact(Regularization):

    def functional_term(self, simple_distr):
        return np.sum((simple_distr.multipliers - simple_distr.multipliers[0])**2)

    def gradient_term(self, simple_distr):
        reg_term = 2*(simple_distr.multipliers - simple_distr.multipliers[0])

        return reg_term

    def jacobian_term(self, simple_distr):
        reg = 2
        return reg

    def jacobian_precondition(self, moments_fn, quad_points, quad_weights):
        """
        Jacobian matrix preconditioning
        :return:
        """
        reg_term = 2
        return reg_term


class RegularizationInexact2(Regularization):

    def functional_term(self, simple_distr):
        #print("np.sum(simple_distr.multipliers) ", np.sum(simple_distr.multipliers))
        return np.sum(simple_distr.multipliers**2)

    def gradient_term(self, simple_distr):
        reg_term = 2*simple_distr.multipliers

        return reg_term

    def jacobian_term(self, simple_distr):
        reg = 2
        return reg

    def jacobian_precondition(self, moments_fn, quad_points, quad_weights):
        """
        Jacobian matrix preconditioning
        :return:
        """
        reg_term = 2
        return reg_term



class RegularizationTV(Regularization):

    def functional_term(self, simple_distr):
        return self._reg_term(simple_distr.density, simple_distr.domain)
        #return total_variation_int(simple_distr.density, simple_distr.domain[0], simple_distr.domain[1])

    def _reg_term(self, density, domain):
        return total_variation_int(density, domain[0], domain[1])

    def gradient_term(self, simple_distr):
        #return total_variation_int(simple_distr.density_derivation, simple_distr.domain[0], simple_distr.domain[1])

        print("egrad(self.functional_term(simple_distr)) ", egrad(self.functional_term)(simple_distr))
        return egrad(self._reg_term)(simple_distr.density, simple_distr.domain)

    def jacobian_term(self, simple_distr):

        #return total_variation_int(simple_distr.density_second_derivation,  simple_distr.domain[0], simple_distr.domain[1])

        #print("hessian(self.functional_term(simple_distr)) ", hessian(self.functional_term)(simple_distr))
        return hessian(self._reg_term)(simple_distr.density, simple_distr.domain)


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
    pt, w = numpy.polynomial.legendre.leggauss(GAUSS_DEGREE)
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


# def hessian_reg_term(moments_fn, density, reg_param, tol=1e-10):
#     import numdifftools as nd
#     a, b = moments_fn.domain
#     integral = np.zeros((moments_fn.size, moments_fn.size))
#
#     density_derivation = nd.Derivative(density, n=1)
#     density_2nd_derivation = nd.Derivative(density, n=2)
#
#     for i in range(moments_fn.size):
#         for j in range(i + 1):
#             def fn(x):
#                 mom = moments_fn.eval_all(x)[0, :]
#                 mom_derivative = moments_fn.eval_all_der(x, degree=1)[0, :]
#                 mom_second_derivative = moments_fn.eval_all_der(x, degree=2)[0, :]
#
#                 mult_mom = -np.log(density(x))
#                 mult_mom_der = -density_derivation(x) / density(x)
#                 mult_mom_second_der = (-density_2nd_derivation(x) + (-mult_mom_der) ** 2 * density(x)) / density(x)
#
#                 # print("mult mom der ", mult_mom_der)
#                 # print("mult mom second der ", mult_mom_second_der)
#                 # print("mom ", mom)
#
#                 # first_bracket = -mom * (-mult_mom_second_der + mult_mom_der ** 2) + (-mom_second_derivative + 2 * mult_mom_der * mom_derivative)
#                 # second_bracket = -2 * mom_second_derivative + 4 * mult_mom * mom + mom * mom_second_derivative + mult_mom_der ** 2
#                 # third_bracket = -mult_mom_second_der + mult_mom_der ** 2
#                 # fourth_bracket = 4 * mom ** 2 + mom * mom_second_derivative + 2 * mult_mom_der * mom_derivative
#
#                 # first_bracket = -mom[i] * (-mult_mom_second_der + mult_mom_der**2) + (-mom_second_derivative + 2*mult_mom_der*mom_derivative)
#                 # second_bracket = -2*mom_second_derivative[j] + 4*mult_mom*mom + mom*mom_second_derivative + mult_mom_der**2
#                 # third_bracket = -mult_mom_second_der + mult_mom_der**2
#                 # fourth_bracket = 4*mom**2 + mom[i]*mom_second_derivative[j] + 2*mult_mom_der*mom_derivative
#
#                 first_bracket = -mom[i] * (np.sum(-mult_mom_second_der) + np.sum(mult_mom_der ** 2)) +\
#                                 (-mom_second_derivative[i] + np.sum(2 * mult_mom_der * mom_derivative))
#                 #print("first bracket ", first_bracket)
#
#                 second_bracket = -2 * mom_second_derivative[j] + np.sum(4 * mult_mom * mom) + np.sum(mom * mom_second_derivative)\
#                                  + np.sum(mult_mom_der) ** 2
#                 #print("second bracket ", second_bracket)
#
#                 third_bracket = -np.sum(mult_mom_second_der) + np.sum(mult_mom_der) ** 2
#                 fourth_bracket = np.sum(4 * mom ** 2) + mom[i] * mom_second_derivative[j] + 2 * np.sum(mult_mom_der * mom_derivative)
#
#                 reg = first_bracket * second_bracket + third_bracket * fourth_bracket

#                 # print("moments[i] ", mom[i])
#                 # print("moments[j] ", mom[j])
#                 #return result * density(x)
#
#                 #exit()
#
#                 moments = moments_fn.eval_all(x)[0, :]
#                 # print("HESS REG ", (reg_param * np.sum(moments[i] * moments[j] * density(x))))
#                 return (moments[i] * moments[j] + (reg_param * reg)) * density(x)  # + reg_param * hessian_reg_term(moments[i], moments[j], density(x))
#                 # return moments[i] * moments[j] * density(x) + (reg_param * 2)
#
#             integral[j][i] = integral[i][j] = integrate.quad(fn, a, b, epsabs=tol)[0]
#     return integral


# def compute_exact_hessian(moments_fn, density, tol=1e-10, reg_param=0, multipliers=None):
#     """
#     Compute approximation of covariance matrix using exact density.
#     :param moments_fn: Moments function.
#     :param density: Density function (must accept np vectors).
#     :param tol: Tolerance of integration.
#     :return: np.array, moment values
#     """
#     a, b = moments_fn.domain
#     integral = np.zeros((moments_fn.size, moments_fn.size))
#     integral_reg = np.zeros((moments_fn.size, moments_fn.size))
#
#     for i in range(moments_fn.size):
#         for j in range(i+1):
#             def fn_reg_term(x):
#                 moments_2nd_der = moments_fn.eval_all_der(x, degree=2)[0, :]
#
#                 return moments_fn.eval_all(x)[0, :][i]
#
#                 #return moments_2nd_der[i] **2 * density(x)
#                 return moments_2nd_der[i] * moments_2nd_der[j]# * density(x)
#
#             def fn(x):
#                 moments = moments_fn.eval_all(x)[0, :]
#
#                 density_value = density(x)
#                 if type(density_value).__name__ == 'ArrayBox':
#                     density_value = density_value._value
#
#                 # density_derivation = nd.Derivative(density, n=1)
#                 # density_2nd_derivation = nd.Derivative(density, n=2)
#                 # mult_mom_der = -density_derivation(x) / density(x)
#                 # mult_mom_second_der = (-density_2nd_derivation(x) + (-mult_mom_der) ** 2 * density(x)) / density(x)
#
#                 #print("HESS REG ", (reg_param * np.sum(moments[i] * moments[j] * density(x))))
#                 return moments[i] * moments[j] * density_value + 2#* hessian_reg_term(moments[i], moments[j], density(x))
#                 #return moments[i] * moments[j] * density(x) + (reg_param * 2)
#             integral[j][i] = integral[i][j] = integrate.quad(fn, a, b, epsabs=tol)[0]
#             integral_reg[j][i] = integral_reg[i][j] = integrate.quad(fn_reg_term, a, b, epsabs=tol)[0]
#
#     #integral = hessian_reg_term(moments_fn, density, reg_param, tol)
#
#     integral = integral + (reg_param * (multipliers.T * integral_reg * multipliers))# * integral)
#
#     return integral


# def compute_exact_cov(moments_fn, density, tol=1e-10):
#     """
#     Compute approximation of covariance matrix using exact density.
#     :param moments_fn: Moments function.
#     :param density: Density function (must accept np vectors).
#     :param tol: Tolerance of integration.
#     :return: np.array, moment values
#     """
#     a, b = moments_fn.domain
#     integral = np.zeros((moments_fn.size, moments_fn.size))
#
#     for i in range(moments_fn.size):
#         for j in range(i+1):
#             def fn(x):
#                 moments = moments_fn.eval_all(x)[0, :]
#
#                 density_value = density(x)
#                 if type(density_value).__name__ == 'ArrayBox':
#                     density_value = density_value._value
#
#                 return moments[i] * moments[j]* density_value # * density(x)
#             integral[j][i] = integral[i][j] = integrate.quad(fn, a, b, epsabs=tol)[0]
#
#
#     # print("integral ", integral)
#     # print("integral shape ", integral.shape)
#     # exit()
#     #
#     # integral +=
#
#     return integral


def compute_exact_cov(moments_fn, density, tol=1e-10, reg_param=0, domain=None):
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
    int_reg = np.zeros((moments_fn.size, moments_fn.size))

    print("a_2: {}, b_2: {}".format(a_2, b_2))

    for i in range(moments_fn.size):
        for j in range(i+1):

            def fn_moments_der(x):
                moments = moments_fn.eval_all_der(x, degree=2)[0, :]
                return moments[i] * moments[j]

            def fn(x):
                moments = moments_fn.eval_all(x)[0, :]
                #print("moments ", moments)

                density_value = density(x)
                if type(density_value).__name__ == 'ArrayBox':
                    density_value = density_value._value

                return moments[i] * moments[j] * density_value # * density(x)

            integral[j][i] = integral[i][j] = integrate.quad(fn, a, b, epsabs=tol)[0]

            int_2 = integrate.quad(fn_moments_der, a_2, b_2, epsabs=tol)[0]
            int_reg[j][i] = int_reg[i][j] = int_2

    int_reg = 2 * reg_param * int_reg
    return integral, int_reg


def compute_semiexact_cov_2(moments_fn, density, tol=1e-10, reg_param=0, mom_size=None, regularization=None):
    """
    Compute approximation of covariance matrix using exact density.
    :param moments_fn: Moments function.
    :param density: Density function (must accept np vectors).
    :param tol: Tolerance of integration.
    :return: np.array, moment values
    """
    print("COMPUTE SEMIEXACT COV")

    a, b = moments_fn.domain
    if mom_size is not None:
        moments_fn.size = mom_size
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
    pt, w = numpy.polynomial.legendre.leggauss(GAUSS_DEGREE)

    K = info['last']
    # print("Update Quad: {} {} {} {}".format(K, y, abserr, message))
    a = info['alist'][:K, None]
    b = info['blist'][:K, None]

    points = (pt[None, :] + 1) / 2 * (b - a) + a
    weights = w[None, :] * (b - a) / 2

    quad_points = points.flatten()
    quad_weights = weights.flatten()
    quad_moments = moments_fn.eval_all(quad_points)
    quad_moments_2nd_der = moments_fn.eval_all_der(quad_points, degree=2)
    q_density = density(quad_points)
    q_density_w = q_density * quad_weights

    jacobian_matrix = (quad_moments.T * q_density_w) @ quad_moments

    reg_matrix = np.zeros(jacobian_matrix.shape)
    print("regularization ", regularization)

    if regularization is not None:
        reg_term = regularization.jacobian_precondition(moments_fn, quad_points, quad_weights)
        #reg_term = (quad_moments_2nd_der.T * quad_weights) @ quad_moments_2nd_der
        reg_matrix += reg_param * reg_term

        print("reg matrix ")
        print(pd.DataFrame(reg_matrix))

    return jacobian_matrix, reg_matrix


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
    pt, w = numpy.polynomial.legendre.leggauss(GAUSS_DEGREE)
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
        p = prior_density(x)
        # posterior
        #print("p ", p)
        q = max(posterior_density(x), 1e-300)
        #print("q ", q)
        # modified integrand to provide positive value even in the case of imperfect normalization
        return p * np.log(p / q) - p + q

    value = integrate.quad(integrand, a, b)#, epsabs=1e-10)

    return value[0]
    #return max(value[0], 1e-10)


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


# def total_variation_int(func, a, b):
#     import numdifftools as nd
#
#     def integrand(x):
#         return hubert_l1_norm(nd.Derivative(func), x)
#
#     return integrate.quad(integrand, a, b)[0]


# def total_variation_int(func, a, b):
#     import numdifftools as nd
#     from autograd import grad, elementwise_grad
#     import matplotlib.pyplot as plt
#
#     f = grad(func)
#
#     fun_y = []
#     f_y = []
#
#     x = numpy.linspace(-10, 10, 200)
#     #
#     for i in x:
#         print("func(i) ", func(i))
#         print("f(i) ", f(i))
#     #     # fun_y.append(func(i))
#         # f_y.append(f(i))
#
#     # plt.plot(x, fun_y, '-')
#     # plt.plot(x, f_y, ":")
#     # plt.show()
#
#
#     def integrand(x):
#         return hubert_l1_norm(f, x)
#
#     return integrate.quad(integrand, a, b)[0]


def l1_norm(func, x):
    import numdifftools as nd
    return numpy.absolute(func(x))
    #return numpy.absolute(nd.Derivative(func, n=1)(x))


def huber_l1_norm(func, x):
    r = func(x)

    mu = HUBER_MU
    y = mu * (numpy.sqrt(1+(r**2/mu**2)) - 1)

    return y


def huber_norm(func, x):
    result = []

    for value in x:
        r = func(value)
        mu = HUBER_MU

        y = mu * (numpy.sqrt(1+(r**2/mu**2)) - 1)

        result.append(y)

    return result
    pass


def total_variation_vec(func, a, b):
    x = numpy.linspace(a, b, 1000)
    x1 = x[1:]
    x2 = x[:-1]

    #print("tv ", sum(abs(func(x1) - func(x2))))

    return sum(abs(func(x1) - func(x2)))


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

def print_cumul(eval):
    import matplotlib.pyplot as plt
    tot = sum(eval)
    var_exp = [(i / tot) * 100 for i in sorted(eval, reverse=True)]
    print("var_exp ", var_exp)
    cum_var_exp = np.cumsum(var_exp)
    #print("cum_var_exp ", cum_var_exp)

    # threshold = np.argmin(cum_var_exp > 99.99)
    # print("new threshold ", threshold)

    #with plt.style.context('seaborn-whitegrid'):
    # plt.figure(figsize=(6, 4))
    #
    # plt.bar(range(len(eval)), var_exp, alpha=0.5, align='center',
    #         label='individual explained variance')
    # plt.step(range(len(eval)), cum_var_exp, where='mid',
    #          label='cumulative explained variance')
    # plt.ylabel('Explained variance ratio')
    # plt.xlabel('Principal components')
    # plt.legend(loc='best')
    # plt.tight_layout()
    #
    # plt.show()

    return cum_var_exp, var_exp


def _cut_eigenvalues(cov_center, tol):
    print("CUT eigenvalues")

    eval, evec = np.linalg.eigh(cov_center)

    print("original evec ")
    print(pd.DataFrame(evec))

    #eval = np.abs(eval)

    #print_cumul(eval)

    original_eval = eval
    print("original eval ", eval)
    # print("cut eigenvalues tol ", tol)

    # eig_pairs = [(np.abs(eval[i]), evec[:, i]) for i in range(len(eval))]
    #
    # # Sort the (eigenvalue, eigenvector) tuples from high to low
    # eig_pairs.sort(key=lambda x: x[0], reverse=True)

    # for pair in eig_pairs:
    #     print("pair ", pair)
    #
    # for pair in eig_pairs[:10]:
    #     print("pair[] ", pair)
    #
    # exit()

    # Visually confirm that the list is correctly sorted by decreasing eigenvalues
    # print('Eigenvalues in descending order:')
    # for i in eig_pairs:
    #     print(i[0])
    #
    # print("sorted(eval, reverse=True) ", sorted(eval, reverse=True))

    # print("EVAL SORTED ", sorted(eval, reverse=True))
    # print("EVAL EIG PAIR ", np.hstack(np.array([eig_pair[0] for eig_pair in eig_pairs[:]])))
    # cum_var_exp = print_cumul(np.hstack(np.array([eig_pair[0] for eig_pair in eig_pairs[:]])))

    if tol is None:
        # treshold by statistical test of same slopes of linear models
        threshold, fixed_eval = detect_treshold_slope_change(eval, log=True)
        threshold = np.argmax(eval - fixed_eval[0] > 0)
    else:
        # threshold given by eigenvalue magnitude
        threshold = np.argmax(eval > tol)

    # print("[eig_pair[1].reshape(len(eval), 1) for eig_pair in eig_pairs[:-5]]",
    #       [eig_pair[1].reshape(len(eval), 1) for eig_pair in eig_pairs[:-5]])

    #threshold = 30
    # print("threshold ", threshold)
    # print("eval ", eval)

    #print("eig pairs ", eig_pairs[:])

    #threshold_above = len(original_eval) - np.argmax(eval > 1)

    #print("threshold above ", threshold_above)

    # threshold = np.argmax(cum_var_exp > 110)
    # if threshold == 0:
    #     threshold = len(cum_var_exp)
    #
    # print("max eval index: {}, threshold: {}".format(len(eval) - 1, threshold))

    # matrix_w = np.hstack(np.array([eig_pair[1].reshape(len(eval), 1) for eig_pair in eig_pairs[:-30]]))
    #
    # print("matrix_w.shape ", matrix_w.shape)
    # print("matrix_w ")
    # print(pd.DataFrame(matrix_w))

    # matrix_w = np.hstack(np.array([eig_pair[1].reshape(len(eval), 1) for eig_pair in eig_pairs[:threshold]]))
    #
    # new_eval = np.hstack(np.array([eig_pair[0] for eig_pair in eig_pairs[:threshold]]))
    #
    # threshold -= 1

    # print("matrix_w.shape final ", matrix_w.shape)
    # print("matrix_w final ")
    # print(pd.DataFrame(matrix_w))

    # add the |smallest eigenvalue - tol(^2??)| + eigenvalues[:-1]

    #threshold = 0
    # print("threshold ", threshold)
    # print("eval ", eval)

    #treshold, _ = self.detect_treshold(eval, log=True, window=8)

    # tresold by MSE of eigenvalues
    #treshold = self.detect_treshold_mse(eval, std_evals)

    # treshold

    #self.lsq_reconstruct(cov_center, fixed_eval, evec, treshold)

    # cut eigen values under treshold
    new_eval = eval[threshold:]
    new_evec = evec[:, threshold:]

    eval = np.flip(new_eval, axis=0)
    evec = np.flip(new_evec, axis=1)

    print_cumul(eval)

    # for ev in evec:
    #     print("np.linalg.norm(ev) ", np.linalg.norm(ev))
    #     #testing.assert_array_almost_equal(1.0, np.linalg.norm(ev), decimal=0)
    # print('Everything ok!')

    return eval, evec, threshold, original_eval


def _svd_cut(cov_center, tol):
    print("CUT eigenvalues")
    u, s, vh = np.linalg.svd(cov_center)

    print("u")
    print(pd.DataFrame(u))

    print("s")
    print(pd.DataFrame(s))

    print("vh")
    print(pd.DataFrame(vh))
    exit()

    # print("EVAL SORTED ", sorted(eval, reverse=True))
    # print("EVAL EIG PAIR ", np.hstack(np.array([eig_pair[0] for eig_pair in eig_pairs[:]])))
    # cum_var_exp = print_cumul(np.hstack(np.array([eig_pair[0] for eig_pair in eig_pairs[:]])))

    if tol is None:
        # treshold by statistical test of same slopes of linear models
        threshold, fixed_eval = detect_treshold_slope_change(eval, log=True)
        threshold = np.argmax(eval - fixed_eval[0] > 0)
    else:
        # threshold given by eigenvalue magnitude
        threshold = np.argmax(eval > tol)

    # print("[eig_pair[1].reshape(len(eval), 1) for eig_pair in eig_pairs[:-5]]",
    #       [eig_pair[1].reshape(len(eval), 1) for eig_pair in eig_pairs[:-5]])

    #threshold = 30
    # print("threshold ", threshold)
    # print("eval ", eval)

    #print("eig pairs ", eig_pairs[:])

    #threshold_above = len(original_eval) - np.argmax(eval > 1)

    #print("threshold above ", threshold_above)

    # threshold = np.argmax(cum_var_exp > 110)
    # if threshold == 0:
    #     threshold = len(cum_var_exp)
    #
    # print("max eval index: {}, threshold: {}".format(len(eval) - 1, threshold))

    # matrix_w = np.hstack(np.array([eig_pair[1].reshape(len(eval), 1) for eig_pair in eig_pairs[:-30]]))
    #
    # print("matrix_w.shape ", matrix_w.shape)
    # print("matrix_w ")
    # print(pd.DataFrame(matrix_w))

    # matrix_w = np.hstack(np.array([eig_pair[1].reshape(len(eval), 1) for eig_pair in eig_pairs[:threshold]]))
    #
    # new_eval = np.hstack(np.array([eig_pair[0] for eig_pair in eig_pairs[:threshold]]))
    #
    # threshold -= 1

    # print("matrix_w.shape final ", matrix_w.shape)
    # print("matrix_w final ")
    # print(pd.DataFrame(matrix_w))

    # add the |smallest eigenvalue - tol(^2??)| + eigenvalues[:-1]

    #threshold = 0
    # print("threshold ", threshold)
    # print("eval ", eval)

    #treshold, _ = self.detect_treshold(eval, log=True, window=8)

    # tresold by MSE of eigenvalues
    #treshold = self.detect_treshold_mse(eval, std_evals)

    # treshold

    #self.lsq_reconstruct(cov_center, fixed_eval, evec, treshold)

    # cut eigen values under treshold
    new_eval = eval[threshold:]
    new_evec = evec[:, threshold:]

    eval = np.flip(new_eval, axis=0)
    evec = np.flip(new_evec, axis=1)

    print_cumul(eval)

    # for ev in evec:
    #     print("np.linalg.norm(ev) ", np.linalg.norm(ev))
    #     #testing.assert_array_almost_equal(1.0, np.linalg.norm(ev), decimal=0)
    # print('Everything ok!')

    return eval, evec, threshold, original_eval

def my_ceil(a, precision=0):
    return np.round(a + 0.5 * 10**(-precision), precision)

def my_floor(a, precision=0):
    return np.round(a - 0.5 * 10**(-precision), precision)


# def _pca(cov_center, tol):
#     from numpy import ma
#     eval, evec = np.linalg.eigh(cov_center)
#
#     original_eval = eval
#     print("original eval ", original_eval)
#     #
#     # print("original evec ")
#     # print(pd.DataFrame(evec))
#
#     cum_var_exp, var_exp = print_cumul(sorted(eval, reverse=True))
#     print("CUM VAR EXP ", cum_var_exp)
#
#     eval = np.flip(eval, axis=0)
#     evec = np.flip(evec, axis=1)
#
#     eig_pairs = [(np.abs(eval[i]), evec[:, i]) for i in range(len(eval))]
#
#     # threshold = np.argmax(cum_var_exp > 110)
#     # if threshold == 0:
#     #     threshold = len(eval)
#     # #threshold = len(eval)
#
#     cumul_hundred = np.argmax(cum_var_exp == 100)
#     #print("cumul hundred ", cumul_hundred)
#
#     # cut_threshold = np.argmax(np.array(var_exp) < 1e-5)
#     # cum_var_exp, var_exp = print_cumul(eval[:cut_threshold])
#     # print("new cum var exp ", cum_var_exp)
#
#     ######!!!!!! previous
#
#     print("np.max(np.floor(cum_var_exp))", )
#
#     threshold = 0
#
#     import decimal
#     d = decimal.Decimal(str(tol))
#     dec = d.as_tuple().exponent
#
#     # print("exp 10 ", 10**(-2))
#     #
#     # print("exp 10 ", 10 ** (-dec*(-1)))
#     #
#     # exit()
#
#     raw_floor_max = np.max(np.floor(cum_var_exp))
#     #decimal_floor_max = np.max(my_floor(cum_var_exp, dec * (-1)))
#     decimal_floor_max = np.max(np.round(cum_var_exp, dec * (-1)))
#
#     if raw_floor_max > 100:
#         threshold = np.argmax(np.floor(cum_var_exp))
#     elif raw_floor_max == 100:
#         if decimal_floor_max > 100:
#             threshold = np.argmax(my_floor(cum_var_exp, dec * (-1)))
#             threshold = np.argmax(np.round(cum_var_exp, dec * (-1)))
#
#         elif decimal_floor_max == 100:
#             for idx in range(len(cum_var_exp)):
#                 if cum_var_exp[idx] > (100 + tol * 10):
#                     # print("cum var exp threshold ", idx)
#                     threshold = idx
#                     print("cum var exp threshold FOR ", threshold)
#                     break
#
#             if threshold <= 0:
#                 threshold = len(eval) - 1
#
#             print("ALL <= (100 + tol * 10)) threshold ", print("ALL <= (100 + tol * 10)) threshold ", threshold))
#             if all(cum_var_exp[threshold:] <= (100 + (tol * 10))):
#                 threshold = len(eval) - 1
#                 print("ALL <= (100 + tol * 10)) threshold ", threshold)
#             else:
#                 print("tol ", tol)
#                 print("np.min([1e-5, tol]) ", np.min([1e-5, tol]))
#                 cut_threshold = np.argmax(np.array(var_exp) < np.min([1e-5, tol]))  # 1e-5)
#                 cut_threshold -= 1
#                 print("CUT threshold ", cut_threshold)
#                 if cut_threshold < threshold:  # and not threshold_set:
#                     threshold = cut_threshold
#
#                 threshold = cut_threshold
#
#     #
#     #
#     #
#     #
#     # if np.max(np.floor(cum_var_exp)) == 100:
#     #     threshold = np.argmax(my_floor(cum_var_exp, dec * (-1)))
#     #     print("MY floor threshold ", threshold)
#     #
#     #     for idx in range(len(cum_var_exp)):
#     #         if cum_var_exp[idx] > (100 + tol * 10):
#     #             #print("cum var exp threshold ", idx)
#     #             threshold = idx
#     #             print("cum var exp threshold FOR ", threshold)
#     #             break
#     #
#     #     if threshold <= 0:
#     #         threshold = len(eval) - 1
#     #
#     #     print("ALL <= (100 + tol * 10)) threshold ", print("ALL <= (100 + tol * 10)) threshold ", threshold))
#     #     if all(cum_var_exp[threshold:] <= (100 + (tol * 10))):
#     #         threshold = len(eval) - 1
#     #         print("ALL <= (100 + tol * 10)) threshold ", threshold)
#     #     else:
#     #         print("tol ", tol)
#     #         print("np.min([1e-5, tol]) ", np.min([1e-5, tol]))
#     #         cut_threshold = np.argmax(np.array(var_exp) < np.min([1e-5, tol]))#1e-5)
#     #         cut_threshold -= 1
#     #         print("CUT threshold ", cut_threshold)
#     #         if cut_threshold < threshold:  # and not threshold_set:
#     #             threshold = cut_threshold
#     #
#     #         threshold = cut_threshold
#     #
#     # else:
#     #     threshold = np.argmax(np.floor(cum_var_exp))
#     #     print("floor threshold ", threshold)
#     #
#     # max_cum = np.max(my_floor(cum_var_exp, dec*(-1)))
#     # if max_cum > 100:
#     #     threshold = np.argmax(my_floor(cum_var_exp, dec*(-1)))#np.floor(cum_var_exp))
#     #     print("floor threshold ", threshold)
#     # else:
#     #     for idx in range(len(cum_var_exp)):
#     #         if cum_var_exp[idx] > (100 + tol * 10):
#     #             #print("cum var exp threshold ", idx)
#     #             threshold = idx
#     #             print("cum var exp threshold FOR ", threshold)
#     #             break
#     #
#     #     if threshold <= 0:
#     #         threshold = len(eval) - 1
#     #
#     #     print("ALL <= (100 + tol * 10)) threshold ", print("ALL <= (100 + tol * 10)) threshold ", threshold))
#     #     if all(cum_var_exp[threshold:] <= (100 + (tol * 10))):
#     #         threshold = len(eval) - 1
#     #         print("ALL <= (100 + tol * 10)) threshold ", threshold)
#     #     else:
#     #         print("tol ", tol)
#     #         print("np.min([1e-5, tol]) ", np.min([1e-5, tol]))
#     #         cut_threshold = np.argmax(np.array(var_exp) < np.min([1e-5, tol]))#1e-5)
#     #         cut_threshold -= 1
#     #         print("CUT threshold ", cut_threshold)
#     #         if cut_threshold < threshold:  # and not threshold_set:
#     #             threshold = cut_threshold
#     #
#     #         threshold = cut_threshold
#     #
#     # print("computed threshold ", threshold)
#
#     threshold_set = False
#     # if threshold == len(eval)-1:
#     #     threshold_set = True
#
#     #threshold = 0#cut_threshold -10
#
#     if threshold <= 0:
#         threshold = len(eval) - 1
#         threshold_set = True
#
#     # if threshold > 30:
#     #     threshold = 30
#
#
#
#     cum_var_exp = np.floor(cum_var_exp)#, 2)
#     #print("np.round(cum_var_exp, 2) ", cum_var_exp)
#
#     # threshold = 0
#     # maximum = 0
#     # for idx in range(len(cum_var_exp)):
#     #     if cum_var_exp[idx] > maximum:
#     #         print("cum var exp threshold ", idx)
#     #         threshold = idx
#     #         maximum = cum_var_exp[idx]
#     #         break
#     #
#     # print("maximum ", maximum)
#     # print("maximum threshold ", maximum)
#
#     #threshold = np.argmax(cum_var_exp)
#
#     # print("np.floor(cum_var_exp) ",cum_var_exp)
#     # print("np.floor(cum_var_exp).argmax(axis=0) ", cum_var_exp.argmax(axis=0))
#
#     ##########!!!!! previous version
#     #mx = np.max(cum_var_exp)
#     ############!!!!!!!!!!
#
#
#     # mx_index = np.argmax(cum_var_exp < (100.1))
#     # if mx_index == 0:
#     #     mx_index = len(eval) - 1
#     # print("mx index ", mx_index)
#     # mx = cum_var_exp[mx_index]
#     # print("mx ", mx)
#
#     #threshold = np.max([i for i, j in enumerate(cum_var_exp) if j == mx])
#
#
#     # print("all(cum_var_exp[threshold:] == mx) ", all(cum_var_exp[threshold:] == mx))
#     #
#     # cut_threshold = np.argmax(np.array(var_exp) < 1e-5)
#     # # cut_threshold = np.argmax(np.array(var_exp) < tol)
#     #
#     # print("cut threshold ", cut_threshold)
#
#     ### !!!! previous
#     # if all(cum_var_exp[threshold:] == mx):
#     #     threshold = len(cum_var_exp) - 1
#     #     #print("np.array(np.abs(var_exp)) ", np.array(np.abs(var_exp)))
#     #     threshold = np.argmax(np.array(np.abs(var_exp)) < 1e-5)
#     # else:
#     #     ##### !!!!!
#     #
#     #     threshold = mx_index
#
#     # if threshold == 0:
#     #     threshold = len(eval) - 1
#     #
#     # print("threshold ", threshold)
#
#     # print("threshold if threshold < cut_threshold else cut_threshold ", threshold if threshold < cut_threshold else cut_threshold)
#     # if cut_threshold < threshold:# and not threshold_set:
#     #     threshold = cut_threshold
#     #
#     # #threshold = threshold if threshold < cut_threshold else cut_threshold
#     # print("threshold after if ", threshold)
#
#     #threshold = cut_threshold
#
#     # if threshold == 0:
#     #     threshold = len(eval) - 1
#     #
#     # #exit()
#
#     threshold += 1
#
#     #threshold = 35
#
#     print("tol ", tol)
#
#     #threshold = 9#len(new_eig_pairs)
#     print("THreshold ", threshold)
#
#     # for pair in eig_pairs:
#     #     print("evec ", pair[1])
#
#     new_evec = np.hstack(np.array([eig_pair[1].reshape(len(eval), 1) for eig_pair in eig_pairs[:threshold]]))
#     new_eval = np.hstack(np.array([eig_pair[0] for eig_pair in eig_pairs[:threshold]]))
#
#     threshold = len(new_eval)-1
#
#     print_cumul(new_eval)
#
#     # cut eigen values under treshold
#     # new_eval = eval[threshold:]
#     # new_evec = evec[:, threshold:]
#
#     eval = np.flip(new_eval, axis=0)
#     evec = np.flip(new_evec, axis=1)
#
#     eval = new_eval
#     evec = new_evec
#
#
#
#     #print("evec ", evec)
#
#     # for i in range(len(original_eval)):
#     #     threshold = len(original_eval) - i
#     #     print("THRESHOLD ", threshold)
#     #
#     #     evec = np.hstack(np.array([eig_pair[1].reshape(len(eval), 1) for eig_pair in eig_pairs[:threshold]]))
#     #
#     #     for ev in evec:
#     #         print("np.linalg.norm(ev) ", np.linalg.norm(ev))
#     #         testing.assert_array_almost_equal(1.0, np.linalg.norm(ev), decimal=0)
#     #     print('Everything ok!')
#     #
#     # exit()
#
#     # print("evec ", evec)
#     #
#     #
#     # for ev in evec:
#     #     print("ev")
#     #     print("np.linalg.norm(ev) ", np.linalg.norm(ev))
#     #     #testing.assert_array_almost_equal(1.0, np.linalg.norm(ev), decimal=0)
#     # print('Everything ok!')
#
#     return eval, evec, threshold, original_eval,None# new_evec

def _pca_(cov_center, tol):
    from numpy import ma
    eval, evec = np.linalg.eigh(cov_center)

    original_eval = eval
    print("original eval ", original_eval)
    #
    # print("original evec ")
    # print(pd.DataFrame(evec))

    cum_var_exp, var_exp = print_cumul(sorted(eval, reverse=True))
    print("CUM VAR EXP ", cum_var_exp)

    eval = np.flip(eval, axis=0)
    evec = np.flip(evec, axis=1)

    eig_pairs = [(np.abs(eval[i]), evec[:, i]) for i in range(len(eval))]

    # threshold = np.argmax(cum_var_exp > 110)
    # if threshold == 0:
    #     threshold = len(eval)
    # #threshold = len(eval)

    cumul_hundred = np.argmax(cum_var_exp == 100)
    #print("cumul hundred ", cumul_hundred)

    # cut_threshold = np.argmax(np.array(var_exp) < 1e-5)
    # cum_var_exp, var_exp = print_cumul(eval[:cut_threshold])
    # print("new cum var exp ", cum_var_exp)

    ######!!!!!! previous

    print("np.max(np.floor(cum_var_exp))", )

    threshold = 0

    max_cum = np.max(np.floor(cum_var_exp))
    if max_cum > 100:
        threshold = np.argmax(np.floor(cum_var_exp))
    else:
        for idx in range(len(cum_var_exp)):
            if cum_var_exp[idx] > (100 + tol * 10):
                #print("cum var exp threshold ", idx)
                threshold = idx
                print("cum var exp threshold FOR ", threshold)
                break

        if threshold <= 0:
            threshold = len(eval) - 1

        print("ALL <= (100 + tol * 10)) threshold ", print("ALL <= (100 + tol * 10)) threshold ", threshold))
        if all(cum_var_exp[threshold:] <= (100 + tol * 10)):
            threshold = len(eval) - 1
            print("ALL <= (100 + tol * 10)) threshold ", threshold)
        else:
            print("tol ", tol)
            print("np.min([1e-5, tol]) ", np.min([1e-5, tol]))
            cut_threshold = np.argmax(np.array(var_exp) < np.min([1e-5, tol]))#1e-5)
            cut_threshold -= 1
            print("CUT threshold ", cut_threshold)
            if cut_threshold < threshold:  # and not threshold_set:
                threshold = cut_threshold

            threshold = cut_threshold

    print("computed threshold ", threshold)

    threshold_set = False
    # if threshold == len(eval)-1:
    #     threshold_set = True

    #threshold = cut_threshold -10

    if threshold <= 0:
        threshold = len(eval) - 1
        threshold_set = True

    cum_var_exp = np.floor(cum_var_exp)#, 2)
    #print("np.round(cum_var_exp, 2) ", cum_var_exp)

    # threshold = 0
    # maximum = 0
    # for idx in range(len(cum_var_exp)):
    #     if cum_var_exp[idx] > maximum:
    #         print("cum var exp threshold ", idx)
    #         threshold = idx
    #         maximum = cum_var_exp[idx]
    #         break
    #
    # print("maximum ", maximum)
    # print("maximum threshold ", maximum)

    #threshold = np.argmax(cum_var_exp)

    # print("np.floor(cum_var_exp) ",cum_var_exp)
    # print("np.floor(cum_var_exp).argmax(axis=0) ", cum_var_exp.argmax(axis=0))

    ##########!!!!! previous version
    #mx = np.max(cum_var_exp)
    ############!!!!!!!!!!


    # mx_index = np.argmax(cum_var_exp < (100.1))
    # if mx_index == 0:
    #     mx_index = len(eval) - 1
    # print("mx index ", mx_index)
    # mx = cum_var_exp[mx_index]
    # print("mx ", mx)

    #threshold = np.max([i for i, j in enumerate(cum_var_exp) if j == mx])


    # print("all(cum_var_exp[threshold:] == mx) ", all(cum_var_exp[threshold:] == mx))
    #
    # cut_threshold = np.argmax(np.array(var_exp) < 1e-5)
    # # cut_threshold = np.argmax(np.array(var_exp) < tol)
    #
    # print("cut threshold ", cut_threshold)

    ### !!!! previous
    # if all(cum_var_exp[threshold:] == mx):
    #     threshold = len(cum_var_exp) - 1
    #     #print("np.array(np.abs(var_exp)) ", np.array(np.abs(var_exp)))
    #     threshold = np.argmax(np.array(np.abs(var_exp)) < 1e-5)
    # else:
    #     ##### !!!!!
    #
    #     threshold = mx_index

    # if threshold == 0:
    #     threshold = len(eval) - 1
    #
    # print("threshold ", threshold)

    # print("threshold if threshold < cut_threshold else cut_threshold ", threshold if threshold < cut_threshold else cut_threshold)
    # if cut_threshold < threshold:# and not threshold_set:
    #     threshold = cut_threshold
    #
    # #threshold = threshold if threshold < cut_threshold else cut_threshold
    # print("threshold after if ", threshold)

    #threshold = cut_threshold

    # if threshold == 0:
    #     threshold = len(eval) - 1
    #
    # #exit()

    threshold += 1

    print("tol ", tol)

    #threshold = 9#len(new_eig_pairs)
    print("THreshold ", threshold)

    # for pair in eig_pairs:
    #     print("evec ", pair[1])

    new_evec = np.hstack(np.array([eig_pair[1].reshape(len(eval), 1) for eig_pair in eig_pairs[:threshold]]))
    new_eval = np.hstack(np.array([eig_pair[0] for eig_pair in eig_pairs[:threshold]]))

    threshold = len(new_eval)-1

    print_cumul(new_eval)

    # cut eigen values under treshold
    # new_eval = eval[threshold:]
    # new_evec = evec[:, threshold:]

    eval = np.flip(new_eval, axis=0)
    evec = np.flip(new_evec, axis=1)

    eval = new_eval
    evec = new_evec

    #print("evec ", evec)

    # for i in range(len(original_eval)):
    #     threshold = len(original_eval) - i
    #     print("THRESHOLD ", threshold)
    #
    #     evec = np.hstack(np.array([eig_pair[1].reshape(len(eval), 1) for eig_pair in eig_pairs[:threshold]]))
    #
    #     for ev in evec:
    #         print("np.linalg.norm(ev) ", np.linalg.norm(ev))
    #         testing.assert_array_almost_equal(1.0, np.linalg.norm(ev), decimal=0)
    #     print('Everything ok!')
    #
    # exit()

    # print("evec ", evec)
    #
    #
    # for ev in evec:
    #     print("ev")
    #     print("np.linalg.norm(ev) ", np.linalg.norm(ev))
    #     #testing.assert_array_almost_equal(1.0, np.linalg.norm(ev), decimal=0)
    # print('Everything ok!')

    return eval, evec, threshold, original_eval,None# new_evec


def _pca(cov_center, tol):
    from numpy import ma
    print("tol ", tol)
    eval, evec = np.linalg.eigh(cov_center)

    original_eval = eval
    print("original eval ", original_eval)
    #
    # print("original evec ")
    # print(pd.DataFrame(evec))

    cum_var_exp, var_exp = print_cumul(sorted(eval, reverse=True))
    print("CUM VAR EXP ", cum_var_exp)

    # cum_var_exp, var_exp = print_cumul(sorted(np.abs(eval), reverse=True))
    # print("ABS CUM VAR EXP ", cum_var_exp)

    eval = np.flip(eval, axis=0)
    evec = np.flip(evec, axis=1)

    eig_pairs = [(np.abs(eval[i]), evec[:, i]) for i in range(len(eval))]

    # threshold = np.argmax(cum_var_exp > 110)
    # if threshold == 0:
    #     threshold = len(eval)
    # #threshold = len(eval)

    cumul_hundred = np.argmax(cum_var_exp == 100)
    #print("cumul hundred ", cumul_hundred)

    # cut_threshold = np.argmax(np.array(var_exp) < 1e-5)
    # cum_var_exp, var_exp = print_cumul(eval[:cut_threshold])
    # print("new cum var exp ", cum_var_exp)

    cut = False

    ######!!!!!! previous
    threshold = 0
    for idx in range(len(cum_var_exp)):
        if cum_var_exp[idx] > (100 + tol * 10):
            #print("cum var exp threshold ", idx)
            threshold = idx
            print("cum var exp threshold FOR ", threshold)
            break

    if threshold == 0:
        threshold = len(eval) - 1

    #print("ALL <= (100 + tol * 10)) threshold ", print("ALL <= (100 + tol * 10)) threshold ", threshold))
    if all(cum_var_exp[threshold:] <= (100 + tol * 10)):
        threshold = len(eval) - 1
        print("ALL <= (100 + tol * 10)) threshold ", threshold)
    else:
        print("np.min([1e-5, tol]) ", np.min([1e-5, tol]))
        cut_threshold = np.argmax(np.array(var_exp) < np.min([1e-5, tol]))#1e-5)
        print("CUT threshold ", cut_threshold)
        if cut_threshold < threshold:  # and not threshold_set:
            threshold = cut_threshold
            cut = True
    # threshold = cut_threshold
    # print("computed threshold ", threshold)

    threshold_set = False
    # if threshold == len(eval)-1:
    #     threshold_set = True

    print("cut: {}, threshold: {}".format(cut, threshold))

    # There is cut on cumul value, so cut it from original eig pairs
    if cut is False and threshold != (len(eval) - 1):
        eig_pairs = [(eval[i], evec[:, i]) for i in range(len(eval))]

    if threshold == 0:
        threshold = len(eval) - 1
        threshold_set = True

    cum_var_exp = np.floor(cum_var_exp)#, 2)
    #print("np.round(cum_var_exp, 2) ", cum_var_exp)

    threshold += 1

    #threshold = 35


    #threshold = 9#len(new_eig_pairs)
    print("THreshold ", threshold)

    # for pair in eig_pairs:
    #     print("evec ", pair[1])

    print("cut ", cut)



    new_evec = np.hstack(np.array([eig_pair[1].reshape(len(eval), 1) for eig_pair in eig_pairs[:threshold]]))
    new_eval = np.hstack(np.array([eig_pair[0] for eig_pair in eig_pairs[:threshold]]))

    threshold = len(new_eval)-1

    print_cumul(new_eval)

    # cut eigen values under treshold
    # new_eval = eval[threshold:]
    # new_evec = evec[:, threshold:]

    eval = np.flip(new_eval, axis=0)
    evec = np.flip(new_evec, axis=1)

    eval = new_eval
    evec = new_evec



    #print("evec ", evec)

    # for i in range(len(original_eval)):
    #     threshold = len(original_eval) - i
    #     print("THRESHOLD ", threshold)
    #
    #     evec = np.hstack(np.array([eig_pair[1].reshape(len(eval), 1) for eig_pair in eig_pairs[:threshold]]))
    #
    #     for ev in evec:
    #         print("np.linalg.norm(ev) ", np.linalg.norm(ev))
    #         testing.assert_array_almost_equal(1.0, np.linalg.norm(ev), decimal=0)
    #     print('Everything ok!')
    #
    # exit()

    # print("evec ", evec)
    #
    #
    # for ev in evec:
    #     print("ev")
    #     print("np.linalg.norm(ev) ", np.linalg.norm(ev))
    #     #testing.assert_array_almost_equal(1.0, np.linalg.norm(ev), decimal=0)
    # print('Everything ok!')

    return eval, evec, threshold, original_eval,None# new_evec


def _pca_add_one(cov_center, tol, moments):
    eval, evec = np.linalg.eigh(cov_center)

    cum_var_exp = print_cumul(sorted(eval, reverse=True))

    original_eval = eval
    diag_value = tol - np.min([np.min(eval), 0])  # np.abs((np.min(eval) - tol))
    diagonal = np.zeros(moments.size)
    print("diag value ", diag_value)

    diagonal[1:] += diag_value
    diag = np.diag(diagonal)
    eval += diagonal

    #cum_var_exp = print_cumul(sorted(eval, reverse=True))

    eig_pairs = [(eval[i], evec[:, i]) for i in range(len(eval))]

    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs.sort(key=lambda x: x[0], reverse=True)

    # print("EVAL SORTED ", sorted(eval, reverse=True))
    # print("EVAL EIG PAIR ", np.hstack(np.array([eig_pair[0] for eig_pair in eig_pairs[:]])))
    # cum_var_exp = print_cumul(np.hstack(np.array([eig_pair[0] for eig_pair in eig_pairs[:]])))

    threshold = np.argmax(cum_var_exp > 100)
    if threshold == 0:
        threshold = len(cum_var_exp)

    print("max eval index: {}, threshold: {}".format(len(eval) - 1, threshold))

    new_evec = np.hstack(np.array([eig_pair[1].reshape(len(eval), 1) for eig_pair in eig_pairs[:threshold]]))

    new_eval = np.hstack(np.array([eig_pair[0] for eig_pair in eig_pairs[:threshold]]))

    threshold -= 1

    print_cumul(new_eval)

    # self.lsq_reconstruct(cov_center, fixed_eval, evec, treshold)

    # cut eigen values under treshold
    # new_eval = eval[threshold:]
    # new_evec = evec[:, threshold:]

    print("new eval", new_eval)
    print("new evec", new_evec)


    eval = np.flip(new_eval, axis=0)
    evec = np.flip(new_evec, axis=1)

    eval = new_eval
    evec = new_evec

    # print("eval flipped ", eval)
    # print("evec flipped ", evec)
    # exit()

    for ev in evec:
        print("np.linalg.norm(ev) ", np.linalg.norm(ev))
        testing.assert_array_almost_equal(1.0, np.linalg.norm(ev), decimal=0)
    print('Everything ok!')

    return eval, evec, threshold, original_eval, None#, matrix_w


def _cut_eigenvalues_to_constant(cov_center, tol):
    eval, evec = np.linalg.eigh(cov_center)
    original_eval = eval
    print("cut eigenvalues tol ", tol)

    # threshold given by eigenvalue magnitude
    threshold = np.argmax(eval > tol)

    # add the |smallest eigenvalue - tol(^2??)| + eigenvalues[:-1]

    #threshold = 0
    print("threshold ", threshold)

    #treshold, _ = self.detect_treshold(eval, log=True, window=8)

    # tresold by MSE of eigenvalues
    #treshold = self.detect_treshold_mse(eval, std_evals)

    # treshold

    #self.lsq_reconstruct(cov_center, fixed_eval, evec, treshold)
    print("original eval ", eval)
    print("threshold ", threshold)

    # cut eigen values under treshold
    eval[:threshold] = tol#eval[threshold]
    #new_evec = evec[:, threshold:]

    print("eval ")
    print(pd.DataFrame(eval))

    eval = np.flip(eval, axis=0)
    #print("eval ", eval)
    evec = np.flip(evec, axis=1)
    #print("evec ", evec)

    return eval, evec, threshold, original_eval


def _add_to_eigenvalues(cov_center, tol, moments):
    eval, evec = np.linalg.eigh(cov_center)

    # we need highest eigenvalues first
    eval = np.flip(eval, axis=0)
    evec = np.flip(evec, axis=1)

    original_eval = eval

    for ev in evec:
        print("np.linalg.norm(ev) ", np.linalg.norm(ev))
        testing.assert_array_almost_equal(1.0, np.linalg.norm(ev), decimal=0)
    print('Everything ok!')

    print_cumul(eval)


    # # Permutation
    # index = (np.abs(eval - 1)).argmin()
    # first_item = eval[0]
    # eval[0] = eval[index]
    # eval[index] = first_item
    #
    # selected_evec = evec[:, index]
    # first_evec = evec[:, 0]
    #
    # evec[:, 0] = selected_evec[:]
    # evec[:, index] = first_evec[:]

    alpha = 5
    diag_value = tol - np.min([np.min(eval), 0])  # np.abs((np.min(eval) - tol))

    #diag_value += diag_value * 5

    #print("diag value ", diag_value)
    diagonal = np.zeros(moments.size)

    #diag_value = 10

    print("diag value ", diag_value)

    diagonal[1:] += diag_value
    diag = np.diag(diagonal)
    eval += diagonal

    for ev in evec:
        print("np.linalg.norm(ev) ", np.linalg.norm(ev))
        testing.assert_array_almost_equal(1.0, np.linalg.norm(ev), decimal=0)
    print('Everything ok!')

    # print_cumul(eval)
    # exit()

    return eval, evec, original_eval


def construct_orthogonal_moments(moments, cov, tol=None, reg_param=0, orth_method=1, exact_cov=None):
    """
    For given moments find the basis orthogonal with respect to the covariance matrix, estimated from samples.
    :param moments: moments object
    :return: orthogonal moments object of the same size.
    """
    threshold = 0
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print("cov ")
        print(pd.DataFrame(cov))

    # print("cov matrix rank ", numpy.linalg.matrix_rank(cov))

    # centered covariance
    M = np.eye(moments.size)
    M[:, 0] = -cov[:, 0]
    cov_center = M @ cov @ M.T

    projection_matrix = None

    # print("centered cov ")
    # print(pd.DataFrame(cov_center))

    # Add const to eigenvalues
    if orth_method == 1:
        eval_flipped, evec_flipped, original_eval = _add_to_eigenvalues(cov_center, tol=tol, moments=moments)
        # print("eval flipped ")
        # print(pd.DataFrame(eval_flipped))
        # print("evec flipped ")
        # print(pd.DataFrame(evec_flipped))

    # Cut eigenvalues below threshold
    elif orth_method == 2:
        eval_flipped, evec_flipped, threshold, original_eval = _cut_eigenvalues(cov_center, tol=tol)
        # print("eval flipped ")
        # print(pd.DataFrame(eval_flipped))
        # print("evec flipped ")
        # print(pd.DataFrame(evec_flipped))
        # print("threshold ", threshold)
        #original_eval = eval_flipped

    # Add const to eigenvalues below threshold
    elif orth_method == 3:
        eval_flipped, evec_flipped, threshold, original_eval = _cut_eigenvalues_to_constant(cov_center, tol=tol)
        # print("eval flipped ")
        # print(pd.DataFrame(eval_flipped))
        # print("evec flipped ")
        # print(pd.DataFrame(evec_flipped))
        # print("threshold ", threshold)
        #original_eval = eval_flipped
    elif orth_method == 4:
        eval_flipped, evec_flipped, threshold, original_eval, projection_matrix = _pca(cov_center, tol=tol)
    elif orth_method == 5:
        eval_flipped, evec_flipped, threshold, original_eval, projection_matrix = \
            _pca_add_one(cov_center, tol=tol, moments=moments)
    elif orth_method == 6:
        eval_flipped, evec_flipped, threshold, original_eval, projection_matrix = \
            _svd_cut(cov_center, tol=tol)

    else:
        raise Exception("No eigenvalues method")

    #original_eval, _ = np.linalg.eigh(cov_center)

    # Compute eigen value errors.
    #evec_flipped = np.flip(evec, axis=1)
    #L = (evec_flipped.T @ M)
    #rot_moments = mlmc.moments.TransformedMoments(moments, L)
    #std_evals = eigenvalue_error(rot_moments)

    if projection_matrix is not None:
        icov_sqrt_t = projection_matrix
    else:
        # print("evec flipped ", evec_flipped)
        # print("eval flipped ", eval_flipped)
        #
        # print("evec_flipped * (1 / np.sqrt(eval_flipped))[None, :]")
        # print(pd.DataFrame(evec_flipped * (1 / np.sqrt(eval_flipped))[None, :]))

        icov_sqrt_t = M.T @ (evec_flipped * (1 / np.sqrt(eval_flipped))[None, :])

        # print("icov_sqrt_t")
        # print(pd.DataFrame(icov_sqrt_t))

        # try:
        #     eval, evec = np.linalg.eigh(icov_sqrt_t)
        #     cum_var_exp = print_cumul(sorted(eval, reverse=True))
        #     print("ICOV CUM ", cum_var_exp)
        # except:
        #     pass

    R_nm, Q_mm = sc.linalg.rq(icov_sqrt_t, mode='full')

    # check
    L_mn = R_nm.T
    if L_mn[0, 0] < 0:
        L_mn = -L_mn

    # if exact_cov is not None:
    #     print("H")
    #     print(pd.DataFrame(exact_cov))
    #
    #     cov_eval, cov_evec = np.linalg.eigh(cov)
    #     exact_cov_eval, exact_cov_evec = np.linalg.eigh(exact_cov)
    #
    #     cov_evec = np.flip(cov_evec, axis=1)
    #     exact_cov_evec = np.flip(exact_cov_evec, axis=1)
    #
    #     #print("cov evec ", cov_evec)
    #     #
    #     #print("exact_cov_evec ", exact_cov_evec)
    #
    #     #print("np.dot(cov_evec, exact_cov_evec) ", np.dot(cov_evec[-1], exact_cov_evec[-1]))
    #     print("einsum('ij,ij->i', cov_evec, exact_cov_evec) ", np.einsum('ij,ij->i', cov_evec, exact_cov_evec))
    #     #print("np.dot(cov_evec, exact_cov_evec) ", np.sum(np.dot(cov_evec, exact_cov_evec), axis=0))
    #     #exit()
    #
    #     print("Hn")
    #     print(pd.DataFrame(cov))
    #
    #     print("inv(L) @ inv(L.T)")
    #     print(pd.DataFrame(numpy.linalg.pinv(L_mn) @ numpy.linalg.pinv(L_mn.T)))
    #
    #     # print("inv(L) @ cov @ inv(L.T)")
    #     # print(pd.DataFrame(numpy.linalg.pinv(L_mn) @ cov @ numpy.linalg.pinv(L_mn.T)))
    #
    #     # print("M @ inv(L) @ cov @ inv(L.T) @ M")
    #     # print(pd.DataFrame(np.linalg.inv(M) @ numpy.linalg.pinv(L_mn) @ cov @ numpy.linalg.pinv(L_mn.T) @ np.linalg.inv(M)))
    #
    #     print("Cov centered")
    #     print(pd.DataFrame(cov_center))


    ortogonal_moments = mlmc.moments.TransformedMoments(moments, L_mn)

    #mlmc.tool.plot.moments(ortogonal_moments, size=ortogonal_moments.size, title=str(reg_param), file=None)
    #exit()

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
    info = (original_eval, eval_flipped, threshold, L_mn)
    return ortogonal_moments, info, cov_center


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
