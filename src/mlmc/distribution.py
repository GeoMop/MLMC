import numpy as np
import scipy as sc
import scipy.integrate as integrate
import scipy.stats as stats

class Distribution:
    """
    Calculation of the distribution
    TODO:
    - moments estimates must ignore outlayers, just count fraction of them
    - use mean and variance estimate (Monomials), to estimate the domain
    - 
    """
    def __init__(self, moments_obj, moment_data, mean, std, is_positive = False):
        """
        :param moments_fn: Function for calculating moments
        :param moment_data: Array  of moments or tuple (moments, variance).
        :param positive_distr: Indication of distribution for a positive variable.
        """
        # Family of moments basis functions.
        self.moments_basis = moments_obj

        # Moment evaluation function with bounded number of moments and their domain.
        self.moments_fn = None

        # Domain of the density approximation (and moment functions).
        self.domain = None

        # Approximation of moment values.
        self.moment_means = moment_data[:, 0]
        self.moment_vars = moment_data[:, 1]

        self.ref_norm_distr = stats.norm(loc=mean, scale= std)

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

    def choose_parameters_from_moments(self, mean, variance, quantile=0.99):
        """
        Determine model hyperparameters, in particular domain of the density function,
        from given samples.
        :param samples: np array of samples from the distribution or its approximation.
        :return: None
        """
        if self.is_positive:
            # approximate by log normal
            # compute mu, sigma parameters from observed mean and variance
            sigma_sq = np.log( np.exp(np.log(variance) - 2.0 * np.log(mean)) + 1.0)
            mu = np.log(mean) - sigma_sq / 2.0
            sigma = np.sqrt(sigma_sq)
            self.domain = tuple(sc.stats.lognorm.ppf([1.0 - quantile, quantile], s=sigma, scale = np.exp(mu)))
            assert np.isclose(mean, sc.stats.lognorm.mean(s=sigma, scale = np.exp(mu)))
            assert np.isclose(variance, sc.stats.lognorm.var(s=sigma, scale=np.exp(mu)))
        else:
            self.domain = tuple(sc.stats.norm.ppf([1.0 - quantile, quantile], loc=mean, scale=np.sqrt(variance)))


    def choose_parameters_from_approximation(self):
        pass



    def estimate_density(self, tol=None):
        """
        Run nonlinear iterative solver to estimate density, use previous solution as initial guess.
        :return: None
        """
        assert self.domain is not None
        if self.is_positive:
            self.domain = (max(0.0, self.domain[0]), self.domain[1])

        #self.moments_fn = lambda x, size=self.approx_size, a=self.domain[0], b=self.domain[1] : self.moments_basis(x, size, a, b)
        # tODO: How to choose number of QP, solution is quite sensitive to it
        self._n_quad_points = 200 * self.approx_size

        self.restricted_moments = self.approx_size
        assert tol is not None
        if self.multipliers is None:
            self.multipliers = np.ones(self.approx_size)



        result = sc.optimize.root(
            # Seems that system may have more inexact solutions as different methods
            method='hybr', # Best method of the package. Still need to introduce kind of regularisation to prevent oscilatory solutions.
            #method='lm', # better but when osciation appears it become even worse, posibly need some stabilization term,
            #method='broyden1', # Fails
            #method='broyden2',  # Fails
            #method='anderson', # Severe oscilations.

            fun=self._calculate_moments_approximation,
            x0=self.multipliers,
            jac=self._calculate_jacobian_matrix,
            tol=tol,
            callback = self._iteration_monitor
        )
        #result = sc.optimize.minimize

            #print(result)
        if result.success:
            self.multipliers = result.x
        else:
            print("Failed to converge.")
            print("Res: {}", np.linalg.norm(result.fun))
            print(result.message)

        return result

    def _iteration_monitor(self, x, f):
        print("Norm: {} x: {}".format(np.linalg.norm(f), x))


    def density(self, value):
        """
        :param value: float or np.array
        :return: density for passed value

        TODO:
        """
        moments = self.moments_fn(value)
        return np.exp( - np.sum(moments * self.multipliers, axis=1))

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

    def _calculate_cost_functional(self, approx):
        cost = np.sum(0.5 * self._calculate_moments_approximation(approx)**2 / self.moment_vars[:self.restricted_moments])
        + 0.01 * KL_divergence(self.ref_norm_distr, self.density, self.domain[0], self.domain[1])
        return cost

    def _calculate_moments_approximation(self, approx):
        """
        :param lagrangians: array, lagrangians parameters
        :return: array, moments approximation
        """

        def integrand(value, lg=approx):
            moments =  self.moments_fn(value)[:, :self.restricted_moments]
            density = np.exp( - np.sum(moments * lg, axis=1))
            return moments.T * density


        integral = sc.integrate.fixed_quad(integrand, self.domain[0], self.domain[1],
                                                n=self._n_quad_points)

        return (integral[0] - self.moment_means[:self.restricted_moments])

    def _calculate_jacobian_matrix(self, lagrangians):
        """
        :param lagrangians: np.array, lambda
        :return: jacobian matrix, symmetric,
        """
        triu_idx = np.triu_indices(self.restricted_moments)
        def integrand(value, lg=lagrangians, triu_idx = triu_idx):
            """
            Upper triangle of the matrix, flatten.
            """
            moments =  self.moments_fn(value)[:, :self.restricted_moments]
            density = np.exp( - np.sum(moments * lg, axis=1))
            moment_outer = np.einsum('ki,kj->ijk', moments, moments)
            triu_outer = moment_outer[triu_idx[0], triu_idx[1], :]
            return triu_outer * density

        # Initialization of matrix


        integral = sc.integrate.fixed_quad(integrand, self.domain[0], self.domain[1],
                                           n=self._n_quad_points)
        jacobian_matrix = np.empty(shape=(self.restricted_moments, self.restricted_moments))
        jacobian_matrix[triu_idx[0], triu_idx[1]] = -integral[0]
        jacobian_matrix[triu_idx[1], triu_idx[0]] = -integral[0]
        return jacobian_matrix



def compute_exact_moments(moments_fn, density,  tol=1e-4):
    """
    Compute approximation of moments using exact density.
    :param moments_fn: Moments function.
    :param n_moments: Number of mements to compute.
    :param density: Density function (must accept np vectors).
    :param a, b: Integral bounds, approximate integration over R.
    :param tol: Tolerance of integration.
    :return: np.array, moment values
    """
    integrand = lambda x: moments_fn(x).T * density(x)
    a, b = moments_fn.domain
    last_integral = integrate.fixed_quad(integrand, a, b, n=moments_fn.size)[0]

    n_points = 2*moments_fn.size
    integral = integrate.fixed_quad(integrand, a, b, n=n_points)[0]
    while np.linalg.norm(integral - last_integral) > tol:
        last_integral = integral
        n_points *= 2
        integral = integrate.fixed_quad(integrand, a, b, n=n_points)[0]
    return integral


def KL_divergence(prior_density, posterior_density, a, b):
    """
    \int_R P(x) \log( P(X)/Q(x)) \dx
    :param prior_density: Q
    :param posterior_density: P
    :return: KL divergence value
    """
    integrand = lambda x: posterior_density(x) * np.log( posterior_density(x) / prior_density(x) )
    return integrate.quad(integrand, a, b)

def L2_distance(prior_density, posterior_density, a, b):
    integrand = lambda x: (posterior_density(x) -  prior_density(x))**2
    return np.sqrt( integrate.quad(integrand, a, b) )

