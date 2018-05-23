import numpy as np
import scipy as sc

class Distribution:
    """
    Calculation of the distribution
    """
    def __init__(self, moments_fn, moment_data, positive_distr = False):
        """
        :param moments_fn: Function for calculating moments
        :param moment_data: Array  of moments or tuple (moments, variance).
        :param positive_distr: Indication of distribution for a positive variable.
        """
        # Family of moments basis functions.
        self.moments_basis = moments_fn

        # Moment evaluation function with bounded number of moments and their domain.
        self.moments_fn = None

        # Domain of the density approximation (and moment functions).
        self.domain = None

        # Approximation of moment values.
        if type(moment_data) is tuple:
            self.moment_means, self.moment_vars = moment_data
        else:
            self.moment_means, self.moment_vars = moment_data, np.ones_like(self.moment_means)


        # Force density with positive support.
        self.is_positive = positive_distr

        # Approximation paramters. Lagrange multipliers for moment equations.
        self.multipliers = None
        # Number of basis functions to approximate the density.
        # In future can be smaller then number of provided approximative moments.
        self.approx_size = len(self.moment_means)


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


    def estimate_density(self, tolerance=None):
        """
        Run nonlinear iterative solver to estimate density, use previous solution as initial guess.
        :return: None
        """
        assert self.domain is not None
        if self.is_positive:
            self.domain = (max(0.0, self.domain[0]), self.domain[1])

        self.moments_fn = lambda x, size=self.approx_size, a=self.domain[0], b=self.domain[1] : self.moments_basis(x, size, a, b)
        self._n_quad_points = 4 * self.approx_size


        assert tolerance is not None
        if self.multipliers is None:
            self.multipliers = np.ones(self.approx_size)



        result = sc.optimize.root(
            fun=self._calculate_moments_approximation,
            x0=self.multipliers,
            jac=self._calculate_jacobian_matrix,
            tol=tolerance,
            callback = self._iteration_monitor
        )

        if result.success:
            self.multipliers = result.x
        else:
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

    def _calculate_moments_approximation(self, approx):
        """
        :param lagrangians: array, lagrangians parameters
        :return: array, moments approximation
        """

        def integrand(value, lg=approx):
            moments =  self.moments_fn(value)
            density = np.exp( - np.sum(moments * lg, axis=1))
            return moments.T * density


        integral = sc.integrate.fixed_quad(integrand, self.domain[0], self.domain[1],
                                                n=self._n_quad_points)
        return (integral[0] - self.moment_means)

    def _calculate_jacobian_matrix(self, lagrangians):
        """
        :param lagrangians: np.array, lambda
        :return: jacobian matrix, symmetric,
        """
        triu_idx = np.triu_indices(self.approx_size)
        def integrand(value, lg=lagrangians, triu_idx = triu_idx):
            """
            Upper triangle of the matrix, flatten.
            """
            moments =  self.moments_fn(value)
            density = np.exp( - np.sum(moments * lg, axis=1))
            moment_outer = np.einsum('ki,kj->ijk', moments, moments)
            triu_outer = moment_outer[triu_idx[0], triu_idx[1], :]
            return triu_outer * density

        # Initialization of matrix


        integral = sc.integrate.fixed_quad(integrand, self.domain[0], self.domain[1],
                                           n=self._n_quad_points)
        jacobian_matrix = np.empty(shape=(self.approx_size, self.approx_size))
        jacobian_matrix[triu_idx[0], triu_idx[1]] = -integral[0]
        jacobian_matrix[triu_idx[1], triu_idx[0]] = -integral[0]
        return jacobian_matrix
