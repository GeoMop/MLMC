import autograd.numpy as np
import numpy
from scipy import optimize
from autograd import elementwise_grad as egrad
import matplotlib.pyplot as plt
import scipy.stats as st

class SplineApproximation:

    def __init__(self, mlmc, domain, poly_degree, accuracy):
        self.mlmc = mlmc
        self.domain = domain
        self.poly_degree = poly_degree
        self.smoothing_factor = np.zeros(self.mlmc.n_levels)
        self.accuracy = accuracy
        self.interpolation_points = []
        self.polynomial = None
        self.moments_fn = None

    def determine_interpolation_points(self, n_points):
        self.interpolation_points = np.linspace(self.domain[0], self.domain[1], n_points) #@TODO: how to determine number of interpolation points

    def compute_smoothing_factor(self, data, level_id):
        result = []
        res = 0

        print("self polynomial ", self.polynomial)

        def functional(x, data, s):
            return np.abs(np.sum(self.polynomial((data - s) / x) - self.indicator(s, data))) / len(data) - self.accuracy/2

        for s in self.interpolation_points:
            try:
                res = optimize.newton(functional, x0=0.1, args=(data, s), tol=1e-5)
                #res = optimize.minimize(functional, [0.01], jac=egrad(functional), method='trust-ncg', args=(data, s), tol=1e-5)
            except Exception as e:
                print("Exceptation ", e)
                print("res= ", res)
                print("Compute smoothing factor optimization failed")

            result.append(res)

        print("smoothing factor result ", result)

        self.smoothing_factor[level_id] = np.max(result)

    def _create_smooth_polynomial(self):
        coeficients_matrix = np.empty((self.poly_degree, self.poly_degree))
        constants_matrix = np.empty(self.poly_degree)

        coeficients_matrix[0] = np.ones(self.poly_degree)
        coeficients_matrix[1] = [1 if i % 2 == 0 else -1 for i in range(self.poly_degree)]
        constants_matrix[0] = 0
        constants_matrix[1] = 1

        for j in range(self.poly_degree - 2):
            coeficients_matrix[j + 2] = np.array(
                [(1 ** (i + j + 1) - (-1) ** (i + j + 1)) / (i + j + 1) for i in range(self.poly_degree)])
            constants_matrix[j + 2] = (-1) ** j / (j + 1)

        poly_coefs = numpy.linalg.solve(coeficients_matrix, constants_matrix)
        self.polynomial = np.poly1d(poly_coefs)

    def smooth(self, interpolation_point, data):
        data = (data - interpolation_point) / self._level_smoothing_factor

        if self.polynomial is None:
            self._create_smooth_polynomial()

        return self._polynomial_smoothing(data)

    def _polynomial_smoothing(self, data):
        result = np.zeros(len(data))
        result[(data < -1)] = 1
        indices = (-1 <= data) & (data <= 1)
        data = data[indices]

        if len(data) > 0:
            result[indices] = self.polynomial(data)
        return result

    def indicator(self, interpolation_point, data):
        return np.array([int(d <= interpolation_point) for d in data])

    def lagrange_basis_polynomial(self, x, j):
        product = 1
        data = self.interpolation_points

        for m in range(len(data)):
            if j == m:
                continue
            product *= (x - data[m]) / (data[j] - data[m])

        return product

    def density(self, X):
        return egrad(self.cdf)(X)
        # dx = X[1] - X[0]
        # print("X[1] ", X[1])
        # print("X[0] ", X[0])
        # print("X[1] - X[0] ", X[1] - X[0])
        # print("dx" ,dx)
        # exit()
        # return np.diff(self.cdf(X)) / dx

    # def cdf(self, X, data):
    #     self.determine_interpolation_points(len(data) / 10)
    #
    #     distribution = np.empty(len(X))
    #     for index, x in enumerate(X):
    #         sum_ind_lag = 0
    #         for n, s in enumerate(self.interpolation_points):
    #             sum_ind_lag += np.sum(self.indicator(s, data))/len(data) * self.lagrange_basis_polynomial(x, n)
    #
    #         if type(sum_ind_lag).__name__ == 'ArrayBox':
    #             sum_ind_lag = sum_ind_lag._value
    #         distribution[index] = sum_ind_lag
    #
    #     #return distribution
    #     mask = (distribution >= 0) & (distribution <= 1)
    #     distr_sorted = np.sort(distribution[(distribution >= 0) & (distribution <= 1)])
    #     self.mask = mask
    #     return distr_sorted

    # def cdf_smoothing(self, X):
    #     self.determine_interpolation_points(len(self.data)/10)
    #     # print("interpolation points ", self.interpolation_points)
    #
    #     distribution = np.empty(len(X))
    #     for index, x in enumerate(X):
    #         sum_ind_lag = 0
    #         for n, s in enumerate(self.interpolation_points):
    #             print("self.smoothing_function(s) ", self.smoothing_function(s))
    #             print("self.indicator_func(s) ", self.indicator_func(s))
    #
    #             sum_ind_lag += self.smoothing_function(s) * self.lagrange_basis_polynomial(x, n)
    #         print("sum ind lag ", sum_ind_lag)
    #
    #         if type(sum_ind_lag).__name__ == 'ArrayBox':
    #             sum_ind_lag = sum_ind_lag._value
    #         distribution[index] = sum_ind_lag
    #
    #     #return distribution
    #
    #     print("distribution ", distribution)
    #     mask = (distribution >= 0) & (distribution <= 1)
    #     distr_sorted = np.sort(distribution[(distribution >= 0) & (distribution <= 1)])
    #     print("distr_sorted ", distr_sorted)
    #
    #     self.mask = mask
    #     return distr_sorted

    def cdf(self, x):
        sum_ind_level = np.zeros(len(self.interpolation_points))
        lagrange_poly = []
        lagrange_poly_set = False

        for level in self.mlmc.levels:
            moments = level.evaluate_moments(self.moments_fn)
            fine_values = np.squeeze(moments[0])[:, 1]
            fine_values = self.moments_fn.inv_linear(fine_values)
            coarse_values = np.squeeze(moments[1])[:, 1]
            coarse_values = self.moments_fn.inv_linear(coarse_values)
            #
            # if self.smoothing_factor[level._level_idx] == 0 and self.ind_method.__name__ == "smooth":
            #     self.compute_smoothing_factor(fine_values, level._level_idx)

            # self._level_smoothing_factor = self.accuracy**(1/(self.poly_degree + 1))
            self._level_smoothing_factor = 1#self.smoothing_factor[level._level_idx]

            #print("self. level smoothing factor ", self._level_smoothing_factor)

            for n, s in enumerate(self.interpolation_points):
                if level._level_idx == 0:
                    fine_indic = self.ind_method(s, fine_values)
                    int_point_mean = np.sum(fine_indic) / len(fine_values)
                else:
                    fine_indic = self.ind_method(s, fine_values)
                    coarse_indic = self.ind_method(s, coarse_values)
                    int_point_mean = np.sum(fine_indic - coarse_indic) / len(fine_values)

                sum_ind_level[n] += int_point_mean

                if not lagrange_poly_set:
                    lagrange_poly.append(self.lagrange_basis_polynomial(x, n))
            lagrange_poly_set = True

            #print("sum ind level ", sum_ind_level)

        lagrange_poly = np.array(lagrange_poly)


        # print("lagrange poly T", lagrange_poly.T)
        #
        # print("sum_ind_level * self.lagrange_basis_polynomial(x, n) ", sum_ind_level * lagrange_poly.T)


        #print("np.sum(sum_ind_level * self.lagrange_basis_polynomial(x, n)) ", np.sum(sum_ind_level * self.lagrange_basis_polynomial(x, n)))
        return np.sum(sum_ind_level * lagrange_poly.T)

    def cdf_(self, x):
        sum_ind_lagr = 0
        for n, s in enumerate(self.interpolation_points):

            sum_ind_level = 0
            for level in self.mlmc.levels:
                moments = level.evaluate_moments(self.moments_fn)
                fine_values = np.squeeze(moments[0])[:, 1]
                fine_values = self.moments_fn.inv_linear(fine_values)
                coarse_values = np.squeeze(moments[1])[:, 1]
                coarse_values = self.moments_fn.inv_linear(coarse_values)

                # if self.smoothing_factor[level._level_idx] == 0 and self.ind_method.__name__ == "smooth":
                #     self.compute_smoothing_factor(fine_values, level._level_idx)
                #     print("delta determined")
                #     print("smoothing factor ", self.smoothing_factor)

                self._level_smoothing_factor = self.accuracy**(1/(self.poly_degree + 1)) # self.accuracy#self.smoothing_factor[level._level_idx]
                if level._level_idx == 0:
                    fine_indic = self.ind_method(s, fine_values)
                    level_indic_mean = np.sum(fine_indic) / len(fine_values)
                    sum_ind_level += level_indic_mean
                else:
                    fine_indic = self.ind_method(s, fine_values)
                    coarse_indic = self.ind_method(s, coarse_values)

                    level_indic_mean = np.sum(fine_indic - coarse_indic) / len(fine_values)
                    sum_ind_level += level_indic_mean

            print("self.lagrange_basis_polynomial(x, n) ", self.lagrange_basis_polynomial(x, n))


            sum_ind_lagr += sum_ind_level * self.lagrange_basis_polynomial(x, n)

        return sum_ind_lagr

    # def _test_smoothing_polynomial(self):
    #     degrees = [3, 5, 7, 9, 11]
    #     X = np.linspace(-1, 1, 1000)
    #
    #     for d in degrees:
    #         self.poly_degree = d
    #         self._create_smooth_polynomial()
    #
    #         Y = self.polynomial(X)
    #         plt.plot(X, Y, label="r = {}".format(d))
    #
    #     plt.title("Smoothing polynomial")
    #     plt.legend()
    #     plt.show()
    #     exit()

    def mlmc_cdf(self, X, moments_fn, ind_method="indicator", int_points=10):
        self.determine_interpolation_points(int_points)
        self.moments_fn = moments_fn

        if ind_method == "smooth":
            #self._test_smoothing_polynomial()
            self._create_smooth_polynomial()
        self.ind_method = getattr(self, ind_method)

        distribution = np.empty(len(X))
        for index, x in enumerate(X):
            sum_ind_lagr = self.cdf(x)

            if type(sum_ind_lagr).__name__ == 'ArrayBox':
                sum_ind_lagr = sum_ind_lagr._value
            distribution[index] = sum_ind_lagr

        #return distribution
        mask = (distribution >= 0) & (distribution <= 1)
        distr_sorted = np.sort(distribution[mask])
        self.mask = mask
        return distr_sorted

    # def run_from_mlmc_smoothing(self, X, moments_fn):
    #     self.determine_interpolation_points(20)
    #
    #     distribution = np.empty(len(X))
    #     for index, x in enumerate(X):
    #         sum_ind_lag = 0
    #         for n, s in enumerate(self.interpolation_points):
    #             sum_ind_level = 0
    #             for level in self.mlmc.levels[:1]:
    #                 moments = level.evaluate_moments(moments_fn)
    #
    #                 fine_values = np.squeeze(moments[0])[:, 1]
    #                 coarse_values = np.squeeze(moments[1])[:, 1]
    #
    #                 if level._level_idx == 0:
    #                     fine_indic = self.smooth(s, fine_values)
    #                     level_indic_mean = np.sum(fine_indic) / len(fine_values)
    #                     sum_ind_level += level_indic_mean
    #                 else:
    #                     fine_indic = self.smooth(s, fine_values)
    #                     coarse_indic = self.smooth(s, coarse_values)
    #
    #                     level_indic_mean = np.sum(fine_indic - coarse_indic) / len(fine_values)
    #                     sum_ind_level += level_indic_mean
    #
    #             sum_ind_lag += sum_ind_level * self.lagrange_basis_polynomial(x, n)
    #
    #         if type(sum_ind_lag).__name__ == 'ArrayBox':
    #             sum_ind_lag = sum_ind_lag._value
    #         distribution[index] = sum_ind_lag
    #
    #     #return distribution
    #     mask = (distribution >= 0) & (distribution <= 1)
    #     distr_sorted = np.sort(distribution[(distribution >= 0) & (distribution <= 1)])
    #     self.mask = mask
    #     return distr_sorted
