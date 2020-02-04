import numpy as np
from scipy import integrate, optimize
from scipy.interpolate import interp1d, CubicSpline, splrep, splev


class SplineApproximation:

    def __init__(self, mlmc, inter_points_domain, poly_degree, accuracy):
        """
        Cdf and pdf spline approximation
        :param mlmc: MLMC instance
        :param inter_points_domain: interpolation points inter_points_domain
        :param poly_degree: degree of polynomial
        :param accuracy: RMSE accurancy, used to smooth
        """
        self.mlmc = mlmc
        self.inter_points_domain = inter_points_domain
        self.poly_degree = poly_degree
        self.accuracy = accuracy

        self.smoothing_factor = np.zeros(self.mlmc.n_levels)
        self.interpolation_points = []
        self.polynomial = None
        self.moments_fn = None
        self.indicator_method_name = "indicator"
        self.n_interpolation_points = 10

        self.sampling_error = None
        self.smoothing_error = None

        self.distribution = None
        self.pdf = None

        self.distr_mask = None
        self.density_mask = None

        # It is necessary for distribution plot
        self.domain = None

    def determine_interpolation_points(self, n_points):
        """
        Determine equidistant points at which the cdf (or pdf) is calculated
        :param n_points: number of interpolation points
        :return: list
        """
        self.interpolation_points = np.linspace(self.inter_points_domain[0], self.inter_points_domain[1], n_points)

    def compute_smoothing_factor(self, data, level_id):
        """
        Compute smoothing factor - not stable at the moment
        :param data: Level fine or coarse data
        :param level_id: Level id
        :return: Smoothing factor for particular level
        """
        result = []
        res = 0

        def functional(x, data, s):
            return np.abs(np.sum(self.polynomial((data - s) / x) - self.indicator(s, data))) / len(data) - self.accuracy/2

        for s in self.interpolation_points:
            try:
                res = optimize.root(functional, x0=0.01, args=(data, s), tol=1e-5)#, full_output=True, disp=True)
                #res = optimize.minimize(functional, [0.01], jac=egrad(functional), method='trust-ncg', args=(data, s), tol=1e-5)
            except Exception as e:
                print("Compute smoothing factor optimization failed")

            if res.success is True:
                result.append(np.squeeze(res.x))

        result.remove(max(result))
        result.remove(min(result))

        self.smoothing_factor[level_id] = np.max(result)

        self._test_smoothing_factor(data, self.smoothing_factor[level_id])

    def _test_smoothing_factor(self, data, smoothing_factor):

        for s in self.interpolation_points:
            res = np.abs(np.sum(self.polynomial((data - s) / smoothing_factor))) / len(data)
            print("res ", res)
            print("accuracy / 2 ", self.accuracy/2)
            assert np.isclose(res, self.accuracy / 2, atol=1e-5)
        exit()

    def _create_smooth_polynomial(self):
        """
        Calculate smoothing polynomial according to Giles
        Set global variable polynomial
        :return: None
        """
        coeficients_matrix = np.empty((self.poly_degree+1, self.poly_degree+1))
        constants_matrix = np.empty(self.poly_degree+1)

        # g(1) = 0, g(-1) = 1
        coeficients_matrix[0] = np.ones(self.poly_degree+1)
        coeficients_matrix[1] = [1 if i % 2 != 0 or i == self.poly_degree else -1 for i in range(self.poly_degree+1)]
        constants_matrix[0] = 0
        constants_matrix[1] = 1

        for j in range(self.poly_degree - 1):
            coeficients_matrix[j+2] = np.flip(np.array([(1 ** (i + j + 1) - (-1) ** (i + j + 1)) / (i + j + 1) for i
                                                        in range(self.poly_degree+1)]))
            constants_matrix[j + 2] = (-1) ** j / (j + 1)

        poly_coefs = np.linalg.solve(coeficients_matrix, constants_matrix)
        self.polynomial = np.poly1d(poly_coefs)

        self._test_poly()

    def _test_poly(self):
        """
        Test calculated polynomial
        :return: None
        """
        for degree in range(0, self.poly_degree):
            def integrand(x):
                return x**degree * self.polynomial(x)
            result = integrate.quad(integrand, -1, 1)[0]
            expected_result = (-1)**degree / (degree + 1)

            assert np.isclose(result, expected_result, atol=1e-5)

    def smooth(self, interpolation_point, data):
        data = (data - interpolation_point) / self._level_smoothing_factor
        if self.polynomial is None:
            self._create_smooth_polynomial()

        return self._polynomial_smoothing(data)

    def _polynomial_smoothing(self, data):
        """
        Smooth
        :param data: Given data, e.g. fine data from MLMC level
        :return: numpy array
        """
        result = np.zeros(len(data))
        result[(data < -1)] = 1
        indices = (-1 <= data) & (data <= 1)
        data = data[indices]

        if len(data) > 0:
            result[indices] = self.polynomial(data)
        return result

    def indicator(self, interpolation_point, data):
        """
        Initial state without smoothing technique
        :param interpolation_point: list
        :param data:
        :return:
        """
        d = np.zeros(len(data))
        d[data <= interpolation_point] = 1
        return d

    def lagrange_basis_polynomial_derivative(self, x, j):
        """
        Derivation of lagrange basis polynomial
        :param x: Given point
        :param j: point index
        :return: number
        """
        product = 1
        summation = 0
        data = self.interpolation_points

        for m in range(len(data)):
            if j == m:
                continue
            product *= (x - data[m]) / (data[j] - data[m])
            summation += 1/(x - data[m])
        return product * summation

    def lagrange_basis_polynomial(self, x, j):
        """
        Lagrange basis polynomial
        :param x: Given point
        :param j: Index of given point
        :return:
        """
        product = 1
        data = self.interpolation_points

        for m in range(len(data)):
            if j == m:
                continue
            product *= (x - data[m]) / (data[j] - data[m])

        return product

    def indicator_mean(self):
        """
        Mean value for indicator method - either indicator function or smoothing function
        :return:
        """
        self.all_levels_indicator = np.zeros(len(self.interpolation_points))

        sampling_error = np.zeros(len(self.interpolation_points))
        smooting_err = np.zeros(len(self.interpolation_points))

        for level in self.mlmc.levels:
            moments = level.evaluate_moments(self.moments_fn)
            fine_values = np.squeeze(moments[0])[:, 1]
            fine_values = self.moments_fn.inv_linear(fine_values)
            coarse_values = np.squeeze(moments[1])[:, 1]
            coarse_values = self.moments_fn.inv_linear(coarse_values)
            #
            # if self.smoothing_factor[level._level_idx] == 0 and self.ind_method.__name__ == "smooth":
            #     self.compute_smoothing_factor(fine_values, level._level_idx)

            self._level_smoothing_factor = self.accuracy**(1/(self.poly_degree + 1))
            #self._level_smoothing_factor = self.smoothing_factor[level._level_idx]

            print("_level_smoothing_factor ", self._level_smoothing_factor)

            for n, s in enumerate(self.interpolation_points):
                if level._level_idx == 0:
                    fine_indic = self.ind_method(s, fine_values)
                    int_point_mean = np.sum(fine_indic) / len(fine_values)
                else:
                    fine_indic = self.ind_method(s, fine_values)
                    coarse_indic = self.ind_method(s, coarse_values)
                    int_point_mean = np.sum(fine_indic - coarse_indic) / len(fine_values)
                    sampling_error[n] += (np.var(fine_indic - coarse_indic) / len(fine_indic))

                self.all_levels_indicator[n] += int_point_mean

        self.sampling_error = np.max(sampling_error)

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

    def _setup(self):
        """
        Set interpolation points, smoothing factor and polynomial (for smoothing technique), indicator method
        Also run indicator method for all MLMC levels and calculate expected value of particular indicator function
        :return: None
        """
        self.determine_interpolation_points(self.n_interpolation_points)

        if self.indicator_method_name == "smooth":
            self.smoothing_factor = np.zeros(self.mlmc.n_levels)
            self._create_smooth_polynomial()

        self.ind_method = getattr(self, self.indicator_method_name)
        self.indicator_mean()

    def cdf(self, points):
        """
        Cumulative distribution function at points X
        :param points: list of points (1D)
        :return: distribution
        """
        if self.distribution is not None:
            return self.distribution

        self._setup()

        distribution = np.empty(len(points))
        for index, x in enumerate(points):
            lagrange_poly = []

            # Lagrange polynomials at interpolation points
            for n, s in enumerate(self.interpolation_points):
                lagrange_poly.append(self.lagrange_basis_polynomial(x, n))

            distribution[index] = np.sum(self.all_levels_indicator * np.array(lagrange_poly).T)

        #return distribution

        #return np.sort(distribution)

        mask = (distribution >= 0) & (distribution <= 1)
        distr_sorted = distribution[mask]#np.sort(distribution[mask])
        self.distr_mask = mask
        return distr_sorted

    def density(self, points):
        """
        Calculate probability density function at points X
        :param points: 1D list of points
        :return: density
        """
        if self.pdf is not None:
            return self.pdf

        self._setup()
        density = np.empty(len(points))

        for index, x in enumerate(points):
            lagrange_poly = []

            # Derivative of lagrange polynomials at interpolation points
            for n, s in enumerate(self.interpolation_points):
                lagrange_poly.append(self.lagrange_basis_polynomial_derivative(x, n))

            density[index] = np.sum(self.all_levels_indicator * np.array(lagrange_poly).T)

        mask = (density >= 0)
        self.mask = mask
        return density[mask]

    def cdf_pdf(self, points):
        """
        Calculate cdf and pdf at same time
        :param points:
        :return:
        """
        self._setup()

        distribution = np.empty(len(points))
        density = np.empty(len(points))
        for index, x in enumerate(points):
            lagrange_poly = []
            lagrange_poly_der = []

            # Lagrange polynomials at interpolation points
            for n, s in enumerate(self.interpolation_points):
                lagrange_poly.append(self.lagrange_basis_polynomial(x, n))
                lagrange_poly_der.append(self.lagrange_basis_polynomial_derivative(x, n))

            distribution[index] = np.sum(self.all_levels_indicator * np.array(lagrange_poly).T)
            density[index] = np.sum(self.all_levels_indicator * np.array(lagrange_poly_der).T)

        mask = (distribution >= 0) & (distribution <= 1)
        distr_sorted = np.sort(distribution)#[mask])
        self.distribution = distr_sorted
        #self.distr_mask = mask

        # mask = (density >= 0)
        # self.density_mask = mask
        # self.pdf = density[mask]
        return distr_sorted, density#[mask]


class BSplineApproximation(SplineApproximation):

    def cdf(self, points):
        """
        Cumulative distribution function at points X
        :param points: list of points (1D)
        :return: distribution
        """
        import scipy.interpolate as si
        self._setup()

        print("BSpline cdf poly degree ", self.poly_degree)
        print("all levels indicator ", self.all_levels_indicator)
        print("self.interpolation points ", self.interpolation_points)
        #bspline = si.BSpline(self.interpolation_points, self.all_levels_indicator, k=self.poly_degree)

        spl = splrep(self.interpolation_points, self.all_levels_indicator)

        # distribution = np.zeros(len(points))
        # for index, x in enumerate(points):
        #     distribution[index] = bspline(x)
        #
        # return distribution

        #res = bspline(points)
        return splev(points, spl)

        print("BSpline CDF")
        print("res ", res)
        print("res ", res)
        return res

    def density(self, points):
        self._setup()
        spl = splrep(self.interpolation_points, self.all_levels_indicator)

        return splev(points, spl, der=1)

    # def density(self, points):
    #     import numdifftools as nd
    #     import scipy.interpolate as si
    #
    #     #return nd.Derivative(self.cdf)(points)
    #     self._setup()
    #     bspline = si.BSpline(self.interpolation_points, self.all_levels_indicator, k=self.poly_degree)
    #     bspline_derivative = bspline.derivative()
    #     res = bspline_derivative(points)
    #     print("BSpline PDF")
    #     return res
