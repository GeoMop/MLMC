import numpy as np
import scipy as sc

class Distribution:
    """
    Calculation of the distribution
    """
    def __init__(self, moments_fce, moments_number, moments, toleration=0.05):
        """
        Getting basis function
        """
        self.moments_function = moments_fce
        self.integral_lower_limit = self.moments_function.bounds[0]
        self.integral_upper_limit = self.moments_function.bounds[1]
        self.moments_number = moments_number
        self.moments = moments
        self.lagrangian_parameters = []
        self.toleration = toleration


    def newton_method(self):
        """
        Newton method
        :return: None
        """
        lagrangians = []
        first_lambda = np.ones(self.moments_number) * 0
        lagrangians.append(first_lambda)
        damping = 0.1
        steps = 0
        max_steps = 1000

        try:
            while steps < max_steps:
                error = 0
                # Calculate moments approximation
                moments_approximation = self.calculate_moments_approximation(lagrangians[steps])

                # Calculate jacobian matrix
                jacobian_matrix = self.calculate_jacobian_matrix(lagrangians[steps])
                jacobian_matrix = np.linalg.inv(jacobian_matrix)

                # Result - add new lagrangians
                lagrangians.append(lagrangians[steps] +
                                   np.dot(jacobian_matrix, np.subtract(self.moments, moments_approximation)) * damping)

                for degree in range(self.moments_number):
                    try:
                        error += pow((self.moments[degree] - moments_approximation[degree]) / (self.moments[degree] + 1), 2)
                    except ZeroDivisionError:
                        print("Division by zero")

                if error < self.toleration ** 2:
                    break
                steps += 1
            self.lagrangian_parameters = lagrangians[steps - 1]
        except TypeError:
            return self.lagrangian_parameters, steps

        return self.lagrangian_parameters, steps

    def density(self, value):
        """
        :param value: float
        :return: density for passed value
        """
        result = 0
        for degree in range(self.moments_number):
            result += self.lagrangian_parameters[degree] * self.moments_function.get_moments(value, degree)

        return np.exp(-result)

    def calculate_moments_approximation(self, lagrangians):
        """
        :param lagrangians: array, lagrangians parameters
        :return: array, moments approximation
        """
        moments_approximation = []

        def integrand(value, lagrangians, moment_degree):
            """
            Integral function
            """
            total = 0
            for degree in range(self.moments_number):
                total += lagrangians[degree] * self.moments_function.get_moments(value, degree)
            return self.moments_function.get_moments(value, moment_degree) * np.exp(-total)

        for moment_degree in range(self.moments_number):
            integral = sc.integrate.fixed_quad(integrand, self.integral_lower_limit, self.integral_upper_limit,
                                               args=(lagrangians, moment_degree), n=self.moments_function.fixed_quad_n)
            moments_approximation.append(integral[0])

        return moments_approximation

    def calculate_jacobian_matrix(self, lagrangians):
        """
        :param lagrangians: lambda
        :return: matrix, jacobian matrix
        """

        def integrand(value, lagrangians, moment_degree_r, moment_degree_s):
            """
            Integral function
            """
            total = 0
            for degree in range(self.moments_number):
                total += lagrangians[degree] * self.moments_function.get_moments(value, degree)
            return self.moments_function.get_moments(value, moment_degree_s) * \
                   self.moments_function.get_moments(value, moment_degree_r) * np.exp(-total)

        # Initialization of matrix
        jacobian_matrix = np.zeros(shape=(self.moments_number, self.moments_number))

        for moment_degree_r in range(self.moments_number):
            for moment_degree_s in range(self.moments_number):
                integral = sc.integrate.fixed_quad(integrand, self.integral_lower_limit, self.integral_upper_limit,
                           args=(lagrangians, moment_degree_r, moment_degree_s), n=self.moments_function.fixed_quad_n)
                jacobian_matrix[moment_degree_r, moment_degree_s] = -integral[0]
                jacobian_matrix[moment_degree_s, moment_degree_r] = -integral[0]

        return jacobian_matrix
