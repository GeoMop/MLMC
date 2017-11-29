import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from  BasisFunctions import BasisFunctions
import time as t
import math


class Distribution:
    def __init__(self, basis_fce, moments_number, bounds, moments, toleration=0.05, shape=0.1):
        '''
        Getting basis function 
        '''
        self.bf = basis_fce
        self.integral_lower_limit = bounds[0]
        self.integral_upper_limit = bounds[1]
        self.moments_number = moments_number
        self.moments = moments
        self.lagrangien_parameters = []
        self.toleration = toleration
        # Shape of distribution
        self.shape = shape


    def newton_method(self):
        '''
        Newton method
        :return: None
        '''

        lam = []
        l_0 = np.ones(self.moments_number)
        lam.append(l_0)
        damping = 0.1
        steps = 0
        max_steps = 1000

        try:
            while steps < max_steps:
                error = 0
                # Calculate G matrix
                G = self.calculate_G(lam[steps])
                # print("G", G, self.moments)

                # Calculate H matrix
                H = self.calculate_H(lam[steps])
                #print("H", H)
                H = np.linalg.inv(H)

                # Result - add new lamba
                lam.append(lam[steps] + np.dot(H, np.subtract(self.moments, G)) * damping)

                for i in range(self.moments_number):
                    try:
                        error += pow((self.moments[i] - G[i]) / (self.moments[i]), 2)
                    except ZeroDivisionError:
                        print("Division by zero")

                if error < self.toleration ** 2:
                    break
                steps += 1

            self.lagrangien_parameters = lam[steps - 1]

        except:
            return (self.lagrangien_parameters, steps)

        return (self.lagrangien_parameters, steps)


    def density(self, x):
        """       
        :param x: 
        :return: density for passed x
        """
        s = 0
        for r in range(self.moments_number):
            s += self.lagrangien_parameters[r] * self.bf.get_moments(x, r)

        return np.exp(-s)


    def calculate_G(self, lam):
        """
        :param l: lambda
        :return: array G
        """
        G = []

        def integrand(x, lam, s):
            sum = 0
            for r in range(self.moments_number):
                sum += lam[r] * self.bf.get_moments(x, r)
            return self.bf.get_moments(x, s) * np.exp(-sum)

        for s in range(self.moments_number):
            integral = sc.integrate.quad(integrand, self.integral_lower_limit, self.integral_upper_limit, args=(lam, s),
                                         limit=100)
            G.append(integral[0])

        return G


    def calculate_H(self, lam):
        '''
        :param l: lambda
        :return: matrix H
        '''

        def integrand(x, lam, r, s):
            sum = 0
            for p in range(self.moments_number):
                sum += lam[p] * self.bf.get_moments(x, p)
            return self.bf.get_moments(x, s) * self.bf.get_moments(x, r) * np.exp(-sum)

        H = np.zeros(shape=(self.moments_number, self.moments_number))
        for r in range(self.moments_number):
            for s in range(self.moments_number):
                integral = sc.integrate.quad(integrand, self.integral_lower_limit, self.integral_upper_limit,
                                             args=(lam, r, s), limit=100)
                H[r, s] = -integral[0]
                H[s, r] = -integral[0]

        return H
