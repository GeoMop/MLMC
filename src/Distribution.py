import numpy as np
import BasisFunctions
import scipy as sc
import matplotlib.pyplot as plt


class Distribution():
    def __init__(self):
        '''
        Getting basis function 
        '''

        self.bf = BasisFunctions.BasisFunctions()
        self.integral_lower_limit = -1
        self.integral_upper_limit = 2
        self.R = 5
        self.toleration = 0.5

    def newton_method(self):
        '''
        Newton method
        :return: None
        '''

        lam = []
        l_0 = np.ones(self.R) * 0
        lam.append(l_0)
        damping = 0.1
        error = 0

        moments, stemps = self.bf.get_moments(self.R)
        steps = 0
        max_steps = 65
        while steps < max_steps:
            # Calculate G matrix
            G = self.calculate_G(lam[steps])
            print("G", G, moments)

            # Calculate H matrix
            H = self.calculate_H(lam[steps])
            H = np.linalg.inv(H)
            # print("H", H)

            # Result - add new lamba
            lam.append(lam[steps] + np.dot(H, np.subtract(moments, G)) * damping)

            for i in range(self.R):
                try:
                    error += pow(1 - (G[i] / moments[i]), 2)
                except ZeroDivisionError:
                    print()

            if error < self.toleration ** 2:
                break

            steps += 1

            print("lambda", lam[steps - 1])

        for x in range(10):
            self.density(lam[steps - 1], x)

        self.show_density(lam[steps - 1])

        '''
        Hustota 

        density_function = 0
        for x in range(10):
            self.density(lam[steps -1],x)
        '''


    def show_density(self, lam):
        '''
        Show density
        :param lam: lambdas
        :return: None
        '''
        approximate_density = []
        X = np.linspace(self.integral_lower_limit, self.integral_upper_limit, 100)
        for x in X:
            approximate_density.append(self.density(lam, x))

        plt.plot(X, approximate_density, 'r')
        plt.plot(X, [sc.stats.lognorm.pdf(x, 0.1) for x in X])
        plt.ylim((0, 10))
        #plt.xlim(0.25, 2)

        plt.show()


    def density(self, l, x):
        """       
        :param l: lambda
        :param x: 
        :return: density for passed x
        """
        s = 0
        for r in range(self.R):
            s += l[r] * self.function(x, r)

        return np.exp(-s)


    def calculate_G(self, lam):
        """
        :param l: lambda
        :return: array G
        """
        G = []

        def integrand(x, lam, s):
            sum = 0
            for r in range(self.R):
                sum += lam[r] * self.function(x, r)
            return self.function(x, s) * np.exp(-sum)

        for s in range(self.R):
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
            for p in range(self.R):
                sum += lam[p] * self.function(x, p)
            return self.function(x, s) * self.function(x, r) * np.exp(-sum)

        H = np.zeros(shape=(self.R, self.R))
        for r in range(self.R):
            for s in range(self.R):
                integral = sc.integrate.quad(integrand, self.integral_lower_limit, self.integral_upper_limit,
                                             args=(lam, r, s), limit=100)
                H[r, s] = -integral[0]
                H[s, r] = -integral[0]

        return H


    def function(self, x, r):
        '''
        :param x: 
        :param r: 
        :return: 
        '''
        #return self.fourier(x, r);

        return pow(x,r)


    def change_interval(self, x):

        """
        Fitting value from first interval to the second interval
        :param x: 
        :return: value x remapped to second interval
        """
        """
        z interval (c, d) na interval (a, b)
        x = (b-a)*(x-c)/(d-c) + a
        (a, b) = (0,2pi)
        """
        c, d = (self.integral_lower_limit, self.integral_upper_limit)

        return (2 * np.pi * (x - c)) / (d - c)


    def fourier(self, x, r):
        '''
        Calculating fourier functions for passed params
        :param x:  
        :param r: 
        :return: fourier function result
        '''
        x = self.change_interval(x)

        if r == 0:
            return 0
        if r % 2 != 0:
            return np.sin(r * x)
        if r % 2 == 0:
            return np.cos(r * x)


# Run density
d = Distribution()
d.newton_method()
