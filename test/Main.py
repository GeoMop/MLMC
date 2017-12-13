import time as t
import SimulationSetting as Sim
import sys

sys.path.insert(0, '/home/martin/Documents/MLMC/src')
from result import Result
from mlmc import MLMC
from monomials import Monomials
from fourier_functions import FourierFunctions
from distribution import Distribution
import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
from distribution_fixed_quad import DistributionFixedQuad


class Main:
    '''
    Class launchs MLMC
    '''
    def main(*args):

        start = t.time()
        Y, n, v, N, cas = ([] for i in range(5))

        pocet_urovni = 3
        pocet_vykonani = 1
        moments_number = 10
        bounds = [0, 2]
        toleration = 0.05
        eps = 1e-8

        # param type -> type of simulation
        sim = Sim.SimulationSetting(1)

        result = Result(moments_number)
        result.set_levels_number(pocet_urovni)
        result.set_execution_number(pocet_vykonani)

        for i in range(pocet_vykonani):
            start_MC = t.time()
            # number of levels, n_fine, n_coarse, simulation
            m = MLMC(pocet_urovni, 100, 10, sim)

            # type, time or variance
            m.monte_carlo(1, 0.01)
            end_MC = t.time()

            result.add_average(m.get_Y())
            result.add_arrays(m.get_arrays())
            result.append_result(m.formatting_result())
            result.add_time(end_MC - start_MC)

        mc_moments, mc_data = result.result()

        bounds = sc.stats.mstats.mquantiles(mc_data, prob=[eps, 1 - eps])
        mean = np.mean(mc_data)

        print(bounds)


        basis_function = FourierFunctions(mean)
        #basis_function = Monomials(mean)
        basis_function.set_bounds(bounds)
        basis_function.fixed_quad_n = moments_number * 10

        moments = []

        for k in range(moments_number):
            val = []
            for value in mc_data:
                val.append(basis_function.get_moments(value, k))
            moments.append(np.mean(val))

        print("momenty", moments)

        # Run distribution
        distribution = DistributionFixedQuad(basis_function, moments_number, moments, toleration)
        distribution.newton_method()

        # Empirical distribution function
        ecdf = ECDF(mc_data)
        samples = np.linspace(bounds[0], bounds[1], len(mc_data))
        distribution_function = []
        difference = 0
        ## Set approximate density values
        approximate_density = []

        def integrand(x):
            return distribution.density(x)

        for step, value in enumerate(samples):
            integral = sc.integrate.quad(integrand, bounds[0], value)
            approximate_density.append(distribution.density(value))
            distribution_function.append(integral[0])
            difference += abs(ecdf.y[step] - integral[0])**2

        print(distribution_function)

        plt.figure(1)
        plt.plot(ecdf.x, ecdf.y)
        plt.plot(samples, distribution_function, 'r')

        ## Show approximate and exact density
        plt.figure(2)
        plt.plot(samples, approximate_density, 'r')
        plt.show()

        print(difference)

        '''
        sum = 0
        stredni_hodnota = 50
        for i in range(1):
            sum = sum + pow((stredni_hodnota -Y[i]),2)
        chyba = np.sqrt(sum/10)
        end = t.time()

        print("Y = ", m.average(Y))
        print("Chyba = ", chyba)

        print("cas", end - start)
        '''

    if __name__ == "__main__":
        main()
