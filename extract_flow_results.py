import time as t
import os
import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../src/')
import shutil
import math
#from test.simulation_test import SimulationTest as TestSim
from result import Result
from mlmc.mlmc import MLMC
from mlmc.moments import Monomials, FourierFunctions
import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
from mlmc.distribution import Distribution
import pynverse as pv


class ExtractResult:
    """
    Class launchs MLMC
    """
    def main(*args):
        moments_number = 20
        bounds = []
        toleration = 1e-15
        eps = 1e-10

        files = [#"/home/martin/Desktop/FlowResults/FourierFunctions/Results/data_2L_20M"
                 #"/home/martin/Desktop/FlowResults/FourierFunctions/Results/DATA_MLMC/data_5L_2"
                 #"/home/martin/Desktop/FlowResults/FourierFunctions/Results/data_5L_e5"
                 #"/home/martin/Desktop/FlowResults/Monomials/Results/data_5L",
                 #"/home/martin/Desktop/FlowResults/Monomials/Results/data_5L",
                 #"/home/martin/Desktop/FlowResults/Monomials/Results/data_3L",
                 "/home/martin/Desktop/FlowResults/Monomials/Results/data_2L",
                 #"/home/martin/Desktop/FlowResults/Monomials/Results/data_1L_50000N"
                ]
        distributions = []
        densities = []
        all_samples = []
        levels =[]

        for file in files:
            result = Result(moments_number)
            #result.levels_number = pocet_urovni
            #result.execution_number = pocet_vykonani

            moments_object = FourierFunctions(moments_number)

            result.extract_data(file)
            result.process_data()
            mean = result.average
            moments_object.mean = mean
            moments_object.bounds = [0, 6]
            result.add_level_moments_object(moments_object)
            result.format_result()
            moments = result.level_moments()

            result._mc_levels = [result._mc_levels[0]]

            cost = 0
            for index, level in enumerate(result._mc_levels):
                if index > 0:
                    cost += (level.n_ops_estimate() * level.number_of_simulations)
                            #(result._mc_levels[index - 1].n_ops_estimate() * level.number_of_simulations)
                else:
                    cost += (level.n_ops_estimate() * level.number_of_simulations)

            for level in result.mc_levels:
                level.moments_object.mean = np.mean(level.result)

            average = 0
            maximum = 0
            for index, level_data in enumerate(result.levels_data):
                if index > 0:
                    abs_hod = [abs(data) for data in level_data]
                    average += np.mean(level_data)
                    maximum += np.amax(abs_hod)


            #bounds = sc.stats.mstats.mquantiles(result.levels_data[0], prob=[eps, 1 - eps])
            bounds = [0, 6]

            moments_object.bounds = bounds
            moments_object.fixed_quad_n = moments_number ** 2
            moments_object.mean = mean

            # Run distribution
            distribution = Distribution(moments_object, moments_number, moments, toleration)
            distribution.estimate_density()

            # Empirical distribution function
            # ecdf = ECDF(mc_data)
            samples = np.linspace(bounds[0], bounds[1], 100000)
            distribution_function = []
            difference = 0
            KL_divergence = 0
            approximate_density = []

            def integrand(x):
                return distribution.density(x)

            def distribution_integral(value):
                return sc.integrate.quad(integrand, bounds[0], value, limit=500)[0]

            def kullback_leibler_distance(x, exact_value):
                return (distribution.density(x) * math.log(distribution.density(x) / exact_value))

            for step, value in enumerate(samples):
                integral = distribution_integral(value)
                approximate_density.append(distribution.density(value))
                distribution_function.append(integral)
                #difference += abs(lognorm.cdf(value, s=sigma, loc=0, scale=np.exp(mu)) - integral) ** 2
                # if sc.stats.norm.pdf(value, 2, 1) != 0:
                #    integ = sc.integrate.quad(kullback_leibler_distance, bounds[0], value, args=(sc.stats.norm.pdf(value,2,1)), limit=500)
                #    KL_divergence += integ[0]

            for step, value in enumerate(samples):
                KL_divergence += distribution.density(value) * np.log(distribution.density(value) /approximate_density[step])
            distributions.append(distribution_function)
            densities.append(approximate_density)
            all_samples.append(samples)
            levels.append(str(result.extract_levels))

            distr = lambda x: distribution_integral(x)
            # print(distr(10))
            # [bounds[0], bounds[1]]

            inverse = pv.inversefunc(distr)
            """
            print("aproximované momenty")
            print("Kvantil 0.1", inverse(0.1))
            print("Kvantil 0.2", inverse(0.2))
            print("Kvantil 0.3", inverse(0.3))
            print("Kvantil 0.4", inverse(0.4))
            print("Kvantil 0.25", inverse(0.25))
            print("Kvantil 0.1", inverse(0.4))

            print("Median", inverse(0.5))
            print("Kvantil 0.6", inverse(0.6))
            print("Kvantil 0.7", inverse(0.7))
            print("Kvantil 0.75", inverse(0.75))
            print("Kvantil 0.8", inverse(0.8))
            print("Kvantil 0.9", inverse(0.9))
            """
            quant_approx = np.arange(0, 1, 0.001)


        for index, samples in enumerate(all_samples):
            plt.plot(samples, distributions[index], label="distribuční funkce")
            plt.ylabel(r'F(y)')
            plt.xlabel(r'y($m^2/s)$')

        #plt.savefig(path +"/distribution.png")
        ## Show approximate and exact density
        plt.xlim(-0.5, 6)
        #plt.xlabel(r'$m^2/s$')
        plt.legend()

        plt.figure(2)
        for index, samples in enumerate(all_samples):
            plt.plot(samples, densities[index], label="hustota pravděpodobnosti")

        #plt.savefig(path +"/density.png")
        plt.xlim(-0.5, 6)
        plt.ylabel(r'f(y)')
        plt.xlabel(r'y($m^2/s)$')
        plt.legend()
        plt.show()


    if __name__ == "__main__":
        main()
