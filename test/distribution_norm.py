import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/home/martin/Documents/MLMC/src')
from distribution import Distribution
from distribution_fixed_quad import DistributionFixedQuad
from monomials import Monomials
from fourier_functions import FourierFunctions

shape = 0.1
values = np.random.normal(0, shape, 100000)

moments_number = 15
bounds = [0, 2]
toleration = 0.05
eps = 1e-6

bounds = sc.stats.mstats.mquantiles(values, prob=[eps, 1 - eps])

mean = np.mean(values)

basis_function = FourierFunctions(mean)
basis_function.set_bounds(bounds)
basis_function.fixed_quad_n = moments_number * 2
"""
basis_function = Monomials(mean)
basis_function.set_bounds(bounds)
basis_function.fixed_quad_n = moments_number + 1
"""

moments = []
for k in range(moments_number):
    val = []

    for value in values:
        val.append(basis_function.get_moments(value, k))

    moments.append(np.mean(val))


# Run distribution
distribution = DistributionFixedQuad(basis_function, moments_number, moments, toleration)
#d.set_values(values)
lagrangian_parameters = distribution.newton_method()

print(moments)
print(lagrangian_parameters)


## Difference between approximate and exact density
sum = 0
X = np.linspace(bounds[0], bounds[1], 100)
for x in X:
   sum += abs(distribution.density(x) - sc.stats.norm.pdf(x))
print(sum)


## Set approximate density values
approximate_density = []
X = np.linspace(bounds[0], bounds[1], 100)
for x in X:
    approximate_density.append(distribution.density(x))


## Show approximate and exact density
plt.plot(X, approximate_density, 'r')
plt.plot(X, [sc.stats.norm.pdf(x, 0, shape) for x in X])
plt.ylim((0, 10))
plt.show()

"""
## Show approximate and exact density in logaritmic scale
X = np.linspace(bounds[0], bounds[1], 100)
plt.plot(X, -np.log(approximate_density), 'r')
plt.plot(X, -np.log([sc.stats.norm.pdf(x) for x in X]))
plt.ylim((-10, 10))
plt.show()
"""




