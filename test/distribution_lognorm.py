import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/home/martin/Documents/MLMC/src')
from distribution import Distribution
from monomials import Monomials
from fourier_functions import FourierFunctions
from distribution_fixed_quad import DistributionFixedQuad
import time as t


shape = 0.1
values = np.random.lognormal(0, shape, 100000)

moments_number = 10
bounds = [0, 2]
toleration = 0.05
eps = 1e-6

bounds = sc.stats.mstats.mquantiles(values, prob=[eps, 1 - eps])
print(bounds)


mean = np.mean(values)
print(mean)

basis_function = FourierFunctions(mean)
basis_function.set_bounds(bounds)
basis_function.fixed_quad_n = moments_number * 2
"""
basis_function = Monomials(mean)
basis_function.set_bounds(bounds)
basis_function.fixed_quad_n = moments_number + 1
"""
#print(np.mean(np.sin(values)))
moments = []
for k in range(moments_number):
    val = []

    for value in values:
        val.append(basis_function.get_moments(value, k))

    moments.append(np.mean(val))

print("momenty", moments)


zacatek = t.time()
# Run distribution
distribution = DistributionFixedQuad(basis_function, moments_number, moments, toleration)
#d.set_values(values)
lagrangian_parameters = distribution.newton_method()

konec = t.time()

print("celkovy cas", konec- zacatek)
print(lagrangian_parameters)


## Difference between approximate and exact density
sum = 0
X = np.linspace(bounds[0], bounds[1], 100)
for x in X:
   sum += abs(distribution.density(x) - sc.stats.lognorm.pdf(x, shape))**2
print(sum)


## Set approximate density values
approximate_density = []
X = np.linspace(bounds[0], bounds[1], 100)
for x in X:
    approximate_density.append(distribution.density(x))


## Show approximate and exact density
plt.plot(X, approximate_density, 'r')
plt.plot(X, [sc.stats.lognorm.pdf(x, shape) for x in X])
plt.ylim((0, 10))
plt.show()

"""
## Show approximate and exact density in logaritmic scale
X = np.linspace(bounds[0], bounds[1], 100)
plt.plot(X, -np.log(approximate_density), 'r')
plt.plot(X, -np.log([sc.stats.lognorm.pdf(x, shape) for x in X]))
plt.ylim((-10, 10))
plt.show()
"""




