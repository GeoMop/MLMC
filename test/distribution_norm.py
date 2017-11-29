import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from distribution import Distribution
from monomials import Monomials
from fourier_functions import FourierFunctions

shape = 0.5
values =  np.random.normal(0, shape, 100000)

moments_number = 15
bounds = [0,2]
toleration = 0.05

bounds = sc.stats.mstats.mquantiles(values,prob=[0.00001, 0.99999])

mean = np.mean(values)
basis_function = FourierFunctions(mean)
basis_function.set_bounds(bounds)

moments = []
for k in range(moments_number):
    val = []

    for value in values:
        val.append(basis_function.get_moments(value, k))

    moments.append(np.mean(val))

#basis_function = FourierFunctions()
#basis_function.set_bounds()

# Run distribution
distribution = Distribution(basis_function, moments_number, bounds, moments, toleration, shape)
#d.set_values(values)
lagrangian_parameters = distribution.newton_method()

print(moments)
print(lagrangian_parameters)


## Difference between approximate and exact density
sum = 0
X = np.linspace(bounds[0], bounds[1], 100)
for x in X:
   sum += abs(distribution.density(x) - sc.stats.norm.pdf(x, shape))
print(sum)


## Set approximate density values
approximate_density = []
X = np.linspace(bounds[0], bounds[1], 100)
for x in X:
    approximate_density.append(distribution.density(x))


## Show approximate and exact density
plt.plot(X, approximate_density, 'r')
plt.plot(X, [sc.stats.norm.pdf(x) for x in X])
plt.ylim((0, 2))
plt.show()

"""
## Show approximate and exact density in logaritmic scale
X = np.linspace(bounds[0], bounds[1], 100)
plt.plot(X, -np.log(approximate_density), 'r')
plt.plot(X, -np.log([sc.stats.norm.pdf(x) for x in X]))
plt.ylim((-10, 10))
plt.show()
"""




