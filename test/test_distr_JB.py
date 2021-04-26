"""


Implementation TODO:
- support for possitive distributions
- compute approximation of more moments then used for approximation, turn problem into
  overdetermined non-linear least square problem
- Make TestMLMC a production class to test validity of MLMC estimatioon on any sampleset
  using subsampling.

Tests:
For given exact distribution with known density.
and given moment functions.

- compute "exact" moments (Gaussian quadrature, moment functions, exact density)
- construct approximation of the density for given exact moments
- compute L2 norm of density and KL divergence for the resulting density approximation

- compute moments approximation using MC and sampling from the dirstirbution
- compute approximation of the density
- compute L2 and KL
- compute sensitivity to the precision of input moments, estimate precision of the result,
  compute requested precision of the moments


"""
import os
import shutil
import time
import pytest

import numpy as np
from scipy import stats
from scipy import integrate

import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.patheffects as mpe
from scipy.interpolate import interp1d

import mlmc.tool.plot
import mlmc.archive.estimate
import mlmc.tool.simple_distribution
#import mlmc.tool.simple_distribution_total_var
from mlmc import moments
from test.fixtures import benchmark_distribution as benchmark
from mlmc.tool.restrict_distribution import RestrictDistribution as Restrict

import mlmc.tool.plot as plot
import test.fixtures.mlmc_test_run
import mlmc.spline_approx as spline_approx
from mlmc.moments import Legendre
from mlmc import estimator
from mlmc.quantity_spec import ChunkSpec
import mlmc.quantity
import pandas as pd
import pickle
import test.plot_numpy

from memoization import cached

"""
List of distributions with compact support used in tests. 
"""
quantile = 0.1
distribution_list = [

    ######
    #  Benchmark distributions
    ######
    (benchmark.TwoGaussians(), quantile),
    (benchmark.FiveFingers(), quantile),  # Covariance matrix decomposition failed
    (benchmark.Cauchy(), quantile), # pass, check exact
    (stats.lognorm(scale=np.exp(1), s=1), quantile),
    (benchmark.Gamma(), quantile),
    (benchmark.Steps(), quantile),
    (benchmark.ZeroValue(), quantile),
    (benchmark.SmoothZV(), quantile),
    (benchmark.Abyss(), quantile),
    ]
    ############

    # # # # # # # # # # # # # # # # # # # #(bd.Gamma(name='gamma'), False) # pass
    # # # # # # # # # # # # # # # # # # # #(stats.norm(loc=1, scale=2), False),

    # Quite hard but peak is not so small comparet to the tail.
    # # (stats.lognorm(scale=np.exp(-3), s=2), False),  # Extremely difficult to fit due to very narrow peak and long tail.
    # (stats.lognorm(scale=np.exp(-3), s=2), True),    # Still difficult for Lagrange with many moments.
    # (stats.chi2(df=10), False),# Monomial: s1=nan, Fourier: s1= -1.6, Legendre: s1=nan
    # (stats.chi2(df=5), True), # Monomial: s1=-10, Fourier: s1=-1.6, Legendre: OK
    # (stats.weibull_min(c=0.5), False),  # Exponential # Monomial stuck, Fourier stuck
    # (stats.weibull_min(c=1), False),  # Exponential
    # (stats.weibull_min(c=2), False),  # Rayleigh distribution
    # (stats.weibull_min(c=5, scale=4), False),   # close to normal
    # (stats.weibull_min(c=1.5), True),  # Infinite derivative at zero

class ecdf:
    def __init__(self, samples):
        self._samples = samples
    def __call__(self, x):
        x = np.atleast_1d(x)


def _test_correct_distribution(distr):
    """
    Test correct and compatible implementation of _pdf, _cdf, _rvs, _ppf.
    """
    assert(isinstance(distr, Restrict))
    a, b = distr.a, distr.b
    # Correct CDF
    assert np.isclose(distr.cdf(a), 0)      # default atol=1e-5, rtol=1e-8
    assert np.isclose(distr.cdf(b), 1.0)
    # PDF compatible with CDF
    points = stats.uniform.rvs(a,b,size=10)
    points[-1] = b
    for p in points:
        numerical_cdf = integrate.quad(distr.pdf, a, p, limit = 100)[0]
        proper_cdf = distr.cdf(p)
        assert np.isclose(numerical_cdf, proper_cdf), f"x: {p} cdf:{proper_cdf} approx:{numerical_cdf}"
    # RVS produce matching ECDF
    n_samples = 1000
    samples = distr.rvs(n_samples)
    samples.sort()
    # compute L2 norm
    norm = np.linalg.norm(distr.cdf(samples) - np.arange(n_samples) / n_samples) / np.sqrt(n_samples)
    assert norm < 2e-2, f"{norm} > tol"


def _plot_distribution(distr):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    X = np.linspace(distr.a, distr.b, 500)
    ax.plot(X, distr.cdf(X), label='cdf')
    ax.plot(X, distr.pdf(X), label='pdf')
    X = distr.rvs(1000)
    X.sort()
    Y = [i/len(X) for i,v in enumerate(X)]
    ax.plot(X, Y, label='ecdf')
    ax.legend()
    fig.suptitle(distr.distr_name)
    plt.show()



np.random.seed(123)

@pytest.mark.parametrize('distr', distribution_list)
def test_all(distribution):
    distr_class, quantile = distribution
    distr = Restrict.from_quantiles(distr_class, quantile)
    #_plot_distribution(distr)
    _test_correct_distribution(distr)




if __name__ == "__main__":
    for d in distribution_list:
        test_all(d)
