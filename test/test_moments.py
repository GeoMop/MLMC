"""
Test class monomials
"""
import numpy as np
import mlmc.moments


def test_monomials():
    # Natural domain (0,1).
    moments_fn  = mlmc.moments.monomial_moments
    values = np.array([-2, -1, -0.5, 0, 0.5, 1, 2])
    size = 5
    moments = moments_fn(values, size)
    for r in range(size):
        assert np.allclose(moments[:, r], values**r)


    # Given domain (a,b).
    a, b = (-1, 3)
    moments = moments_fn((b - a)*values + a , size, a, b)
    for r in range(size):
        assert np.allclose(moments[:, r], values**r)

    # Approximate mean.
    values = np.random.randn(1000)
    moments = moments_fn(values, 2)
    assert np.abs( np.mean(moments[:, 1]) ) < 0.1




def test_fourier():
    # Natural domain (0,1).
    moments_fn = mlmc.moments.fourier_moments

    values = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    size = 6
    moments = moments_fn(values, size)
    values = 2*np.pi* values
    ref = [ np.ones_like(values), np.cos(values), np.sin(values), np.cos(2*values), np.sin(2*values), np.cos(3*values)]
    assert np.allclose(np.array(ref).T, moments)

    # Given domain (a,b).
    a, b = (-1, 3)
    moments = moments_fn((b - a)*values + a , size, a, b)
    values = 2 * np.pi * values
    ref = [ np.ones_like(values), np.cos(values), np.sin(values), np.cos(2*values), np.sin(2*values), np.cos(3*values)]
    assert np.allclose(np.array(ref).T, moments)


def test_legendere():
    # Natural domain (0,1).
    moments_fn = mlmc.moments.legendre_moments

    values = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    size = 4
    moments = moments_fn(values, size, -1.0, 1.0)
    ref = [ np.ones_like(values), values, (3*values**2 - 1.0) / 2.0, (5*values**3 - 3 * values) / 2.0]
    assert np.allclose(np.array(ref).T, moments)
