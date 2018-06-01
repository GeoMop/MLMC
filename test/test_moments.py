"""
Test class monomials
"""
import numpy as np
import mlmc.moments


def test_monomials():
    # Natural domain (0,1).
    size = 5
    values = np.array([-2, -1, -0.5, 0, 0.5, 1, 2])
    ref = [values**r for r in range(size)]

    moments_fn  = mlmc.moments.Monomial(size, safe_eval=False)
    moments = moments_fn(values)
    assert np.allclose(np.array(ref).T, moments)


    # Given domain (a,b).
    a, b = (-1, 3)
    moments_fn = mlmc.moments.Monomial(size, (a,b), safe_eval=False )
    moments = moments_fn((b - a)*values + a)
    assert np.allclose(np.array(ref).T, moments)

    # Approximate mean.
    values = np.random.randn(1000)
    moments_fn = mlmc.moments.Monomial(2, safe_eval=False)
    moments = moments_fn(values)
    assert np.abs( np.mean(moments[:, 1]) ) < 0.1




def test_fourier():
    # Natural domain (0,1).
    size = 6
    moments_fn = mlmc.moments.Fourier(size, (0,1))

    values = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    values_ = 2*np.pi* values
    ref = [ np.ones_like(values_), np.cos(values_), np.sin(values_),
            np.cos(2*values_), np.sin(2*values_), np.cos(3*values_)]

    moments = moments_fn(values)
    assert np.allclose(np.array(ref).T, moments)

    # Given domain (a,b).
    a, b = (-1, 3)
    moments_fn = mlmc.moments.Fourier(size, (a, b))
    moments = moments_fn((b - a)*values + a)
    #values = 2 * np.pi * values
    #ref = [ np.ones_like(values), np.cos(values), np.sin(values), np.cos(2*values), np.sin(2*values), np.cos(3*values)]
    assert np.allclose(np.array(ref).T, moments)


def test_legendere():
    # Natural domain (0,1).
    size = 4
    moments_fn = mlmc.moments.Legendre( size, (-1.0, 1.0))

    values = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    moments = moments_fn(values)
    ref = [ np.ones_like(values), values, (3*values**2 - 1.0) / 2.0, (5*values**3 - 3 * values) / 2.0]
    assert np.allclose(np.array(ref).T, moments)
