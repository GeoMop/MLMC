"""
Test class monomials
"""
import math
import mlmc.moments

MONOMIALS = mlmc.moments.Monomials()

def test_get_moments():
    """
    Test get moments method
    """
    assert MONOMIALS.get_moments(2, 2) == 4
    assert MONOMIALS.get_moments(-2, 2) == 4
    assert MONOMIALS.get_moments(0, 1) == 0
    assert MONOMIALS.get_moments(0.1, 1) == 0.1

    MONOMIALS.mean = 1
    assert MONOMIALS.get_moments(2, 2) == 1
    assert MONOMIALS.get_moments(-2, 2) == 9
    assert MONOMIALS.get_moments(0, 1) == -1


"""
Test fourier functions class
"""

FOURIER = mlmc.moments.FourierFunctions()

def test_get_moments():
    """
    Test get moments method
    """
    FOURIER.bounds = [-10, 10]
    assert FOURIER.get_moments(10, 0) == 1
    assert FOURIER.get_moments(0, 2) == -1
    assert FOURIER.get_moments(10, 4) == 1
    assert FOURIER.get_moments(-10, 4) == 1
    assert FOURIER.get_moments(0, 3) < 0
    assert FOURIER.get_moments(10, 5) < 0
    assert FOURIER.get_moments(-2, 5) > 0


def test_change_interval():
    """
    Test change interval method
    """
    FOURIER.bounds = [-10, 10]
    assert FOURIER.change_interval(-10) == 0
    assert FOURIER.change_interval(10) == 2*math.pi
    assert math.pi == FOURIER.change_interval(0)
