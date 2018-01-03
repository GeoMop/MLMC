"""
Test class monomials
"""
from src.monomials import Monomials

MONOMIALS = Monomials()

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
