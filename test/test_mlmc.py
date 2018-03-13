#import sys, os
#myPath = os.path.dirname(os.path.abspath(__file__))
#sys.path.insert(0, myPath + '/../')
import pytest
from src.mlmc.flow_sim_generic import FlowSimGeneric as Sim
import src.mlmc
from test.result import Result
import src.mlmc.mlmc
from src.mlmc.moments import Monomials, FourierFunctions
import numpy as np

@pytest.mark.parametrize('levels, n_moments, moments_variance, final_variance, moments_function', [
    (1, 5, [1e-5, 1e-5, 1e-5, 1e-5, 1e-5], [1e-3, 1e-3, 1e-2, 1e-2, 1e-1], Monomials),
    (1, 5, [1e-5, 1e-5, 1e-5, 1e-5, 1e-5], [1e-1, 1e-1, 1e-1, 1e-1, 1e-1], FourierFunctions),
    (5, 5, [1e-5, 1e-5, 1e-5, 1e-5, 1e-5], [1e-3, 1e-3, 1e-2, 1e-2, 1e-1], Monomials),
    (5, 5, [1e-5, 1e-5, 1e-5, 1e-5, 1e-5], [1e-1, 0, 0, 0, 0], FourierFunctions)
])
def test_mlmc(levels, n_moments, moments_variance, final_variance, moments_function):
    moments_number = n_moments
    eps = 1e-10
    sim = Sim()
    sim.sim_param_range = (10, 100)
    mo = moments_function(moments_number)

    all_moments = []
    for _ in range(0, 10):
        mo.moments_number = moments_number
        mo.eps = eps
        result = Result(moments_number)
        result.level_number = levels
        # number of levels, n_fine, n_coarse, simulation
        m = src.mlmc.mlmc.MLMC(levels, sim, mo)
        m.set_target_variance(moments_variance)
        m.refill_samples()

        result.mc_levels = m.levels
        result.process_data()
        mo.mean = result.average
        moments = result.level_moments()
        all_moments.append(moments)

    moments = [(np.mean(m), np.var(m)) for m in zip(*all_moments)]

    print(moments)
    for index, (moment_mean, moment_variance) in enumerate(moments):
        print(moments[index])
        print(final_variance[index])
        assert moment_variance <= final_variance[index]

    if isinstance(moments_function, Monomials):
        assert moments[0][0] == result.average
