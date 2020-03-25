from scipy import stats
import numpy as np
import pytest
#import mlmc.estimate

@pytest.mark.skip
@pytest.mark.parametrize("n_levels, n_samples, failed_fraction", [
    (1, [100], 0.2),
    (2, [200, 100], 0.5),
    (5, [300, 250, 200, 150, 100], 0.3)
])
def test_estimate(n_levels, n_samples, failed_fraction):
    """
    Test mlmc.estimate.Estimate
    :return: None
    """
    # MLMC instance
    estimator = create_estimator(n_levels, n_samples, failed_fraction)

    estimate_covariance(estimator)
    estimate_n_samples_for_target_variance(estimator)
    estimate_cost(estimator)


def create_estimator(n_levels, n_samples, failed_fraction):
    mc = test.test_level.create_mc(n_levels=n_levels, n_samples=n_samples, failed_fraction=failed_fraction)
    mc.wait_for_simulations()
    return mlmc.estimate.Estimate(mc)


def estimate_n_samples_for_target_variance(estimator):
    """
    Check if number of estimated samples is increasing while target variance is decreasing
    :param estimator: mlmc.estimate.Estimate instance
    :return: None
    """
    target_vars = [1e-3, 1e-5, 1e-7, 1e-9]
    n_moments = 15
    moments_fn = mlmc.moments.Legendre(n_moments, estimator.estimate_domain(estimator.mlmc), safe_eval=True, log=False)

    prev_n_samples = np.zeros(len(estimator.levels))
    for var in target_vars:
        n_samples = estimator.estimate_n_samples_for_target_variance(var, moments_fn)

        for prev_n, curr_n in zip(prev_n_samples, n_samples):
            assert prev_n < curr_n


def estimate_cost(estimator):
    """
    Test estimate total cost
    :param estimator: mlmc.Estimate instance
    :return: None
    """
    n_ops_estimate = 1
    for level in estimator.mlmc.levels:
        level.n_ops_estimate = n_ops_estimate
    cost = estimator.estimate_cost()
    assert sum([n_sam * n_ops_estimate for n_sam in estimator.mlmc.n_samples]) == cost


def estimate_covariance(estimator):
    """
    Test covariance matrix symmetry
    :param estimator:
    :return: None
    """
    n_moments = 15
    moments_fn = mlmc.moments.Legendre(n_moments, estimator.estimate_domain(estimator.mlmc), safe_eval=True, log=False)
    cov = estimator.estimate_covariance(moments_fn, estimator.mlmc.levels)
    assert np.allclose(cov, cov.T, atol=1e-6)


@pytest.mark.skip
def test_target_var_adding_samples():
    """
    Test if adding samples converge to expected values
    :return: None
    """
    np.random.seed(2)
    distr = (stats.norm(loc=1, scale=2), False, '_sample_fn')

    n_levels = [1, 2, 5]
    n_moments = 31

    # Level samples for target variance = 1e-4 and 31 moments
    ref_level_samples = {1e-3: {1: [100],  2: [180, 110],  5: [425, 194, 44, 7, 3]},
                         1e-4: {1: [704],  2: [1916, 975],  5: [3737, 2842, 516, 67, 8]},
                         1e-5: {1: [9116],  2: [20424, 26154],  5: [40770, 34095, 4083, 633, 112]}
                         }

    target_var = [1e-3, 1e-4, 1e-5]

    for t_var in target_var:
        for nl in n_levels:
            d, il, sim = distr
            mc_test = TestMLMC(nl, n_moments, d, il, sim)

            mc_test.mc.set_initial_n_samples()
            mc_test.mc.refill_samples()
            mc_test.mc.wait_for_simulations()
            mc_test.estimator.target_var_adding_samples(t_var, mc_test.moments_fn, sleep=0)
            mc_test.mc.wait_for_simulations()

            assert sum(ref_level_samples[t_var][nl]) == sum([level.finished_samples for level in mc_test.mc.levels])


if __name__ == "__main__":
    test_estimate(3, [10000, 5000, 1000], failed_fraction=0.0)
    #test_target_var_adding_samples()
