import numpy as np
import scipy.stats as st


def t_test(mu_0, samples, max_p_val=0.01):
    """
    Test that mean of samples is mu_0, false
    failures with probability max_p_val.

    Perform the two-tailed t-test and
    Assert that p-val is smaller then given value.
    :param mu_0: Exact mean.
    :param samples: Samples to test.
    :param max_p_val: Probability of failed t-test for correct samples.
    """
    T, p_val = st.ttest_1samp(samples, mu_0)
    assert p_val < max_p_val


def chi2_test(var_0, samples, max_p_val=0.01, tag=""):
    """
    Test that variance of samples is sigma_0, false
    failures with probability max_p_val.
    :param sigma_0: Exact mean.
    :param samples: Samples to test.
    :param max_p_val: Probability of failed t-test for correct samples.
    """
    N = len(samples)
    var = np.var(samples)
    T = var * N / var_0
    pst = st.chi2.cdf(T, df=len(samples)-1)
    p_val = 2 * min(pst, 1 - pst)
    print("{}\n var: {} var_0: {} p-val: {}".format(tag, var, var_0, p_val))
    assert p_val > max_p_val


def anova(level_moments):
    """
    Analysis of variance
    :param level_moments: moments values per level
    :return: bool
    """
    # H0: all levels moments have same mean value
    f_value, p_value = st.f_oneway(*level_moments)

    # Significance level
    alpha = 0.05
    # Same means, can not be rejected H0
    if p_value > alpha:
        print("Same means, cannot be rejected H0")
        return True
    # Different means (reject H0)
    print("Different means, reject H0")
    return False
