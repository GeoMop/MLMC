import numpy as np
import scipy.stats as st


def t_test(mu_0, samples, max_p_val=0.01):
    """
    Perform a one-sample two-tailed t-test to check if the sample mean equals mu_0.

    This test ensures that false positives (rejecting H0 when true) occur with probability <= max_p_val.

    :param mu_0: Expected mean value of the samples.
    :param samples: Array-like of sample values to test.
    :param max_p_val: Maximum allowed p-value for false rejection (significance threshold).
    :raises AssertionError: if the p-value is larger than max_p_val, indicating the mean is statistically equal to mu_0.
    """
    # Perform one-sample t-test
    T, p_val = st.ttest_1samp(samples, mu_0)

    # Assert that the p-value is smaller than threshold, otherwise fail the test
    assert p_val < max_p_val, f"T-test failed: p_val={p_val}, threshold={max_p_val}"


def chi2_test(var_0, samples, max_p_val=0.01, tag=""):
    """
    Perform a chi-squared test to check if the sample variance equals var_0.

    False rejections should occur with probability <= max_p_val.

    :param var_0: Expected variance of the samples.
    :param samples: Array-like of sample values.
    :param max_p_val: Maximum allowed p-value for false rejection (significance threshold).
    :param tag: Optional string tag for printing/debugging.
    :raises AssertionError: if the p-value indicates variance significantly differs from var_0.
    """
    N = len(samples)
    var = np.var(samples, ddof=0)  # population variance
    T = var * N / var_0            # chi-squared statistic
    pst = st.chi2.cdf(T, df=N-1)   # cumulative probability
    p_val = 2 * min(pst, 1 - pst)  # two-tailed p-value

    # Print debug info
    print(f"{tag}\nvar: {var}, var_0: {var_0}, p-val: {p_val}")

    # Assert variance is consistent with expected variance
    assert p_val > max_p_val, f"Chi2-test failed: p_val={p_val}, threshold={max_p_val}"


def anova(level_moments):
    """
    Perform one-way ANOVA (analysis of variance) across multiple levels.

    Tests the null hypothesis H0: all levels have the same mean.

    :param level_moments: List of arrays, each array containing moments/samples for a level.
    :return: True if H0 cannot be rejected (means are statistically equal),
             False if H0 is rejected (at least one mean differs).
    """
    # Compute F-statistic and p-value
    f_value, p_value = st.f_oneway(*level_moments)

    alpha = 0.05  # significance level

    if p_value > alpha:
        # H0 cannot be rejected: means are statistically the same
        print("Same means, cannot reject H0")
        return True
    else:
        # H0 rejected: means differ significantly
        print("Different means, reject H0")
        return False
