"""
Test class monomials
"""
import numpy as np
import mlmc.moments
import mlmc.tool.distribution
import scipy.integrate as integrate
import scipy.stats as stats


def test_monomials():
    # Natural domain (0,1).
    size = 5  # Number of moments
    values = np.array([-2, -1, -0.5, 0, 0.5, 1, 2])
    # Reference values of moments
    ref = [values**r for r in range(size)]

    # Monomials moments object
    moments_fn = mlmc.moments.Monomial(size, safe_eval=False)

    # Calculated moments
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
    assert np.abs(np.mean(moments[:, 1])) < 0.1


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


def test_legendre():
    # Natural domain (0,1).
    size = 4
    moments_fn = mlmc.moments.Legendre(size, (-1.0, 1.0))

    values = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    moments = moments_fn(values)
    ref = [np.ones_like(values), values, (3*values**2 - 1.0) / 2.0, (5*values**3 - 3 * values) / 2.0]

    assert np.allclose(np.array(ref).T, moments)


def test_moments():
    # Natural domain (0,1).
    size = 5  # Number of moments
    values = np.array([-2, -1, -0.5, 0, 0.5, 1, 2])
    # Reference values of moments
    ref = [values ** r for r in range(size)]

    # Monomials moments object
    moments_fn = mlmc.moments.Monomial(size, safe_eval=False)

    # Calculated moments
    moments = moments_fn(values)
    assert np.allclose(np.array(ref).T, moments)

    # Given domain (a,b).
    a, b = (-1, 3)
    moments_fn = mlmc.moments.Monomial(size, (a, b), safe_eval=False)
    moments = moments_fn((b - a) * values + a)
    assert np.allclose(np.array(ref).T, moments)

    # Approximate mean.
    values = np.random.randn(1000)
    moments_fn = mlmc.moments.Monomial(2, safe_eval=False)
    moments = moments_fn(values)
    assert np.abs(np.mean(moments[:, 1])) < 0.1


# def compare_norm_lognorm():
#     """
#     Compare moments from lognorm and equivalent norm distribution
#     :return: None
#     """
#     # from statsmodels.distributions.empirical_distribution import ECDF
#
#     norm = stats.norm(loc=-5, scale=1)
#     lognorm = stats.lognorm(scale=np.exp(-5), s=1)
#
#     size = 9
#     samples_size = 10000
#     #Moments function domain for norm distribution
#     domain = norm.ppf([0.001, 0.999])
#     norm_mean_moments, norm_var_moments, moments_fn_norm = get_moments(norm, size, domain, False, True, samples_size)
#
#     # Exact norm moments
#     exact_norm_moments = get_exact_moments(norm, moments_fn_norm)
#     #print("exact norm moments ", exact_norm_moments)
#
#     # Moments function domain for lognorm distribution
#     domain = lognorm.ppf([0.001, 0.999])
#     lognorm_mean_moments, lognorm_var_moments, moments_fn_lognorm = get_moments(lognorm, size, domain, True, True, samples_size)
#
#
#     # Exact lognorm moments
#     exact_lognorm_moments = get_exact_moments(lognorm, moments_fn_lognorm)
#
#     x = np.arange(0, len(lognorm_mean_moments), 1)
#
#     mc_mean, mc_var = test_one_level()
#     print(mc_var)
#
#     test_mlmc_moments()
#
#     plt.plot(x, norm_mean_moments, 'bo', label="norm")
#     plt.errorbar(x, norm_mean_moments, yerr=norm_var_moments, fmt='o', capsize=3, color='blue')
#     plt.plot(x, lognorm_mean_moments, 'ro', label="lognorm")
#     plt.errorbar(x, lognorm_mean_moments, yerr=lognorm_var_moments, fmt='o', capsize=3, color='red')
#     plt.plot(x, exact_norm_moments, 'o', label="norm exact")
#     plt.plot(x, exact_lognorm_moments, 'o', label="lognorm exact")
#     plt.plot(x, mc_mean, 'og', label="MC")
#     plt.errorbar(x, mc_mean, yerr=mc_var, fmt='o', capsize=3, color='green')
#     plt.legend()
#     plt.show()


def get_moments(distr, size, domain, log=False, safe_eval=True, samples_size=10000):
    """
    Get moments from distribution 
    :param distr: distribution type
    :param size: number of moments
    :param domain: domain of moments function
    :param log: logaritmic transformation
    :param safe_eval: mask outliers
    :param samples_size: number of samples from distribution
    :return: tuple (moments_mean, moments_var)
    """
    moments_fn = mlmc.moments.Legendre(size, domain, log=log, safe_eval=safe_eval)
    moments = moments_fn(distr.rvs(size=samples_size))

    return np.mean(moments, axis=0)[1:], np.var(moments, axis=0)[1:]/samples_size, moments_fn


def get_exact_moments(distribution, moments_fn):
    """
    Get exact moments from given distribution
    :param distribution: Distribution object
    :param moments_fn: Function for generating moments
    :return: Exact moments
    """
    integrand = lambda x: moments_fn(x).T * distribution.pdf(x)

    a, b = moments_fn.domain
    integral = integrate.fixed_quad(integrand, a, b, n=moments_fn.size*5)[0]
    return integral[1:]


def _test_one_level():
    mc_samples = 10000
    size = 9
    lognorm = stats.lognorm(scale=np.exp(-5), s=1)
    domain = lognorm.ppf([0.001, 0.999])
    moments_fn_lognorm = mlmc.moments.Legendre(size, domain, log=True, safe_eval=True)

    all_moments = []
    samples = []
    for i in range(mc_samples):
        s = lognorm.rvs(size=1)
        samples.append(s)
        all_moments.append(moments_fn_lognorm(s))

    last_moments = []
    for moments in all_moments:
        print(moments[0][-1])
        last_moments.append(moments[0][-1])

    #return np.mean(all_moments, axis=0)[0][1:], np.var(all_moments, axis=0)[0][1:]/mc_samples

    # Exact lognorm moments
    exact_lognorm_moments = get_exact_moments(lognorm, moments_fn_lognorm)
    x = np.arange(0, len(exact_lognorm_moments), 1)

    plt.plot(x, np.mean(all_moments, axis=0)[0], 'og', label="MC")
    plt.errorbar(x, np.mean(all_moments, axis=0)[0], yerr=np.var(all_moments, axis=0)[0], fmt='o', capsize=3,
                 color='green')
    plt.legend()
    plt.show()


def _test_legendre():
    """
    Does not work
    :return: None 
    """
    t = np.random.rand(1000000)*2-1
    t= stats.lognorm(loc=np.exp(1), s=1).rvs(size=100000)
    print(t[0:30])
    print(np.mean(t))
    print(np.sqrt(np.var(t)))
    print("min ", np.min(t))
    print("max ", np.max(t))
    n_moments = 20
    moments = np.polynomial.legendre.legvander(t, deg=(n_moments - 1))

    print("number of samples ", len(moments))
    x = np.arange(0, n_moments)

    print("moments means ", np.mean(moments, axis=0))
    plt.plot(x, np.mean(moments, axis=0), 'og', label="MC")
    plt.errorbar(x, np.mean(moments, axis=0), yerr=np.var(moments, axis=0), fmt='o', capsize=3,
                 color='green')
    plt.legend()
    plt.show()


def plot_distribution():
    size = 100000
    x = stats.norm(loc=42.0, scale=5).rvs(size=size)
    # (stats.norm(loc=-5, scale=1), False),
    #(stats.lognorm(scale=np.exp(-5), s=1), True)  # worse conv of higher moments
    # (stats.lognorm(scale=np.exp(-5), s=0.5), True)         # worse conv of higher moments
    # (stats.chi2(df=10), True)
    x = stats.weibull_min(c=3).rvs(size=size)   # Exponential
    # (stats.weibull_min(c=1.5), True)  # Infinite derivative at zero
    # (stats.weibull_min(c=3), True)    # Close to normal
    print("t mean ", np.mean(x))
    print("t var ", np.var(x))
    print("delka t ", len(x))
    plt.hist(x, bins=10000, normed=1)
    plt.show()
    exit()


def test_transform():
    size = 5
    domain = [-1.0, 1.0]
    values = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

    moments_fn = mlmc.moments.Legendre(size, domain, log=False, safe_eval=True)

    matrix = np.eye(size)
    transformed_moments = mlmc.moments.TransformedMoments(moments_fn, matrix)
    mom = moments_fn(values)
    trans_mom = transformed_moments(values)
    expected_trans_mom = np.matmul(mom, matrix.T)

    np.allclose(mom, trans_mom)
    np.allclose(mom, expected_trans_mom)

    matrix = np.ones((size, size))
    transformed_moments = mlmc.moments.TransformedMoments(moments_fn, matrix)
    mom = moments_fn(values)
    trans_mom = transformed_moments(values)
    expected_trans_mom = np.matmul(mom, matrix.T)

    np.allclose(expected_trans_mom, trans_mom)


test_legendre()
