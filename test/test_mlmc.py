import os
import sys
import numpy as np
import scipy.integrate as integrate
import scipy.stats as st
import scipy.stats as stats
# import statprof
import matplotlib.pyplot as plt
import matplotlib.cm as cm

src_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, src_path + '/../src/')
import mlmc.mlmc
import mlmc.simulation
import mlmc.moments
import mlmc.distribution
import mlmc.correlated_field as cf
from test.fixtures.mlmc_test_run import TestMLMC
from test.fixtures.synth_simulation import SimulationTest
import pbs as pb
import copy

import shutil

#from test.simulations.simulation_shooting import SimulationShooting





# def impl_var_estimate(n_levels, n_moments, target_var, distr, is_log=False):
#
#
#     raw_vars = []
#     reg_vars = []
#
#     n_loops=5
#     print("--")
#     for _ in range(n_loops):
#         mc.clean_levels()
#         mc.set_initial_n_samples()
#         mc.refill_samples()
#         mc.wait_for_simulations()
#         est_domain = mc.estimate_domain()
#         domain_err = [distr.cdf(est_domain[0]), 1-distr.cdf(est_domain[1])]
#         print("Domain error: ", domain_err )
#         assert np.sum(domain_err) < 0.01
#         moments_fn = lambda x, n=n_moments, a=est_domain[0], b=est_domain[1]: mlmc.moments.legendre_moments(x, n, a, b)
#         raw_var_estimate, n_col_samples = mc.estimate_diff_vars(moments_fn)
#
#         reg_var_estimate = mc.estimate_diff_vars_regression(moments_fn)
#         raw_vars.append(raw_var_estimate)
#         reg_vars.append(reg_var_estimate)
#     print("n samples   : ", n_col_samples)
#     print("ref var min    : ", np.min(vars[:, 1:], axis=1) )
#     print("ref var max    : ", np.max(vars, axis=1) )
#
#     # relative max over loops
#     print("max raw err: ", np.max( np.abs(np.array(raw_vars) - vars) / (1e-4 + vars), axis=0))
#     print("max reg err: ", np.max( np.abs(np.array(reg_vars) - vars) / (1e-4 + vars), axis=0))
#
#     # std over loops
#     # max (of relative std) over moments
#     print("std  raw var: ", np.max( np.sqrt(np.mean( (np.array(raw_vars) - vars)**2, axis=0)) / (1e-4 + vars) , axis=1) )
#     print("std  reg var: ", np.max( np.sqrt(np.mean( (np.array(reg_vars) - vars)**2, axis=0)) / (1e-4 + vars) , axis=1) )
#     #print("Linf raw var: ", np.max( np.abs(np.array(raw_vars) - vars), axis=0) / (1e-4 + vars))
#

def err_tuple(x):
    return (np.min(x), np.median(x), np.max(x))


def test_mlmc():
    """
    Test if mlmc moments correspond to exact moments from distribution
    :return: None
    """
    np.random.seed(3)   # To be reproducible
    n_levels = [9] #[1, 2, 3, 5, 7]
    n_moments = [8]

    distr = [
        (stats.norm(loc=1, scale=2), False, '_sample_fn'),
        (stats.norm(loc=1, scale=10), False, '_sample_fn'),
        (stats.lognorm(scale=np.exp(5), s=1), True, '_sample_fn'),  # worse conv of higher moments
        (stats.lognorm(scale=np.exp(-3), s=2), True, '_sample_fn'),  # worse conv of higher moments
        (stats.chi2(df=10), True, '_sample_fn'),
        (stats.chi2(df=5), True, '_sample_fn'),
        (stats.weibull_min(c=0.5), False, '_sample_fn'),  # Exponential
        (stats.weibull_min(c=1), False, '_sample_fn'),  # Exponential
        (stats.weibull_min(c=2), False, '_sample_fn'),  # Rayleigh distribution
        (stats.weibull_min(c=5, scale=4), False, '_sample_fn'),   # close to normal
        (stats.weibull_min(c=1.5), True, '_sample_fn'),  # Infinite derivative at zero
        #(stats.lognorm(scale=np.exp(-5), s=1), True, '_sample_fn_no_error'),
        #(stats.weibull_min(c=20), True, '_sample_fn_no_error'),   # Exponential
        # (stats.weibull_min(c=20), True, '_sample_fn_no_error'),   # Exponential
        #(stats.weibull_min(c=3), True, '_sample_fn_no_error')    # Close to normal
        ]

    level_moments_mean = []
    level_moments = []
    level_moments_var = []
    level_variance_diff = []
    var_mlmc = []
    number = 100

    for nl in n_levels:
        for nm in n_moments:
            for d, il, sim in distr:
                mc_test = TestMLMC(nl, nm, d, il, sim)
                # number of samples on each level

                total_samples = mc_test.mc.sample_range(10000, 100)
                mc_test.generate_samples(total_samples)
                total_samples = mc_test.mc.n_samples
                n_rep = 50
                subsamples = mc_test.mc.sample_range(1000, 3)
                mc_test.collect_subsamples(n_rep, subsamples)
                mc_test.test_variance_of_varaince()
                mc_test.test_mean_var_consistency()

                # test regression for initial sample numbers
                print("n_samples:", mc_test.mc.n_samples)
                mc_test.test_variance_regression()
                mc_test.mc.clean_subsamples()
                n_samples = mc_test.mc.estimate_n_samples_for_target_variance(0.0005, mc_test.moments_fn)
                n_samples = np.round(np.max(n_samples, axis=1)).astype(int)
                # n_samples by at most 0.8* total_samples
                scale = min(np.max(n_samples / total_samples) / 0.8, 1.0)
                # avoid to small number of samples
                n_samples = np.maximum((n_samples / scale).astype(int), 2)
                mc_test.collect_subsamples(n_rep, n_samples)
                # test regression for real sample numbers
                print("n_samples:", mc_test.mc.n_samples)
                mc_test.test_variance_regression()

                # level_var_diff = []
                # all_stdevs = []
                # all_means = []
                # var_mlmc_pom = []
                # for i in range(number):
                #     size = 3000
                #     mc_test.mc.subsample(nl*[size])
                #     # Moments as tuple (means, vars)
                #     means, vars = mc_test.mc.estimate_moments(mc_test.moments_fn)
                #
                #     # Remove first moment
                #     means = means[1:]
                #     vars = vars[1:]
                #
                #
                #     # Variances
                #     stdevs = np.sqrt(vars) * 3
                #
                #     var_mlmc_pom.append(vars)
                #
                #     all_stdevs.append(stdevs)
                #     all_means.append(means)
                #
                # # Exact moments from distribution
                # exact_moments = mlmc.distribution.compute_exact_moments(mc_test.moments_fn, d.pdf, 1e-10)[1:]
                #
                # means = np.mean(all_means, axis=0)
                # errors = np.mean(all_stdevs, axis=0)
                # var_mlmc.append(np.mean(var_mlmc_pom, axis=0))
                #
                # assert all(means[index] + errors[index] >= exact_mom >= means[index] - errors[index]
                #            for index, exact_mom in enumerate(exact_moments))

            #     if len(level_var_diff) > 0:
            #         # Average from more iteration
            #         level_variance_diff.append(np.mean(level_var_diff, axis=0))
            #
            # if len(level_var_diff) > 0:
            #     moments = []
            #     level_var_diff = np.array(level_var_diff)
            #     for index in range(len(level_var_diff[0])):
            #         moments.append(level_var_diff[:, index])
            #
            # level_moments.append(moments)
            # level_moments_mean.append(means)
            # level_moments_var.append(vars)

    # if len(level_moments) > 0:
    #     level_moments = np.array(level_moments)
    #     print("level_moments ", level_moments)
    #     print("level_moments 0 ", level_moments[0])
    #
    #     for index in range(len(level_moments[0])):
    #         anova(level_moments[:, index])
    #
    #     for l_mom in level_moments:
    #         anova(l_mom)
    #
    # if len(level_variance_diff) > 0:
    #     plot_diff_var(level_variance_diff, n_levels)
    #
    # Plot moment values
    #plot_vars(level_moments_mean, level_moments_var, n_levels, exact_moments)


#
# def test_var_subsample_regresion():
#     var_exp = -5
#
#     # n_levels = [1, 2, 3, 4, 5, 7, 9]
#     n_levels = [8]
#     n_moments = [20]  # ,4,5] #,7, 10,14,19]
#     distr = [
#         # (stats.norm(loc=42.0, scale=5), False)
#         (stats.lognorm(scale=np.exp(5), s=1), True)  # worse conv of higher moments
#         # (stats.lognorm(scale=np.exp(-5), s=0.5), True)         # worse conv of higher moments
#         # (stats.chi2(df=10), True)
#         # (stats.weibull_min(c=1), True)    # Exponential
#         # (stats.weibull_min(c=1.5), True)  # Infinite derivative at zero
#         # (stats.weibull_min(c=3), True)    # Close to normal
#     ]
#
#     level_moments_mean = []
#     level_moments_var = []
#
#     for nl in n_levels:
#         for nm in n_moments:
#             for d, il in distr:
#                 mc_test = TestMLMC(nl, nm, d, il)
#
#                 n_samples = mc_test.mc.set_target_variance(10 - 4)
#
#                 print("n of samples ", n_samples)
#
#
#
#     target_var = 10 ** var_exp
#     means_el = []
#     vars_el = []
#     n_loops = 30
#     for t_var in target_var:
#         n_samples = mc.set_target_variance(t_var, prescribe_vars=self.ref_diff_vars)
#         n_samples = np.max(n_samples, axis=1).astype(int)
#
#     pass


def variance_level(mlmc_test):
    x = np.arange(0, mlmc_test.n_moments, 1)

    for level in mlmc_test.mc.levels:
        print(level.level_idx)
        print(level.estimate_diff_var(mlmc_test.moments_fn)[0])
        #print("len vars ", len(level.estimate_diff_var(mlmc_test.moments_fn)[0]))
        plt.plot(x, level.estimate_diff_var(mlmc_test.moments_fn)[0], label="%d level" % (level.level_idx + 1))

    plt.legend()
    plt.show()


def var_subsample_independent():
    n_moments = [8]
    n_levels = [1, 2, 3, 5, 7, 9]

    distr = [
        (stats.norm(loc=1, scale=2), False, '_sample_fn'),
        (stats.lognorm(scale=np.exp(5), s=1), True),            # worse conv of higher moments
        (stats.lognorm(scale=np.exp(-5), s=0.5), True)         # worse conv of higher moments
        (stats.chi2(df=10), True)
        (stats.weibull_min(c=20), True)    # Exponential
        (stats.weibull_min(c=1.5), True)  # Infinite derivative at zero
        (stats.weibull_min(c=3), True)    # Close to normal
        ]

    level_variance_diff = []
    level_moments = []
    level_moments_mean = []
    level_moments_var = []
    for nl in n_levels:
        for nm in n_moments:
            for d, il, sim in distr:
                var_n = []
                mean_n = []
                level_var_diff = []

                for _ in range(10):
                    mc_test = TestMLMC(nl, nm, d, il, sim)
                    # 10 000 samples on each level
                    mc_test.mc.set_initial_n_samples(nl*[200])
                    mc_test.mc.refill_samples()
                    mc_test.mc.wait_for_simulations()
                    # Moments as tuple (means, vars)
                    moments = mc_test.mc.estimate_moments(mc_test.moments_fn)
                    # Remove first moment
                    exact_moments = moments[0][1:], moments[1][1:]

                    var_n.append(exact_moments[1])
                    mean_n.append(exact_moments[0])

                    means = np.mean(mean_n, axis=0)
                    vars = np.mean(var_n, axis=0)

                    mc_test = TestMLMC(nl, nm, d, il, sim)
                    # 10 000 samples on each level
                    mc_test.mc.set_initial_n_samples(nl * [1000])
                    mc_test.mc.refill_samples()
                    mc_test.mc.wait_for_simulations()
                    # Moments as tuple (means, vars)
                    moments = mc_test.mc.estimate_moments(mc_test.moments_fn)
                    # Remove first moment
                    exact_moments = moments[0][1:], moments[1][1:]

                    var_n.append(exact_moments[1])
                    mean_n.append(exact_moments[0])

                    #level_var_diff.append(var_subsample(mom, mc_test.mc, mc_test.moments_fn, n_subsamples=1000, sub_n=200, tol=10))

                    if len(level_var_diff) > 0:
                        # Average from more iteration
                        level_variance_diff.append(np.mean(level_var_diff, axis=0))

            if len(level_var_diff) > 0:
                moments = []
                level_var_diff = np.array(level_var_diff)
                for index in range(len(level_var_diff[0])):
                    moments.append(level_var_diff[:, index])

        level_moments.append(moments)
        level_moments_mean.append(means)
        level_moments_var.append(vars)


    # if len(level_moments) > 0:
    #     level_moments = np.array(level_moments)
    #     for index in range(len(level_moments[0])):
    #         anova(level_moments[:, index])
    #
    # if len(level_variance_diff) > 0:
    #     plot_diff_var(level_variance_diff, n_levels)

    exact_moments = mlmc.distribution.compute_exact_moments(mc_test.moments_fn, d.pdf, 1e-10)[1:]
    # Plot moment values
    #plot_vars(level_moments_mean, level_moments_var, n_levels, exact_moments)


    #
    #
    #
    # n_subsamples = 1000
    # sub_means = []
    # subsample_variance = np.zeros(n_moments[0]-1)
    # for _ in range(n_subsamples):
    #     for nl in n_levels:
    #         for nm in n_moments:
    #             for d, il in distr:
    #                 mc_test = TestMLMC(nl, nm, d, il)
    #                 # 10 000 samples on each level
    #                 mc_test.generate_samples(500)
    #                 # Moments as tuple (means, vars)
    #                 moments = mc_test.mc.estimate_moments(mc_test.moments_fn)
    #                 # Remove first moment
    #                 moments = moments[0][1:], moments[1][1:]
    #
    #                 sub_means.append(moments[0])
    #
    #     subsample_variance = np.add(subsample_variance,
    #                             (np.subtract(np.array(exact_moments[0]), np.array(moments[0]))) ** 2)
    #
    # # Sample variance
    # variance = subsample_variance / (n_subsamples - 1)
    # result = np.sqrt([a / b if b != 0 else a for a, b in zip(exact_moments[1], variance)])
    #
    # print("result sqrt(V/V*) ", result)
    # exact_moments = np.mean(sub_means, axis=0)
    #
    # result = np.sqrt([a / b if b != 0 else a for a, b in zip(exact_moments, variance)])
    # exit()
    # plot_diff_var(result, n_levels)


def var_subsample(moments, mlmc, moments_fn, n_subsamples=1000, sub_n=None, tol=2):
    """
    Compare moments variance from whole mlmc and from subsamples
    :param moments: moments from mlmc
    :param mlmc_test: mlmc object
    :param n_subsamples: number of subsamples (J in theory)
    :return: array, absolute difference between samples variance and mlmc moments variance
    """
    assert len(moments) == 2
    tolerance = tol

    # Variance from subsamples
    subsample_variance = np.zeros(len(moments[0]-1))
    all_means = []

    # Number of subsamples (J from theory)
    for _ in range(n_subsamples):
        subsamples = []
        # Clear number of subsamples
        mlmc.clear_subsamples()

        # Generate subsamples for all levels
        if sub_n is None:
            subsamples = [int(n_sub) for n_sub in mlmc.n_samples / 2]
        else:
            subsamples = np.ones(len(mlmc.n_samples), dtype=np.int8) * sub_n

        print("subsamples ", subsamples)
        print("n sub ", sub_n)

        # Process mlmc with new number of level samples
        if subsamples is not None:
            mlmc.subsample(subsamples)
        mlmc.refill_samples()
        mlmc.wait_for_simulations()

        print("mlmc n samples ", mlmc.n_samples)
        # Moments estimation from subsamples
        moments_mean_subsample = (mlmc.estimate_moments(moments_fn)[0])[1:]
        all_means.append(moments_mean_subsample)

    # SUM[(EX - X)**2]
    #subsample_variance = np.add(subsample_variance, (np.subtract(np.array(moments[0]), np.array(moments_mean_subsample))) ** 2)

    means = np.mean(all_means, axis=0)

    variance = np.sum([(means - m)**2 for m in all_means], axis=0)/(n_subsamples-1)

    # Sample variance
    #variance = subsample_variance/(n_subsamples-1)

    # Difference between arrays
    variance_abs_diff = np.abs(np.subtract(variance, moments[1]))

    result = np.sqrt([a/b if b != 0 else a for a, b in zip(moments[1], variance)])

    # Difference between variances must be in tolerance
    #assert all(var_diff < tolerance for var_diff in result)
    return result


def _test_shooting():
    """
    Test mlmc with shooting simulation
    :return: None
    """
    #np.random.seed(3)
    n_levels = [1, 2, 3, 5, 7]
    n_moments = [8]

    level_moments_mean = []
    level_moments = []
    level_moments_var = []
    level_variance_diff = []
    var_mlmc = []
    number = 1

    corr_field_object = cf.SpatialCorrelatedField(corr_exp='gauss', dim=1, corr_length=1,
                                                  aniso_correlation=None, mu=0.0, sigma=1, log=False)
    # Simulation configuration
    config = {'coord': np.array([0, 0]),
              'speed': np.array([10, 0]),
              'extremes': np.array([-200, 200, -200, 200]),
              'time': 10,
              'fields': corr_field_object
              }
    step_range = (0.1, 0.02)

    # Moments function
    true_domain = [-100, 50]

    for nl in n_levels:
        for nm in n_moments:
            level_var_diff = []
            all_variances = []
            all_means = []
            var_mlmc_pom = []
            for i in range(number):
                pbs = pb.Pbs()
                simulation_factory = SimulationShooting.factory(step_range, config=config)
                mc = mlmc.mlmc.MLMC(nl, simulation_factory, pbs)
                mc.create_levels()
                moments_fn = mlmc.moments.Legendre(nm, true_domain, False)

                mc.clean_levels()
                # Set initialize samples
                mc.set_initial_n_samples(nl* [500])
                mc.refill_samples()
                mc.wait_for_simulations()

                mc.set_target_variance(1e-1, moments_fn)
                mc.refill_samples()
                mc.wait_for_simulations()
                print("N samples ", mc.n_samples)

                # Moments as tuple (means, vars)
                moments = mc.estimate_moments(moments_fn)

                samples = mc.levels[0].sample_values

                # Remove first moment
                moments = moments[0], moments[1]

                #level_var_diff.append(var_subsample(moments, mc, moments_fn, sub_n=200))

                # Variances
                variances = np.sqrt(moments[1]) * 3

                var_mlmc_pom.append(moments[1])

                all_variances.append(variances)
                all_means.append(moments[0])

            moments_data = np.empty((len(all_means[0]), 2))

            moments_data[:, 0] = np.mean(all_means, axis=0)
            moments_data[:, 1] = np.var(all_variances, axis=0)

            print("moments data ", moments_data)
            print("moments function size ", moments_fn.size)

            # print("moments data ", moments_data)
            distr_obj = mlmc.distribution.Distribution(moments_fn, moments_data)
            # distr_obj.choose_parameters_from_samples()
            distr_obj.domain = moments_fn.domain
            # result = distr_obj.estimate_density(tol=0.0001)
            result = distr_obj.estimate_density_minimize(tol=1e-15)

            size = int(1e5)

            x = np.linspace(distr_obj.domain[0], distr_obj.domain[1], size)
            density = distr_obj.density(x)

            tol = 10
            last_density = density

            print("samples ", samples)

            print("density ", density)
            plt.plot(x, density, label="entropy density", color='red')
            plt.hist(samples, bins=10000)
            #plt.plot(x, d.pdf(x), label="pdf")
            plt.legend()
            plt.show()

            if nl == 1:
                avg = 0
                var = 0
                for i, level in enumerate(mc.levels):
                #print("level samples ", level._sample_values)
                            result = np.array(level._sample_values)[:, 0] - np.array(level._sample_values)[:, 1]
                            avg += np.mean(result)
                            var += np.var(result)/len(level._sample_values)

                distr = stats.norm(loc=avg, scale=np.sqrt(var))

            # Exact moments from distribution
            exact_moments = mlmc.distribution.compute_exact_moments(moments_fn, distr.pdf, 1e-10)[1:]

            means = np.mean(all_means, axis=0)
            vars = np.mean(all_variances, axis=0)
            var_mlmc.append(np.mean(var_mlmc_pom, axis=0))

            print(all(means[index] + vars[index] >= exact_mom >= means[index] - vars[index]
                      for index, exact_mom in enumerate(exact_moments)))

            if len(level_var_diff) > 0:
                # Average from more iteration
                level_variance_diff.append(np.mean(level_var_diff, axis=0))

        if len(level_var_diff) > 0:
            moments = []
            level_var_diff = np.array(level_var_diff)
            for index in range(len(level_var_diff[0])):
                moments.append(level_var_diff[:, index])

        level_moments.append(moments)
        level_moments_mean.append(means)
        level_moments_var.append(vars)

    if len(level_moments) > 0:
        level_moments = np.array(level_moments)
        for index in range(len(level_moments[0])):
            anova(level_moments[:, index])

    if len(level_variance_diff) > 0:
        plot_diff_var(level_variance_diff, n_levels)

    #plot_vars(level_moments_mean, level_moments_var, n_levels, exact_moments)

    # for _ in range(number):
    #         # Create MLMC object
    #         pbs = flow_pbs.FlowPbs()
    #         simulation_factory = SimulationShooting.factory(step_range, config=config)
    #         mc = mlmc.mlmc.MLMC(nl, simulation_factory, pbs)
    #         moments_fn = mlmc.moments.Monomial(nm, true_domain, False)
    #
    #         #for _ in range(10):
    #         mc.clean_levels()
    #         # Set initialize samples
    #         mc.set_initial_n_samples(nl * [500])
    #         mc.refill_samples()
    #         mc.wait_for_simulations()
    #
    #
    #
    #         #print("n samples ", mc.n_samples)
    #
    #
    #
    #         # Set other samples if it is necessary
    #         # mc.set_target_variance(1e-4, moments_fn)
    #         # mc.refill_samples()
    #         # mc.wait_for_simulations()
    #         # print("N ", mc.n_samples)
    #
    #         # Moments as tuple (means, vars)
    #         moments = mc.estimate_moments(moments_fn)
    #         level_var_diff.append(var_subsample(moments, mc))
    #         #level_variance_diff.append(var_subsample(moments, mc_test))

    #         avg = 0
    #         var = 0
    #
    #         for i, level in enumerate(mc.levels):
    #             #print("level samples ", level._sample_values)
    #             result = np.array(level._sample_values)[:, 0] - np.array(level._sample_values)[:, 1]
    #             avg += np.mean(result)
    #             var += np.var(result)/len(level._sample_values)
    #         print("avg ", avg)
    #         print("var ", var)
    #
    #         if nl == 1:
    #             distr = stats.norm(loc=avg, scale=np.sqrt(var))
    #
    #         # Remove first moment
    #         moments = moments[0][1:], moments[1][1:]
    #
    #         # Variances
    #         variances = np.sqrt(moments[1]) * 4
    #
    #         print("moments fn domain ", moments_fn.domain)
    #
    #         # Exact moments from distribution
    #         exact_moments = mlmc.distribution.compute_exact_moments(moments_fn, distr.pdf, 1e-10)[1:]
    #
    #         # all(moments[0][index] + variances[index] >= exact_mom >= moments[0][index] - variances[index]
    #         #          for index, exact_mom in enumerate(exact_moments))
    #
    #     level_moments_mean.append(moments[0])
    #     level_moments_var.append(variances)
    #
    # # Plot moment values
    # plot_vars(level_moments_mean, level_moments_var, n_levels, exact_moments)





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


def plot_diff_var(level_variance_diff, n_levels):
    """
    Plot diff between V* and V
    :param level_variance_diff: array of each moments sqrt(V/V*)
    :param n_levels: array, number of levels
    :return: None
    """
    if len(level_variance_diff) > 0:
        colors = iter(cm.rainbow(np.linspace(0, 1, len(level_variance_diff) + 1)))
        x = np.arange(0, len(level_variance_diff[0]))
        [plt.plot(x, var_diff, 'o', label="%dLMC" % n_levels[index], color=next(colors)) for index, var_diff in enumerate(level_variance_diff)]
        plt.legend()
        plt.ylabel(r'$ \sqrt{\frac{V}{V^{*}}}$', rotation=0)
        plt.xlabel("moments")
        plt.show()

        # Levels on x axes
        moments = []
        level_variance_diff = np.array(level_variance_diff)

        print("level variance diff ", level_variance_diff)
        for index in range(len(level_variance_diff[0])):
            moments.append(level_variance_diff[:, index])

        print("moments ", moments)
        print("n levels ", n_levels)
        colors = iter(cm.rainbow(np.linspace(0, 1, len(moments) + 1)))
        #x = np.arange(1, len(moments[0]) +1)
        [plt.plot(n_levels, moment, 'o', label=index+1, color=next(colors)) for index, moment in
         enumerate(moments)]
        plt.ylabel(r'$ \sqrt{\frac{V}{V^{*}}}$', rotation=0)
        plt.xlabel("levels method")
        plt.legend(title="moments")
        plt.show()


def plot_vars(moments_mean, moments_var, n_levels, exact_moments=None, ex_moments=None):
    """
    Plot means with variance whiskers
    :param moments_mean: array, moments mean
    :param moments_var: array, moments variance
    :param n_levels: array, number of levels
    :param exact_moments: array, moments from distribution
    :param ex_moments: array, moments from distribution samples
    :return: 
    """
    colors = iter(cm.rainbow(np.linspace(0, 1, len(moments_mean) + 1)))

    x = np.arange(0, len(moments_mean[0]))
    x = x - 0.3
    default_x = x

    for index, means in enumerate(moments_mean):
        if index == int(len(moments_mean)/2) and exact_moments is not None:
            plt.plot(default_x, exact_moments, 'ro', label="Exact moments")
        else:
            x = x + (1 / (len(moments_mean)*1.5))
            plt.errorbar(x, means, yerr=moments_var[index], fmt='o', capsize=3, color=next(colors), label="%dLMC" % n_levels[index])

    if ex_moments is not None:
        plt.plot(default_x-0.125, ex_moments, 'ko', label="Exact moments")

    plt.legend()
    plt.show()
    exit()


def plot_distribution(distr, size=20000):
    x = distr.rvs(size=size)
    plt.hist(x, bins=50, normed=1)
    plt.show()
    exit()


def check_estimates_for_nans(mc, distr):
    # test that estimates work even with nans
    n_moments = 4
    true_domain = distr.ppf([0.001, 0.999])
    moments_fn = mlmc.moments.Fourier(n_moments, true_domain)
    moments, vars = mc.estimate_moments(moments_fn)
    assert not np.any(np.isnan(moments))
    assert not np.any(np.isnan(vars))


def test_save_load_samples():
    # 1. make some samples in levels
    # 2. copy key data from levels
    # 3. clean levels
    # 4. create new mlmc object
    # 5. read stored data
    # 6. check that they match the reference copy

    work_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '_test_tmp')
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    n_levels = 5
    distr = stats.norm()
    step_range = (0.8, 0.01)

    simulation_config = dict(
        distr=distr, complexity=2, nan_fraction=0.4, sim_method='_sample_fn')
    simulation_factory = SimulationTest.factory(step_range, config=simulation_config)

    mlmc_options = {'output_dir': work_dir,
                    'keep_collected': True,
                    'regen_failed': False}

    mc = mlmc.mlmc.MLMC(n_levels, simulation_factory, step_range, mlmc_options)
    mc.create_new_execution()
    mc.set_initial_n_samples()
    mc.refill_samples()
    mc.wait_for_simulations()
    check_estimates_for_nans(mc, distr)

    level_data = []
    # Levels collected samples
    for level in mc.levels:
        l_data = (copy.deepcopy(level.scheduled_samples),
                  copy.deepcopy(level.collected_samples),
                  level.sample_values)
        assert not np.isnan(level.sample_values).any()
        level_data.append(l_data)

    mc.clean_levels()
    # Check NaN values
    for level in mc.levels:
        assert not np.isnan(level.sample_values).any()

    # New mlmc
    mc = mlmc.mlmc.MLMC(n_levels, simulation_factory, step_range, mlmc_options)
    mc.load_from_file()
    check_estimates_for_nans(mc, distr)

    # Test
    for level, data in zip(mc.levels, level_data):
        # Collected sample results must be same
        scheduled, collected, values = data
        # Compare scheduled and collected samples with saved one
        _compare_samples(scheduled, level.scheduled_samples)
        _compare_samples(collected, level.collected_samples)


def _compare_samples(saved_samples, current_samples):
    """
    Compare two list of samples
    :param saved_samples: List of tuples - [(fine Sample(), coarse Sample())], from log
    :param current_samples: List of tuples - [(fine Sample(), coarse Sample())]
    :return: None
    """
    saved_samples = sorted(saved_samples, key=lambda sample_tuple: sample_tuple[0].sample_id)
    current_samples = sorted(current_samples, key=lambda sample_tuple: sample_tuple[0].sample_id)
    for (coll_fine, coll_coarse), (coll_fine_s, coll_coarse_s) in zip(saved_samples, current_samples):
        assert coll_coarse == coll_coarse_s
        assert coll_fine == coll_fine_s


def _test_regression(distr_cfg, n_levels, n_moments):
    step_range = (0.8, 0.01)
    distr, log_distr, simulation_fn = distr_cfg
    simulation_config = dict(distr=distr, complexity=2, nan_fraction=0, sim_method=simulation_fn)
    simultion_factory = SimulationTest.factory(step_range, config=simulation_config)

    mlmc_options = {'output_dir': None,
                    'keep_collected': True,
                    'regen_failed': False}
    mc = mlmc.mlmc.MLMC(n_levels, simultion_factory, step_range, mlmc_options)
    mc.create_levels()
    sims = [level.fine_simulation for level in mc.levels]

    mc.set_initial_n_samples()
    mc.refill_samples()
    mc.wait_for_simulations()
    check_estimates_for_nans(mc, distr)

    # Copy level data
    level_data = []
    for level in mc.levels:
        l_data = (level.running_simulations.copy(),
                   level.finished_simulations.copy(),
                   level.sample_values)
        assert not np.isnan(level.sample_values).any()
        level_data.append(l_data)
    mc.clean_levels()

    # New mlmc
    mc = mlmc.mlmc.MLMC(n_levels, simulation_factory, step_range, mlmc_options)
    mc.load_from_setup()

    check_estimates_for_nans(mc, distr)

    # Test
    for level, data in zip(mc.levels, level_data):
        run, fin, values = data

        assert run == level.running_simulations
        assert fin == level.finished_simulations
        assert np.allclose(values, level.sample_values)

# def test_regression():
#     n_levels = [1, 2, 3, 5, 7]
#     n_moments = [5]
#
#     distr = [
#         (stats.norm(loc=1, scale=2), False, '_sample_fn'),
#         (stats.lognorm(scale=np.exp(5), s=1), True, '_sample_fn'),            # worse conv of higher moments
#         (stats.lognorm(scale=np.exp(-5), s=1), True, '_sample_fn_no_error'),
#         (stats.chi2(df=10), True, '_sample_fn'),
#         (stats.weibull_min(c=20), True, '_sample_fn_no_error'),   # Exponential
#         (stats.weibull_min(c=1.5), True, '_sample_fn'),  # Infinite derivative at zero
#         (stats.weibull_min(c=3), True, '_sample_fn_no_error')    # Close to normal
#         ]
#
#     level_moments_mean = []
#     level_moments = []
#     level_moments_var = []
#     level_variance_diff = []
#     var_mlmc = []
#     number = 10
#
#     for nl in n_levels:
#         for nm in n_moments:
#             for d, il, sim in distr:
#                 _test_regression(distr, n_levels, n_moments)

if __name__ == '__main__':
    # @TODO fox subsample error
    #test_mlmc()
    test_save_load_samples()
    #_test_shooting()

