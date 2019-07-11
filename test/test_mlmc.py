import os
import sys
import shutil
import numpy as np
import scipy.stats as stats
import re
import test.stats_tests

src_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, src_path + '/../src/')
import mlmc.mlmc
import mlmc.simulation
import mlmc.moments
import mlmc.distribution
import mlmc.estimate
import mlmc.plot
import mlmc.correlated_field as cf
from test.fixtures.mlmc_test_run import TestMLMC
from test.fixtures.synth_simulation import SimulationTest
from test.simulations.simulation_shooting import SimulationShooting
import mlmc.pbs as pb
import copy
from memory_profiler import profile


#@profile
def test_mlmc():
    """
    Test if mlmc moments correspond to exact moments from distribution
    :return: None
    """
    #np.random.seed(3)   # To be reproducible
    n_levels = [1] #[1, 2, 3, 5, 7]
    n_moments = [8]

    clean = True

    work_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'fixtures')

    if clean:
        for f in os.listdir(work_dir):
            if re.search(r".\.hdf5", f):
                os.remove(os.path.join(work_dir, f))

    distr = [
        (stats.norm(loc=10, scale=2), False, '_sample_fn'),
        # (stats.norm(loc=1, scale=10), False, '_sample_fn'),
        # (stats.lognorm(scale=np.exp(5), s=1), True, '_sample_fn'),  # worse conv of higher moments
        # (stats.lognorm(scale=np.exp(-3), s=2), True, '_sample_fn'),  # worse conv of higher moments
        # (stats.chi2(df=10), True, '_sample_fn'),
        # (stats.chi2(df=5), True, '_sample_fn'),
        # (stats.weibull_min(c=0.5), False, '_sample_fn'),  # Exponential
        # (stats.weibull_min(c=1), False, '_sample_fn'),  # Exponential
        # (stats.weibull_min(c=2), False, '_sample_fn'),  # Rayleigh distribution
        # (stats.weibull_min(c=5, scale=4), False, '_sample_fn'),   # close to normal
        # (stats.weibull_min(c=1.5), True, '_sample_fn'),  # Infinite derivative at zero
        #(stats.lognorm(scale=np.exp(-5), s=1), True, '_sample_fn_no_error'),
        #(stats.weibull_min(c=20), True, '_sample_fn_no_error'),   # Exponential
        # (stats.weibull_min(c=20), True, '_sample_fn_no_error'),   # Exponential
        #(stats.weibull_min(c=3), True, '_sample_fn_no_error')    # Close to normal
        ]

    for nl in n_levels:
        for nm in n_moments:
            for d, il, sim in distr:
                mc_test = TestMLMC(nl, nm, d, il, sim)
                # number of samples on each level
                estimator = mlmc.estimate.Estimate(mc_test.mc)

                mc_test.mc.set_initial_n_samples()#[10000])
                mc_test.mc.refill_samples()
                mc_test.mc.wait_for_simulations()

                mc_test.mc.select_values({"quantity": (b"quantity_1", "=")})
                estimator.target_var_adding_samples(0.00001, mc_test.moments_fn)

                #mc_test.mc.clean_select()
                #mc_test.mc.select_values({"quantity": (b"quantity_1", "=")})

                cl = mlmc.estimate.CompareLevels([mc_test.mc],
                                   output_dir=src_path,
                                   quantity_name="Q [m/s]",
                                   moment_class=mlmc.moments.Legendre,
                                   log_scale=False,
                                   n_moments=nm, )

                cl.plot_densities()

                mc_test.mc.update_moments(mc_test.moments_fn)

                #total_samples = mc_test.mc.sample_range(10000, 100)
                #mc_test.generate_samples(total_samples)
                total_samples = mc_test.mc.n_samples

                mc_test.collect_subsamples(1, 1000)
                #
                # mc_test.test_variance_of_variance()
                mc_test.test_mean_var_consistency()

                #mc_test._test_min_samples() # No asserts, just diff var plot and so on

                # test regression for initial sample numbers

                print("n_samples:", mc_test.mc.n_samples)
                mc_test.test_variance_regression()
                mc_test.mc.clean_subsamples()
                n_samples = mc_test.estimator.estimate_n_samples_for_target_variance(0.0005, mc_test.moments_fn)
                n_samples = np.round(np.max(n_samples, axis=0)).astype(int)
                # n_samples by at most 0.8* total_samples
                scale = min(np.max(n_samples / total_samples) / 0.8, 1.0)
                # avoid to small number of samples
                n_samples = np.maximum((n_samples / scale).astype(int), 2)
                #mc_test.collect_subsamples(n_rep, n_samples)
                # test regression for real sample numbers
                print("n_samples:", mc_test.mc.n_samples)
                mc_test.test_variance_regression()


def _test_shooting():
    """
    Test mlmc with shooting simulation
    :return: None
    """
    #np.random.seed(3)
    n_levels = [1, 2]#, 2, 3, 5, 7]
    n_moments = [8]

    level_moments_mean = []
    level_moments = []
    level_moments_var = []
    level_variance_diff = []
    var_mlmc = []
    number = 1

    for nl in n_levels:
        for nm in n_moments:
            level_var_diff = []
            all_variances = []
            all_means = []
            var_mlmc_pom = []
            for i in range(number):
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

                simulation_factory = SimulationShooting.factory(step_range, config=config)

                work_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '_test_tmp')
                if os.path.exists(work_dir):
                        shutil.rmtree(work_dir)
                os.makedirs(work_dir)

                mlmc_options = {'output_dir': work_dir,
                                'keep_collected': True,
                                'regen_failed': False}

                mc = mlmc.mlmc.MLMC(nl, simulation_factory, step_range, mlmc_options)
                mc.create_new_execution()
                mc.set_initial_n_samples(nl * [500])
                mc.refill_samples()
                mc.wait_for_simulations()

                estimator = mlmc.estimate.Estimate(mc)

                moments_fn = mlmc.moments.Legendre(nm, true_domain, False)
                estimator.target_var_adding_samples(1e-3, moments_fn)
                print("N samples ", mc.n_samples)

                # Moments as tuple (means, vars)
                moments = estimator.estimate_moments(moments_fn)
                samples = mc.levels[0].sample_values

                print("moments ", moments)

                # Remove first moment
                moments = moments[0], moments[1]

                # Variances
                variances = np.sqrt(moments[1]) * 3

                var_mlmc_pom.append(moments[1])

                all_variances.append(variances)
                all_means.append(moments[0])

            moments_data = np.empty((len(all_means[0]), 2))

            moments_data[:, 0] = np.mean(all_means, axis=0)
            moments_data[:, 1] = np.var(all_variances, axis=0)

            # print("moments data ", moments_data)
            # print("moments function size ", moments_fn.size)
            #
            # print("moments data ", moments_data)
            # distr_obj = mlmc.simple_distribution.Distribution(moments_fn, moments_data)
            # # distr_obj.choose_parameters_from_samples()
            # distr_obj.domain = moments_fn.domain
            # # result = distr_obj.estimate_density(tol=0.0001)
            # result = distr_obj.estimate_density_minimize(tol=1e-15)
            #
            # size = int(1e5)
            #
            # x = np.linspace(distr_obj.domain[0], distr_obj.domain[1], size)
            # density = distr_obj.density(x)
            #
            # tol = 10
            # last_density = density

            # print("samples ", samples)
            #
            # print("density ", density)
            # plt.plot(x, density, label="entropy density", color='red')
            # plt.hist(samples, bins=10000)
            # #plt.plot(x, d.pdf(x), label="pdf")
            # plt.legend()
            # plt.show()

            if nl == 1:
                # avg = 0
                # var = 0
                # for i, level in enumerate(mc.levels):
                # #print("level samples ", level._sample_values)
                #             result = np.array(level._sample_values)[:, 0] - np.array(level._sample_values)[:, 1]
                #             avg += np.mean(result)
                #             var += np.var(result)/len(level._sample_values)
                #
                # distr = stats.norm(loc=avg, scale=np.sqrt(var))

                exact_moments = moments_data[:, 0]

                # Exact moments from distribution
                #exact_moments = mlmc.distribution.compute_exact_moments(moments_fn, distr.pdf, 1e-10)[1:]

            means = np.mean(all_means, axis=0)
            vars = np.mean(all_variances, axis=0)
            var_mlmc.append(np.mean(var_mlmc_pom, axis=0))

            for index, exact_mom in enumerate(exact_moments):
                print(means[index] + vars[index] >= exact_mom >= means[index] - vars[index])

            print(all(means[index] + vars[index] >= exact_mom >= means[index] - vars[index]
                      for index, exact_mom in enumerate(exact_moments)))



    #         if len(level_var_diff) > 0:
    #             # Average from more iteration
    #             level_variance_diff.append(np.mean(level_var_diff, axis=0))
    #
    #     if len(level_var_diff) > 0:
    #         moments = []
    #         level_var_diff = np.array(level_var_diff)
    #         for index in range(len(level_var_diff[0])):
    #             moments.append(level_var_diff[:, index])
    #
    #     level_moments.append(moments)
    #     level_moments_mean.append(means)
    #     level_moments_var.append(vars)
    #
    # if len(level_moments) > 0:
    #     level_moments = np.array(level_moments)
    #     for index in range(len(level_moments[0])):
    #         stats_tests.anova(level_moments[:, index])
    #
    # if len(level_variance_diff) > 0:
    #     mlmc.plot.plot_diff_var(level_variance_diff, n_levels)
    #
    # # # Plot moment values
    # mlmc.plot.plot_vars(level_moments_mean, level_moments_var, n_levels, exact_moments)


def check_estimates_for_nans(mc, distr):
    # test that estimates work even with nans
    n_moments = 4
    true_domain = distr.ppf([0.001, 0.999])
    moments_fn = mlmc.moments.Legendre(n_moments, true_domain)
    mlmc_est = mlmc.estimate.Estimate(mc)
    moments, vars = mlmc_est.estimate_moments(moments_fn)
    assert not np.any(np.isnan(moments))
    assert not np.any(np.isnan(vars))


def test_save_load_samples():
    # 1. make some samples in levels
    # 2. copy key data from levels
    # 3. clean levels
    # 4. create new mlmc object
    # 5. read stored data
    # 6. check that they match the reference copy
    clean = True
    work_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '_test_tmp')
    if os.path.exists(work_dir):
        if clean:
            shutil.rmtree(work_dir)
            #os.makedirs(work_dir)

    os.makedirs(work_dir)

    n_levels = 1
    distr = stats.norm()
    step_range = (0.8, 0.01)

    simulation_config = dict(
        distr=distr, complexity=2, nan_fraction=0.2, sim_method='_sample_fn')
    simulation_factory = SimulationTest.factory(step_range, config=simulation_config)

    mlmc_options = {'output_dir': work_dir,
                    'keep_collected': True,
                    'regen_failed': False}

    mc = mlmc.mlmc.MLMC(n_levels, simulation_factory, step_range, mlmc_options)
    if clean:
        mc.create_new_execution()
    else:
        mc.load_from_file()
    mc.set_initial_n_samples()
    mc.refill_samples()
    mc.wait_for_simulations()

    #check_estimates_for_nans(mc, distr)

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
    # import pstats
    #import cProfile

    test_mlmc()

    # cProfile.run('test_mlmc()', 'mlmctest')
    # p = pstats.Stats('mlmctest')
    # p.sort_stats('cumulative').print_stats()

    #test_save_load_samples()
