import os.path
import numpy as np
from mlmc.mlmc import MLMC
from mlmc import moments
import mlmc.plot
import mlmc.estimate
from test.fixtures.synth_simulation import SimulationTest


class TestMLMC:
    def __init__(self, n_levels, n_moments, distr, is_log=False, sim_method=None, quantile=None):
        """
        Create TestMLMC object instance
        :param n_levels: number of levels
        :param n_moments: number of moments
        :param distr: distribution object
        :param is_log: use logarithm of moments
        :param sim_method: name of simulation method
        :param quantile: quantiles of domain determination
        """
        # Not work for one level method
        print("\n")
        print("L: {} R: {} distr: {} sim: {}".format(n_levels, n_moments, distr.dist.__class__.__name__, sim_method))

        self.distr = distr
        self.n_levels = n_levels
        self.n_moments = n_moments
        self.is_log = is_log

        # print("var: ", distr.var())
        step_range = (0.8, 0.01)

        if self.n_levels == 1:
            self.steps = step_range[1]
        else:
            coef = (step_range[1]/step_range[0])**(1.0/(self.n_levels - 1))
            self.steps = step_range[0] * coef**np.arange(self.n_levels)

        # All levels simulations objects and MLMC object
        self.mc, self.sims = self.make_simulation_mc(step_range, sim_method)
        self.estimator = mlmc.estimate.Estimate(self.mc)

        # reference variance
        if quantile is not None:
            true_domain = distr.ppf([quantile, 1 - quantile])
        else:
            true_domain = distr.ppf([0.0001, 0.9999])

        self.moments_fn = moments.Legendre(n_moments, true_domain, is_log)

        # Exact means and vars estimation from distribution
        sample_size = 10000
        # Prefer to use numerical quadrature to get moments,
        # but check if it is precise enough and possibly switch back to MC estimates

        means, vars = self.estimator.direct_estimate_diff_var(self.sims, self.distr, self.moments_fn)
        have_nan = np.any(np.isnan(means)) or np.any(np.isnan(vars))
        self.ref_means = np.sum(np.array(means), axis=0)
        self.exact_means = self.estimator.estimate_exact_mean(self.distr, self.moments_fn, 5 * sample_size)
        rel_error = np.linalg.norm(self.exact_means - self.ref_means) / np.linalg.norm(self.exact_means)

        if have_nan or rel_error > 1 / np.sqrt(sample_size):
            # bad match, probably bad domain, use MC estimates instead
            # TODO: still getting NaNs constantly, need to determine inversion of Simultaion._sample_fn and
            # map the true domain for which the moments fn are constructed into integration domain so that
            # integration domain mapped by _sample_fn is subset of true_domain.
            means, vars = self.estimator.estimate_diff_var(self.sims, self.distr, self.moments_fn, sample_size)
            self.ref_means = np.sum(np.array(means), axis=0)

        self.ref_level_vars = np.array(vars)
        self.ref_level_means = np.array(means)
        self.ref_vars = np.sum(np.array(vars) / sample_size, axis=0)
        self.ref_mc_diff_vars = None

    def make_simulation_mc(self, step_range, sim_method=None):
        """
        Used by constructor to create mlmc and simulation objects for given exact distribution.
        :param step_range: simulation steps, tuple
        :param sim_method: simulation method name
        :return: mlmc.MLMC instance, list of level fine simulations
        """
        simulation_config = dict(distr=self.distr, complexity=2, nan_fraction=0, sim_method=sim_method)
        simulation_factory = SimulationTest.factory(step_range, config=simulation_config)

        mlmc_options = {'output_dir': os.path.dirname(os.path.realpath(__file__)),
                        'keep_collected': True,
                        'regen_failed': False}
        mc = MLMC(self.n_levels, simulation_factory, step_range, mlmc_options)
        mc.create_new_execution()

        sims = [level.fine_simulation for level in mc.levels]
        return mc, sims

    def generate_samples(self, n_samples, variance=None):
        """
        Generate samples
        :param n_samples: list, number of samples on each level
        :param variance: target variance
        :return:
        """
        # generate samples
        self.mc.set_initial_n_samples(n_samples)
        self.mc.refill_samples()
        self.mc.wait_for_simulations()

        if variance is not None:
            self.estimator.target_var_adding_samples(variance, self.moments_fn)
        # Force moments evaluation to deal with bug in subsampling.
        self.mc.update_moments(self.moments_fn)
        print("Collected n_samples: ", self.mc.n_samples)

    def test_variance_of_variance(self):
        """
        Standard deviance of log of level variances should behave like log chi-squared,
        which is computed by MLCM._variance_of_variance.
        We test both correctness of the MLCM._variance_of_variance method as wel as
        Validity of the assumption for variety of sampling distributions.
        """
        if self.n_levels > 2 and np.amax(self.ref_level_vars) > 1e-16:
            # Variance of level diff variances estimate should behave like log chi-squared
            est_var_of_var_est = np.sqrt(self.estimator._variance_of_variance()[1:])
            for i_mom in range(self.n_moments-1):
                # omit abs var of level 0
                mom_level_vars = np.array([v[1:, i_mom] for v in self.all_level_vars])
                if np.min(mom_level_vars) < 1e-16:
                    continue
                diff_vars = np.log(mom_level_vars)
                std_diff_var = np.std(diff_vars, axis=0, ddof=1)
                fraction = std_diff_var / est_var_of_var_est
                mean_frac = np.mean(fraction)
                fraction /= mean_frac

                # Ratio between observed std of variance estimate and theoretical std of variance estimate
                # should be between 0.3 to 3.
                # Theoretical std do not match well for log norm distr.
                assert 0.2 < np.min(fraction) < 3, "{}; {}".format(fraction, std_diff_var/np.mean(std_diff_var))

    def test_variance_regression(self):
        """
        Test that MLMC._varinace_regression works well producing
        correct estimate of level variances even for small number of
        samples in
        :return:
        """
        # Test variance regression
        # 1. use MC direct estimates to determine level counts for a target variance
        # 2. subsample and compute regression, compute RMS for exact variances
        # 3. compute total
        sim_steps = np.array([lvl.fine_simulation.step for lvl in self.mc.levels])
        mean_level_vars = np.mean(np.array(self.all_level_vars), axis=0)  # L x (R-1)
        all_diffs = []
        vars = np.zeros((self.n_levels, self.n_moments))
        for vars_sample in self.all_level_vars:
            vars[:, 1:] = vars_sample
            reg_vars = self.estimator._variance_regression(vars, sim_steps)
            #diff = reg_vars[:, 1:] - mean_level_vars[1:, :]
            diff = reg_vars[:, 1:] - self.ref_level_vars[:, 1:]
            all_diffs.append(diff)

        # compute error
        print("RMS:", np.linalg.norm(np.array(all_diffs).ravel()))

        reg_vars = self.estimator._variance_regression(vars, sim_steps)
        #mlmc.plot.plot_var_regression(self.ref_level_vars, reg_vars, self.n_levels, self.n_moments)
        #mlmc.plot.plot_regression_diffs(all_diffs, self.n_moments)

    def collect_subsamples(self, n_times, n_samples):
        """
        Subsample n_times from collected samples using n_samples array to specify
        number of samples on individual levels.
        :param n_times: Number of repetitions.
        :param n_samples: Array, shape L.
        :return: None, fill variables:
        self.all_means, list  n_times x array R-1
        self.all_vars,  list  n_times x array R-1
        self.all_level_vars, list  n_times x array L x (R-1)
        """
        self.all_level_vars = []
        self.all_means = []
        self.all_vars = []

        for i in range(n_times):
            # Moments as tuple (means, vars)
            means, vars = self.estimator.ref_estimates_bootstrap(n_samples, moments_fn=self.moments_fn)
            diff_vars, n_samples = self.estimator.estimate_diff_vars(self.moments_fn)
            # Remove first moment
            means = np.squeeze(means)[1:]
            vars = np.squeeze(vars)[1:]
            diff_vars = diff_vars[:, :, 1:]

            self.all_vars.append(vars)
            self.all_means.append(means)
            self.all_level_vars.append(diff_vars)

    def test_mean_var_consistency(self):
        """
        Test that estimated means are at most 3 sigma far from the exact
        moments, and that variance estimate is close to the true variance of the mean estimate.
        :return: None
        """
        mean_means = np.mean(self.all_means, axis=0)

        all_stdevs = 3 * np.sqrt(np.array(self.all_vars))
        mean_std_est = np.mean(all_stdevs, axis=0)

        # Variance estimates match true
        # 95% of means are within 3 sigma
        exact_moments = self.ref_means[1:]
        for i_mom, exact_mom in enumerate(exact_moments):

            assert np.abs(mean_means[i_mom] - exact_mom) < mean_std_est[i_mom], \
                "moment: {}, diff: {}, std: {}".format(i_mom, np.abs(mean_means[i_mom] - exact_mom), mean_std_est[i_mom])

    def check_lindep(self, x, y, slope):
        fit = np.polyfit(np.log(x), np.log(y), deg=1)
        print("MC fit: ", fit, slope)
        assert np.isclose(fit[0], slope, rtol=0.2), (fit, slope)

    def convergence_test(self):
        # subsamples
        var_exp = np.linspace(-1, -4, 10)
        target_var = 10**var_exp
        means_el = []
        vars_el = []
        n_loops = 2

        for t_var in target_var:
            self.estimator.target_var_adding_samples(t_var, self.moments_fn)

            n_samples = np.max(self.mc.n_samples, axis=1).astype(int)
            n_samples = np.minimum(n_samples, (self.mc.n_samples * 0.9).astype(int))
            n_samples = np.maximum(n_samples, 1)
            for i in range(n_loops):
                self.mc.subsample(n_samples)
                means_est, vars_est = self.estimator.estimate_moments(self.moments_fn)
                means_el.append(means_est)
                vars_el.append(vars_est)
        self.means_est = np.array(means_el).reshape(len(target_var), n_loops, self.n_moments)
        self.vars_est = np.array(vars_el).reshape(len(target_var), n_loops, self.n_moments)

        #self.plot_mlmc_conv(self.n_moments, self.vars_est, self.exact_means, self.means_est, target_var)

        for m in range(1, self.n_moments):
            Y = np.var(self.means_est[:, :, m], axis=1)

            self.check_lindep(target_var, Y, 1.0)
            Y = np.mean(self.vars_est[:, :, m], axis=1)
            self.check_lindep(target_var, Y, 1.0)

            X = np.tile(target_var, n_loops)
            Y = np.mean(np.abs(self.exact_means[m] - self.means_est[:, :, m])**2, axis=1)
            self.check_lindep(target_var, Y,  1.0)

    def show_diff_var(self):
        """
        Plot moments variance
        :return: None
        """
        if self.ref_mc_diff_vars is None:
            self.ref_mc_diff_vars, _ = self.estimator.estimate_diff_vars(self.moments_fn)

        mlmc.plot.plot_diff_var(self.ref_mc_diff_vars, self.n_moments, self.steps)

    def _test_min_samples(self):
        """
        How many samples we need on every level to get same Nl or higher but
        with at most 10% cost increase in 99%
        :return: None
        """
        self.ref_mc_diff_vars, _ = self.estimator.estimate_diff_vars(self.moments_fn)
        #self.show_diff_var()

        t_var = 0.0002
        ref_n_samples, _ = self.estimator.n_sample_estimate_moments(t_var, self.moments_fn)#, prescribe_vars)
        ref_n_samples = np.max(ref_n_samples, axis=1)
        ref_cost = self.estimator.estimate_cost(n_samples=ref_n_samples.astype(int))
        ref_total_var = np.sum(self.ref_mc_diff_vars / ref_n_samples[:, None]) / self.n_moments
        n_samples = self.n_levels*[100]
        n_loops = 10

        print("ref var: {} target var: {}".format(ref_total_var, t_var))
        print(ref_n_samples.astype(int))

        # subsamples
        l_cost_err = []
        l_total_std_err = []
        l_n_samples_err = []
        for i in range(n_loops):
            fractions = [0, 0.001, 0.01, 0.1,  1]
            for fr in fractions:
                if fr == 0:
                    nL, n0 = 3, 30
                    L = max(2, self.n_levels)
                    factor = (nL / n0) ** (1 / (L - 1))
                    n_samples = (n0 * factor ** np.arange(L)).astype(int)
                else:
                    n_samples = np.maximum( n_samples, (fr*max_est_n_samples).astype(int))
                # n_samples = np.maximum(n_samples, 1)

                self.mc.subsample(n_samples)
                est_diff_vars, _ = self.estimator.estimate_diff_vars(self.moments_fn)
                est_n_samples, _ = self.estimator.n_sample_estimate_moments(t_var, self.moments_fn, est_diff_vars)
                max_est_n_samples = np.max(est_n_samples, axis=1)
                est_cost = self.estimator.estimate_cost(n_samples=max_est_n_samples.astype(int))
                est_total_var = np.sum(self.ref_mc_diff_vars / max_est_n_samples[:, None]) / self.n_moments

                n_samples_err = np.min( (max_est_n_samples - ref_n_samples) /ref_n_samples)
                #total_std_err =  np.log2(est_total_var/ref_total_var)/2
                total_std_err = (np.sqrt(est_total_var) - np.sqrt(ref_total_var)) / np.sqrt(ref_total_var)
                cost_err = (est_cost - ref_cost)/ref_cost
                print("Fr: {:6f} NSerr: {} Tstderr: {} cost_err: {}".format(fr, n_samples_err, total_std_err, cost_err))
            print("est cost: {} ref cost: {}".format(est_cost, ref_cost))
            print(n_samples)
            print(np.maximum( n_samples, (max_est_n_samples).astype(int)))
            print(ref_n_samples.astype(int))
            print("\n")
            l_n_samples_err.append(n_samples_err)
            l_total_std_err.append(total_std_err)
            l_cost_err.append((ref_cost - est_cost)/ref_cost)

        l_cost_err.sort()
        l_total_std_err.sort()
        l_n_samples_err.sort()
        mlmc.plot.plot_n_sample_est_distributions(l_cost_err, l_total_std_err, l_n_samples_err)

    def clear_subsamples(self):
        for level in self.mc.levels:
            level.sample_indices = None
