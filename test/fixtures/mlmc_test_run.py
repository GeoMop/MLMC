
import pytest
import numpy as np
import scipy.stats as st
import scipy.integrate as integrate
import mlmc
from test.fixtures.synth_simulation import SimulationTest


class TestMLMC:
    def __init__(self, n_levels, n_moments, distr, is_log=False, sim_method=None):
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

        # reference variance
        true_domain = distr.ppf([0.001, 0.999])


        self.moments_fn = mlmc.moments.Legendre(n_moments, true_domain, is_log)



        # Exact means and vars estimation from distribution
        sample_size = 10000
        # Prefer to use numerical quadrature to get moments,
        # but check if it is precise enough and possibly switch back to MC estimates

        means, vars = self.direct_estimate_diff_var(self.moments_fn)
        have_nan = np.any(np.isnan(means)) or np.any(np.isnan(vars))
        self.ref_means = np.sum(np.array(means), axis=0)
        mc_exact_means = self.mc_estimate_exact_mean(self.moments_fn, 5 * sample_size)
        rel_error =  np.linalg.norm(mc_exact_means - self.ref_means) / np.linalg.norm(mc_exact_means)
        if have_nan or rel_error > 1 / np.sqrt(sample_size):
            # bad match, probably bad domain, use MC estimates instead
            # TODO: still getting NaNs constantly, need to determine inversion of Simultaion._sample_fn and
            # map the true domain for which the moments fn are constructed into integration domain so that
            # integration domain mapped by _sample_fn is subset of true_domain.
            means, vars = self.mc_estimate_diff_var(self.moments_fn, sample_size)
            self.ref_means = np.sum(np.array(means), axis=0)

        self.ref_level_vars = np.array(vars)
        self.ref_level_means = np.array(means)
        self.ref_vars = np.sum(np.array(vars) / sample_size, axis=0)



    def make_simulation_mc(self, step_range, sim_method=None):
        """
        Used by constructor to create mlmc and simulation objects for given exact distribution.
        :param step_range:
        :param sim_method:
        :return:
        """
        simulation_config = dict(distr=self.distr, complexity=2, nan_fraction=0, sim_method=sim_method)
        simultion_factory = SimulationTest.factory(step_range, config=simulation_config)

        mlmc_options = {'output_dir': None,
                        'keep_collected': True,
                        'regen_failed': False}
        mc = mlmc.mlmc.MLMC(self.n_levels, simultion_factory, step_range, mlmc_options)
        mc.create_levels()
        sims = [level.fine_simulation for level in mc.levels]
        return mc, sims

    # def make_simulation_shooting(self, step_range):
    #     pbs = pb.Pbs()
    #     simulation_config = dict(distr=self.distr, complexity=2, nan_fraction=0)
    #     simultion_factory = SimulationTest.factory(step_range, config=simulation_config)
    #
    #     mc = mlmc.mlmc.MLMC(self.n_levels, simultion_factory, pbs)
    #     mc.create_levels()
    #     sims = [level.fine_simulation for level in mc.levels]
    #     return mc, sims

    def direct_estimate_diff_var(self, moments_fn):
        """
        Calculate variances of level differences using numerical quadrature.
        :param moments_fn:
        :param domain:
        :return:
        """
        mom_domain = moments_fn.domain

        means = []
        vars = []
        sim_l = None
        for l in range(len(self.sims)):
            # TODO: determine intergration domain as _sample_fn ^{-1} (  mom_domain )
            domain = mom_domain

            sim_0 = sim_l
            sim_l = lambda x, h=self.sims[l].step: self.sims[l]._sample_fn(x, h)
            if l == 0:
                md_fn = lambda x: moments_fn(sim_l(x))
            else:
                md_fn = lambda x: moments_fn(sim_l(x)) - moments_fn(sim_0(x))
            fn = lambda x: (md_fn(x)).T * self.distr.pdf(x)
            moment_means = integrate.fixed_quad(fn, domain[0], domain[1], n=100)[0]
            fn2 = lambda x: ((md_fn(x) - moment_means[None, :]) ** 2).T * self.distr.pdf(x)
            moment_vars = integrate.fixed_quad(fn2, domain[0], domain[1], n=100)[0]
            means.append(moment_means)
            vars.append(moment_vars)
        return means, vars

    def mc_estimate_diff_var(self, moments_fn, size=10000):
        """
        Calculate variances of level differences using MC method.
        :param moments_fn:
        :param domain:
        :param distr:
        :param sims:
        :return: means, vars ; shape = (n_levels, n_moments)
        """
        means = []
        vars = []
        sim_l = None
        # Loop through levels simulation objects
        for l in range(len(self.sims)):
            # Previous level simulations
            sim_0 = sim_l
            # Current level simulations
            sim_l = lambda x, h=self.sims[l].step: self.sims[l]._sample_fn(x, h)
            # Samples from distribution
            X = self.distr.rvs(size=size)
            if l == 0:
                MD = moments_fn(sim_l(X))
            else:
                MD = (moments_fn(sim_l(X)) - moments_fn(sim_0(X)))

            moment_means = np.nanmean(MD, axis=0)
            moment_vars = np.nanvar(MD, axis=0, ddof=1)
            means.append(moment_means)
            vars.append(moment_vars)
        return np.array(means), np.array(vars)

    def mc_estimate_exact_mean(self, moments_fn, size=200000):
        """
        Calculate exact means using MC method.
        :param moments_fn:
        :param size:
        :return:
        """
        X = self.distr.rvs(size=size)
        return np.nanmean(moments_fn(X), axis=0)



    def generate_samples(self, n_samples):
        # generate samples
        self.mc.set_initial_n_samples(n_samples)
        self.mc.refill_samples()
        self.mc.wait_for_simulations()
        # Force moments evaluation to deal with bug in subsampling.
        self.mc.update_moments(self.moments_fn)
        print("Collected n_samples: ", self.mc.n_samples)

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
            self.mc.subsample(n_samples)
            # Moments as tuple (means, vars)
            means, vars = self.mc.estimate_moments(self.moments_fn)
            diff_vars, n_samples = self.mc.estimate_diff_vars(self.moments_fn)
            # Remove first moment
            means = means[1:]
            vars = vars[1:]
            diff_vars = diff_vars[:, 1:]

            self.all_vars.append(vars)
            self.all_means.append(means)
            self.all_level_vars.append(diff_vars)

    @staticmethod
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

    @staticmethod
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
        T =  var * N / var_0
        pst = st.chi2.cdf(T, df=len(samples)-1)
        p_val = 2 * min(pst, 1 - pst)
        print("{}\n var: {} var_0: {} p-val: {}".format(tag, var, var_0, p_val))
        assert p_val > max_p_val


    def test_variance_of_varaince(self):
        """
        Standard deviance of log of level variances should behave like log chi-squared,
        which is computed by MLCM._variance_of_variance.
        We test both correctness of the MLCM._variance_of_variance method as wel as
        Validity of the assumption for variety of sampling distributions.
        """
        if self.n_levels > 2 and np.amax(self.ref_level_vars) > 1e-16:
            # Variance of level diff variances estimate should behave like log chi-squared
            est_var_of_var_est = np.sqrt(self.mc._variance_of_variance()[1:])
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

                #plt.plot(fraction, label="est_var" + str(i_mom) )

            #plt.legend()
            #plt.show()

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
            reg_vars = self.mc._varinace_regression(vars, sim_steps)
            #diff = reg_vars[:, 1:] - mean_level_vars[1:, :]
            diff = reg_vars[:, 1:] - self.ref_level_vars[:, 1:]
            all_diffs.append(diff)

        # compute error
        print( "RMS:", np.linalg.norm(np.array(all_diffs).ravel()))

        def plot_var_regression():
            # Plot variance regression for exact level variances
            vars[:, 1:] = self.ref_level_vars[:, 1:]
            reg_vars = self.mc._varinace_regression(vars, sim_steps)
            X = np.outer( np.arange(self.n_levels), np.ones(self.n_moments -1 )) + 0.1 * np.outer( np.ones(self.n_levels), np.arange(self.n_moments -1 ) )
            col = np.outer( np.ones(self.n_levels), np.arange(self.n_moments -1 ) )
            plt.scatter(X.ravel(), self.ref_level_vars[:, 1:].ravel(), c=col.ravel(), cmap=plt.cm.tab10, norm=plt.Normalize(0, 10), marker='o')
            for i_mom in range(self.n_moments - 1):
                col = plt.cm.tab10(plt.Normalize(0, 10)(i_mom))
                plt.plot(X[:, i_mom], reg_vars[:, i_mom + 1], c=col)
            plt.legend()
            plt.yscale('log')
            plt.ylim(1e-10, 1)
            plt.show()

        def plot_regression_diffs():
            for i_mom in range(self.n_moments-1):
                diffs =  [ sample[:, i_mom]  for sample in all_diffs]
                diffs = np.array(diffs)
                N, L = diffs.shape
                X = np.outer( np.ones(N), np.arange(L) ) + i_mom * 0.1
                col = np.ones_like(diffs) * i_mom
                plt.scatter(X, diffs, c=col, cmap=plt.cm.tab10, norm=plt.Normalize(0, 10), marker='o', label=str(i_mom))
            plt.legend()
            plt.yscale('log')
            plt.ylim(1e-10, 1)
            plt.show()


    def test_mean_var_consistency(self):
        """
        Test that estimated means are at most 3 sigma far from the exact
        moments, and that variance estimate is close to the true variance of the mean estimate.
        :return:
        """

        # TODO: Use statistical tests to check correctness of mlmc results

        # Variance estimates vs. MC estimate using exact distribution.
        #
        # for i_mom in range(self.n_moments-1):
        #     for l in range(self.n_levels):
        #
        #     self.t_test(mc_vars[i_mom], vars)

        # Exact moments from distribution

        mean_means = np.mean(self.all_means, axis=0)
        #est_var_of_mean_est = np.var(self.all_means, axis=0, ddof=1)

        all_stdevs = 3 * np.sqrt(np.array(self.all_vars))
        mean_std_est = np.mean(all_stdevs, axis=0)


        # Variance estimates match true
        # 95% of means are within 3 sigma
        exact_moments = self.ref_means[1:]
        for i_mom, exact_mom in enumerate(exact_moments):
            assert np.abs(mean_means[i_mom] - exact_mom) <  mean_std_est[i_mom], \
                "moment: {}, diff: {}, std: {}".format(i_mom, np.abs(mean_means[i_mom] - exact_mom), mean_std_est[i_mom])




    # @staticmethod
    # def box_plot(ax, X, Y):
    #     bp = boxplot(column='age', by='pclass', grid=False)
    #     for i in [1, 2, 3]:
    #         y = titanic.age[titanic.pclass == i].dropna()
    #         # Add some random "jitter" to the x-axis
    #         x = np.random.normal(i, 0.04, size=len(y))
    #         plot(x, y, 'r.', alpha=0.2)

    def plot_mlmc_conv(self, target_var):

        fig = plt.figure(figsize=(10,20))

        for m in range(1, self.n_moments):
            ax = fig.add_subplot(2, 2, m)
            color = 'C' + str(m)
            Y = np.var(self.means_est[:,:,m], axis=1)
            ax.plot(target_var, Y, 'o', c=color, label=str(m))


            Y = np.percentile(self.vars_est[:, :, m],  [10, 50, 90], axis=1)
            ax.plot(target_var, Y[1,:], c=color)
            ax.plot(target_var, Y[0,:], c=color, ls='--')
            ax.plot(target_var, Y[2, :], c=color, ls ='--')
            Y = (self.exact_mean[m] - self.means_est[:, :, m])**2
            Y = np.percentile(Y, [10, 50, 90], axis=1)
            ax.plot(target_var, Y[1,:], c='gray')
            ax.plot(target_var, Y[0,:], c='gray', ls='--')
            ax.plot(target_var, Y[2, :], c='gray', ls ='--')
            ax.set_yscale('log')
            ax.set_xscale('log')
            ax.legend()
            ax.set_ylabel("observed var. of mean est.")

        plt.show()

    def plot_error(self, arr, ax, label):
        ax.hist(arr, normed=1)
        ax.set_xlabel(label)
        prc = np.percentile(arr, [99])
        ax.axvline(x=prc, label=str(prc), c='red')
        ax.legend()

    def plot_n_sample_est_distributions(self, title, cost, total_std, n_samples, rel_moments):
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(30,10))
        ax1 = fig.add_subplot(2, 2, 1)
        self.plot_error(cost, ax1, "cost err")

        ax2 = fig.add_subplot(2, 2, 2)
        self.plot_error(total_std, ax2, "total std err")

        ax3 = fig.add_subplot(2, 2, 3)
        self.plot_error(n_samples, ax3, "n. samples err")

        ax4 = fig.add_subplot(2, 2, 4)
        self.plot_error(rel_moments, ax4, "moments err")
        fig.suptitle(title)
        plt.show()

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
        n_loops = 30
        for t_var in target_var:
            n_samples = self.mc.set_target_variance(t_var, prescribe_vars=self.ref_mc_diff_vars)
            n_samples = np.max(n_samples, axis=1).astype(int)

            print(t_var, n_samples)
            n_samples = np.minimum(n_samples, (self.mc.n_samples*0.9).astype(int))
            n_samples = np.maximum(n_samples, 1)
            for i in range(n_loops):
                self.mc.subsample(n_samples)
                means_est, vars_est = self.mc.estimate_moments(self.moments_fn)
                means_el.append(means_est)
                vars_el.append(vars_est)
        self.means_est = np.array(means_el).reshape(len(target_var), n_loops, self.n_moments)
        self.vars_est = np.array(vars_el).reshape(len(target_var), n_loops, self.n_moments)

        #self.plot_mlmc_conv(target_var)

        for m in range(1, self.n_moments):
            Y = np.var(self.means_est[:,:,m], axis=1)
            self.check_lindep(target_var, Y, 1.0)
            Y = np.mean(self.vars_est[:, :, m], axis=1)
            self.check_lindep(target_var, Y , 1.0)

            X = np.tile(target_var, n_loops)
            Y = np.mean(np.abs(self.exact_mean[m] - self.means_est[:, :, m])**2, axis=1)
            self.check_lindep(target_var, Y,  1.0)

    def plot_diff_var(self):
        fig = plt.figure(figsize=(10, 20))
        ax = fig.add_subplot(1, 1, 1)

        print(self.ref_mc_diff_vars)

        error_power = 2.0
        for m in range(1, self.n_moments):
            color = 'C' + str(m)
            print(self.ref_mc_diff_vars)

            Y = self.ref_mc_diff_vars[:, m] / (self.steps ** error_power)

            ax.plot(self.steps[1:], Y[1:], c=color, label=str(m))
            ax.plot(self.steps[0], Y[0], 'o', c=color)

            # Y = np.percentile(self.vars_est[:, :, m],  [10, 50, 90], axis=1)
            # ax.plot(target_var, Y[1,:], c=color)
            # ax.plot(target_var, Y[0,:], c=color, ls='--')
            # ax.plot(target_var, Y[2, :], c=color, ls ='--')
            # Y = (self.exact_mean[m] - self.means_est[:, :, m])**2
            # Y = np.percentile(Y, [10, 50, 90], axis=1)
            # ax.plot(target_var, Y[1,:], c='gray')
            # ax.plot(target_var, Y[0,:], c='gray', ls='--')
            # ax.plot(target_var, Y[2, :], c='gray', ls ='--')
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.legend()
        ax.set_ylabel("observed var. of mean est.")

        plt.show()

    def plot_n_sample_est_distributions(self, cost, total_std, n_samples):
        print(cost)
        print(total_std)
        print(n_samples)

        fig = plt.figure(figsize=(30,10))
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.hist(cost, normed=1)
        ax1.set_xlabel("cost")
        cost_99 = np.percentile(cost, [99])
        ax1.axvline(x=cost_99, label=str(cost_99), c='red')
        ax1.legend()

        ax2 = fig.add_subplot(1, 3, 2)
        ax2.hist(total_std, normed=1)
        ax2.set_xlabel("total var")
        tstd_99 = np.percentile(total_std, [99])
        ax2.axvline(x=tstd_99, label=str(tstd_99), c='red')
        ax2.legend()

        ax3 = fig.add_subplot(1, 3, 3)
        ax3.hist(n_samples, normed=1)
        ax3.set_xlabel("n_samples")
        ns_99 = np.percentile(n_samples, [99])
        ax3.axvline(x=ns_99, label=str(ns_99), c='red')
        ax3.legend()

        plt.show()

    def _test_min_samples(self):
        """
        How many samples we need on every level to get same Nl or higher but
        with at most 10% cost increase in 99%
        :return:
        """
        #self.plot_diff_var()

        t_var = 0.0002
        ref_n_samples = self.mc.set_target_variance(t_var, prescribe_vars=self.ref_mc_diff_vars)
        ref_n_samples = np.max(ref_n_samples, axis=1)
        ref_cost = self.mc.estimate_cost(n_samples=ref_n_samples.astype(int))
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
                est_diff_vars, _ = self.mc.estimate_diff_vars(self.moments_fn)
                est_n_samples = self.mc.set_target_variance(t_var, prescribe_vars=est_diff_vars)
                max_est_n_samples = np.max(est_n_samples, axis=1)
                est_cost = self.mc.estimate_cost(n_samples=max_est_n_samples.astype(int))
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
            l_total_std_err.append(total_std_err )
            l_cost_err.append((ref_cost - est_cost)/ref_cost)

        l_cost_err.sort()
        l_total_std_err.sort()
        l_n_samples_err.sort()
        self.plot_n_sample_est_distributions(l_cost_err, l_total_std_err, l_n_samples_err)

    def clear_subsamples(self):
        for level in self.mc.levels:
            level.sample_indices = None
