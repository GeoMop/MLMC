import os
import sys

import scipy.integrate as integrate
# import statprof
import scipy.stats as stats

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/../src/')
import mlmc.mlmc
import mlmc.simulation
import mlmc.moments
import mlmc.distribution
import numpy as np
import flow_pbs
import matplotlib.pyplot as plt
import matplotlib.cm as cm

src_path = os.path.dirname(os.path.abspath(__file__))


def aux_sim(x,  h):
    return x + h*np.sqrt(x)


class SimulationTest(mlmc.simulation.Simulation):
    # Artificial simulation. Just random parameter + numerical error."""
    def __init__(self, step, config):
        """
        :param config: Dict:
            'distr' scipy.stats.XYZ freezed distribution
            'complexity' number of FLOPS as function of step
        :param step:
        """
        super().__init__()
        self.config = config
        self.nan_fraction = config.get('nan_fraction', 0.0)
        self.n_nans = 0
        self.step = step
        self._result_dict = {}
        self._coarse_simulation = None

    def _sample_fn(self, x, h):
        """
        Calculates the simulation sample
        :param x: Distribution sample
        :param h: Simluation step
        :return: sample
        """
        # This function can cause many outliers depending on chosen domain of moments function
        return x + h * np.sqrt(1e-4 + np.abs(x))

    def simulation_sample(self, tag):
        """
        Run simulation
        :param sim_id:    Simulation id
        """
        x = self._input_sample

        h = self.step

        y = self._sample_fn(x, h)
        if (self.n_nans/(1e-10 + len(self._result_dict)) < self.nan_fraction) :
            self.n_nans += 1
            y = np.nan

        self._result_dict[tag] = float(y)

        return tag

    def generate_random_sample(self):
        distr = self.config['distr']
        self._input_sample = distr.rvs(size=1)
        if self._coarse_simulation is not None:
            self._coarse_simulation._input_sample = self._input_sample

    # def get_coarse_sample(self):
    #     return self._input_sample

    def n_ops_estimate(self):
        return (1/self.step)**self.config['complexity']*np.log(max(1/self.step, 2.0))

    # @property
    # def mesh_step(self):
    #     return self.sim_param
    #
    # @mesh_step.setter
    # def mesh_step(self, step):
    #     self.sim_param = step

    def set_coarse_sim(self, coarse_simulation=None):
        self._coarse_simulation = coarse_simulation

    def extract_result(self, sim_id):
        return self._result_dict[sim_id]


def impl_var_estimate(n_levels, n_moments, target_var, distr, is_log=False):
    print("\n")
    print("L: {} R: {} var: {} distr: {}".format(n_levels, n_moments, target_var, distr.dist.__class__.__name__))
    # if is_log:
    #     q1,q3 = np.log(distr.ppf([0.25, 0.75]))
    #     iqr = 2*(q3-q1)
    #     domain =np.exp( [q1 - iqr, q3 + iqr ] )
    # else:
    #     q1,q3 = distr.ppf([0.25, 0.75])
    #     iqr = 2*(q3-q1)
    #     domain = [q1 - iqr, q3 + iqr ]

    step_range = (0.8, 0.01)
    mc, sims = make_simulation_mc(step_range, distr, n_levels)

    true_domain = distr.ppf([0.001, 0.999])
    true_moments_fn = lambda x, n=n_moments, a=true_domain[0], b=true_domain[1]: mlmc.moments.legendre_moments(x, n, a, b)
    means, vars = mc_estimate_diff_var(true_moments_fn, true_domain, distr, sims)
    vars = np.array(vars)

    raw_vars = []
    reg_vars = []

    n_loops=5
    print("--")
    for _ in range(n_loops):
        mc.clean_levels()
        mc.set_initial_n_samples()
        mc.refill_samples()
        mc.wait_for_simulations()
        est_domain = mc.estimate_domain()
        domain_err = [distr.cdf(est_domain[0]), 1-distr.cdf(est_domain[1])]
        print("Domain error: ", domain_err )
        assert np.sum(domain_err) < 0.01
        moments_fn = lambda x, n=n_moments, a=est_domain[0], b=est_domain[1]: mlmc.moments.legendre_moments(x, n, a, b)
        raw_var_estimate, n_col_samples = mc.estimate_diff_vars(moments_fn)

        reg_var_estimate = mc.estimate_diff_vars_regression(moments_fn)
        raw_vars.append(raw_var_estimate)
        reg_vars.append(reg_var_estimate)
    print("n samples   : ", n_col_samples)
    print("ref var min    : ", np.min(vars[:, 1:], axis=1) )
    print("ref var max    : ", np.max(vars, axis=1) )

    # relative max over loops
    print("max raw err: ", np.max( np.abs(np.array(raw_vars) - vars) / (1e-4 + vars), axis=0))
    print("max reg err: ", np.max( np.abs(np.array(reg_vars) - vars) / (1e-4 + vars), axis=0))

    # std over loops
    # max (of relative std) over moments
    print("std  raw var: ", np.max( np.sqrt(np.mean( (np.array(raw_vars) - vars)**2, axis=0)) / (1e-4 + vars) , axis=1) )
    print("std  reg var: ", np.max( np.sqrt(np.mean( (np.array(reg_vars) - vars)**2, axis=0)) / (1e-4 + vars) , axis=1) )
    #print("Linf raw var: ", np.max( np.abs(np.array(raw_vars) - vars), axis=0) / (1e-4 + vars))

def err_tuple(x):
    return (np.min(x), np.median(x), np.max(x))


class TestMLMC:
    def __init__(self, n_levels, n_moments, distr, is_log=False):
        # Not work for one level method
        #print("\n")
        #print("L: {} R: {} distr: {}".format(n_levels, n_moments, distr.dist.__class__.__name__))

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
        self.mc, self.sims = self.make_simulation_mc(step_range)

        # reference variance
        true_domain = distr.ppf([0.001, 0.999])

        self.moments_fn = mlmc.moments.Legendre(n_moments, true_domain, is_log)

        sample_size = 10000
        # Exact mean estimation from distribution
        self.exact_mean = self.mc_estimate_exact_mean(self.moments_fn, 5*sample_size)
        means, vars = self.mc_estimate_diff_var(self.moments_fn, sample_size)
        self.ref_diff_vars = np.array(vars)[:, :]
        self.ref_vars = np.sum(np.array(vars)/sample_size, axis=0)
        self.ref_means = np.sum(np.array(means), axis=0)
        #print("Ref means: ", ref_means)
        #print("Ref vars: ", ref_vars)

    def make_simulation_mc(self, step_range):
        pbs = flow_pbs.FlowPbs()
        simulation_config = dict(distr=self.distr, complexity=2, nan_fraction=0)
        simultion_factory = SimulationTest.factory(step_range, config=simulation_config)

        mc = mlmc.mlmc.MLMC(self.n_levels, simultion_factory, pbs)
        sims = [level.fine_simulation for level in mc.levels]
        return mc, sims

    def direct_estimate_diff_var(self, moments_fn, domain):
        means = []
        vars = []
        sim_l = None
        for l in range(len(self.sims)):
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
        :param moments_fn:
        :param domain:
        :param distr:
        :param sims:
        :return: means, vars ; shape n_levels, n_moments
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

            moment_means = np.mean(MD, axis=0)
            moment_vars = np.var(MD, axis=0, ddof=1)
            means.append(moment_means)
            vars.append(moment_vars)
        return np.array(means), np.array(vars)

    def mc_estimate_exact_mean(self, moments_fn, size=200000):
        X = self.distr.rvs(size=size)

        return np.mean([mom for mom in moments_fn(X) if mom is not None], axis=0)

    def generate_samples(self, size):
        # generate samples
        self.mc.set_initial_n_samples(self.n_levels*[size])
        self.mc.refill_samples()
        self.mc.wait_for_simulations()

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
            n_samples = self.mc.set_target_variance(t_var, prescribe_vars=self.ref_diff_vars)
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

        print(self.ref_diff_vars)

        error_power = 2.0
        for m in range(1, self.n_moments):
            color = 'C' + str(m)
            print(self.ref_diff_vars)

            Y = self.ref_diff_vars[:,m]/(self.steps**error_power)

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
        ref_n_samples = self.mc.set_target_variance(t_var, prescribe_vars=self.ref_diff_vars)
        ref_n_samples = np.max(ref_n_samples, axis=1)
        ref_cost = self.mc.estimate_cost(n_samples=ref_n_samples.astype(int))
        ref_total_var = np.sum(self.ref_diff_vars / ref_n_samples[:, None])/self.n_moments
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
                est_total_var = np.sum(self.ref_diff_vars / max_est_n_samples[:, None])/self.n_moments

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


def var_subsample(moments, mlmc_test, n_subsamples=1000):
    """
    Compare moments variance from whole mlmc and from subsamples
    :param moments: moments from mlmc
    :param mlmc_test: mlmc object
    :param n_subsamples: number of subsamples (J in theory)
    :return: array, absolute difference between samples variance and mlmc moments variance
    """
    assert len(moments) == 2
    tolerance = 1

    # Variance from subsamples
    subsample_variance = np.zeros(len(moments[0]))

    # Number of subsamples (J from theory)
    for _ in range(n_subsamples):
        subsamples = []
        # Clear number of subsamples
        mlmc_test.clear_subsamples()

        # Generate subsamples for all levels
        for _ in mlmc_test.mc.n_samples:
            subsamples.append(5000)

        # Process mlmc with new number of level samples
        if subsamples is not None:
            mlmc_test.mc.subsample(subsamples)
        mlmc_test.mc.refill_samples()
        mlmc_test.mc.wait_for_simulations()

        # Moments estimation from subsamples
        moments_mean_subsample = (mlmc_test.mc.estimate_moments(mlmc_test.moments_fn)[0])

        # SUM[(EX - X)**2]
        subsample_variance = np.add(subsample_variance, (np.subtract(np.array(moments[0]), np.array(moments_mean_subsample))) ** 2)

    # Sample variance
    variance = subsample_variance/(n_subsamples-1)

    # Difference between arrays
    variance_abs_diff = np.abs(np.subtract(variance, moments[1]))

    result = np.sqrt([a/b if b != 0 else a for a, b in zip(moments[1], variance)])

    # Difference between variances must be in tolerance
    assert all(var_diff < tolerance for var_diff in result)
    return result


def test_var_estimate():
    """
    Test if mlmc moments correspond to exact moments from distribution
    :return: None
    """
    #np.random.seed(3)
    n_levels = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    n_moments = [10]

    distr = [
        #(stats.norm(loc=5, scale=1), False),
        (stats.lognorm(scale=np.exp(5), s=1), True),            # worse conv of higher moments
        #(stats.lognorm(scale=np.exp(-5), s=0.5), True)         # worse conv of higher moments
        (stats.chi2(df=10), True)
        #(stats.weibull_min(c=20), True)    # Exponential
        #(stats.weibull_min(c=1.5), True)  # Infinite derivative at zero
        #(stats.weibull_min(c=3), True)    # Close to normal
        ]

    level_moments_mean = []
    level_moments_var = []
    level_variance_diff = []

    for nl in n_levels:
        for nm in n_moments:
            for d, il in distr:
                mc_test = TestMLMC(nl, nm, d, il)
                # 10 000 samples on each level
                mc_test.generate_samples(1000)
                # Moments as tuple (means, vars)
                moments = mc_test.mc.estimate_moments(mc_test.moments_fn)
                #level_variance_diff.append(var_subsample(moments, mc_test))

                # Variances
                variances = np.sqrt(moments[1]) * 4

                # Exact moments from distribution
                exact_moments = mlmc.distribution.compute_exact_moments(mc_test.moments_fn, d.pdf, 1e-10)[1:]

                assert all(moments[0][index] + variances[index] >= exact_mom >= moments[0][index] - variances[index]
                           for index, exact_mom in enumerate(exact_moments))

            level_moments_mean.append(moments[0])
            level_moments_var.append(variances)

    #plot_diff_var(level_variance_diff, n_levels)
    # Plot moment values
    #plot_vars(level_moments_mean, level_moments_var, n_levels, exact_moments)


def plot_diff_var(level_variance_diff, n_levels):
    """
    Plot diff between V* and V
    :param level_variance_diff: array of each moments sqrt(V/V*)
    :param n_levels: array, number of levels
    :return: None
    """
    if len(level_variance_diff) > 0:
        colors = iter(cm.rainbow(np.linspace(0, 1, len(level_variance_diff) + 1)))
        x = np.arange(1, len(level_variance_diff[0]) +1)
        [plt.plot(x, var_diff, 'o', label="%dLMC" % n_levels[index], color=next(colors)) for index, var_diff in enumerate(level_variance_diff)]
        plt.legend()
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

    x = np.arange(1, len(moments_mean[0]) + 1)
    x = x - 0.3
    default_x = x

    for index, means in enumerate(moments_mean):
        if index == int(len(moments_mean)/2):
            if exact_moments is not None:
                plt.plot(default_x, exact_moments, 'ro', label="Exact moments")
        else:
            x = x + (1 / (len(moments_mean)*1.5))
            plt.errorbar(x, means, yerr=moments_var[index], fmt='o', capsize=3, color=next(colors), label="%dLMC" % n_levels[index])

    if ex_moments is not None:
        plt.plot(default_x-0.125, ex_moments, 'ko', label="Exact moments")

    plt.legend()
    plt.show()


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

    #print("var: ", distr.var())
    work_dir = '_test_tmp'
    work_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '_test_tmp')

    n_levels = 5
    distr = stats.norm()
    step_range = (0.8, 0.01)
    pbs = flow_pbs.FlowPbs(work_dir=work_dir, clean=True)
    simulation_config = dict(
        distr= distr, complexity=2, nan_fraction=0.1)
    simulation_factory = SimulationTest.factory(step_range, config=simulation_config)
    mc = mlmc.mlmc.MLMC(n_levels, simulation_factory, pbs)
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
    pbs.close()
    # New mlmc
    pbs = flow_pbs.FlowPbs(work_dir=work_dir)
    pbs.reload_logs()
    mc = mlmc.mlmc.MLMC(n_levels, simulation_factory, pbs)

    check_estimates_for_nans(mc, distr)

    # Test
    for level, data in zip(mc.levels, level_data):
        run, fin, values = data

        assert run == level.running_simulations
        assert fin == level.finished_simulations
        assert np.allclose(values, level.sample_values)


# class TestMLMC(mlmc.mlmc.MLMC):
#     def __init__(self):
#         n_levels = 3
#         n_samples = 1000
#         self.levels = 2**(-np.arange(n_levels))
#         x = np.random.lognormal(0.0, 0.5, n_samples)
#         self.samples = np.empty((n_levels, n_samples, 2))
#         for i, h in enumerate(self.levels):
#             if i == 0:
#                 self.samples[i, :, : ] = np.array([aux_sim(x, h), 0]).T
#             else:
#                 self.samples[i, :, :] = np.array([aux_sim(x, h), aux_sim(x, 2*h)]).T
#
#
#
#     def estimate_diff_vars(self, moments_fn):
#         vars = []
#         for l in range(len(self.levels)):
#             samples = self.samples[l, :, :]
#             var_vec = np.var(moments_fn(samples[:, 0]) - moments_fn(samples[:, 1]), axis=0, ddof=1)
#             vars.append(var_vec)
#         return vars
#
#     def test_var_estimate(self):
#         n_moments = 5
#         domain = self.distribution.percentile([0.001, 0.999], self.distr_args)
#         moments_fn = lambda x, n=n_moments, a=domain[0], b=domain[1]: mlmc.moments.legendre_moments(x, n, a, b)
#
#         self.set_target_variance(0.01, moments_fn)
if __name__ == '__main__':
    test_save_load_samples()
    #test_var_estimate()

