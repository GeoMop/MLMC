import os
import statprof
import scipy.stats as stats
import scipy.integrate as integrate

import mlmc.mlmc
import mlmc.simulation
import mlmc.moments
import mlmc.distribution
import numpy as np
import flow_pbs



src_path = os.path.dirname(os.path.abspath(__file__))


def aux_sim(x,  h):
    return x +  h*np.sqrt(x)


class SimulationTest(mlmc.simulation.Simulation):
    # Artificial simulation. Just random parameter + numerical error."""
    def __init__(self, config, step):
        """
        :param config: Dict:
            'distr' scipy.stats.XYZ freezed distribution
            'complexity' numer of FLOPS as function of step
        :param step:
        """
        super().__init__()
        self.config = config
        self.step = step
        self._result_dict = {}
        self._coarse_simulation = None

    def _sample_fn(self, x, h):
        return x + h * np.sqrt(1e-4 + np.abs(x))

    def simulation_sample(self, tag):
        """
        Run simulation
        :param sim_id:    Simulation id
        """
        x = self._input_sample
        h = self.step
        y = self._sample_fn(x, h)
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
        print("\n")
        print("L: {} R: {} distr: {}".format(n_levels, n_moments, distr.dist.__class__.__name__))

        self.distr = distr
        self.n_levels = n_levels
        self.n_moments = n_moments

        # print("var: ", distr.var())
        step_range = (0.8, 0.01)
        self.mc, self.sims = self.make_simulation_mc(step_range)

        # reference variance
        true_domain = distr.ppf([0.001, 0.999])
        self.moments_fn = lambda x, n=n_moments, a=true_domain[0], b=true_domain[1]: \
                                    mlmc.moments.legendre_moments(x, n, a, b)

        sample_size = 10000
        self.exact_mean = self.mc_estimate_exact_mean(self.moments_fn, 5*sample_size)
        means, vars = self.mc_estimate_diff_var(self.moments_fn, sample_size)
        self.ref_diff_vars = np.array(vars)[:, 1:]
        self.ref_vars = np.sum(np.array(vars)/sample_size, axis=0)
        self.ref_means = np.sum(np.array(means), axis=0)
        #print("Ref means: ", ref_means)
        #print("Ref vars: ", ref_vars)

    def make_simulation_mc(self, step_range):
        pbs = flow_pbs.FlowPbs()
        simulation_config = dict(distr=self.distr, complexity=2)
        simultion_factory = lambda t_level: SimulationTest.make_sim(simulation_config, step_range, t_level)

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
        for l in range(len(self.sims)):
            sim_0 = sim_l
            sim_l = lambda x, h=self.sims[l].step: self.sims[l]._sample_fn(x, h)
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

    def mc_estimate_exact_mean(self, moments_fn, size= 20000):
        X = self.distr.rvs(size=size)
        return np.mean(moments_fn(X), axis=0)


    # @staticmethod
    # def box_plot(ax, X, Y):
    #     bp = boxplot(column='age', by='pclass', grid=False)
    #     for i in [1, 2, 3]:
    #         y = titanic.age[titanic.pclass == i].dropna()
    #         # Add some random "jitter" to the x-axis
    #         x = np.random.normal(i, 0.04, size=len(y))
    #         plot(x, y, 'r.', alpha=0.2)

    def plot_mlmc_conv(self, target_var):
        import matplotlib.pyplot as plt

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








    def convergence_test(self):

        # generate samples
        total_samples = 10000
        mc = self.mc
        mc.set_initial_n_samples(self.n_levels*[total_samples])
        mc.refill_samples()
        mc.wait_for_simulations()

        var_exp = np.linspace(-1, -4, 10)
        target_var = 10**var_exp
        means_el = []
        vars_el = []
        n_loops = 30
        for t_var in target_var:
            n_samples = mc.set_target_variance(t_var, prescribe_vars=self.ref_diff_vars)
            n_samples = np.max(n_samples, axis=1).astype(int)

            print( t_var, n_samples)
            n_samples = np.minimum(n_samples, int(total_samples*0.9))
            n_samples = np.maximum(n_samples, 1)
            for i in range(n_loops):
                mc.subsample(n_samples)
                means_est, vars_est = mc.estimate_moments(self.moments_fn)
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


    def check_lindep(self, x, y, slope):
        fit = np.polyfit(np.log(x), np.log(y), deg=1)
        print("MC fit: ", fit, slope)
        assert np.isclose(fit[0], slope, rtol=0.2), (fit, slope)

def test_var_estimate():
    np.random.seed(3)
    n_levels=[4] #, 2, 3] #, 4, 5, 7, 9]
    n_moments=[20] #,4,5] #,7, 10,14,19]
    distr = [
        (stats.norm(loc=42.0, scale=5), False)
        #,(stats.lognorm(scale=np.exp(-5), s=1), True)            # worse conv of higher moments
        #, (stats.lognorm(scale=np.exp(-5), s=0.5), True)         # worse conv of higher moments
        #,(stats.chi2(df=10), True)
        #,(stats.weibull_min(c=1), True)    # Exponential
        #,(stats.weibull_min(c=1.5), True)  # Infinite derivative at zero
        #,(stats.weibull_min(c=3), True)    # Close to normal
        #,(stats.alpha(a=0.5), True),  # infinite mean, distribution of 1/Y if Y is Norm(mu, sig); a=mu/sig
        ]
    #statprof.start()

    # import cProfile, pstats, io
    # pr = cProfile.Profile()
    # pr.enable()

    for nl in n_levels:
        for nm in n_moments:
            for d, il in distr:
                mc = TestMLMC(nl, nm, d, il)

                #impl_var_estimate(nl, nm, tv, d, il)
                mc.convergence_test()

    # pr.disable()
    # s = io.StringIO()
    # sortby = 'cumulative'
    # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # ps.print_stats()
    # print(s.getvalue())

    #statprof.stop()
    #statprof.display()



def test_save_load_samples():
    # 1. make some somples in levels
    # 2. copy key data from levels
    # 3. clean levels
    # 4. create new mlmc object
    # 5. read stored data
    # 6. check that they match the reference copy

    #print("var: ", distr.var())
    work_dir = '_test_tmp'
    work_dir = os.path.join( os.path.dirname(os.path.realpath(__file__)), '_test_tmp')


    n_levels = 5
    distr = stats.norm()
    step_range = (0.8, 0.01)
    pbs = flow_pbs.FlowPbs(work_dir=work_dir)
    simulation_config = dict(
        distr= distr, complexity=2)
    simultion_factory = lambda t_level: SimulationTest.make_sim(simulation_config, step_range, t_level)
    mc = mlmc.mlmc.MLMC(n_levels, simultion_factory, pbs)
    mc.set_initial_n_samples()
    mc.refill_samples()
    mc.wait_for_simulations()

    # Copy level data
    level_data = []
    for level in mc.levels:
        l_data  = (level.running_simulations.copy(),
                   level.finished_simulations.copy(),
                   level.sample_values)
        assert not np.isnan(level.sample_values).any()
        level_data.append(l_data)

    mc.clean_levels()
    pbs.close()
    # New mlmc
    pbs = flow_pbs.FlowPbs(work_dir=work_dir, reload = True)
    mc = mlmc.mlmc.MLMC(n_levels, simultion_factory, pbs)

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