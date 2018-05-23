import os
import scipy.stats as stats
import scipy.integrate as integrate

import mlmc.mlmc
import mlmc.simulation
import mlmc.moments
import mlmc.distribution
import numpy as np
import flow_pbs

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
        return x + h * np.sqrt(x)

    def simulation_sample(self, tag):
        """
        Run simulation
        :param sim_id:    Simulation id
        """
        x = self._input_sample
        h = self.step
        y = self._sample_fn(x, h)
        self._result_dict[tag] = y

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


def direct_estimate_diff_var(moments_fn, domain, distr, sims):
    means=[]
    vars=[]
    sim_l=None
    for l in range(len(sims)):
        sim_0 = sim_l
        sim_l = lambda x, h=sims[l].step: sims[l]._sample_fn(x, h)
        if l == 0:
            md_fn = lambda x : moments_fn(sim_l(x))
        else:
            md_fn = lambda x: moments_fn(sim_l(x)) - moments_fn(sim_0(x))
        fn = lambda x: (md_fn(x)).T * distr.pdf(x)
        moment_means = integrate.fixed_quad(fn, domain[0], domain[1],n=100)[0]
        fn2 = lambda x : ((md_fn(x) - moment_means[None,:])**2).T*distr.pdf(x)
        moment_vars = integrate.fixed_quad(fn2, domain[0], domain[1], n=100)[0]
        means.append(moment_means)
        vars.append(moment_vars)
    return means, vars

def mc_estimate_diff_var(moments_fn, domain, distr, sims):
    means=[]
    vars=[]
    sim_l = None
    for l in range(len(sims)):
        sim_0 = sim_l
        sim_l = lambda x, h=sims[l].step: sims[l]._sample_fn(x, h)
        X = distr.rvs(size = 1000)
        if l==0:
            MD = moments_fn(sim_l(X))
        else:
            MD = (moments_fn(sim_l(X)) - moments_fn(sim_0(X)))

        moment_means = np.mean(MD, axis=0)
        moment_vars = np.var(MD, axis=0, ddof=1)
        means.append(moment_means)
        vars.append(moment_vars)
    return means, vars



def impl_estimate_n_samples(n_levels, n_moments, target_var, distr, is_log=False):

    print("\n")
    print("L: {} R: {} var: {} distr: {}".format(n_levels, n_moments, target_var, distr.dist.__class__.__name__))
    if is_log:
        q1,q3 = np.log(distr.ppf([0.25, 0.75]))
        iqr = 2*(q3-q1)
        domain =np.exp( [q1 - iqr, q3 + iqr ] )
    else:
        q1,q3 = distr.ppf([0.25, 0.75])
        iqr = 2*(q3-q1)
        domain = [q1 - iqr, q3 + iqr ]

    step_range = (0.8, 0.01)

    file_dir = os.path.dirname(os.path.realpath(__file__))
    scripts_dir = os.path.join(file_dir, 'aux_sim_work')

    pbs = flow_pbs.FlowPbs(work_dir=scripts_dir)
    simulation_config = dict(
        distr= distr, complexity=2)
    simultion_factory = lambda t_level: SimulationTest.make_sim(simulation_config, step_range, t_level)

    moments_fn = lambda x, n=n_moments, a=domain[0], b=domain[1]: mlmc.moments.legendre_moments(x, n, a, b)

    mc = mlmc.mlmc.MLMC(n_levels, simultion_factory, pbs)
    sims = [level.fine_simulation for level in mc.levels ]
    true_domain = distr.ppf([0.001, 0.999])
    means, vars = mc_estimate_diff_var(moments_fn, true_domain, distr, sims)
    vars = np.array(vars)

    raw_vars = []
    reg_vars = []

    n_loops=5
    for _ in range(n_loops):
        mc.clean_levels()
        mc.set_initial_n_samples()
        mc.refill_samples()
        mc.wait_for_simulations()
        #est_domain = mc.estimate_domain()
        #assert est_domain[0] < true_domain[0]
        #assert est_domain[1] > true_domain[1]
        #print("Domain: {}, est: {} true: {}".format(domain, est_domain, true_domain))

        raw_var_estimate, n_col_samples = mc.estimate_diff_vars(moments_fn)

        reg_var_estimate = mc.estimate_diff_vars_regression(moments_fn)
        raw_vars.append(raw_var_estimate)
        reg_vars.append(reg_var_estimate)
    print("n samples   : ", n_col_samples)
    print("ref var min    : ", np.min(vars[:, 1:], axis=1) )
    print("ref var max    : ", np.max(vars, axis=1) )
    #print("mean raw var: ", np.mean( np.array(raw_vars), axis=0) )
    print("std  raw var: ", np.max( np.sqrt(np.mean( (np.array(raw_vars) - vars)**2, axis=0)) / (1e-4 + vars) , axis=1) )

    print("std  reg var: ", np.max( np.sqrt(np.mean( (np.array(reg_vars) - vars)**2, axis=0)) / (1e-4 + vars) , axis=1) )
    #print("Linf raw var: ", np.max( np.abs(np.array(raw_vars) - vars), axis=0) / (1e-4 + vars))

    # n_samples = mc.set_target_variance(target_var, moments_fn)
    # min_n = np.min(n_samples, axis=1)
    # max_n = np.max(n_samples, axis=1)
    # mm = np.array([min_n, max_n]).T
    # print(mm)
    #vars, n_samples = self.estimate_diff_vars(moments_fn)
    #mc.refill_samples()
    #mc.wait_for_simulations()

    #moment_data = mc.estimate_moments(moments_fn)
    #mlmc.distribution.Distribution(mlmc.moments.legendre_moments, moment_data, positive_distr=True)

def test_estimate_n_samples():
    n_levels=[5] #, 2, 3] #, 4, 5, 7, 9]
    n_moments=[11] #,4,5] #,7, 10,14,19]
    target_var = [0.1]
    distr = [
        (stats.lognorm(scale=np.exp(-5), s=2), True),
        (stats.norm(42.0, 5), False),
        (stats.chi2(df=10), True),
        (stats.alpha(a=0.1), True),
        ]
    for nl in n_levels:
        for nm in n_moments:
            for tv in target_var:
                for d, il in distr:
                    impl_estimate_n_samples(nl, nm, tv, d, il)

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