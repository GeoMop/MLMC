import os
import sys
import json
import shutil
import copy
#import statprof
import numpy as np
import scipy.stats as stats
import scipy.integrate as integrate

src_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append( os.path.join(src_path, '..', '..', 'src'))

import mlmc.mlmc
import mlmc.simulation
import mlmc.moments
import mlmc.distribution
import flow_pbs
import flow_mc as flow_mc
import mlmc.correlated_field as cf


class ProcessMLMC:
    
    def __init__(self, work_dir):        
        self.work_dir = os.path.abspath(work_dir)
        self._serialize = ['n_levels', 'step_range']

    def get_root_dir(self):
        root_dir = os.path.abspath(self.work_dir)
        while root_dir != '/':
            root_dir, tail = os.path.split(root_dir)
        return tail

    def setup_environment(self):
        self.pbs_config = dict(
            package_weight=250000,  # max number of elements per package
            n_cores=1,
            n_nodes=1,
            select_flags=['cgroups=cpuacct'],
            mem='4gb',
            queue='charon')

        print("root: '", self.get_root_dir(),"'")
        if self.get_root_dir() == 'storage':
            # Metacentrum
            self.sample_sleep = 30
            self.init_sample_timeout = 600
            self.sample_timeout = 0
            self.pbs_config['qsub'] = '/usr/bin/qsub'
            flow123d = "/storage/praha1/home/jan_brezina/local/flow123d_2.2.0/flow123d"
            gmsh = "/storage/liberec1-tul/home/martin_spetlik/astra/gmsh/bin/gmsh"
        else:
            # Local
            self.sample_sleep = 1
            self.init_sample_timeout = 60
            self.sample_timeout = 60
            #self.pbs_config['qsub'] = os.path.join(src_path, '..', 'mocks', 'qsub')
            self.pbs_config['qsub'] = None
            flow123d = "/home/jb/workspace/flow123d/bin/fterm flow123d dbg"
            #flow123d = os.path.join(src_path, '..', 'mocks', 'flow_mock')
            gmsh = "/home/jb/local/gmsh-3.0.5-git-Linux/bin/gmsh"

        self.env = dict(
            flow123d=flow123d,
            gmsh=gmsh,
        )

    def _set_n_levels(self, nl):
        self.n_levels = nl
        self.output_dir = os.path.join(self.work_dir, "output_{}".format(nl))
        self._setup_file = os.path.join(self.output_dir, "setup.json")

    def setup(self, n_levels):
        self._set_n_levels(n_levels)
        self.output_dir = os.path.join(self.work_dir, "output_{}".format(n_levels))

        self.setup_environment()

        fields = cf.Fields([
            cf.Field('conductivity', cf.SpatialCorrelatedField('gauss', mu=1e-8, sigma=1, corr_length=0.1, log=True)),
        ])

        self.step_range = (1, 0.02)





        yaml_path = os.path.join(self.work_dir, '01_conductivity.yaml')
        geo_path = os.path.join(self.work_dir, 'square_1x1.geo')
        self.simulation_config = {
            'env': self.env,  # The Environment.
            'output_dir': self.output_dir,
            'fields': fields,  # correlated_field.FieldSet object
            'yaml_file': yaml_path,  # The template with a mesh and field placeholders
            'sim_param_range': self.step_range,  # Range of MLMC simulation parametr. Here the mesh step.
            'geo_file': geo_path,  # The file with simulation geometry (independent of the step)
            'field_template': "!FieldElementwise {mesh_data_file: \"${INPUT}/%s\", field_name: %s}"
            #'field_template': "!FieldElementwise {gmsh_file: \"${INPUT}/%s\", field_name: %s}"
        }

    @staticmethod
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    def initialize(self, clean):
        print('init')
        self.pbs_work_dir = os.path.join(self.output_dir, "scripts")
        self.pbs = flow_pbs.FlowPbs(self.pbs_work_dir,
                       qsub=self.pbs_config['qsub'],
                       clean=clean)
        self.pbs.pbs_common_setting(**self.pbs_config)
        if not clean:
            print('read logs')
            self.pbs.reload_logs()
            print('done')
        self.env['pbs'] = self.pbs

        flow_mc.FlowSim.total_sim_id = 0
        self.simultion_factory = flow_mc.FlowSim.factory(self.step_range,
            config = self.simulation_config, clean=clean)

        self.mc = mlmc.mlmc.MLMC(self.n_levels, self.simultion_factory, self.pbs)
        if clean:
            #assert ProcessMLMC.is_exe(self.env['flow123d'])
            assert ProcessMLMC.is_exe(self.env['gmsh'])
            #assert ProcessMLMC.is_exe(self.pbs_config['qsub'])
            self.save()

    def collect(self):
        return self.mc.wait_for_simulations(sleep=self.sample_sleep, timeout=0.1)

    def set_moments(self, n_moments, log=False):
        self.moments_fn = mlmc.moments.Legendre(n_moments, self.domain, safe_eval=True, log=log)
        #self.moments_fn = mlmc.moments.Fourier(n_moments, self.domain, safe_eval=True, log=log)
        return self.moments_fn

    def n_sample_estimate(self, target_variance):
        self.n_samples = self.mc.set_initial_n_samples([30, 3])
        self.mc.refill_samples()
        self.mc.wait_for_simulations(sleep=self.sample_sleep, timeout=self.init_sample_timeout)

        self.domain = self.mc.estimate_domain()
        self.mc.set_target_variance(0.001, self.moments_fn, 2.0)

    def generate_jobs(self, n_samples=None, target_variance = None):
        if n_samples is not None:
            self.mc.set_initial_n_samples(n_samples)
        self.mc.refill_samples()
        self.mc.wait_for_simulations(sleep=self.sample_sleep, timeout=self.sample_timeout)


    def save(self):

        setup={}
        for key in self._serialize:
            setup[key] = self.__dict__.get(key, None)
        with open(self._setup_file, 'w') as f:
            json.dump(setup, f)

      
    def load(self, n_levels):
        
        self._set_n_levels(n_levels)
        self.setup(n_levels)
        print('read  setup')
        # read setup
        with open(self._setup_file, 'r') as f:
            setup = json.load(f)
        for key in self._serialize:

            self.__dict__[key] = setup.get(key, None)


        self.initialize(clean=False)
        
        
    #     self.distr = distr
    #     self.n_levels = n_levels
    #     self.n_moments = n_moments
    #
    #     # print("var: ", distr.var())
    #     step_range = (0.8, 0.01)
    #     coef = (step_range[1]/step_range[0])**(1.0/(self.n_levels - 1))
    #     self.steps = step_range[0] * coef**np.arange(self.n_levels)
    #     self.mc, self.sims = self.make_simulation_mc(step_range)
    #
    #     # reference variance
    #     true_domain = distr.ppf([0.001, 0.999])
    #     self.moments_fn = lambda x, n=n_moments, a=true_domain[0], b=true_domain[1]: \
    #                                 mlmc.moments.legendre_moments(x, n, a, b)
    #
    #     sample_size = 10000
    #     self.exact_mean = self.mc_estimate_exact_mean(self.moments_fn, 5*sample_size)
    #     means, vars = self.mc_estimate_diff_var(self.moments_fn, sample_size)
    #     self.ref_diff_vars = np.array(vars)[:, :]
    #     self.ref_vars = np.sum(np.array(vars)/sample_size, axis=0)
    #     self.ref_means = np.sum(np.array(means), axis=0)
    #     #print("Ref means: ", ref_means)
    #     #print("Ref vars: ", ref_vars)
    #
    # def make_simulation_mc(self, step_range):
    #
    # def direct_estimate_diff_var(self, moments_fn, domain):
    #     means = []
    #     vars = []
    #     sim_l = None
    #     for l in range(len(self.sims)):
    #         sim_0 = sim_l
    #         sim_l = lambda x, h=self.sims[l].step: self.sims[l]._sample_fn(x, h)
    #         if l == 0:
    #             md_fn = lambda x: moments_fn(sim_l(x))
    #         else:
    #             md_fn = lambda x: moments_fn(sim_l(x)) - moments_fn(sim_0(x))
    #         fn = lambda x: (md_fn(x)).T * self.distr.pdf(x)
    #         moment_means = integrate.fixed_quad(fn, domain[0], domain[1], n=100)[0]
    #         fn2 = lambda x: ((md_fn(x) - moment_means[None, :]) ** 2).T * self.distr.pdf(x)
    #         moment_vars = integrate.fixed_quad(fn2, domain[0], domain[1], n=100)[0]
    #         means.append(moment_means)
    #         vars.append(moment_vars)
    #     return means, vars
    #
    # def mc_estimate_diff_var(self, moments_fn, size=10000):
    #     """
    #     :param moments_fn:
    #     :param domain:
    #     :param distr:
    #     :param sims:
    #     :return: means, vars ; shape n_levels, n_moments
    #     """
    #     means = []
    #     vars = []
    #     sim_l = None
    #     for l in range(len(self.sims)):
    #         sim_0 = sim_l
    #         sim_l = lambda x, h=self.sims[l].step: self.sims[l]._sample_fn(x, h)
    #         X = self.distr.rvs(size=size)
    #         if l == 0:
    #             MD = moments_fn(sim_l(X))
    #         else:
    #             MD = (moments_fn(sim_l(X)) - moments_fn(sim_0(X)))
    #
    #         moment_means = np.mean(MD, axis=0)
    #         moment_vars = np.var(MD, axis=0, ddof=1)
    #         means.append(moment_means)
    #         vars.append(moment_vars)
    #     return np.array(means), np.array(vars)
    #
    # def mc_estimate_exact_mean(self, moments_fn, size= 20000):
    #     X = self.distr.rvs(size=size)
    #     return np.mean(moments_fn(X), axis=0)
    #
    #
    # def generate_samples(self, size):
    #     # generate samples
    #     self.mc.set_initial_n_samples(self.n_levels*[size])
    #     self.mc.refill_samples()
    #     self.mc.wait_for_simulations()
    #
    # # @staticmethod
    # # def box_plot(ax, X, Y):
    # #     bp = boxplot(column='age', by='pclass', grid=False)
    # #     for i in [1, 2, 3]:
    # #         y = titanic.age[titanic.pclass == i].dropna()
    # #         # Add some random "jitter" to the x-axis
    # #         x = np.random.normal(i, 0.04, size=len(y))
    # #         plot(x, y, 'r.', alpha=0.2)
    #
    # def plot_mlmc_conv(self, target_var):
    #     import matplotlib.pyplot as plt
    #
    #     fig = plt.figure(figsize=(10,20))
    #
    #     for m in range(1, self.n_moments):
    #         ax = fig.add_subplot(2, 2, m)
    #         color = 'C' + str(m)
    #         Y = np.var(self.means_est[:,:,m], axis=1)
    #         ax.plot(target_var, Y, 'o', c=color, label=str(m))
    #
    #
    #         Y = np.percentile(self.vars_est[:, :, m],  [10, 50, 90], axis=1)
    #         ax.plot(target_var, Y[1,:], c=color)
    #         ax.plot(target_var, Y[0,:], c=color, ls='--')
    #         ax.plot(target_var, Y[2, :], c=color, ls ='--')
    #         Y = (self.exact_mean[m] - self.means_est[:, :, m])**2
    #         Y = np.percentile(Y, [10, 50, 90], axis=1)
    #         ax.plot(target_var, Y[1,:], c='gray')
    #         ax.plot(target_var, Y[0,:], c='gray', ls='--')
    #         ax.plot(target_var, Y[2, :], c='gray', ls ='--')
    #         ax.set_yscale('log')
    #         ax.set_xscale('log')
    #         ax.legend()
    #         ax.set_ylabel("observed var. of mean est.")
    #
    #     plt.show()
    #
    #
    #
    #
    #
    #
    #
    # def check_lindep(self, x, y, slope):
    #     fit = np.polyfit(np.log(x), np.log(y), deg=1)
    #     print("MC fit: ", fit, slope)
    #     assert np.isclose(fit[0], slope, rtol=0.2), (fit, slope)
    #
    # def convergence_test(self):
    #
    #
    #     # subsamples
    #     var_exp = np.linspace(-1, -4, 10)
    #     target_var = 10**var_exp
    #     means_el = []
    #     vars_el = []
    #     n_loops = 30
    #     for t_var in target_var:
    #         n_samples = self.mc.set_target_variance(t_var, prescribe_vars=self.ref_diff_vars)
    #         n_samples = np.max(n_samples, axis=1).astype(int)
    #
    #         print( t_var, n_samples)
    #         n_samples = np.minimum(n_samples, (self.mc.n_samples*0.9).astype(int))
    #         n_samples = np.maximum(n_samples, 1)
    #         for i in range(n_loops):
    #             self.mc.subsample(n_samples)
    #             means_est, vars_est = self.mc.estimate_moments(self.moments_fn)
    #             means_el.append(means_est)
    #             vars_el.append(vars_est)
    #     self.means_est = np.array(means_el).reshape(len(target_var), n_loops, self.n_moments)
    #     self.vars_est = np.array(vars_el).reshape(len(target_var), n_loops, self.n_moments)
    #
    #     #self.plot_mlmc_conv(target_var)
    #
    #     for m in range(1, self.n_moments):
    #         Y = np.var(self.means_est[:,:,m], axis=1)
    #         self.check_lindep(target_var, Y, 1.0)
    #         Y = np.mean(self.vars_est[:, :, m], axis=1)
    #         self.check_lindep(target_var, Y , 1.0)
    #
    #         X = np.tile(target_var, n_loops)
    #         Y = np.mean(np.abs(self.exact_mean[m] - self.means_est[:, :, m])**2, axis=1)
    #         self.check_lindep(target_var, Y,  1.0)
    #
    def plot_diff_var(self):
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(10,20))
        ax = fig.add_subplot(1, 1, 1)

        error_power = 2.0
        for m in range(1, self.n_moments):
            color = 'C' + str(m)
            Y = self.ref_diff_vars[:,m]/(self.steps**error_power)

            ax.plot(self.steps[1:], Y[1:], c=color, label=str(m))
            ax.plot(self.steps[0], Y[0], 'o', c=color)
        #
        #
        #     Y = np.percentile(self.vars_est[:, :, m],  [10, 50, 90], axis=1)
        #     ax.plot(target_var, Y[1,:], c=color)
        #     ax.plot(target_var, Y[0,:], c=color, ls='--')
        #     ax.plot(target_var, Y[2, :], c=color, ls ='--')
        #     Y = (self.exact_mean[m] - self.means_est[:, :, m])**2
        #     Y = np.percentile(Y, [10, 50, 90], axis=1)
        #     ax.plot(target_var, Y[1,:], c='gray')
        #     ax.plot(target_var, Y[0,:], c='gray', ls='--')
        #     ax.plot(target_var, Y[2, :], c='gray', ls ='--')
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

    def plot_n_sample_est_distributions(self, cost, total_std, n_samples, rel_moments):
        import matplotlib.pyplot as plt
        title = "N levels = {}".format(self.mc.n_levels)

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
        fig.savefig("L{}_subsample_estimates.pdf".format(self.mc.n_levels))





    @staticmethod
    def ecdf(x):
        xs = np.sort(x)
        ys = np.arange(1, len(xs)+1)/float(len(xs))
        return xs, ys

    def plot_pdf_approx(self, ax1, ax2, mc0_samples):
        import matplotlib.pyplot as plt
        
        X = np.exp( np.linspace(np.log(self.domain[0]), np.log(self.domain[1]), 1000) )

        
        n_levels = self.mc.n_levels
        color = "C{}".format(n_levels)
        label = "L{}: ".format(n_levels) #+ str(self.mc.n_samples)
        Y = self.distr_obj.density(X)
        ax1.plot(X, Y, c=color, label=label)
        #ax1.plot(X, -np.log(Y), c=color, label=label)
        
        Y = self.distr_obj.cdf(X)
        ax2.plot(X, Y, c=color, label=label)
        
        if n_levels == 1:
            ax1.axvline(x=self.domain[0], c='black')
            ax1.axvline(x=self.domain[1], c='black')
            ax1.axvline(x=np.exp(np.mean(np.log(mc0_samples))), c='black')

            bins = np.exp(np.linspace(np.log(self.domain[0]), np.log(10), 100))
            ax1.hist(mc0_samples, normed=1,  bins=bins, alpha = 0.3, label='full MC', color='gray')
            X, Y = ProcessMLMC.ecdf(mc0_samples)
            ax2.plot(X, Y, 'black')

            #Y = stats.lognorm.pdf(X, s=1, scale=np.exp(0.0))
            #ax1.plot(X, Y, c='gray', label="stdlognorm")
            #Y = stats.lognorm.cdf(X, s=1, scale=np.exp(0.0))
            #ax2.plot(X, Y, c='gray')
          
        #ax1.axvline(x=self.est_domain[0], c=color)
        #ax1.axvline(x=self.est_domain[1], c=color)
          
        
        

    @staticmethod    
    def align_array(arr):
        return "[" + ", ".join([ "{:10.5f}".format(x) for x in arr]) + "]" 
  


    def compute_results(self, mlmc_0, n_moments):
        self.domain = mlmc_0.ref_domain         
        self.est_domain = self.mc.estimate_domain()       
        moments_fn = self.set_moments(n_moments, log=True)       

        #####
        # Compute global characteristics of full sample set
        t_var = 1e-5
        self.ref_diff_vars, ns = self.mc.estimate_diff_vars(moments_fn)

        # std how it may be if we are able to keep all diff vars at minimal contributor value

        self.ref_n_samples = ns
        self.ref_diff_vars /= ns[:, None]
        self.ref_std_min = np.sqrt( len(ns) * self.ref_diff_vars[0])
        self.max_var_level = np.argmax(self.ref_diff_vars, axis=0)
        self.ref_moments, self.ref_vars = self.mc.estimate_moments(moments_fn)

        self.ref_std = np.sqrt(self.ref_vars)
        self.ref_diff_vars_max = np.max(self.ref_diff_vars, axis =1)
        ref_n_samples = self.mc.set_target_variance(t_var, moments_fn)
        self.ref_est_n_samples = np.max(ref_n_samples, axis=1)
        self.ref_cost = self.mc.estimate_cost(n_samples = self.ref_est_n_samples)
        self.ref_total_std = np.sqrt(np.mean(np.sum(self.ref_diff_vars, axis=0)))
        self.ref_total_std_x = np.sqrt(np.mean(self.ref_vars))



        print("\nLevels : ", self.mc.n_levels, "---------")
        print("moments:  ", self.align_array(self.ref_moments))
        print("std:      ", self.align_array(self.ref_std ))
        print("err:      ", self.align_array(self.ref_moments - mlmc_0.ref_moments ))
        print("domain:   ", self.est_domain)
        print("cost:     ", self.ref_cost)
        print("tot. std: ", self.ref_total_std, self.ref_total_std_x)
        print("dif_vars: ", self.align_array(self.ref_diff_vars_max), )
        print("ns :      ", self.align_array(self.ref_est_n_samples))
        print("")

        diff = self.ref_moments - mlmc_0.ref_moments
        str_moments = ["{:8.3f}".format(m) for m in self.ref_moments]
        str_err = ["{:8.2f}".format(m) for m in np.abs(100*diff)]
        str_std = ["{:8.2f}".format(m) for m in np.abs(100*self.ref_std)]
        #mm_line = np.stack([self.ref_moments, diff, self.ref_std], axis=1).ravel()
        print(" & ".join([" & ".join(triple) for triple in zip(str_moments, str_err, str_std)]))

        steps = np.array([ l.fine_simulation.step for l in self.mc.levels])

        table = np.stack([np.arange(self.n_levels),
                          steps,
                          self.ref_diff_vars_max,
                          self.ref_est_n_samples], axis=0)
        print(" \\\\\n".join([" & ".join(map(str, line)) for line in table]))
        print("")


        print("SUBSAMPLES")
        
              
        #a, b = self.domain              
        #distr_mean = self.distr_mean = moments_fn.inv_linear(ref_means[1])
        #distr_var  = ((2*ref_means[2] + 1)/3 - ref_means[1]**2) / 4 * ((b-a)**2)        
        #self.distr_std  = np.sqrt(distr_var)

        #####
        # Distribution approximation

        moments_data = np.stack( (self.ref_moments, self.ref_vars), axis=1)
        ref_mean, ref_var = self.mc.estimate_norm_approx()

        self.distr_obj = mlmc.distribution.Distribution(moments_fn, moments_data, ref_mean, ref_var)
        self.distr_obj.domain = self.domain
        result = self.distr_obj.estimate_density(tol=0.001)
        print(result)

        n_loops = 0

        # use subsampling to:
        # - simulate moments estimation without appriori knowledge about n_samples (but still on fixed domain)
        # - estimate error in final number of samples
        # - estimate error of estimated varinace of moments
        # - estimate cost error (due to n_sample error)
        # - estimate average error against reference momemnts
        l_cost_err = []
        l_total_std_err = []
        l_n_samples_err = []
        l_rel_mom_err = []
        for i in range(n_loops):
            fractions = [0, 0.001, 0.01, 0.1]
            for fr in fractions:
                if fr == 0:
                    nL, n0 = 3, 30
                    if self.mc.n_levels > 1:
                        L = self.n_levels
                        factor = (nL / n0) ** (1 / (L - 1))
                        n_samples = (n0 * factor ** np.arange(L)).astype(int)
                    else:
                        n_samples = [ n0]
                else:
                    n_samples = np.maximum( n_samples, (fr*max_est_n_samples).astype(int))
                # n_samples = np.maximum(n_samples, 1)

                self.mc.subsample(n_samples)
                est_diff_vars, _ = self.mc.estimate_diff_vars(self.moments_fn)
                est_n_samples = self.mc.set_target_variance(t_var, prescribe_vars=est_diff_vars)
                max_est_n_samples = np.max(est_n_samples, axis=1)
                
                #est_cost = self.mc.estimate_cost(n_samples=max_est_n_samples.astype(int))
                #est_total_var = np.sum(self.ref_diff_vars / max_est_n_samples[:, None])/self.n_moments

                #n_samples_err = np.min( (max_est_n_samples - ref_n_samples) /ref_n_samples)
                ##total_std_err =  np.log2(est_total_var/ref_total_var)/2
                #total_std_err = (np.sqrt(est_total_var) - np.sqrt(ref_total_var)) / np.sqrt(ref_total_var)
                #cost_err = (est_cost - ref_cost)/ref_cost
                #print("Fr: {:6f} NSerr: {} Tstderr: {} cost_err: {}".format(fr, n_samples_err, total_std_err, cost_err))
            
            est_diff_vars, _ = self.mc.estimate_diff_vars(self.moments_fn)
            est_moments, est_vars = self.mc.estimate_moments(self.moments_fn)
            #print("Vars:", est_vars)
            #print("em:", est_moments)
            #print("rm:", self.ref_moments)
            
            n_samples_err = np.min((n_samples - self.ref_est_n_samples) / self.ref_est_n_samples)
            est_total_std = np.sqrt(np.mean(np.sum(est_diff_vars / n_samples[:, None], axis=0)))
            # est_total_std = np.sqrt(np.mean(est_vars))
            total_std_err =  np.log2(est_total_std/self.ref_total_std)
            est_cost = self.mc.estimate_cost(n_samples = n_samples)
            cost_err = (est_cost - self.ref_cost)/self.ref_cost
            
            #print("MM: ", (est_moments[1:] - self.ref_moments[1:]), "\n    ",  np.sqrt(est_vars[1:]))
            
            relative_moments_err = np.linalg.norm((est_moments[1:] - self.ref_moments[1:]) / np.sqrt(est_vars[1:]))
            #print("est cost: {} ref cost: {}".format(est_cost, ref_cost))
            #print(n_samples)
            #print(np.maximum( n_samples, (max_est_n_samples).astype(int)))
            #print(ref_n_samples.astype(int))
            #print("\n")
            l_n_samples_err.append(n_samples_err)
            l_total_std_err.append(total_std_err)
            l_cost_err.append(cost_err)
            l_rel_mom_err.append(relative_moments_err)
            
        l_cost_err.sort()
        l_total_std_err.sort()
        l_n_samples_err.sort()
        l_rel_mom_err.sort()
        
        # def describe(arr):
        #     q1, q3 = np.percentile(arr, [25,75])
        #     print(np.min(arr), q1, np.mean(arr), q3, np.max(arr))
        #     return "{:8.2f} < {:8.2f} | {:8.2f} | {:8.2f} < {:8.2f}".format(
        #         np.min(arr), q1, np.mean(arr), q3, np.max(arr))
        #
        # print("Cost err:       ", describe(l_cost_err))
        # print("Total std err:  ", describe(l_total_std_err))
        # print("N. samples err: ", describe(l_n_samples_err))
        # print("Rel. Mom. err:  ", describe(l_rel_mom_err))
        #
        # #print(l_rel_mom_err)
        # self.plot_n_sample_est_distributions(l_cost_err, l_total_std_err, l_n_samples_err, l_rel_mom_err)


        # moments_data = np.stack( (est_moments, est_vars), axis=1)
        # self.distr_obj = mlmc.distribution.Distribution(moments_fn, moments_data)
        # self.distr_obj.domain = self.domain
        # result = self.distr_obj.estimate_density(tol=0.001)
        #print(result)
        self.mc.subsample(None)
        
        




def all_results(mlmc_list):
        import matplotlib.pyplot as plt
        n_moments = 9
        mc0_samples = mlmc_list[0].mc.levels[0].sample_values[:, 0]
        center = np.mean(np.log(mc0_samples))
        err = 5*np.std(np.log(mc0_samples))
        q_min, q_max = np.percentile(mc0_samples, [0.01, 99.99])
        #mlmc_list[0].ref_domain = (q_min, q_max-1)
        mlmc_list[0].ref_domain = (np.exp(center - err), np.exp(center + err))
        #mlmc_list[0].ref_domain = (np.min(mc0_samples), np.max(mc0_samples) )

        # Plot density
        fig = plt.figure(figsize=(15,10))
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_xscale('log')
        ax1.set_ylim([0, 2])
        ax1.set_xlim([0.1, 10])

        # Plot distribution
        fig2 = plt.figure(figsize=(15, 10))
        ax2 = fig2.add_subplot(1, 1, 1)
        ax2.set_xscale('log')
        ax2.set_ylim(-0.1, 1.1)
        ax2.set_xlim(0.1, 10)


        for prmc in mlmc_list:
            prmc.compute_results(mlmc_list[0], n_moments)
            prmc.plot_pdf_approx(ax1, ax2, mc0_samples)



        ax1.legend()
        ax2.legend()

        fig.savefig('compare_density.pdf')
        fig2.savefig('compare_distribution.pdf')

        # Plot estimated moments and their error
        space = 2*  len(mlmc_list)
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(1, 1, 1)
        for prmc in mlmc_list:
            X = len(prmc.mc.levels) + space*np.arange(1,n_moments)
            Y = prmc.ref_diff_vars

            # std corresponding to the first level
            # Obervation:
            # - 2 level is bad since V1 is usualy same as V0
            # - so at least 4 level MLMC should be used
            #ax.errorbar(X, prmc.ref_moments[1:], yerr=3 * prmc.ref_std_min[1:], fmt='o', capsize=5, ecolor='grey')
            ax.errorbar(X, prmc.ref_moments[1:], yerr=3 * prmc.ref_std[1:], fmt='o', capsize=3, label=str(len(prmc.mc.levels)))


        x_ticks = space*np.arange(1,n_moments) + 0.7*(len(mlmc_list) +1)
        x_ticks_labels = [str(m) for m in np.arange(1,n_moments)]

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticks_labels)
        ax.set_xlabel("Legendere polynomial moments")
        ax.legend(title="Num. of levels")

        fig.suptitle("Estimating moments of log(Q) using different number of levels.")
        fig.savefig('compare_moments.pdf')


        # Plot diff var and n_samples estimates
        # X - mesh_step (0.0 for V_l); log
        # Y - V_l resp n_l; log
        # color - moment
        # shape - N levels
        # error bars - from number of samples
        cmap = plt.get_cmap('viridis')
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(1, 1, 1)
        #markers = ['*', 'o', 'h', '+', 'v', 's', 'd']
        for i, prmc in enumerate(mlmc_list[-1:]):
            steps = np.array([l.fine_simulation.step for l in prmc.mc.levels])
            #X = len(prmc.mc.levels) + space*np.arange(1,n_moments)


            # std corresponding to the first level
            # Obervation:
            # - 2 level is bad since V1 is usualy same as V0
            # - so at least 4 level MLMC should be used
            #ax.errorbar(X, prmc.ref_moments[1:], yerr=3 * prmc.ref_std_min[1:], fmt='o', capsize=5, ecolor='grey')
            for m in range(1, n_moments):
                Y = prmc.ref_diff_vars[:, m] * prmc.ref_n_samples
                Y_err = 1/prmc.ref_n_samples
                #s = markers[i]
                c = cmap(m/n_moments)
                ax.errorbar(steps, Y,  fmt='o', color=c ,marker='o'  , label=str(m))
            #ax.plot(steps, steps**2)
            ax.plot(steps, 100*steps ** 4, c='orange', label="$h^4$")

            Y_min = np.min(prmc.ref_diff_vars[:, 1:] * prmc.ref_n_samples[:,None], axis=1)
            for i, (x, y) in enumerate(zip(steps, Y_min)):
                #samples_label = "$n_{}$={}".format(i, int(prmc.ref_est_n_samples[i]))
                samples_label = "{}".format(int(prmc.ref_est_n_samples[i]))
                ax.text(x, y*0.1, samples_label)

        #x_ticks = space*np.arange(1, n_moments) + 0.7*(len(mlmc_list) +1)
        #x_ticks_labels = [str(m) for m in np.arange(1,n_moments)]

        ax.set_xscale('log')
        ax.set_yscale('log')
        #ax.set_xticks(x_ticks)
        #ax.set_xticklabels(x_ticks_labels)
        ax.set_ylim(1e-10, 1)
        ax.set_xlabel("$h$ - mesh step")
        ax.set_ylabel("$V_l$")
        ax.legend(title="Moments", loc=4, ncol=2)

        #fig.suptitle("Estimating moments of log(Q) using different number of levels.")
        fig.savefig('est_vars.pdf')



def all_collect(mlmc_list):
    running = 1
    while running > 0:
        running = 0
        for mc in mlmc_list:
            running += mc.collect()
        print("N running: ", running)


def main():
    level_list = [9]
    
    print('main')
    command = sys.argv[1]
    work_dir = os.path.abspath(sys.argv[2])
    if command == 'run':
        os.makedirs(work_dir, mode=0o775, exist_ok=True)

        # copy
        for file_res in os.scandir(src_path):
            if (os.path.isfile(file_res.path)):
                shutil.copy(file_res.path, work_dir)
        
        mlmc_list = []
        for nl in [1]:  #, 2, 3, 4,5, 7, 9]:
            mlmc = ProcessMLMC(work_dir)
            mlmc.setup(nl)
            mlmc.initialize(clean=True)
            ns = { 
                1: [7087],
                2: [14209,  332],
                3: [18979,  487,    2],
                4: [13640,  610,    2,    2],
                5: [12403,  679,   10,    2,    2],
                7: [12102,  807,   11,    2,    2,    2,    2],
                9: [11449,  806,   72,    8,    2,    2,    2,    2,    2]
                }
            
            
            n_samples = 2*np.array(ns[nl])
            #mlmc.generate_jobs(n_samples=n_samples)
            #mlmc.generate_jobs(n_samples=[10000, 100])
            mlmc.mc.levels[0].target_n_samples = 1
            mlmc.generate_jobs(n_samples=[1])
            mlmc_list.append(mlmc)  

        #for nl in [3,4]:
            #mlmc = ProcessMLMC(work_dir)
            #mlmc.load(nl)
            #mlmc_list.append(mlmc)  
            
        all_collect(mlmc_list)
        
    elif command == 'collect':
        assert os.path.isdir(work_dir)
        mlmc_list = []
        for nl in [ 5,7]:
            mlmc = ProcessMLMC(work_dir)
            mlmc.load(nl)
            mlmc_list.append(mlmc)  
        all_collect(mlmc_list)    
    
    elif command == 'process':
        assert os.path.isdir(work_dir)
        mlmc_list = []
        for nl in [ 1,2,3,4,5, 7,9]: #[ 1,2,3,4,5,7,9]:
        #for nl in [ 1,2]:
            prmc = ProcessMLMC(work_dir)
            prmc.load(nl)
            mlmc_list.append(prmc)  

        all_results(mlmc_list)
            
        

main()
