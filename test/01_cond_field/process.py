import os
import sys
import shutil
import yaml
import numpy as np

src_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(src_path, '..', '..', 'src'))

import mlmc.mlmc
import mlmc.simulation
import mlmc.moments
import mlmc.distribution
import pbs
import glob
import flow_mc as flow_mc
import mlmc.correlated_field as cf
from mlmc import moments

from mlmc.estimate import CompareLevels

class FlowProcSim(flow_mc.FlowSim):
    """
    Child from FlowSimulation that defines extract method
    """

    def _extract_result(self, sample):
        """
        Extract the observed value from the Flow123d output.
        :param sample: Sample instance
        :return: None, inf or water balance result (float) and overall sample time
        """
        sample_dir = sample.directory
        if os.path.exists(os.path.join(sample_dir, "FINISHED")):
            # try:
            # extract the flux
            balance_file = os.path.join(sample_dir, "water_balance.yaml")

            with open(balance_file, "r") as f:
                balance = yaml.load(f)

            flux_regions = ['.bc_outflow']
            total_flux = 0.0
            found = False
            for flux_item in balance['data']:
                if flux_item['time'] > 0:
                    break

                if flux_item['region'] in flux_regions:
                    flux = float(flux_item['data'][0])
                    flux_in = float(flux_item['data'][1])
                    if flux_in > 1e-10:
                        raise Exception("Possitive inflow at outlet region.")
                    total_flux += flux  # flux field
                    found = True

            # Get flow123d computing time
            run_time = self.get_run_time(sample_dir)

            if not found:
                raise

            return -total_flux, run_time
        else:
            return None, 0


class UglyMLMC:
    """
    This type of class should not exist. Just few configuration objects
    should be created (PBS, Environment, Sim Factory)
    Postporcessing is done by ProcessMLMC.
    """
    def __init__(self, work_dir, options):
        """
        :param work_dir: Work directory (there will be dir with samples)
        :param options: MLMC options, currently regen_failed (failed realizations will be regenerated)
                                      and keep_collected (keep collected samples dirs)
        """
        self.work_dir = os.path.abspath(work_dir)
        self.mlmc_options = options
        self._serialize = ['work_dir', 'output_dir', 'n_levels', 'step_range']

    def get_root_dir(self):
        """
        Get root directory
        :return: Last pathname component
        """
        root_dir = os.path.abspath(self.work_dir)
        while root_dir != '/':
            root_dir, tail = os.path.split(root_dir)
        return tail

    def setup_environment(self):
        """
        Setup pbs configuration, set flow123d and gmsh commands
        :return: None
        """
        self.pbs_config = dict(
            job_weight=250000,  # max number of elements per job
            n_cores=1,
            n_nodes=1,
            select_flags=['cgroups=cpuacct'],
            mem='4gb',
            queue='charon')

        print("root: '", self.get_root_dir(), "'")
        if self.get_root_dir() == 'storage':
            # Metacentrum
            self.sample_sleep = 30
            self.init_sample_timeout = 600
            self.sample_timeout = 0
            self.pbs_config['qsub'] = '/usr/bin/qsub'
            flow123d = 'flow123d'  # "/storage/praha1/home/jan_brezina/local/flow123d_2.2.0/flow123d"
            gmsh = "/storage/liberec1-tul/home/martin_spetlik/astra/gmsh/bin/gmsh"
        else:
            # Local
            self.sample_sleep = 1
            self.init_sample_timeout = 60
            self.sample_timeout = 60
            # self.pbs_config['qsub'] = os.path.join(src_path, '..', 'mocks', 'qsub')
            self.pbs_config['qsub'] = None
            flow123d = "/home/jb/workspace/flow123d/bin/fterm flow123d dbg"
            # flow123d = os.path.join(src_path, '..', 'mocks', 'flow_mock')
            gmsh = "/home/jb/local/gmsh-3.0.5-git-Linux/bin/gmsh"

        self.env = dict(
            flow123d=flow123d,
            gmsh=gmsh,
        )

    def setup(self, n_levels):
        """
        Set simulation configuration
        :param n_levels: Number of levels
        :return: None
        """
        self.n_levels = n_levels
        self.output_dir = os.path.join(self.work_dir, "output_{}".format(n_levels))

        self.setup_environment()

        fields = cf.Fields([
            cf.Field('conductivity', cf.FourierSpatialCorrelatedField('gauss', dim=2, corr_length=0.125, log=True)),
        ])

        self.step_range = (1, 0.01)

        yaml_path = os.path.join(self.work_dir, '01_conductivity.yaml')
        geo_path = os.path.join(self.work_dir, 'square_1x1.geo')
        self.simulation_config = {
            'env': self.env,  # The Environment.
            'output_dir': self.output_dir,
            'fields': fields,  # correlated_field.FieldSet object
            'yaml_file': yaml_path,  # The template with a mesh and field placeholders
            'sim_param_range': self.step_range,  # Range of MLMC simulation parametr. Here the mesh step.
            'geo_file': geo_path,  # The file with simulation geometry (independent of the step)
            # 'field_template': "!FieldElementwise {mesh_data_file: \"${INPUT}/%s\", field_name: %s}"
            'field_template': "!FieldElementwise {mesh_data_file: \"$INPUT_DIR$/%s\", field_name: %s}"
        }

    @staticmethod
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    def initialize(self, clean):
        """
        Initialize output directory and pbs script creater 
        :param clean: bool, if true then remove current directory and create new one
        :return: None
        """
        # Remove log files
        if clean:
            if os.path.isdir(self.output_dir):
                shutil.rmtree(self.output_dir, ignore_errors=True)
            os.makedirs(self.output_dir, mode=0o775, exist_ok=True)

            try:
                for log in glob.glob(self.output_dir + "/*_log_*"):
                    os.remove(log)
            except OSError:
                pass

        self.pbs_work_dir = os.path.join(self.output_dir, "scripts")
        num_jobs = 0
        if os.path.isdir(self.pbs_work_dir):
            num_jobs = len([_ for _ in os.listdir(self.pbs_work_dir)])

        self.pbs = pbs.Pbs(self.pbs_work_dir,
                           job_count=num_jobs,
                           qsub=self.pbs_config['qsub'],
                           clean=clean)
        self.pbs.pbs_common_setting(flow_3=True, **self.pbs_config)
        self.env['pbs'] = self.pbs

        FlowProcSim.total_sim_id = 0

        self.simultion_factory = FlowProcSim.factory(self.step_range,
                                                         config=self.simulation_config, clean=clean)

        self.mlmc_options['output_dir'] = self.output_dir
        self.mc = mlmc.mlmc.MLMC(self.n_levels, self.simultion_factory, self.step_range, self.mlmc_options)

        if clean:
            self.mc.create_new_execution()
            # assert Estimate.is_exe(self.env['flow123d'])
            assert Estimate.is_exe(self.env['gmsh'])
            # assert Estimate.is_exe(self.pbs_config['qsub'])
        else:
            self.mc.load_from_file()

    def collect(self):
        """
        Collect simulation samples
        :return: Number of running simulations
        """
        return self.mc.wait_for_simulations(sleep=self.sample_sleep, timeout=0.1)

    def set_moments(self, n_moments, log=False):
        """
        Create moments function instance
        :param n_moments: int, number of moments
        :param log: bool, If true then apply log transform
        :return: 
        """
        self.moments_fn = mlmc.moments.Legendre(n_moments, self.domain, safe_eval=True, log=log)
        return self.moments_fn

    def n_sample_estimate(self, target_variance=0.001):
        """
        Estimate number of level samples considering target variance
        :param target_variance: float, target variance of moments 
        :return: None
        """
        self.n_samples = self.mc.set_initial_n_samples([30, 3])
        self.mc.refill_samples()
        self.pbs.execute()
        self.mc.wait_for_simulations(sleep=self.sample_sleep, timeout=self.init_sample_timeout)

        self.domain = self.mc.estimate_domain()
        self.mc.set_target_variance(target_variance, self.moments_fn, 2.0)

    def generate_jobs(self, n_samples=None):
        """
        Generate level samples
        :param n_samples: None or list, number of samples for each level
        :return: None
        """
        if n_samples is not None:
            self.mc.set_initial_n_samples(n_samples)
        self.mc.refill_samples()
        self.pbs.execute()
        self.mc.wait_for_simulations(sleep=self.sample_sleep, timeout=self.sample_timeout)


########################################################

    def plot_diff_var(self):
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(10, 20))
        ax = fig.add_subplot(1, 1, 1)

        error_power = 2.0
        for m in range(1, self.n_moments):
            color = 'C' + str(m)
            Y = self.ref_diff_vars[:, m] / (self.steps ** error_power)

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

    def plot_n_sample_est_distributions(self, title, cost, total_std, n_samples, rel_moments):
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(30, 10))
        ax1 = fig.add_subplot(2, 2, 1)
        self.plot_error(cost, ax1, "cost err")

        ax2 = fig.add_subplot(2, 2, 2)
        self.plot_error(total_std, ax2, "total std err")

        ax3 = fig.add_subplot(2, 2, 3)
        self.plot_error(n_samples, ax3, "n. samples err")

        ax4 = fig.add_subplot(2, 2, 4)
        self.plot_error(rel_moments, ax4, "moments err")
        fig.suptitle(title)
        fig.savefig(title + ".pdf")
        plt.show()


    # @staticmethod
    # def align_array(arr):
    #     return "[" + ", ".join(["{:10.5f}".format(x) for x in arr]) + "]"

    # def compute_results(self, mlmc_l0, n_moments):
    #     self.domain = mlmc_l0.ref_domain
    #     self.est_domain = self.domain  # self.mc.estimate_domain()
    #
    #     moments_fn = self.set_moments(n_moments, log=True)
    #
    #     t_var = 1e-5
    #     self.ref_diff_vars, _ = self.mc.estimate_diff_vars(moments_fn)
    #     self.ref_moments, self.ref_vars = self.mc.estimate_moments(moments_fn)
    #
    #     self.ref_std = np.sqrt(self.ref_vars)
    #     self.ref_diff_vars_max = np.max(self.ref_diff_vars, axis=1)
    #     ref_n_samples = self.mc.set_target_variance(t_var, prescribe_vars=self.ref_diff_vars)
    #     self.ref_n_samples = np.max(ref_n_samples, axis=1)
    #     self.ref_cost = self.mc.estimate_cost(n_samples=self.ref_n_samples)
    #     self.ref_total_std = np.sqrt(np.sum(self.ref_diff_vars / self.ref_n_samples[:, None]) / n_moments)
    #     self.ref_total_std_x = np.sqrt(np.mean(self.ref_vars))
    #
    #     # print("\nLevels : ", self.mc.n_levels, "---------")
    #     # print("moments:  ", self.align_array(self.ref_moments))
    #     # print("std:      ", self.align_array(self.ref_std ))
    #     # print("err:      ", self.align_array(self.ref_moments - mlmc_0.ref_moments ))
    #     # print("domain:   ", self.est_domain)
    #     # print("cost:     ", self.ref_cost)
    #     # print("tot. std: ", self.ref_total_std, self.ref_total_std_x)
    #     # print("dif_vars: ", self.align_array(self.ref_diff_vars_max))
    #     # print("ns :      ", self.align_array(self.ref_n_samples))
    #     # print("")
    #     # print("SUBSAMPLES")
    #     #
    #     #
    #     # #a, b = self.domain
    #     # #distr_mean = self.distr_mean = moments_fn.inv_linear(ref_means[1])
    #     # #distr_var  = ((2*ref_means[2] + 1)/3 - ref_means[1]**2) / 4 * ((b-a)**2)
    #     # #self.distr_std  = np.sqrt(distr_var)
    #     #
    #     # n_loops = 10
    #     #
    #     # # use subsampling to:
    #     # # - simulate moments estimation without appriori knowledge about n_samples (but still on fixed domain)
    #     # # - estimate error in final number of samples
    #     # # - estimate error of estimated varinace of moments
    #     # # - estimate cost error (due to n_sample error)
    #     # # - estimate average error against reference momemnts
    #     # l_cost_err = []
    #     # l_total_std_err = []
    #     # l_n_samples_err = []
    #     # l_rel_mom_err = []
    #     # for i in range(n_loops):
    #     #     fractions = [0, 0.001, 0.01, 0.1]
    #     #     for fr in fractions:
    #     #         if fr == 0:
    #     #             nL, n0 = 3, 30
    #     #             if self.mc.n_levels > 1:
    #     #                 L = self.n_levels
    #     #                 factor = (nL / n0) ** (1 / (L - 1))
    #     #                 n_samples = (n0 * factor ** np.arange(L)).astype(int)
    #     #             else:
    #     #                 n_samples = [ n0]
    #     #         else:
    #     #             n_samples = np.maximum(n_samples, (fr*max_est_n_samples).astype(int))
    #     #         # n_samples = np.maximum(n_samples, 1)
    #     #
    #     #         print("n samples ", n_samples)
    #     #         exit()
    #     #         self.mc.subsample(n_samples)
    #     #         est_diff_vars, _ = self.mc.estimate_diff_vars(self.moments_fn)
    #     #         est_n_samples = self.mc.set_target_variance(t_var, prescribe_vars=est_diff_vars)
    #     #         max_est_n_samples = np.max(est_n_samples, axis=1)
    #     #
    #     #         #est_cost = self.mc.estimate_cost(n_samples=max_est_n_samples.astype(int))
    #     #         #est_total_var = np.sum(self.ref_diff_vars / max_est_n_samples[:, None])/self.n_moments
    #     #
    #     #         #n_samples_err = np.min( (max_est_n_samples - ref_n_samples) /ref_n_samples)
    #     #         ##total_std_err =  np.log2(est_total_var/ref_total_var)/2
    #     #         #total_std_err = (np.sqrt(est_total_var) - np.sqrt(ref_total_var)) / np.sqrt(ref_total_var)
    #     #         #cost_err = (est_cost - ref_cost)/ref_cost
    #     #         #print("Fr: {:6f} NSerr: {} Tstderr: {} cost_err: {}".format(fr, n_samples_err, total_std_err, cost_err))
    #     #
    #     #     est_diff_vars, _ = self.mc.estimate_diff_vars(self.moments_fn)
    #     #     est_moments, est_vars = self.mc.estimate_moments(self.moments_fn)
    #     #     #print("Vars:", est_vars)
    #     #     #print("em:", est_moments)
    #     #     #print("rm:", self.ref_moments)
    #     #
    #     #     n_samples_err = np.min( (n_samples - self.ref_n_samples) /self.ref_n_samples )
    #     #     est_total_std = np.sqrt(np.sum(est_diff_vars / n_samples[:, None])/n_moments)
    #     #     # est_total_std = np.sqrt(np.mean(est_vars))
    #     #     total_std_err =  np.log2(est_total_std/self.ref_total_std)
    #     #     est_cost = self.mc.estimate_cost(n_samples = n_samples)
    #     #     cost_err = (est_cost - self.ref_cost)/self.ref_cost
    #     #
    #     #     print("MM: ", (est_moments[1:] - self.ref_moments[1:]), "\n    ",  est_vars[1:])
    #     #
    #     #     relative_moments_err = np.linalg.norm((est_moments[1:] - self.ref_moments[1:]) / est_vars[1:])
    #     #     #print("est cost: {} ref cost: {}".format(est_cost, ref_cost))
    #     #     #print(n_samples)
    #     #     #print(np.maximum( n_samples, (max_est_n_samples).astype(int)))
    #     #     #print(ref_n_samples.astype(int))
    #     #     #print("\n")
    #     #     l_n_samples_err.append(n_samples_err)
    #     #     l_total_std_err.append(total_std_err)
    #     #     l_cost_err.append(cost_err)
    #     #     l_rel_mom_err.append(relative_moments_err)
    #     #
    #     # l_cost_err.sort()
    #     # l_total_std_err.sort()
    #     # l_n_samples_err.sort()
    #     # l_rel_mom_err.sort()
    #
    #
    #     est_moments, est_vars = self.mc.estimate_moments(self.moments_fn)
    #
    #     def describe(arr):
    #         print("arr ", arr)
    #         q1, q3 = np.percentile(arr, [25, 75])
    #         print("q1 ", q1)
    #         print("q2 ", q3)
    #         return "{:f8.2} < {:f8.2} | {:f8.2} | {:f8.2} < {:f8.2}".format(
    #             np.min(arr), q1, np.mean(arr), q3, np.max(arr))
    #
    #     # print("Cost err:       ", describe(l_cost_err))
    #     # print("Total std err:  ", describe(l_total_std_err))
    #     # print("N. samples err: ", describe(l_n_samples_err))
    #     # print("Rel. Mom. err:  ", describe(l_rel_mom_err))
    #
    #     # #print(l_rel_mom_err)
    #     # title = "N levels = {}".format(self.mc.n_levels)
    #     # self.plot_n_sample_est_distributions(title, l_cost_err, l_total_std_err, l_n_samples_err, l_rel_mom_err)
    #
    #
    #     moments_data = np.stack((est_moments, est_vars), axis=1)
    #     self.distr_obj = mlmc.distribution.Distribution(moments_fn, moments_data)
    #     self.distr_obj.domain = self.domain
    #     result = self.distr_obj.estimate_density_minimize(tol=1e-8)
    #     # print(result)






def all_collect(mlmc_list):
    running = 1
    while running > 0:
        running = 0
        for mc in mlmc_list:
            running += mc.collect()
        print("N running: ", running)






def calculate_var(mlmc_list):
    """
    Calculate density, moments (means, vars)
    :param mlmc_list: list of Estimate
    :return: None
    """
    level_moments_mean = []
    level_moments = []
    level_moments_var = []
    level_variance_diff = []
    var_mlmc = []
    n_moments = 5

    # get_var_diff(mlmc_list)
    # exit()

    level_var_diff = []
    # all_results(mlmc_list)

    all_variances = []
    all_means = []
    var_mlmc_pom = []
    n_levels = []
    first_level_samples = []

    # for proc_mlmc in mlmc_list:
    #     print("mlmc samples ", proc_mlmc.mc.n_samples)
    #     print("mlmc estimate cost ", proc_mlmc.mc.estimate_cost())


    # for proc_mlmc in mlmc_list[0:2]:
    #     proc_mlmc.domain = proc_mlmc.mc.estimate_domain()
    #     moments_fn = proc_mlmc.set_moments(n_moments)
    #
    #     proc_mlmc.mc.estimate_moments(moments_fn)
    #     proc_mlmc.mc.set_target_variance(1e-4, moments_fn)
    #
    #     #proc_mlmc.mc.refill_samples()
    #
    #     print("n samples ", proc_mlmc.mc.n_samples)
    #
    #     for level in proc_mlmc.mc.levels:
    #         print("level target n samples ", level.target_n_samples)
    #
    # exit()

    moments = None
    #all_results(mlmc_list)
    densities_x = []
    densities = []

    # Post process each mlmc method
    for proc_mlmc in mlmc_list:
        level_var_diff = []

        proc_mlmc.domain = mlmc_list[0].ref_domain
        n_levels.append(len(proc_mlmc.mc.levels))

        # print("proc mlmc domain ", proc_mlmc.domain)
        moments_fn = proc_mlmc.set_moments(n_moments, log=True)

        moments = proc_mlmc.mc.estimate_moments(moments_fn)
        moments_data = np.empty((len(moments[0]), 2))

        moments_data[:, 0] = moments[0][:]
        moments_data[:, 1] = moments[1][:]

        # Remove first moment
        moments = moments[0][1:], moments[1][1:]

        # Estimate denstity
        #d = estimate_density(moments_fn, moments_data)
        #densities.append(d[0])
        #densities_x.append(d[1])

        level_var_diff.append(test_mlmc.var_subsample(moments, proc_mlmc.mc, moments_fn, n_subsamples=1000))

        # Variances
        variances = np.sqrt(moments[1]) * 3
        var_mlmc_pom.append(moments[1])

        # all_variances.append(variances)
        # all_means.append(moments[0])

        # Exact moments from distribution
        # exact_moments = mlmc.distribution.compute_exact_moments(mc_test.moments_fn, d.pdf, 1e-10)[1:]

        means = np.mean(all_means, axis=0)
        vars = np.mean(all_variances, axis=0)
        # var_mlmc.append(np.mean(var_mlmc_pom, axis=0))

        # print(all(means[index] + vars[index] >= exact_mom >= means[index] - vars[index]
        #           for index, exact_mom in enumerate(exact_moments)))

        level_moments.append(moments)
        level_moments_mean.append(moments[0])
        level_moments_var.append(variances)

    if len(level_var_diff) > 0:
        # Average from more iteration
        level_variance_diff.append(np.mean(level_var_diff, axis=0))

    if len(level_var_diff) > 0:
        moments = []
        level_var_diff = np.array(level_var_diff)
        for index in range(len(level_var_diff[0])):
            moments.append(level_var_diff[:, index])

    # plot_densities(densities, densities_x, n_levels, mlmc_list)
    # all_results(mlmc_list)

    if len(level_moments) > 0 and len(level_var_diff) > 0:
        level_moments = np.array(level_moments)
        for index in range(len(level_moments[0])):
            test_mlmc.anova(level_moments[:, index])

    if len(level_variance_diff) > 0:
        test_mlmc.plot_diff_var(level_variance_diff, n_levels)

    # Plot moment values
    test_mlmc.plot_vars(level_moments_mean, level_moments_var, n_levels)


# def get_var_diff(mlmc_list):
#     """
#     V/V* plot
#     :param mlmc_list: list of ProcessMLMC
#     :return: None
#     """
#     level_moments_mean = []
#     level_moments = []
#     level_moments_var = []
#     level_variance_diff = []
#     var_mlmc = []
#     n_moments = 8
#     number = 10
#
#     level_var_diff = []
#     # all_results(mlmc_list)
#
#     all_variances = []
#     all_means = []
#     var_mlmc_pom = []
#     n_levels = []
#
#     first_level_samples = []
#
#     #     proc_mlmc.domain = proc_mlmc.mc.estimate_domain()
#     #     moments_fn = proc_mlmc.set_moments(10)
#     #     proc_mlmc.mc.estimate_moments(moments_fn)
#     #     proc_mlmc.mc.set_target_variance(1e-5, moments_fn)
#     #
#     #     for level in proc_mlmc.mc.levels:
#     #         print("level target n samples ", level.target_n_samples)
#     #
#     # exit()
#
#     moments = None
#     # all_results(mlmc_list)
#
#     densities_x = []
#     densities = []
#
#     # for proc_mlmc in mlmc_list:
#     #     first_level_samples.append(proc_mlmc.mc.levels[0].sample_values)
#     #
#     #     print("n samples ", proc_mlmc.mc.n_samples)
#     # exit()
#     # mlmc_list = mlmc_list[0:2]
#     for proc_mlmc in mlmc_list:
#         n_levels.append(len(proc_mlmc.mc.levels))
#         level_var_diff = []
#         all_variances = []
#         all_means = []
#         var_mlmc_pom = []
#         for n in range(number):
#             moments_all = []
#             proc_mlmc.mc.clear_subsamples()
#             for k in range(12):
#                 proc_mlmc.domain = proc_mlmc.mc.estimate_domain()
#                 moments_fn = proc_mlmc.set_moments(n_moments, log=True)
#                 # Estimate moments because of eliminating outliers
#                 proc_mlmc.mc.estimate_moments(moments_fn)
#
#                 # Use subsamples as a substitute for independent observations
#                 subsamples = [int(n_sub) for n_sub in proc_mlmc.mc.n_samples / 2]
#                 proc_mlmc.mc.subsample(subsamples)
#                 proc_mlmc.mc.refill_samples()
#                 proc_mlmc.pbs.execute()
#                 proc_mlmc.mc.wait_for_simulations()
#
#                 moments = proc_mlmc.mc.estimate_moments(moments_fn)
#                 moments_data = np.empty((len(moments[0]), 2))
#
#                 moments_data[:, 0] = moments[0][:]
#                 moments_data[:, 1] = moments[1][:]
#
#                 # Remove first moment
#                 moments_all.append(moments)
#
#                 proc_mlmc.mc.clear_subsamples()
#
#             moments = np.mean(moments_all, axis=0)
#             # Remove first moment
#             moments = moments[0][1:], moments[1][1:]
#
#             # Estimate denstity
#             d = estimate_density(moments_fn, moments_data)
#             densities.append(d[0])
#             densities_x.append(d[1])
#
#             level_var_diff.append(test_mlmc.var_subsample(moments, proc_mlmc.mc, moments_fn, n_subsamples=200))
#
#             # Variances
#             variances = np.sqrt(moments[1]) * 3
#             var_mlmc_pom.append(moments[1])
#
#             all_variances.append(variances)
#             all_means.append(moments[0])
#
#             # Exact moments from distribution
#             #exact_moments = mlmc.distribution.compute_exact_moments(mc_test.moments_fn, d.pdf, 1e-10)[1:]
#
#             means = np.mean(all_means, axis=0)
#             vars = np.mean(all_variances, axis=0)
#             # var_mlmc.append(np.mean(var_mlmc_pom, axis=0))
#
#             # print(all(means[index] + vars[index] >= exact_mom >= means[index] - vars[index]
#             #           for index, exact_mom in enumerate(exact_moments)))
#
#         if len(level_var_diff) > 0:
#             # Average from more iteration
#             level_variance_diff.append(np.mean(level_var_diff, axis=0))
#
#         if len(level_var_diff) > 0:
#             moments = []
#             level_var_diff = np.array(level_var_diff)
#             for index in range(len(level_var_diff[0])):
#                 moments.append(level_var_diff[:, index])
#
#         level_moments.append(moments)
#         level_moments_mean.append(moments[0])
#         level_moments_var.append(variances)
#
#     if len(level_moments) > 0:
#         level_moments = np.array(level_moments)
#         print("level_moments ", level_moments)
#
#         # normality_test(level_moments)
#
#         for index in range(len(level_moments[0])):
#             test_mlmc.anova(level_moments[:, index])
#
#         for l_mom in level_moments:
#             test_mlmc.anova(l_mom)
#
#     # print("level variance diff ", level_variance_diff)
#     # print("n levels ", n_levels)
#
#     if len(level_variance_diff) > 0:
#         test_mlmc.plot_diff_var(level_variance_diff, n_levels)
#
#     # Plot moment values
#     test_mlmc.plot_vars(level_moments_mean, level_moments_var, n_levels)
#     plot_densities(densities, densities_x, n_levels, mlmc_list)
#     all_results(mlmc_list)


def normality_test(level_moments):
    """
    Test normality of data
    :param level_moments: moments data
    :return: None
    """
    import pylab
    import scipy.stats as st
    alpha = 1e-3

    for index in range(len(level_moments[0])):
        for lm in level_moments[:, index]:
            k2, p = st.normaltest(lm)
            if p < alpha:  # null hypothesis: x comes from a normal distribution
                print("H0 can be rejected")
                exit()
            else:
                print("H0 cannot be rejected")
                # stats.probplot(, dist="norm", plot=pylab)
                # pylab.show()

    for l_mom in level_moments:
        for lm in l_mom:
            k2, p = st.normaltest(lm)
            if p < alpha:  # null hypothesis: x comes from a normal distribution
                print("H0 can be rejected")
                exit()
            else:
                print("H0 cannot be rejected")





# def show_results(mlmc_list):
#     for mlmc in mlmc_list:
#         print("n samples ", mlmc.mc.n_samples)
#
#         for level in mlmc.mc.levels:
#             print("level id ", level._logger.level_idx)
#             print("level samples ", level._sample_values)





def get_arguments(arguments):
    """
    Getting arguments from console
    :param arguments: list of arguments
    :return: namespace
    """
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('command', choices=['run', 'collect', 'process'], help='Run, collect or process')
    parser.add_argument('work_dir', help='Work directory')
    parser.add_argument("-r", "--regen-failed", default=False, action='store_true', help="Regenerate failed samples",)
    parser.add_argument("-k", "--keep-collected", default=False, action='store_true',
                        help="Keep sample dirs")

    args = parser.parse_args(arguments)
    return args



def analyze_pdf_approx(cl):
    # PDF approximation experiments
    np.random.seed(15)
    cl.set_common_domain(0)
    cl.reinit(n_moments = 10)
    il = 1
    #ns = cl[il].mlmc.estimate_n_samples_for_target_variance(0.01, cl.moments)
    #cl[il].mlmc.subsample(ns)
    #cl.construct_densities(tol = 1.0, reg_param = 1)
    cl[il].construct_density(tol = 0.0001, reg_param = 1)
    cl.plot_densities(i_sample_mlmc=0)


def analyze_regression_of_variance(cl):
    # Plot reference variances as scater and line plot of regression result.
    cl[9].ref_estimates_bootstrap(10)
    sample_vec = [5000, 5000, 1700, 600, 210, 72, 25, 9, 3]
    cl[9].mlmc.subsample(sample_vec)
    cl[9].plot_var_regression([1, 2, 4, 8, 16, 20])


def analyze_error_of_variance(cl):
    # Error of total variance estimator and contribution form individual levels.

    mc = cl[9]
    sample_vec = [5000, 5000, 1700, 600, 210, 72, 25, 9, 3]
    #n_samples = mc.mlmc.estimate_n_samples_for_target_variance(0.0001, cl.moments )
    #sample_vec = np.max(n_samples, axis=1).astype(int)

    mc.ref_estimates_bootstrap(300, sample_vector=sample_vec)
    mc.mlmc.update_moments(cl.moments)
    mc.mlmc.subsample()

    #print("std var. est / var. est.\n", np.sqrt(mc._bs_var_variance) / mc._bs_mean_variance)
    #vv_components = mc._bs_level_mean_variance[:, :] ** 2 / mc._bs_n_samples[:,None] ** 3
    #vv = np.sum(vv_components, axis=0) / mc.n_levels
    #print("err. var. composition\n", vv_components  - vv)
    # cl.plot_var_compare(9)
    mc.plot_bs_var_error_contributions()

def analyze_error_of_regression_variance(cl):
    # Demonstrate that variance of varaince estimates is proportional to
    # sample_vec = [5000, 5000, 1700, 600, 210, 72, 25, 9, 3]
    sample_vec = [5000, 5000, 1700, 600, 210, 72, 25, 9, 3]
    # sample_vec = 9*[80]
    mc = cl[9]
    mc.ref_estimates_bootstrap(300, sample_vector=sample_vec, regression=True)
    # print(mc._bs_level_mean_variance)
    mc.mlmc.update_moments(cl.moments)
    mc.mlmc.subsample()
    # cl.plot_var_compare(9)
    mc.plot_bs_var_error_contributions()


def analyze_error_of_level_variances(cl):
    # Demonstrate that variance of varaince estimates is proportional to

    mc = cl[9]
    # sample_vec = 9*[8]
    sample_vec = [5000, 5000, 1700, 600, 210, 72, 25, 9, 3]
    #n_samples = mc.mlmc.estimate_n_samples_for_target_variance(0.0001, cl.moments )
    #sample_vec = np.max(n_samples, axis=1).astype(int)
    #print(sample_vec)


    mc.ref_estimates_bootstrap(300, sample_vector=sample_vec)
    mc.mlmc.update_moments(cl.moments)
    mc.mlmc.subsample()

    #print("std var. est / var. est.\n", np.sqrt(mc._bs_var_variance) / mc._bs_mean_variance)
    #vv_components = mc._bs_level_mean_variance[:, :] ** 2 / mc._bs_n_samples[:,None] ** 3
    #vv = np.sum(vv_components, axis=0) / mc.n_levels
    #print("err. var. composition\n", vv_components  - vv)
    # cl.plot_var_compare(9)
    mc.plot_bs_level_variances_error()


def analyze_error_of_regression_level_variances(cl):
    # Demonstrate that variance of varaince estimates is proportional to

    mc = cl[9]
    # sample_vec = 9*[8]
    sample_vec = [5000, 5000, 1700, 600, 210, 72, 25, 9, 3]
    #n_samples = mc.mlmc.estimate_n_samples_for_target_variance(0.0001, cl.moments )
    #sample_vec = np.max(n_samples, axis=1).astype(int)
    #print(sample_vec)


    mc.ref_estimates_bootstrap(10, sample_vector=sample_vec, regression=True)
    mc.mlmc.update_moments(cl.moments)
    mc.mlmc.subsample()

    #print("std var. est / var. est.\n", np.sqrt(mc._bs_var_variance) / mc._bs_mean_variance)
    #vv_components = mc._bs_level_mean_variance[:, :] ** 2 / mc._bs_n_samples[:,None] ** 3
    #vv = np.sum(vv_components, axis=0) / mc.n_levels
    #print("err. var. composition\n", vv_components  - vv)
    # cl.plot_var_compare(9)
    mc.plot_bs_level_variances_error()


def analyze_error_of_log_variance(cl):
    # Demonstrate that variance of varaince estimates is proportional to
    # sample_vec = [5000, 5000, 1700, 600, 210, 72, 25, 9, 3]
    sample_vec = [5000, 5000, 1700, 600, 210, 72, 25, 9, 3]
    #sample_vec = 9*[80]
    mc = cl[9]
    mc.ref_estimates_bootstrap(300, sample_vector=sample_vec, log=True)
    mc.mlmc.update_moments(cl.moments)
    mc.mlmc.subsample()
    # cl.plot_var_compare(9)
    mc.plot_bs_var_log_var()



def process_analysis(cl):
    """
    Main analysis function. Particular types of analysis called from here.
    :param cl: Instance of Compare levels.
    :return:
    """
    cl.collected_report()

    analyze_pdf_approx(cl)
    #analyze_regression_of_variance(cl)
    #analyze_error_of_variance(cl)
    #analyze_error_of_regression_variance(cl)
    #analyze_error_of_level_variances(cl)
    #analyze_error_of_regression_level_variances(cl)
    #analyze_error_of_log_variance(cl)




# Demonstrate that variance of varaince estimates based on regression
# TODO:
# sample_vec = [5000, 5000, 1700, 600, 210, 72, 25, 9, 3]
# sample_vec = [5000, 5000, 1700, 600, 210, 72, 25, 9, 3]
# sample_vec = 9*[80]
# mc = cl[9]
# mc.ref_estimates_bootstrap(300, sample_vector=sample_vec)
# mc.mlmc.update_moments(cl.moments)
# mc.mlmc.subsample()
# cl.plot_var_compare(9)
# mc.plot_bootstrap_var_var()


def main():
    args = get_arguments(sys.argv[1:])

    command = args.command
    work_dir = args.work_dir

    options = {'keep_collected': args.keep_collected,
               'regen_failed': args.regen_failed}

    if command == 'run':
        os.makedirs(work_dir, mode=0o775, exist_ok=True)

        mlmc_list = []
        for nl in [1]:  # , 2, 3, 4,5, 7, 9]:
            mlmc = UglyMLMC(work_dir, options)
            mlmc.setup(nl)
            mlmc.initialize(clean=False)
            ns = {
                1: [7087],
                2: [14209, 332],
                3: [18979, 487, 2],
                4: [13640, 610, 2, 2],
                5: [12403, 679, 10, 2, 2],
                7: [12102, 807, 11, 2, 2, 2, 2],
                9: [11449, 806, 72, 8, 2, 2, 2, 2, 2]
            }

            n_samples = 2 * np.array(ns[nl])
            # mlmc.generate_jobs(n_samples=n_samples)
            # mlmc.generate_jobs(n_samples=[10000, 100])
            # mlmc.mc.levels[0].target_n_samples = 1
            mlmc.generate_jobs(n_samples=[8])#, 1, 1])
            mlmc_list.append(mlmc)

            # for nl in [3,4]:
            # mlmc = ProcessMLMC(work_dir)
            # mlmc.load(nl)
            # mlmc_list.append(mlmc)

        all_collect(mlmc_list)

    elif command == 'collect':
        assert os.path.isdir(work_dir)
        mlmc_list = []

        for nl in [1, 2, 3, 4, 5, 7]:  # , 3, 4, 5, 7, 9]:#, 5,7]:
            mlmc = UglyMLMC(work_dir, options)
            mlmc.setup(nl)
            mlmc.initialize(clean=False)
            mlmc_list.append(mlmc)
        all_collect(mlmc_list)
        calculate_var(mlmc_list)
        #show_results(mlmc_list)

    elif command == 'process':
        assert os.path.isdir(work_dir)
        mlmc_list = []
        #for nl in [ 1,2,3,4,5, 7,9]:
        for nl in [1]:
            prmc = UglyMLMC(work_dir, options)
            prmc.setup(nl)
            prmc.initialize(clean=False)
            mlmc_list.append(prmc)

        cl = CompareLevels([pm.mc for pm in mlmc_list],
                           output_dir=src_path,
                           quantity_name="Q [m/s]",
                           moment_class=moments.Legendre,
                           log_scale = False,
                           n_moments=21,)

        process_analysis(cl)

        # statprof.start()
        # try:
        #     cl.ref_estimates_bootstrap(10)
        #     cl.plot_var_var(9)
        # finally:
        #     statprof.stop()
        #     statprof.display()

main()
