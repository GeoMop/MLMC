import os
import sys
import shutil
import numpy as np
from mlmc.moments import Legendre


class ProcessBase:
    """
    Parent class for particular simulation processes
    """
    def __init__(self):
        args = ProcessBase.get_arguments(sys.argv[1:])

        self.step_range = (1, 0.01)

        self.work_dir = args.work_dir
        self.append = False
        self.clean = args.clean
        self.debug = args.debug

        if args.command == 'run':
            self.run()
        else:
            self.append = True
            self.clean = False
            self.run(renew=True) if args.command == 'renew' else self.run()

    @staticmethod
    def get_arguments(arguments):
        """
        Getting arguments from console
        :param arguments: list of arguments
        :return: namespace
        """
        import argparse
        parser = argparse.ArgumentParser()

        parser.add_argument('command', choices=['run', 'collect', 'renew', 'process'],
                            help='run - create new execution,'
                                 'collect - keep collected, append existing HDF file'
                                 'renew - renew failed samples, run new samples with failed sample ids (which determine random seed)')
        parser.add_argument('work_dir', help='Work directory')
        parser.add_argument("-c", "--clean", default=False, action='store_true',
                            help="Clean before run, used only with 'run' command")
        parser.add_argument("-d", "--debug", default=False, action='store_true',
                            help="Keep sample directories")

        args = parser.parse_args(arguments)

        return args

    def run(self, renew=True):
        """
        Run mlmc
        :return: None
        """
        os.makedirs(self.work_dir, mode=0o775, exist_ok=True)

        mlmc_list = []
        for nl in [1]:  # , 2, 3, 4,5, 7, 9]:
            mlmc = self.setup_config(nl, clean=self.clean)
            self.generate_jobs(mlmc, n_samples=[8], sample_sleep=self.sample_sleep, sample_timeout=self.sample_timeout)
            mlmc_list.append(mlmc)

        self.all_collect(mlmc_list)

    # def collect(self):
    #     """
    #     Collect samples
    #     :return: None
    #     """
    #     assert os.path.isdir(self.work_dir)
    #     mlmc_list = []
    #
    #     for nl in [1, 2, 3, 4, 5, 7]:  # , 3, 4, 5, 7, 9]:#, 5,7]:
    #         mlmc = self.setup_config(nl, clean=False)
    #         mlmc_list.append(mlmc)
    #     self.all_collect(mlmc_list)
    #     self.calculate_var(mlmc_list)
    #     # show_results(mlmc_list)

    # def process(self):
    #     """
    #     Use collected data
    #     :return: None
    #     """
    #     assert os.path.isdir(self.work_dir)
    #     mlmc_est_list = []
    #     # for nl in [ 1,3,5,7,9]:
    #     for nl in [3]:  # high resolution fields
    #         mlmc = self.setup_config(nl, clean=False)
    #         # Use wrapper object for working with collected data
    #         mlmc_est_list.append(mlmc)
    #
    #     cl = CompareLevels(mlmc_est_list,
    #                        output_dir=src_path,
    #                        quantity_name="Q [m/s]",
    #                        moment_class=Legendre,
    #                        log_scale=False,
    #                        n_moments=21, )
    #
    #     self.process_analysis(cl)

    def set_environment_variables(self):
        """
        Set pbs config, flow123d, gmsh
        :return: None
        """
        root_dir = os.path.abspath(self.work_dir)
        while root_dir != '/':
            root_dir, tail = os.path.split(root_dir)

        self.pbs_config = dict(
            job_weight=250000,  # max number of elements per job
            n_cores=1,
            n_nodes=1,
            select_flags=['cgroups=cpuacct'],
            mem='4gb',
            queue='charon',
            home_dir='/storage/liberec3-tul/home/martin_spetlik/')

        if tail == 'storage':
            # Metacentrum
            self.sample_sleep = 30
            self.init_sample_timeout = 600
            self.sample_timeout = 0
            self.pbs_config['qsub'] = '/usr/bin/qsub'
            self.flow123d = 'flow123d'  # "/storage/praha1/home/jan_brezina/local/flow123d_2.2.0/flow123d"
            self.gmsh = "/storage/liberec3-tul/home/martin_spetlik/astra/gmsh/bin/gmsh"
        else:
            # Local
            self.sample_sleep = 1
            self.init_sample_timeout = 60
            self.sample_timeout = 60
            self.pbs_config['qsub'] = None
            self.flow123d = "/home/jb/workspace/flow123d/bin/fterm flow123d dbg"
            self.gmsh = "/home/jb/local/gmsh-3.0.5-git-Linux/bin/gmsh"

    def setup_config(self, n_levels, clean):
        """
        Set simulation configuration depends on particular task
        :param n_levels: Number of levels
        :param clean: bool, if False use existing files
        :return: mlmc.MLMC
        """
        raise NotImplementedError("Simulation configuration is not set")

    def rm_files(self, output_dir):
        """
        Rm files and dirs
        :param output_dir: Output directory path
        :return:
        """
        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir, ignore_errors=True)
        os.makedirs(output_dir, mode=0o775, exist_ok=True)

    def create_pbs_object(self, output_dir, clean):
        """
        Initialize object for PBS execution
        :param output_dir: Output directory
        :param clean: bool, if True remove existing files
        :return: None
        """
        pbs_work_dir = os.path.join(output_dir, "scripts")
        num_jobs = 0
        if os.path.isdir(pbs_work_dir):
            num_jobs = len([_ for _ in os.listdir(pbs_work_dir)])

        self.pbs_obj = pbs.Pbs(pbs_work_dir,
                               job_count=num_jobs,
                               qsub=self.pbs_config['qsub'],
                               clean=clean)
        self.pbs_obj.pbs_common_setting(flow_3=True, **self.pbs_config)

    def generate_jobs(self, mlmc, n_samples=None):
        """
        Generate level samples
        :param n_samples: None or list, number of samples for each level
        :return: None
        """
        if n_samples is not None:
            mlmc.set_initial_n_samples(n_samples)
        mlmc.refill_samples()

        if self.pbs_obj is not None:
            self.pbs_obj.execute()
        mlmc.wait_for_simulations(sleep=self.sample_sleep, timeout=self.sample_timeout)

    def set_moments(self, n_moments, log=False):
        """
        Create moments function instance
        :param n_moments: int, number of moments
        :param log: bool, If true then apply log transform
        :return:
        """
        self.moments_fn = Legendre(n_moments, self.domain, safe_eval=True, log=log)
        return self.moments_fn

    def n_sample_estimate(self, mlmc, target_variance=0.001):
        """
        Estimate number of level samples considering target variance
        :param mlmc: MLMC object
        :param target_variance: float, target variance of moments
        :return: None
        """
        mlmc.set_initial_n_samples()
        mlmc.refill_samples()
        self.pbs_obj.execute()
        mlmc.wait_for_simulations(sleep=self.sample_sleep, timeout=self.init_sample_timeout)

        self.domain = mlmc.estimator.estimate_domain()
        self.set_moments(self.n_moments, log=True)

        mlmc.target_var_adding_samples(target_variance, self.moments_fn, pbs=self.pbs_obj)

    def all_collect(self, sampler_list):
        """
        Collect samples
        :param mlmc_list: List of mlmc.MLMC objects
        :return: None
        """
        running = 1
        while running > 0:
            running = 0
            for sampler in sampler_list:
                running += sampler.ask_sampling_pool_for_samples(sleep=self.sample_sleep, timeout=0.1)
            print("N running: ", running)

    def process_analysis(self, cl):
        """
        Main analysis function. Particular types of analysis called from here.
        :param cl: Instance of CompareLevels - list of Estimate objects
        :return:
        """
        cl.collected_report()
        mlmc_level = 1

        #self.analyze_pdf_approx(cl)
        # analyze_regression_of_variance(cl, mlmc_level)
        self.analyze_error_of_variance(cl, mlmc_level)
        # analyze_error_of_regression_variance(cl, mlmc_level)
        # analyze_error_of_level_variances(cl, mlmc_level)
        # analyze_error_of_regression_level_variances(cl, mlmc_level)
        # analyze_error_of_log_variance(cl, mlmc_level)

    def analyze_pdf_approx(self, cl):
        """
        Plot densities
        :param cl: mlmc.estimate.CompareLevels
        :return: None
        """
        # PDF approximation experiments
        np.random.seed(15)
        cl.set_common_domain(0)
        print("cl domain:", cl.domain)

        cl.reinit(n_moments=35)
        il = 1
        # ns = cl[il].mlmc.estimate_n_samples_for_target_variance(0.01, cl.moments)
        # cl[il].mlmc.subsample(ns)
        cl.construct_densities(tol=0.01, reg_param=1)
        # cl[il].construct_density(tol = 0.01, reg_param = 1)
        cl.plot_densities(i_sample_mlmc=0)

    def analyze_regression_of_variance(self, cl, mlmc_level):
        """
        Analyze regression of variance
        :param cl: mlmc.estimate.CompareLevels instance
        :param mlmc_level: selected MC method
        :return: None
        """
        mc = cl[mlmc_level]
        # Plot reference variances as scater and line plot of regression result.
        mc.ref_estimates_bootstrap(10)
        sample_vec = [5000, 5000, 1700, 600, 210, 72, 25, 9, 3]
        mc.mlmc.subsample(sample_vec[mc.n_levels])
        mc.plot_var_regression([1, 2, 4, 8, 16, 20])

    def analyze_error_of_variance(self, cl, mlmc_level):
        """
        Analyze error of variance for particular mlmc method or for all collected methods
        :param cl: mlmc.estimate.CompareLevels instance
        :param mlmc_level: selected MC method
        :return: None
        """
        np.random.seed(20)
        cl.plot_variances()
        cl.plot_level_variances()

        # # Error of total variance estimator and contribution form individual levels.
        # sample_vec = [5000, 5000, 1700, 600, 210, 72, 25, 9, 3]
        # mc = cl[mlmc_level]
        # mc.ref_estimates_bootstrap(300, sample_vector=sample_vec[:mc.n_levels])
        # mc.mlmc.update_moments(cl.moments)
        # mc.mlmc.subsample()

        # print("std var. est / var. est.\n", np.sqrt(mc._bs_var_variance) / mc._bs_mean_variance)
        # vv_components = mc._bs_level_mean_variance[:, :] ** 2 / mc._bs_n_samples[:,None] ** 3
        # vv = np.sum(vv_components, axis=0) / mc.n_levels
        # print("err. var. composition\n", vv_components  - vv)
        # cl.plot_var_compare(9)
        mc.plot_bs_var_error_contributions()

    def analyze_error_of_regression_variance(self, cl, mlmc_level):
        """
        Analyze error of regression variance
        :param cl: CompareLevels
        :param mlmc_level: selected MC method
        :return:
        """
        # Demonstrate that variance of varaince estimates is proportional to
        sample_vec = [5000, 5000, 1700, 600, 210, 72, 25, 9, 3]
        mc = cl[mlmc_level]

        # sample_vec = 9*[80]
        mc.ref_estimates_bootstrap(300, sample_vector=sample_vec[mc.n_levels], regression=True)
        # print(mc._bs_level_mean_variance)
        mc.mlmc.update_moments(cl.moments)
        mc.mlmc.subsample()
        # cl.plot_var_compare(9)
        mc.plot_bs_var_error_contributions()

    def analyze_error_of_level_variances(self, cl, mlmc_level):
        """
        Analyze error of level variances
        :param cl: mlmc.estimate.CompareLevels instance
        :param mlmc_level: selected MC method
        :return: None
        """
        # Demonstrate that variance of varaince estimates is proportional to

        mc = cl[mlmc_level]
        # sample_vec = 9*[8]
        sample_vec = [5000, 5000, 1700, 600, 210, 72, 25, 9, 3]
        # n_samples = mc.mlmc.estimate_n_samples_for_target_variance(0.0001, cl.moments )
        # sample_vec = np.max(n_samples, axis=1).astype(int)
        # print(sample_vec)

        mc.ref_estimates_bootstrap(300, sample_vector=sample_vec[:mc.n_levels])
        mc.mlmc.update_moments(cl.moments)
        mc.mlmc.subsample()

        # print("std var. est / var. est.\n", np.sqrt(mc._bs_var_variance) / mc._bs_mean_variance)
        # vv_components = mc._bs_level_mean_variance[:, :] ** 2 / mc._bs_n_samples[:,None] ** 3
        # vv = np.sum(vv_components, axis=0) / mc.n_levels
        # print("err. var. composition\n", vv_components  - vv)
        # cl.plot_var_compare(9)
        mc.plot_bs_level_variances_error()

    def analyze_error_of_regression_level_variances(self, cl, mlmc_level):
        """
        Analyze error of level variances
        :param cl: mlmc.estimate.CompareLevels instance
        :param mlmc_level: selected MC method
        :return: None
        """
        # Demonstrate that variance of varaince estimates is proportional to
        mc = cl[mlmc_level]
        # sample_vec = 9*[8]
        sample_vec = [5000, 5000, 1700, 600, 210, 72, 25, 9, 3]
        # n_samples = mc.mlmc.estimate_n_samples_for_target_variance(0.0001, cl.moments )
        # sample_vec = np.max(n_samples, axis=1).astype(int)
        # print(sample_vec)

        mc.ref_estimates_bootstrap(10, sample_vector=sample_vec[:mc.n_levels], regression=True)
        mc.mlmc.update_moments(cl.moments)
        mc.mlmc.subsample()

        # print("std var. est / var. est.\n", np.sqrt(mc._bs_var_variance) / mc._bs_mean_variance)
        # vv_components = mc._bs_level_mean_variance[:, :] ** 2 / mc._bs_n_samples[:,None] ** 3
        # vv = np.sum(vv_components, axis=0) / mc.n_levels
        # print("err. var. composition\n", vv_components  - vv)
        # cl.plot_var_compare(9)
        mc.plot_bs_level_variances_error()

    def analyze_error_of_log_variance(self, cl, mlmc_level):
        """
        Analyze error of level variances
        :param cl: mlmc.estimate.CompareLevels instance
        :param mlmc_level: selected MC method
        :return: None
        """
        # Demonstrate that variance of varaince estimates is proportional to
        # sample_vec = [5000, 5000, 1700, 600, 210, 72, 25, 9, 3]
        sample_vec = [5000, 5000, 1700, 600, 210, 72, 25, 9, 3]
        # sample_vec = 9*[80]
        mc = cl[mlmc_level]
        mc.ref_estimates_bootstrap(300, sample_vector=sample_vec[:mc.n_levels], log=True)
        mc.mlmc.update_moments(cl.moments)
        mc.mlmc.subsample()
        # cl.plot_var_compare(9)
        mc.plot_bs_var_log_var()
