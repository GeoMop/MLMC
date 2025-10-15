import os
import sys
import shutil
import numpy as np
from mlmc.moments import Legendre


class ProcessBase:
    """
    Parent class for particular simulation processes.
    Subclasses should implement `setup_config`.
    """
    def __init__(self):
        """
        Parse CLI arguments and run the requested command.

        The constructor reads command-line arguments (sys.argv[1:]) using get_arguments,
        sets default attributes and then either runs or re-runs the workflow based on
        provided arguments.

        :return: None
        """
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
        Parse command-line arguments.

        :param arguments: list of arguments (typically sys.argv[1:])
        :return: argparse.Namespace with parsed arguments:
                 - command: one of ['run', 'collect', 'renew', 'process']
                 - work_dir: str path
                 - clean: bool
                 - debug: bool
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
        High-level entry point to run the MLMC workflow.

        Creates the working directory, sets up MLMC configurations for a set of level
        counts (currently hard-coded to [1]) and schedules/generates jobs. After job creation
        it triggers collection of results via all_collect.

        :param renew: bool, if True indicates renewing failed samples (passed down to setup_config in some subclasses)
        :return: None
        """
        os.makedirs(self.work_dir, mode=0o775, exist_ok=True)

        mlmc_list = []
        for nl in [1]:  # , 2, 3, 4,5, 7, 9]:
            mlmc = self.setup_config(nl, clean=self.clean)
            self.generate_jobs(mlmc, n_samples=[8], sample_sleep=self.sample_sleep, sample_timeout=self.sample_timeout)
            mlmc_list.append(mlmc)

        self.all_collect(mlmc_list)

    def set_environment_variables(self):
        """
        Determine environment-dependent configuration values (PBS config, executables, timeouts).

        The method inspects the work_dir path to decide whether it runs on a cluster or locally and
        sets attributes used later (pbs_config, sample_sleep, init_sample_timeout, sample_timeout, flow123d, gmsh).

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
            # Cluster settings
            self.sample_sleep = 30
            self.init_sample_timeout = 600
            self.sample_timeout = 0
            self.pbs_config['qsub'] = '/usr/bin/qsub'
            self.flow123d = 'flow123d'
            self.gmsh = "/storage/liberec3-tul/home/martin_spetlik/astra/gmsh/bin/gmsh"
        else:
            # Local settings
            self.sample_sleep = 1
            self.init_sample_timeout = 60
            self.sample_timeout = 60
            self.pbs_config['qsub'] = None
            self.flow123d = "/home/jb/workspace/flow123d/bin/fterm flow123d dbg"
            self.gmsh = "/home/jb/local/gmsh-3.0.5-git-Linux/bin/gmsh"

    def setup_config(self, n_levels, clean):
        """
        Set simulation configuration depending on particular task.

        Subclasses **must** override this method and return a configured mlmc.MLMC object.

        :param n_levels: int, number of MLMC levels
        :param clean: bool, whether to clean/create new files or use existing ones
        :return: mlmc.MLMC instance (implementation dependent)
        :raises NotImplementedError: always in base class
        """
        raise NotImplementedError("Simulation configuration is not set")

    def rm_files(self, output_dir):
        """
        Remove (recursively) output_dir and create an empty directory in its place.

        :param output_dir: str path to remove and recreate
        :return: None
        """
        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir, ignore_errors=True)
        os.makedirs(output_dir, mode=0o775, exist_ok=True)

    def create_pbs_object(self, output_dir, clean):
        """
        Initialize PBS helper object for submitting/executing jobs.

        This creates self.pbs_obj and configures it with common PBS settings.

        :param output_dir: str, directory where PBS scripts and job state will be created
        :param clean: bool, if True remove existing scripts before creating new ones
        :return: None
        """
        pbs_work_dir = os.path.join(output_dir, "scripts")
        num_jobs = 0
        if os.path.isdir(pbs_work_dir):
            num_jobs = len([_ for _ in os.listdir(pbs_work_dir)])

        # pbs module is expected to be imported where available
        self.pbs_obj = pbs.Pbs(pbs_work_dir,
                               job_count=num_jobs,
                               qsub=self.pbs_config['qsub'],
                               clean=clean)
        self.pbs_obj.pbs_common_setting(flow_3=True, **self.pbs_config)

    def generate_jobs(self, mlmc, n_samples=None):
        """
        Prepare and kick off sampling jobs for the provided MLMC object.

        The method optionally sets the initial n_samples (if provided), refills the sampler
        queues and triggers the PBS object execution. It then waits for simulations to finish.

        :param mlmc: mlmc.MLMC instance
        :param n_samples: None or list specifying number of samples to request for each level
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
        Create and store a moments function instance (Legendre polynomial family).

        :param n_moments: int, number of moments
        :param log: bool, whether to apply log-transform to quantity prior to moment evaluation
        :return: Legendre moments instance
        """
        self.moments_fn = Legendre(n_moments, self.domain, safe_eval=True, log=log)
        return self.moments_fn

    def n_sample_estimate(self, mlmc, target_variance=0.001):
        """
        Heuristic routine to estimate a good number of initial samples for MLMC using target variance.

        It triggers an initial sampling run, estimates the domain, constructs moments, and requests
        additional samples using mlmc.target_var_adding_samples.

        :param mlmc: mlmc.MLMC instance
        :param target_variance: float target variance for moment estimates
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
        Poll samplers to collect running samples until none are left.

        Repeatedly asks each sampler for the number of running jobs and keeps polling until all complete.

        :param sampler_list: list of sampler-like objects providing ask_sampling_pool_for_samples(sleep, timeout)
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
        Top-level analysis entry point. Calls specific analysis routines (many commented out).

        :param cl: CompareLevels instance (or equivalent) holding estimation/collected data
        :return: None
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
        Perform PDF approximation experiments and plotting.

        :param cl: CompareLevels instance
        :return: None
        """
        np.random.seed(15)
        cl.set_common_domain(0)
        print("cl domain:", cl.domain)

        cl.reinit(n_moments=35)
        il = 1
        cl.construct_densities(tol=0.01, reg_param=1)
        cl.plot_densities(i_sample_mlmc=0)

    def analyze_regression_of_variance(self, cl, mlmc_level):
        """
        Analyze regression of variance for a selected level.

        :param cl: CompareLevels instance
        :param mlmc_level: int index of method/level to analyze
        :return: None
        """
        mc = cl[mlmc_level]
        mc.ref_estimates_bootstrap(10)
        sample_vec = [5000, 5000, 1700, 600, 210, 72, 25, 9, 3]
        mc.mlmc.subsample(sample_vec[mc.n_levels])
        mc.plot_var_regression([1, 2, 4, 8, 16, 20])

    def analyze_error_of_variance(self, cl, mlmc_level):
        """
        Analyze error of variance estimators and plot related diagnostics.

        :param cl: CompareLevels instance
        :param mlmc_level: int index of method/level to analyze
        :return: None
        """
        np.random.seed(20)
        cl.plot_variances()
        cl.plot_level_variances()
        mc = cl[mlmc_level]
        mc.plot_bs_var_error_contributions()

    def analyze_error_of_regression_variance(self, cl, mlmc_level):
        """
        Bootstrap-based analysis of regression variance errors.

        :param cl: CompareLevels instance
        :param mlmc_level: int index of method/level to analyze
        :return: None
        """
        sample_vec = [5000, 5000, 1700, 600, 210, 72, 25, 9, 3]
        mc = cl[mlmc_level]
        mc.ref_estimates_bootstrap(300, sample_vector=sample_vec[mc.n_levels], regression=True)
        mc.mlmc.update_moments(cl.moments)
        mc.mlmc.subsample()
        mc.plot_bs_var_error_contributions()

    def analyze_error_of_level_variances(self, cl, mlmc_level):
        """
        Analyze errors in per-level variance estimates and plot results.

        :param cl: CompareLevels instance
        :param mlmc_level: int index of method/level to analyze
        :return: None
        """
        mc = cl[mlmc_level]
        sample_vec = [5000, 5000, 1700, 600, 210, 72, 25, 9, 3]
        mc.ref_estimates_bootstrap(300, sample_vector=sample_vec[:mc.n_levels])
        mc.mlmc.update_moments(cl.moments)
        mc.mlmc.subsample()
        mc.plot_bs_level_variances_error()

    def analyze_error_of_regression_level_variances(self, cl, mlmc_level):
        """
        Analyze combined regression and level variance errors with bootstrap.

        :param cl: CompareLevels instance
        :param mlmc_level: int index of method/level to analyze
        :return: None
        """
        mc = cl[mlmc_level]
        sample_vec = [5000, 5000, 1700, 600, 210, 72, 25, 9, 3]
        mc.ref_estimates_bootstrap(10, sample_vector=sample_vec[:mc.n_levels], regression=True)
        mc.mlmc.update_moments(cl.moments)
        mc.mlmc.subsample()
        mc.plot_bs_level_variances_error()

    def analyze_error_of_log_variance(self, cl, mlmc_level):
        """
        Analyze bootstrap error of log-variance estimates.

        :param cl: CompareLevels instance
        :param mlmc_level: int index of method/level to analyze
        :return: None
        """
        sample_vec = [5000, 5000, 1700, 600, 210, 72, 25, 9, 3]
        mc = cl[mlmc_level]
        mc.ref_estimates_bootstrap(300, sample_vector=sample_vec[:mc.n_levels], log=True)
        mc.mlmc.update_moments(cl.moments)
        mc.mlmc.subsample()
        mc.plot_bs_var_log_var()
