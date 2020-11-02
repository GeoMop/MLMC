import os
import sys
import numpy as np
import gstools
import time
from mlmc.sampler import Sampler
from mlmc.sample_storage_hdf import SampleStorageHDF
from mlmc.sampling_pool import OneProcessPool, ProcessPool, ThreadPool
from mlmc.sampling_pool_pbs import SamplingPoolPBS
from mlmc.tool.flow_mc import FlowSim
from mlmc.moments import Legendre, Monomial
from mlmc.tool.process_base import ProcessBase
from mlmc.random import correlated_field as cf
#from mlmc.quantity_estimate import QuantityEstimate
from mlmc.quantity import make_root_quantity, estimate_mean, moment, moments, covariance
import mlmc.estimator as new_estimator
import mlmc.tool.simple_distribution


class ProcessSimple:

    def __init__(self):
        args = ProcessBase.get_arguments(sys.argv[1:])

        self.work_dir = os.path.abspath(args.work_dir)
        self.append = False
        # Add samples to existing ones
        self.clean = args.clean
        # Remove HDF5 file, start from scratch
        self.debug = args.debug
        # 'Debug' mode is on - keep sample directories
        self.use_pbs = True
        # Use PBS sampling pool
        self.n_levels = 1
        self.n_moments = 5
        # Number of MLMC levels

        step_range = [1, 0.005]
        # step   - elements
        # 0.1    - 262
        # 0.08   - 478
        # 0.06   - 816
        # 0.055  - 996
        # 0.005  - 106056
        # 0.004  - 165404
        # 0.0035 - 217208

        # step_range [simulation step at the coarsest level, simulation step at the finest level]

        # Determine level parameters at each level (In this case, simulation step at each level) are set automatically
        self.level_parameters = ProcessSimple.determine_level_parameters(self.n_levels, step_range)

        # Determine number of samples at each level
        self.n_samples = ProcessSimple.determine_n_samples(self.n_levels)

        if args.command == 'run':
            self.run()
        elif args.command == "process":
            self.process()
        else:
            self.append = True  # Use 'collect' command (see base_process.Process) to add samples
            self.clean = False
            self.run(renew=True) if args.command == 'renew' else self.run()

    def process(self):
        sample_storage = SampleStorageHDF(file_path=os.path.join(self.work_dir, "mlmc_{}.hdf5".format(self.n_levels)))
        sample_storage.chunk_size = 1024
        result_format = sample_storage.load_result_format()
        root_quantity = make_root_quantity(sample_storage, result_format)


        conductivity = root_quantity['conductivity']
        time = conductivity[1]  # times: [1]
        location = time['0']  # locations: ['0']
        q_value = location[0, 0]

        # @TODO: How to estimate true_domain?
        quantile = 0.001
        true_domain = mlmc.estimator.Estimate.estimate_domain(q_value, sample_storage, quantile=quantile)
        moments_fn = Legendre(self.n_moments, true_domain)


        estimator = mlmc.estimator.Estimate(quantity=q_value, sample_storage=sample_storage, moments_fn=moments_fn)
        means, vars = estimator.estimate_moments(moments_fn)

        moments_quantity = moments(root_quantity, moments_fn=moments_fn, mom_at_bottom=True)
        moments_mean = estimate_mean(moments_quantity, level_means=True)

        conductivity_mean = moments_mean['conductivity']
        time_mean = conductivity_mean[1]  # times: [1]
        location_mean = time_mean['0']  # locations: ['0']
        values_mean = location_mean[0, 0]  # result shape: (1, 1)
        value_mean = values_mean[0]
        assert value_mean() == 1

        # true_domain = [-10, 10]  # keep all values on the original domain
        # central_moments = Monomial(self.n_moments, true_domain, ref_domain=true_domain, mean=means())
        # central_moments_quantity = moments(root_quantity, moments_fn=central_moments, mom_at_bottom=True)
        # central_moments_mean = estimate_mean(central_moments_quantity)

        #self.process_target_var(estimator)
        self.construct_density(estimator, tol=1e-8)

    def process_target_var(self, estimator):
        n0, nL = 100, 3
        n_samples = np.round(np.exp2(np.linspace(np.log2(n0), np.log2(nL), self.n_levels))).astype(int)

        n_estimated = estimator.bs_target_var_n_estimated(target_var=1e-5, sample_vec=n_samples)  # number of estimated sampels for given target variance
        estimator.plot_variances(sample_vec=n_estimated)
        estimator.plot_bs_var_log(sample_vec=n_estimated)

    def construct_density(self, estimator, tol=1.95, reg_param=0.0):
        """
        Construct approximation of the density using given moment functions.
        Args:
            moments_fn: Moments object, determines also domain and n_moments.
            tol: Tolerance of the fitting problem, with account for variances in moments.
                 Default value 1.95 corresponds to the two tail confidency 0.95.
            reg_param: Regularization parameter.
        """
        distr_obj, result, _, _ = estimator.construct_density(tol=tol, reg_param=reg_param)
        #distr_plot = mlmc.tool.plot.Distribution(title="{} levels, {} moments".format(self.n_levels, self.n_moments))
        distr_plot = mlmc.tool.plot.ArticleDistribution(title="{} levels, {} moments".format(self.n_levels, self.n_moments))

        distr_plot.add_distribution(distr_obj, label="#{}".format(self.n_moments))

        if self.n_levels == 1:
            samples = estimator.get_level_samples(level_id=0)[..., 0]
            distr_plot.add_raw_samples(np.squeeze(samples))

        distr_plot.show(None)
        distr_plot.show(file=os.path.join(self.work_dir, "pdf_cdf_{}_moments".format(self.n_moments)))
        distr_plot.reset()

    def run(self, renew=False):
        """
        Run MLMC
        :param renew: If True then rerun failed samples with same sample id
        :return: None
        """
        # Create working directory if necessary
        os.makedirs(self.work_dir, mode=0o775, exist_ok=True)

        if self.clean:
            # Remove HFD5 file
            if os.path.exists(os.path.join(self.work_dir, "mlmc_{}.hdf5".format(self.n_levels))):
                os.remove(os.path.join(self.work_dir, "mlmc_{}.hdf5".format(self.n_levels)))

        # Create sampler (mlmc.Sampler instance) - crucial class which actually schedule samples
        sampler = self.setup_config(clean=True)
        # Schedule samples
        self.generate_jobs(sampler, n_samples=None, renew=renew, target_var=1e-5)
        #self.generate_jobs(sampler, n_samples=[500, 500], renew=renew, target_var=1e-5)
        self.all_collect(sampler)  # Check if all samples are finished
        self.calculate_moments(sampler)  # Simple moment check

    def setup_config(self, clean):
        """
        Simulation dependent configuration
        :param clean: bool, If True remove existing files
        :return: mlmc.sampler instance
        """
        # Set pbs config, flow123d, gmsh, ..., random fields are set in simulation class
        self.set_environment_variables()

        # Create Pbs sampling pool
        sampling_pool = self.create_sampling_pool()

        simulation_config = {
            'work_dir': self.work_dir,
            'env': dict(flow123d=self.flow123d, gmsh=self.gmsh, gmsh_version=1),  # The Environment.
            'yaml_file': os.path.join(self.work_dir, '01_conductivity.yaml'),
            'geo_file': os.path.join(self.work_dir, 'square_1x1.geo'),
            'fields_params': dict(model='TPLgauss'),
            'field_template': "!FieldElementwise {mesh_data_file: \"$INPUT_DIR$/%s\", field_name: %s}"
        }

        # Create simulation factory
        simulation_factory = FlowSim(config=simulation_config, clean=clean)

        # Create HDF sample storage
        sample_storage = SampleStorageHDF(
            file_path=os.path.join(self.work_dir, "mlmc_{}.hdf5".format(self.n_levels)),
            #append=self.append
        )

        # Create sampler, it manages sample scheduling and so on
        sampler = Sampler(sample_storage=sample_storage, sampling_pool=sampling_pool, sim_factory=simulation_factory,
                          level_parameters=self.level_parameters)

        return sampler

    def set_environment_variables(self):
        """
        Set pbs config, flow123d, gmsh
        :return: None
        """
        root_dir = os.path.abspath(self.work_dir)
        while root_dir != '/':
            root_dir, tail = os.path.split(root_dir)

        if tail == 'storage' or tail == 'auto':
            # Metacentrum
            self.sample_sleep = 30
            self.init_sample_timeout = 600
            self.sample_timeout = 0
            self.flow123d = 'flow123d'  # "/storage/praha1/home/jan_brezina/local/flow123d_2.2.0/flow123d"
            self.gmsh = "/storage/liberec3-tul/home/martin_spetlik/astra/gmsh/bin/gmsh"
        else:
            # Local
            self.sample_sleep = 1
            self.init_sample_timeout = 60
            self.sample_timeout = 60
            self.flow123d = "/home/jb/workspace/flow123d/bin/fterm flow123d dbg"
            self.gmsh = "/home/jb/local/gmsh-3.0.5-git-Linux/bin/gmsh"

    def create_sampling_pool(self):
        """
        Initialize sampling pool, object which
        :return: None
        """
        if not self.use_pbs:
            return OneProcessPool(work_dir=self.work_dir, debug=self.debug)  # Everything runs in one process

        # Create PBS sampling pool
        sampling_pool = SamplingPoolPBS(work_dir=self.work_dir, clean=self.clean, debug=self.debug)

        pbs_config = dict(
            n_cores=1,
            n_nodes=1,
            select_flags=['cgroups=cpuacct'],
            mem='4Gb',
            queue='charon',
            pbs_name='flow123d',
            walltime='1:00:00',
            optional_pbs_requests=[],  # e.g. ['#PBS -m ae', ...]
            home_dir='/storage/liberec3-tul/home/martin_spetlik/',
            python='python3',
            env_setting=['cd $MLMC_WORKDIR',
                         'module load python36-modules-gcc',
                         'source env/bin/activate',
                         'pip3 install /storage/liberec3-tul/home/martin_spetlik/MLMC_new_design',
                         'module use /storage/praha1/home/jan-hybs/modules',
                         'module load python36-modules-gcc',
                         'module load flow123d',
                         'module list']
        )

        sampling_pool.pbs_common_setting(flow_3=True, **pbs_config)

        return sampling_pool

    def generate_jobs(self, sampler, n_samples=None, renew=False, target_var=None):
        """
        Generate level samples
        :param n_samples: None or list, number of samples for each level
        :param renew: rerun failed samples with same random seed (= same sample id)
        :return: None
        """
        if renew:
            sampler.ask_sampling_pool_for_samples()
            sampler.renew_failed_samples()
            sampler.ask_sampling_pool_for_samples(sleep=self.sample_sleep, timeout=self.sample_timeout)
        else:
            if n_samples is not None:
                sampler.set_initial_n_samples(n_samples)
            else:
                sampler.set_initial_n_samples()
            sampler.schedule_samples()
            sampler.ask_sampling_pool_for_samples(sleep=self.sample_sleep, timeout=self.sample_timeout)

            if target_var is not None:
                start_time = time.time()
                self.all_collect(sampler)

                moments_fn = self.set_moments(sampler.sample_storage)

                q_estimator = QuantityEstimate(sample_storage=sampler.sample_storage, moments_fn=moments_fn,
                                               sim_steps=self.level_parameters)
                target_var = 1e-5
                sleep = 0
                add_coef = 0.1

                # @TODO: test
                # New estimation according to already finished samples
                variances, n_ops = q_estimator.estimate_diff_vars_regression(sampler._n_scheduled_samples)
                n_estimated = new_estimator.estimate_n_samples_for_target_variance(target_var, variances, n_ops,
                                                                                   n_levels=sampler.n_levels)

                # Loop until number of estimated samples is greater than the number of scheduled samples
                while not sampler.process_adding_samples(n_estimated, sleep, add_coef):
                    with open(os.path.join(self.work_dir, "sampling_info.txt"), "a") as writer:
                        n_target_str = ",".join([str(n_target) for n_target in sampler._n_target_samples])
                        n_scheduled_str = ",".join([str(n_scheduled) for n_scheduled in sampler._n_scheduled_samples])
                        n_estimated_str = ",".join([str(n_est) for n_est in n_estimated])
                        variances_str = ",".join([str(vars) for vars in variances])
                        n_ops_str = ",".join([str(n_o) for n_o in n_ops])

                        writer.write("{}; {}; {}; {}; {}; {}\n".format(n_target_str, n_scheduled_str,
                                                                   n_estimated_str, variances_str,
                                                                   n_ops_str, str(time.time() - start_time)))

                    # New estimation according to already finished samples
                    variances, n_ops = q_estimator.estimate_diff_vars_regression(sampler._n_scheduled_samples)
                    n_estimated = new_estimator.estimate_n_samples_for_target_variance(target_var, variances, n_ops,
                                                                                       n_levels=sampler.n_levels)

    def all_collect(self, sampler):
        """
        Collect samples
        :param sampler: mlmc.Sampler object
        :return: None
        """
        running = 1
        while running > 0:
            running = 0
            running += sampler.ask_sampling_pool_for_samples(sleep=self.sample_sleep, timeout=0.1)
            print("N running: ", running)

    def calculate_moments(self, sampler):
        """
        Calculate moments through the mlmc.QuantityEstimate
        :param sampler: mlmc.Sampler
        :return: None
        """
        # Simple moment evaluation
        moments_fn = self.set_moments(sampler.sample_storage)

        q_estimator = QuantityEstimate(sample_storage=sampler.sample_storage, moments_fn=moments_fn,
                                       sim_steps=self.level_parameters)
        means, vars = q_estimator.estimate_moments(moments_fn)
        # The first moment is in any case 1 and its variance is 0
        assert means[0] == 1
        # assert np.isclose(means[1], 0, atol=1e-2)
        assert vars[0] == 0

    def set_moments(self, sample_storage, n_moments=5):
        true_domain = QuantityEstimate.estimate_domain(sample_storage, quantile=0.01)
        return Legendre(n_moments, true_domain)
    
    @staticmethod
    def determine_level_parameters(n_levels, step_range):
        """
        Determine level parameters,
        In this case, a step of fine simulation at each level
        :param n_levels: number of MLMC levels
        :param step_range: simulation step range
        :return: List
        """
        assert step_range[0] > step_range[1]
        level_parameters = []
        for i_level in range(n_levels):
            if n_levels == 1:
                level_param = 1
            else:
                level_param = i_level / (n_levels - 1)
            level_parameters.append([step_range[0] ** (1 - level_param) * step_range[1] ** level_param])

        return level_parameters

    @staticmethod
    def determine_n_samples(n_levels, n_samples=None):
        """
        Set target number of samples for each level
        :param n_levels: number of levels
        :param n_samples: array of number of samples
        :return: None
        """
        if n_samples is None:
            n_samples = [100, 3]
        # Num of samples to ndarray
        n_samples = np.atleast_1d(n_samples)

        # Just maximal number of samples is set
        if len(n_samples) == 1:
            n_samples = np.array([n_samples[0], 3])

        # Create number of samples for all levels
        if len(n_samples) == 2:
            n0, nL = n_samples
            n_samples = np.round(np.exp2(np.linspace(np.log2(n0), np.log2(nL), n_levels))).astype(int)

        return n_samples


if __name__ == "__main__":
    pr = ProcessSimple()
