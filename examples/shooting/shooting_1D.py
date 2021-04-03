import numpy as np
import mlmc.estimator
from mlmc.sampler import Sampler
from mlmc.sample_storage import Memory
from mlmc.sampling_pool import OneProcessPool
from examples.shooting.simulation_shooting_1D import ShootingSimulation1D
from mlmc.quantity.quantity import make_root_quantity
from mlmc.quantity.quantity_estimate import moments, estimate_mean
from mlmc.moments import Legendre


class ProcessShooting:

    def __init__(self):
        n_levels = 5
        # Number of MLMC levels
        step_range = [1, 0.0005]
        # step_range [simulation step at the coarsest level, simulation step at the finest level]
        level_parameters = ProcessShooting.determine_level_parameters(n_levels, step_range)
        # Determine level parameters at each level (In this case, simulation step at each level) are set automatically.
        # level_parameters should be simulation dependent for MLMC to be
        self._sample_sleep = 30
        self._sample_timeout = 60
        self._adding_samples_coef = 0.1
        self._n_moments = 10
        self._quantile = 0.01
        # Setting parameters that are utilized when scheduling samples

        """
        Run MLMC
        """
        sampler = self.create_sampler(level_parameters=level_parameters)
        # Create sampler (mlmc.Sampler instance) - crucial class that actually schedules MLMC samples
        self.generate_samples(sampler, n_samples=None, target_var=1e-4)
        # Generate MLMC samples, there are two ways:
        # 1) set exact number of samples at each level,
        #    e.g. for 5 levels - self.generate_samples(sampler, n_samples=[1000, 500, 250, 100, 50])
        # 2) set target variance of MLMC estimates,
        #    e.g. self.generate_samples(sampler, n_samples=None, target_var=1e-6)

        self.all_collect(sampler)
        # Check if all samples are finished

        self.process_results(sampler, n_levels)
        # Postprocessing, MLMC is finished at this point

    def create_sampler(self, level_parameters):
        """
        Create:
        # sampling pool - determines how sample simulations are performed
        # sample storage - stores sample results
        # sampler - controls MLMC execution
        :return: mlmc.sampler.Sampler instance
        """
        # Create OneProcessPool - all run in same process
        sampling_pool = OneProcessPool()
        # There is another option mlmc.sampling_pool.ProcessPool() - supports local parallel sample simulation run
        # sampling_pool = ProcessPool(n), n - number of parallel simulations, depends on computer architecture

        # Simulation configuration which is passed to simulation constructor
        simulation_config = {
            "start_position": np.array([0, 0]),
            "start_velocity": np.array([10, 0]),
            "area_borders":  np.array([-100, 200, -300, 400]),
            "max_time": 10,
            "complexity": 2,  # used for initial estimate of number of operations per sample
            'fields_params': dict(model='gauss', dim=1, sigma=1, corr_length=0.1),
        }

        # Create simulation factory, instance of class that inherits from mlmc.sim.simulation
        simulation_factory = ShootingSimulation1D(config=simulation_config)

        # Create simple sample storage
        # Memory keeps samples in computer main memory
        sample_storage = Memory()
        # There is another sample storage mlmc.sample_storage_hdf.SampleStorageHDF() - supports local parallel sample simulation run
        # sample_storage = SampleStorgaeHDF(file_path=path_to_HDF5_file)

        # Create sampler
        # Controls the execution of MLMC
        sampler = Sampler(sample_storage=sample_storage, sampling_pool=sampling_pool, sim_factory=simulation_factory,
                          level_parameters=level_parameters)
        return sampler

    def generate_samples(self, sampler, n_samples=None, target_var=None):
        """
        Generate MLM samples
        :param sampler: mlmc.sampler.Sampler instance
        :param n_samples: None or list, number of samples at each level
        :param target_var: MLMC estimates target variance
        :return: None
        """
        # The number of samples is set by user
        if n_samples is not None:
            sampler.set_initial_n_samples(n_samples)
        # The number of initial samples is determined automatically
        else:
            sampler.set_initial_n_samples()
        # Samples are scheduled and the program is waiting for all of them to be finished
        sampler.schedule_samples()
        sampler.ask_sampling_pool_for_samples(sleep=self._sample_sleep, timeout=self._sample_timeout)
        self.all_collect(sampler)

        # MLMC estimates target variance is set
        if target_var is not None:
            # The mlmc.quantity.quantity.Quantity instance is created
            # parameters 'storage' and 'q_specs' are retrieved from sample_storage,
            # originally 'q_specs' is set in the simulation class
            root_quantity = make_root_quantity(storage=sampler.sample_storage,
                                               q_specs=sampler.sample_storage.load_result_format())

            # Moments functions object is created
            # The MLMC algorithm determine number of samples according to moments variance,
            # Type of moments function (Legendre by default) might affect the total number of MLMC samples
            moments_fn = self.set_moments(root_quantity, sampler.sample_storage, n_moments=self._n_moments)
            estimate_obj = mlmc.estimator.Estimate(root_quantity, sample_storage=sampler.sample_storage,
                                                   moments_fn=moments_fn)

            # Initial estimation of the number of samples at each level
            variances, n_ops = estimate_obj.estimate_diff_vars_regression(sampler.n_finished_samples)
            print("variances ", variances)
            print("n ops ", n_ops)
            # Firstly, a variance of moments and execution time of samples at each level are calculated from already finished samples.
            n_estimated = mlmc.estimator.estimate_n_samples_for_target_variance(target_var, variances, n_ops,
                                                                           n_levels=sampler.n_levels)

            print("n estimated ", n_estimated)

            #####
            # MLMC sampling algorithm - gradually schedules samples and refines the total number of samples
            #####
            # Loop until number of estimated samples is greater than the number of scheduled samples
            while not sampler.process_adding_samples(n_estimated, self._sample_sleep, self._adding_samples_coef,
                                                     timeout=self._sample_timeout):
                # New estimation according to already finished samples
                variances, n_ops = estimate_obj.estimate_diff_vars_regression(sampler._n_scheduled_samples)
                n_estimated = mlmc.estimator.estimate_n_samples_for_target_variance(target_var, variances, n_ops,
                                                                               n_levels=sampler.n_levels)

    def set_moments(self, quantity, sample_storage, n_moments=25):
        true_domain = mlmc.estimator.Estimate.estimate_domain(quantity, sample_storage, quantile=self._quantile)
        return Legendre(n_moments, true_domain)

    def all_collect(self, sampler):
        """
        Collect samples, wait until all samples are finished
        :param sampler: mlmc.sampler.Sampler object
        :return: None
        """
        running = 1
        while running > 0:
            running = 0
            running += sampler.ask_sampling_pool_for_samples()
            print("N running: ", running)

    def process_results(self, sampler, n_levels):
        sample_storage = sampler.sample_storage
        # Load result format from sample storage
        result_format = sample_storage.load_result_format()
        # Create quantity instance representing our real quantity of interest
        root_quantity = make_root_quantity(sample_storage, result_format)

        # It is possible to access item of quantity according to result format
        target = root_quantity['target']
        time = target[10]  # times: [1]
        position = time['0']  # locations: ['0']
        q_value = position[0]

        # Compute moments
        estimated_domain = mlmc.estimator.Estimate.estimate_domain(q_value, sample_storage, quantile=self._quantile)
        moments_fn = Legendre(self._n_moments, estimated_domain)

        # Create estimator for your quantity
        estimator = mlmc.estimator.Estimate(quantity=q_value, sample_storage=sample_storage, moments_fn=moments_fn)
        means, vars = estimator.estimate_moments(moments_fn)

        print("means ", means)
        print("vars ", vars)

        # Generally root quantity has different domain than its items
        root_quantity_estimated_domain = mlmc.estimator.Estimate.estimate_domain(root_quantity, sample_storage,
                                                                                 quantile=self._quantile)
        root_quantity_moments_fn = Legendre(self._n_moments, root_quantity_estimated_domain)

        # There is another possible approach to calculating all moments at once and then select desired quantity
        moments_quantity = moments(root_quantity, moments_fn=root_quantity_moments_fn, mom_at_bottom=True)
        moments_mean = estimate_mean(moments_quantity)
        target_mean = moments_mean['target']
        time_mean = target_mean[10]  # times: [1]
        location_mean = time_mean['0']  # locations: ['0']
        value_mean = location_mean[0]  # result shape: (1,)
        assert value_mean.mean[0] == 1

        # estimator.sub_subselect(sample_vector=[10000])
        self.construct_density(estimator, n_levels, tol=1e-8)

    def construct_density(self, estimator, n_levels, tol=1.95, reg_param=0.0):
        """
        Construct approximation of the density using given moment functions.
        :param estimator: mlmc.estimator.Estimate instance, it contains quantity for which the density is reconstructed
        :param tol: Tolerance of the fitting problem, with account for variances in moments.
        Default value 1.95 corresponds to the two tail confidence 0.95.
        :param reg_param: regularization parameter
        :return: None
        """
        distr_obj, result, _, _ = estimator.construct_density(tol=tol, reg_param=reg_param)
        distr_plot = mlmc.plot.plots.Distribution(title="distribution")
        distr_plot.add_distribution(distr_obj)

        if n_levels == 1:
            samples = estimator.get_level_samples(level_id=0)[..., 0]
            distr_plot.add_raw_samples(np.squeeze(samples))
        distr_plot.show(None)
        distr_plot.reset()

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
    ProcessShooting()

    # import cProfile
    # import pstats
    # pr = cProfile.Profile()
    # pr.enable()
    #
    # my_result = ProcessSimple()
    #
    # pr.disable()
    # ps = pstats.Stats(pr).sort_stats('cumtime')
    # ps.print_stats()
