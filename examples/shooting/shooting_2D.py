import numpy as np
from mlmc.estimator import Estimate, estimate_n_samples_for_target_variance
from mlmc.sampler import Sampler
from mlmc.sample_storage import Memory
from mlmc.sampling_pool import OneProcessPool
from examples.shooting.simulation_shooting_2D import ShootingSimulation2D
from mlmc.quantity.quantity import make_root_quantity
from mlmc.quantity.quantity_estimate import moments, estimate_mean
from mlmc.moments import Legendre
from mlmc.plot.plots import Distribution


# Simple example with OneProcessPool, MultiProcessPool, also show usage of different storages - Memory(), HDFStorage()


class ProcessShooting2D:
    """
    Example driver for a 2D shooting simulation using the MLMC framework.

    Demonstrates:
      - building sampler, sampling pool and sample storage,
      - scheduling and collecting MLMC samples,
      - estimating moments and approximating probability densities for quantities of interest.
    """

    def __init__(self):
        """
        Initialize parameters, create sampler, schedule and collect samples, and postprocess.

        The constructor runs a full example MLMC workflow:
          - determine level parameters,
          - create sampler and sampling pool,
          - generate MLMC samples (either user-specified or adaptively from target variance),
          - wait for completion and postprocess results.
        """
        n_levels = 5
        # Number of MLMC levels
        step_range = [0.05, 0.005]
        # step_range [simulation step at the coarsest level, simulation step at the finest level]
        level_parameters = ProcessShooting2D.determine_level_parameters(n_levels, step_range)
        # Determine each level parameters (in this case, simulation step at each level)

        self._sample_sleep = 0
        # Seconds to sleep while polling sampling pool
        self._sample_timeout = 60
        # Maximum waiting time for sampling pool operations (seconds)
        self._adding_samples_coef = 0.1
        self._n_moments = 20
        # number of generalized statistical moments used for MLMC number of samples estimation
        self._quantile = 0.001
        # quantile used to estimate domain for moment basis functions

        # MLMC run
        sampler = self.create_sampler(level_parameters=level_parameters)
        # Create sampler (mlmc.Sampler instance) - controls MLMC run
        self.generate_samples(sampler, n_samples=None, target_var=1e-3)
        # Generate MLMC samples: either explicit counts or by target variance
        self.all_collect(sampler)
        # Wait until all samples finish

        # Postprocessing
        self.process_results(sampler, n_levels)
        # Postprocessing complete

    def create_estimator(self, quantity, sample_storage):
        """
        Create an Estimate object for a given quantity and storage using Legendre moments.

        :param quantity: Quantity object (mlmc.quantity.quantity.Quantity item) to estimate.
        :param sample_storage: sample storage instance used to estimate domain.
        :return: mlmc.estimator.Estimate instance configured for the quantity.
        """
        estimated_domain = Estimate.estimate_domain(quantity, sample_storage, quantile=self._quantile)
        moments_fn = Legendre(self._n_moments, estimated_domain)
        return Estimate(quantity=quantity, sample_storage=sample_storage, moments_fn=moments_fn)

    def process_results(self, sampler, n_levels):
        """
        Postprocess completed MLMC samples: form quantities, create estimators and approximate distributions.

        Steps performed:
          - load result format and create root Quantity,
          - extract x and y components of the target item,
          - build per-component estimators,
          - compute root-level moments and means,
          - approximate and display distributions for x and y.

        :param sampler: mlmc.sampler.Sampler instance containing completed samples.
        :param n_levels: int, number of MLMC levels used.
        :return: None
        """
        sample_storage = sampler.sample_storage
        # Load result format from sample storage
        result_format = sample_storage.load_result_format()
        # Create quantity instance representing the quantity of interest
        root_quantity = make_root_quantity(sample_storage, result_format)

        # You can access item of quantity according to result format
        target = root_quantity['target']
        time = target[10]  # times: [1]
        position = time['0']  # locations: ['0']
        x_quantity_value = position[0]
        y_quantity_value = position[1]

        # Create estimators for each coordinate component
        x_estimator = self.create_estimator(x_quantity_value, sample_storage)
        y_estimator = self.create_estimator(y_quantity_value, sample_storage)

        # Optionally compute moments for full root quantity (and extract means)
        root_quantity_estimated_domain = Estimate.estimate_domain(root_quantity, sample_storage,
                                                                  quantile=self._quantile)
        root_quantity_moments_fn = Legendre(self._n_moments, root_quantity_estimated_domain)

        moments_quantity = moments(root_quantity, moments_fn=root_quantity_moments_fn, mom_at_bottom=True)
        moments_mean = estimate_mean(moments_quantity)
        target_mean = moments_mean['target']
        time_mean = target_mean[10]  # times: [1]
        location_mean = time_mean['0']  # locations: ['0']
        value_mean = location_mean[0]  # result shape: (1,)

        # Display approximated distributions for both coordinates
        self.approx_distribution(x_estimator, n_levels, tol=1e-8)
        self.approx_distribution(y_estimator, n_levels, tol=1e-8)

    def approx_distribution(self, estimator, n_levels, tol=1.95):
        """
        Approximate and display the probability density function for the estimator's quantity.

        :param estimator: mlmc.estimator.Estimate instance providing construct_density().
        :param n_levels: int, number of MLMC levels (used to optionally overlay raw samples).
        :param tol: float, tolerance for density-fitting routine (affects regularization with moment variances).
        :return: None
        """
        distr_obj, result, _, _ = estimator.construct_density(tol=tol)
        distr_plot = Distribution(title="distributions", error_plot=None)
        distr_plot.add_distribution(distr_obj)

        if n_levels == 1:
            samples = estimator.get_level_samples(level_id=0)[..., 0]
            distr_plot.add_raw_samples(np.squeeze(samples))
        distr_plot.show(None)
        distr_plot.reset()

    def create_sampler(self, level_parameters):
        """
        Create sampler, sampling pool and simulation factory for the 2D shooting example.

        :param level_parameters: list of per-level parameter lists.
        :return: mlmc.sampler.Sampler instance configured with Memory storage and OneProcessPool.
        """
        sampling_pool = OneProcessPool()

        simulation_config = {
            "start_position": np.array([0, 0]),
            "start_velocity": np.array([10, 0]),
            "area_borders": np.array([-100, 200, -300, 400]),
            "max_time": 10,
            "complexity": 2,  # used for initial estimate of number of operations per sample
            "fields_params": dict(model='gauss', dim=1, sigma=1, corr_length=0.1),
        }

        # Create simulation factory
        simulation_factory = ShootingSimulation2D(config=simulation_config)
        # In-memory storage (Memory). Replace with HDF storage if persistence is needed.
        sample_storage = Memory()
        # Create sampler which orchestrates scheduling and storage
        sampler = Sampler(sample_storage=sample_storage, sampling_pool=sampling_pool, sim_factory=simulation_factory,
                          level_parameters=level_parameters)

        return sampler

    def generate_samples(self, sampler, n_samples=None, target_var=None):
        """
        Schedule and generate MLMC samples. If target_var is provided, iteratively determine
        additional samples required to reach the variance target.

        :param sampler: mlmc.sampler.Sampler instance controlling the run.
        :param n_samples: Optional list of exact sample counts per level.
        :param target_var: Optional float target variance for MLMC estimator.
        :return: None
        """
        # Set initial samples either from user or default logic
        if n_samples is not None:
            sampler.set_initial_n_samples(n_samples)
        else:
            sampler.set_initial_n_samples()

        # Schedule and wait for initial batch of samples
        sampler.schedule_samples()
        sampler.ask_sampling_pool_for_samples(sleep=self._sample_sleep, timeout=self._sample_timeout)
        self.all_collect(sampler)

        # If a global target variance is specified, estimate and iteratively add samples
        if target_var is not None:
            # Build a root quantity required for moment estimation
            root_quantity = make_root_quantity(storage=sampler.sample_storage,
                                               q_specs=sampler.sample_storage.load_result_format())

            moments_fn = self.set_moments(root_quantity, sampler.sample_storage, n_moments=self._n_moments)
            estimate_obj = Estimate(root_quantity, sample_storage=sampler.sample_storage,
                                    moments_fn=moments_fn)

            # Estimate per-level variances and costs from finished samples
            variances, n_ops = estimate_obj.estimate_diff_vars_regression(sampler.n_finished_samples)
            n_estimated = estimate_n_samples_for_target_variance(target_var, variances, n_ops,
                                                                 n_levels=sampler.n_levels)

            # Loop until sampler has scheduled enough samples
            while not sampler.process_adding_samples(n_estimated, self._sample_sleep, self._adding_samples_coef,
                                                     timeout=self._sample_timeout):
                variances, n_ops = estimate_obj.estimate_diff_vars_regression(sampler._n_scheduled_samples)
                n_estimated = estimate_n_samples_for_target_variance(target_var, variances, n_ops,
                                                                     n_levels=sampler.n_levels)

    def all_collect(self, sampler):
        """
        Repeatedly poll the sampling pool and collect finished samples until none remain running.

        :param sampler: mlmc.sampler.Sampler instance.
        :return: None
        """
        running = 1
        while running > 0:
            running = 0
            running += sampler.ask_sampling_pool_for_samples()
            print("N running: ", running)

    def set_moments(self, quantity, sample_storage, n_moments=25):
        """
        Build Legendre moments object for a given quantity using the domain estimated from samples.

        :param quantity: quantity object (used to estimate domain).
        :param sample_storage: storage used to compute the domain estimate.
        :param n_moments: number of Legendre polynomials to use.
        :return: Legendre instance configured for the estimated domain.
        """
        true_domain = Estimate.estimate_domain(quantity, sample_storage, quantile=self._quantile)
        return Legendre(n_moments, true_domain)

    @staticmethod
    def determine_level_parameters(n_levels, step_range):
        """
        Determine level parameters for MLMC.

        For this example a single per-level 'step' is returned in a list, interpolating
        (geometrically) between step_range[0] and step_range[1].

        :param n_levels: int number of MLMC levels.
        :param step_range: [coarse_step, fine_step] with coarse_step > fine_step.
        :return: List[List[float]]: per-level parameters (each inner list contains the step value).
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


if __name__ == "__main__":
    ProcessShooting2D()
