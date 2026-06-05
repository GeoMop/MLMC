import numpy as np
import mlmc.estimator as est
from mlmc.estimator import Estimate, estimate_n_samples_for_target_variance
from mlmc.sampler import Sampler
from mlmc.sample_storage import Memory
from mlmc.sampling_pool import OneProcessPool
from examples.shooting.simulation_shooting_1D import ShootingSimulation1D
from mlmc.quantity.quantity import make_root_quantity
from mlmc.quantity.quantity_estimate import moments, estimate_mean
from mlmc.moments import Legendre
from mlmc.plot.plots import Distribution


# Tutorial class for 1D shooting simulation, includes
# - samples scheduling
# - process results:
#           - create Quantity instance
#           - approximate density
class ProcessShooting1D:
    """
    Example driver for a 1D shooting problem using the MLMC framework.

    Demonstrates:
      - creating the Sampler, sampling_pool and sample storage,
      - scheduling and collecting MLMC samples,
      - estimating moments and building an approximate PDF for a quantity of interest.
    """

    def __init__(self):
        """
        Initialize parameters, create sampler, schedule and collect samples, and postprocess.

        The constructor executes a full example run:
          - determine level parameters,
          - create sampler,
          - generate samples (either user-specified or determined from a target variance),
          - collect all results,
          - postprocess and approximate distribution of the chosen quantity.
        """
        n_levels = 3
        # Number of MLMC levels

        step_range = [1, 1e-3]
        # step_range [simulation step at the coarsest level, simulation step at the finest level]

        level_parameters = ProcessShooting1D.determine_level_parameters(n_levels, step_range)
        # Determine each level parameters (in this case, simulation step at each level)

        self._sample_sleep = 0  # seconds to sleep while polling sampling pool
        self._sample_timeout = 60  # maximum waiting time for sampling pool operations (seconds)
        self._adding_samples_coef = 0.1  # coefficient used when adding samples adaptively

        self._n_moments = 20
        # number of generalized statistical moments used for MLMC sample estimation
        self._quantile = 0.01
        # quantile used to estimate domain of distribution for moment basis

        # MLMC run
        sampler = self.create_sampler(level_parameters=level_parameters)
        # Create sampler (mlmc.Sampler instance) - controls MLMC run
        self.generate_samples(sampler, n_samples=None, target_var=1e-3)
        # Generate MLMC samples (either explicit counts or target variance)
        self.all_collect(sampler)
        # Wait for all scheduled samples to finish

        # Postprocessing
        self.process_results(sampler, n_levels)
        # Postprocessing complete

    def create_sampler(self, level_parameters):
        """
        Create and configure sampler components: sampling pool, simulation factory and storage.

        :param level_parameters: list of per-level parameter lists (simulation-dependent).
        :return: mlmc.sampler.Sampler instance configured with Memory storage and OneProcessPool.
        """
        # Use OneProcessPool for sequential runs. Replace with ProcessPool/ThreadPool for parallel runs.
        sampling_pool = OneProcessPool()

        # Simulation configuration passed into the simulation factory
        simulation_config = {
            "start_position": np.array([0, 0]),
            "start_velocity": np.array([10, 0]),
            "area_borders": np.array([-100, 200, -300, 400]),
            "max_time": 10,
            "complexity": 2,  # used for initial estimate of operations per sample
            "fields_params": dict(model='gauss', dim=1, sigma=1, corr_length=0.1),
        }

        # Create simulation factory (produces LevelSimulation instances)
        simulation_factory = ShootingSimulation1D(config=simulation_config)

        # In-memory sample storage (alternative: HDF-based storage)
        sample_storage = Memory()

        # Create and return the Sampler orchestrating MLMC
        sampler = Sampler(sample_storage=sample_storage,
                          sampling_pool=sampling_pool,
                          sim_factory=simulation_factory,
                          level_parameters=level_parameters)
        return sampler

    def generate_samples(self, sampler, n_samples=None, target_var=None):
        """
        Schedule and generate MLMC samples. If target_var is provided, iteratively determine
        and add samples until the MLMC variance target is reached.

        :param sampler: mlmc.sampler.Sampler instance controlling sampling.
        :param n_samples: Optional list of exact sample counts per level. If provided, used as initial counts.
        :param target_var: Optional float target variance for MLMC estimator. If provided, algorithm estimates counts.
        :return: None
        """
        # Set initial number of samples (user-specified or default)
        if n_samples is not None:
            sampler.set_initial_n_samples(n_samples)
        else:
            sampler.set_initial_n_samples()

        # Schedule and start the initial batch of samples, then wait for completion
        sampler.schedule_samples()
        sampler.ask_sampling_pool_for_samples(sleep=self._sample_sleep, timeout=self._sample_timeout)
        self.all_collect(sampler)

        # If a target_var is provided, compute required sample counts and add samples iteratively
        if target_var is not None:
            # Build a root quantity (required for moments estimation)
            root_quantity = make_root_quantity(storage=sampler.sample_storage,
                                               q_specs=sampler.sample_storage.load_result_format())

            # Create moment functions (Legendre) on estimated domain
            moments_fn = self.set_moments(root_quantity, sampler.sample_storage, n_moments=self._n_moments)
            estimate_obj = Estimate(root_quantity, sample_storage=sampler.sample_storage, moments_fn=moments_fn)

            # Estimate variances and costs from finished samples
            variances, n_ops = estimate_obj.estimate_diff_vars_regression(sampler.n_finished_samples)

            # Compute estimated number of samples per level for the target variance
            n_estimated = estimate_n_samples_for_target_variance(target_var, variances, n_ops,
                                                                 n_levels=sampler.n_levels)

            # Iteratively add samples until the scheduler has scheduled enough
            while not sampler.process_adding_samples(n_estimated, self._sample_sleep, self._adding_samples_coef,
                                                     timeout=self._sample_timeout):
                variances, n_ops = estimate_obj.estimate_diff_vars_regression(sampler._n_scheduled_samples)
                n_estimated = estimate_n_samples_for_target_variance(target_var, variances, n_ops,
                                                                     n_levels=sampler.n_levels)

    def set_moments(self, quantity, sample_storage, n_moments=25):
        """
        Build Legendre moment basis on the domain estimated from samples.

        :param quantity: Quantity (or root quantity) used to estimate domain.
        :param sample_storage: Sample storage used to compute domain estimate.
        :param n_moments: Number of Legendre basis functions / moments.
        :return: Legendre(n_moments, domain) instance.
        """
        true_domain = Estimate.estimate_domain(quantity, sample_storage, quantile=self._quantile)
        return Legendre(n_moments, true_domain)

    def all_collect(self, sampler):
        """
        Repeatedly collect finished samples until none are running.

        :param sampler: mlmc.sampler.Sampler instance to poll for finished samples.
        :return: None
        """
        running = 1
        while running > 0:
            running = 0
            running += sampler.ask_sampling_pool_for_samples()
            print("N running: ", running)

    def process_results(self, sampler, n_levels):
        """
        Postprocess completed samples:
          - build Quantity objects,
          - compute moment means/variances,
          - perform checks and consistency tests,
          - approximate distribution for the target quantity.

        :param sampler: mlmc.sampler.Sampler instance (contains sample_storage).
        :param n_levels: int, number of MLMC levels used in the run.
        :return: None
        """
        sample_storage = sampler.sample_storage

        # Load result format and create root quantity
        result_format = sample_storage.load_result_format()
        root_quantity = make_root_quantity(sample_storage, result_format)

        print("N collected ", sample_storage.get_n_collected())

        # Access a nested item (example of how to index Quantity)
        target = root_quantity['target']
        time = target[10]
        position = time['0']
        q_value = position[0]

        # Estimate domain from samples and build moments function
        estimated_domain = Estimate.estimate_domain(q_value, sample_storage, quantile=self._quantile)
        moments_fn = Legendre(self._n_moments, estimated_domain)

        # Estimator for the selected quantity
        estimator = Estimate(quantity=q_value, sample_storage=sample_storage, moments_fn=moments_fn)

        # Compute moment means and variances
        means, vars = estimator.estimate_moments(moments_fn)

        # Diagnostics and consistency checks
        #est.plot_checks(quantity=q_value, sample_storage=sample_storage, moments_fn=moments_fn)
        est.consistency_check(quantity=q_value, sample_storage=sample_storage)
        estimator.kurtosis_check(q_value)

        # Optionally compute moments for full root quantity and extract target mean
        root_quantity_estimated_domain = Estimate.estimate_domain(root_quantity, sample_storage,
                                                                 quantile=self._quantile)
        root_quantity_moments_fn = Legendre(self._n_moments, root_quantity_estimated_domain)
        moments_quantity = moments(root_quantity, moments_fn=root_quantity_moments_fn, mom_at_bottom=True)
        moments_mean = estimate_mean(moments_quantity)
        target_mean = moments_mean['target']
        time_mean = target_mean[10]
        location_mean = time_mean['0']
        value_mean = location_mean[0]

        # Example assertion for tutorial (problem-dependent)
        assert value_mean.mean[0] == 1

        # Build and show approximate density
        self.approx_distribution(estimator, n_levels, tol=1e-8)

    def approx_distribution(self, estimator, n_levels, tol=1.95):
        """
        Approximate and display the probability density function for the estimator's quantity.

        :param estimator: mlmc.estimator.Estimate instance (contains quantity and methods for density construction).
        :param n_levels: int, number of MLMC levels (used for optional raw-sample overlay).
        :param tol: Tolerance for density fitting (accounts for moment variances).
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

    @staticmethod
    def determine_level_parameters(n_levels, step_range):
        """
        Determine parameters for each MLMC level (here a single step size per level).

        Interpolates between step_range[0] (coarse) and step_range[1] (fine) across levels.

        :param n_levels: int number of MLMC levels.
        :param step_range: [coarse_step, fine_step] with coarse_step > fine_step.
        :return: list of lists; each inner list contains the step parameter for that level.
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
    ProcessShooting1D()
