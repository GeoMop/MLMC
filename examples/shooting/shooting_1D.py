import numpy as np
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

    This tutorial class demonstrates how to:
      - create a Sampler with storage and sampling pool,
      - schedule and collect MLMC samples,
      - estimate moments and build an approximate probability density function
        of the quantity of interest.

    The constructor runs a full example MLMC workflow when the module is executed.
    """

    def __init__(self):
        """
        Initialize parameters, create sampler, generate samples, collect them and postprocess.

        The constructor sets up:
        - MLMC level parameters,
        - sampling and storage components,
        - schedule and collection of samples,
        - postprocessing including moment estimation and density approximation.
        """
        n_levels = 2
        # Number of MLMC levels

        step_range = [1, 1e-3]
        # step_range [simulation step at the coarsest level, simulation step at the finest level]

        level_parameters = ProcessShooting1D.determine_level_parameters(n_levels, step_range)
        # Determine each level parameters (in this case, simulation step at each level)

        # Sampling control parameters
        self._sample_sleep = 0  # seconds to sleep between checking sampling pool (was 30)
        self._sample_timeout = 60  # maximum waiting time for running simulations (seconds)
        self._adding_samples_coef = 0.1  # coefficient used when dynamically adding samples

        # Moment / distribution settings
        self._n_moments = 20  # number of generalized moments used for MLMC sample estimation
        self._quantile = 0.01  # quantile used to estimate domain for moment functions

        # MLMC run: create sampler and run sampling workflow
        sampler = self.create_sampler(level_parameters=level_parameters)
        self.generate_samples(sampler, n_samples=None, target_var=1e-3)
        # Generate MLMC samples. Two ways:
        # 1) Provide exact n_samples per level: generate_samples(sampler, n_samples=[...])
        # 2) Provide target variance and let algorithm decide counts: target_var=1e-6

        self.all_collect(sampler)
        # Ensure all scheduled samples finished

        # Postprocessing: compute moments and approximate distributions
        self.process_results(sampler, n_levels)

    def create_sampler(self, level_parameters):
        """
        Create the Sampler with storage and sampling pool and return it.

        Components created:
          - sampling_pool: OneProcessPool (runs all samples in the same process)
          - simulation_factory: instance of ShootingSimulation1D
          - sample_storage: Memory() (in-memory storage)
          - sampler: mlmc.sampler.Sampler

        :param level_parameters: List of level parameter lists (simulation-dependent).
        :return: mlmc.sampler.Sampler instance configured for this example.
        """
        # Use OneProcessPool for simplicity (sequential execution). For parallel local runs,
        # replace with ProcessPool(n) or ThreadPool(n).
        sampling_pool = OneProcessPool()

        # Simulation configuration dictionary passed to simulation factory
        simulation_config = {
            "start_position": np.array([0, 0]),
            "start_velocity": np.array([10, 0]),
            "area_borders":  np.array([-100, 200, -300, 400]),
            "max_time": 10,
            "complexity": 2,  # used as prior for cost estimates
            'fields_params': dict(model='gauss', dim=1, sigma=1, corr_length=0.1),
        }

        # Simulation factory (constructs LevelSimulation instances internally)
        simulation_factory = ShootingSimulation1D(config=simulation_config)

        # Lightweight in-memory sample storage
        sample_storage = Memory()
        # Alternative: HDF storage (persistent) - sample_storage_hdf.SampleStorageHDF(...)

        # Create and return the Sampler that orchestrates MLMC
        sampler = Sampler(sample_storage=sample_storage,
                          sampling_pool=sampling_pool,
                          sim_factory=simulation_factory,
                          level_parameters=level_parameters)
        return sampler

    def generate_samples(self, sampler, n_samples=None, target_var=None):
        """
        Schedule and generate MLMC samples, optionally targetting a global variance.

        :param sampler: mlmc.sampler.Sampler instance controlling sampling.
        :param n_samples: None or list of exact sample counts per level.
        :param target_var: Target variance for MLMC estimator (if not None).
        :return: None (results are stored in sampler.sample_storage).
        """
        # If user provided exact counts, set them; otherwise let sampler pick initial counts
        if n_samples is not None:
            sampler.set_initial_n_samples(n_samples)
        else:
            sampler.set_initial_n_samples()

        # Schedule initial samples and wait for them to complete
        sampler.schedule_samples()
        sampler.ask_sampling_pool_for_samples(sleep=self._sample_sleep, timeout=self._sample_timeout)
        self.all_collect(sampler)

        # If a target variance is requested, iteratively estimate variances and add more samples
        if target_var is not None:
            # Create a Quantity for the root results; needed for moments estimation
            root_quantity = make_root_quantity(storage=sampler.sample_storage,
                                               q_specs=sampler.sample_storage.load_result_format())

            # Build moment functions (Legendre polynomials on estimated domain)
            moments_fn = self.set_moments(root_quantity, sampler.sample_storage, n_moments=self._n_moments)
            estimate_obj = Estimate(root_quantity, sample_storage=sampler.sample_storage, moments_fn=moments_fn)

            # Initial variance and cost estimates from currently finished samples
            variances, n_ops = estimate_obj.estimate_diff_vars_regression(sampler.n_finished_samples)

            # Compute initial recommended number of samples for each level
            n_estimated = estimate_n_samples_for_target_variance(target_var, variances, n_ops,
                                                                 n_levels=sampler.n_levels)

            # Gradually schedule additional samples until the estimate stabilizes
            while not sampler.process_adding_samples(n_estimated,
                                                    self._sample_sleep,
                                                    self._adding_samples_coef,
                                                    timeout=self._sample_timeout):
                # Re-estimate variance / ops using newly finished samples and recompute targets
                variances, n_ops = estimate_obj.estimate_diff_vars_regression(sampler._n_scheduled_samples)
                n_estimated = estimate_n_samples_for_target_variance(target_var, variances, n_ops,
                                                                     n_levels=sampler.n_levels)

    def set_moments(self, quantity, sample_storage, n_moments=25):
        """
        Create a Legendre-moments function object for the given quantity.

        :param quantity: mlmc.quantity.quantity.Quantity instance (root quantity).
        :param sample_storage: sample storage instance to estimate domain from samples.
        :param n_moments: Number of moments (basis size) to construct.
        :return: Legendre(n_moments, domain) instance to be used for moment estimation.
        """
        true_domain = Estimate.estimate_domain(quantity, sample_storage, quantile=self._quantile)
        return Legendre(n_moments, true_domain)

    def all_collect(self, sampler):
        """
        Wait until all scheduled samples finish by repeatedly collecting finished samples.

        :param sampler: mlmc.sampler.Sampler instance.
        :return: None
        """
        running = 1
        while running > 0:
            running = 0
            running += sampler.ask_sampling_pool_for_samples()
            print("N running: ", running)

    def process_results(self, sampler, n_levels):
        """
        Postprocess completed samples: estimate moments and approximate distribution.

        Steps:
          - Load result format and build Quantity objects
          - Choose a target item inside the quantity and check its mean
          - Compute estimated domain, construct moment functions (Legendre)
          - Estimate moments and their variances, compute distribution approximation

        :param sampler: mlmc.sampler.Sampler instance containing sample_storage.
        :param n_levels: int, number of MLMC levels used in the run.
        :return: None
        """
        sample_storage = sampler.sample_storage

        # Load result format (list of QuantitySpec) and create root Quantity
        result_format = sample_storage.load_result_format()
        root_quantity = make_root_quantity(sample_storage, result_format)

        # Access an example item from the nested quantity to demonstrate API
        target = root_quantity['target']
        time = target[10]
        position = time['0']
        q_value = position[0]

        # Estimate domain for moment functions from sample data and build moments function
        estimated_domain = Estimate.estimate_domain(q_value, sample_storage, quantile=self._quantile)
        moments_fn = Legendre(self._n_moments, estimated_domain)

        # Create an estimator and compute moment means and variances
        estimator = Estimate(quantity=q_value, sample_storage=sample_storage, moments_fn=moments_fn)
        means, vars = estimator.estimate_moments(moments_fn)

        # For root-level moments use separate estimated domain
        root_quantity_estimated_domain = Estimate.estimate_domain(root_quantity, sample_storage,
                                                                 quantile=self._quantile)
        root_quantity_moments_fn = Legendre(self._n_moments, root_quantity_estimated_domain)

        # Alternative approach: compute moments for the entire root quantity and then extract target
        moments_quantity = moments(root_quantity, moments_fn=root_quantity_moments_fn, mom_at_bottom=True)
        moments_mean = estimate_mean(moments_quantity)
        target_mean = moments_mean['target']
        time_mean = target_mean[10]  # times: [1]
        location_mean = time_mean['0']  # locations: ['0']
        value_mean = location_mean[0]  # result shape: (1,)

        # Quick assertion expected for the tutorial problem (value_mean should be 1)
        assert value_mean.mean[0] == 1

        # Build approximate density for the selected quantity
        self.approx_distribution(estimator, n_levels, tol=1e-8)

    def approx_distribution(self, estimator, n_levels, tol=1.95):
        """
        Construct and display an approximate probability density function for the quantity.

        :param estimator: mlmc.estimator.Estimate instance which contains the quantity and methods
                          for constructing the density approximation.
        :param n_levels: int, number of MLMC levels (used for optional plotting of raw samples).
        :param tol: Tolerance parameter for the density fitting problem (default 1.95).
        :return: None
        """
        distr_obj, result, _, _ = estimator.construct_density(tol=tol)
        distr_plot = Distribution(title="distributions", error_plot=None)
        distr_plot.add_distribution(distr_obj)

        # Optionally overlay raw samples for single-level case
        if n_levels == 1:
            samples = estimator.get_level_samples(level_id=0)[..., 0]
            distr_plot.add_raw_samples(np.squeeze(samples))

        distr_plot.show(None)
        distr_plot.reset()

    @staticmethod
    def determine_level_parameters(n_levels, step_range):
        """
        Determine level parameters for MLMC from the coarsest and finest step sizes.

        The default strategy interpolates in log-space between step_range[0] and step_range[1]
        across n_levels and returns a list of single-entry lists containing the step for each level.

        :param n_levels: int number of MLMC levels.
        :param step_range: Sequence [coarse_step, fine_step] where coarse_step > fine_step.
        :return: list of lists where each inner list contains the parameter(s) for that level.
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
