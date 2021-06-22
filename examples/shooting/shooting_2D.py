import numpy as np
import mlmc.estimator
from mlmc.sampler import Sampler
from mlmc.sample_storage import Memory
from mlmc.sampling_pool import OneProcessPool
from examples.shooting.simulation_shooting_2D import ShootingSimulation2D
from mlmc.quantity.quantity import make_root_quantity
from mlmc.quantity.quantity_estimate import moments, estimate_mean
from mlmc.moments import Legendre

# Simple example with OneProcessPool, MultiProcessPool, also show usage of different storages - Memory(), HDFStorage()


class ProcessShooting2D:

    def __init__(self):
        n_levels = 5
        # Number of MLMC levels
        step_range = [0.05, 0.005]
        # step_range [simulation step at the coarsest level, simulation step at the finest level]
        # Determine level parameters at each level (In this case, simulation step at each level) are set automatically
        level_parameters = ProcessShooting2D.determine_level_parameters(n_levels, step_range)
        # Determine number of samples at each level

        """
        Run MLMC
        """
        # Create sampler (mlmc.Sampler instance) - crucial class which actually schedule samples
        sampler = self.create_sampler(level_parameters=level_parameters)
        # Schedule samples
        self.generate_jobs(sampler, n_samples=None, target_var=1e-5)
        #self.generate_jobs(sampler, n_samples=[1000],  target_var=1e-5)
        self.all_collect(sampler)  # Check if all samples are finished
        self.process_results(sampler, n_levels)

    def create_estimator(self, quantity, sample_storage, n_moments=25, quantile=0.001):
        estimated_domain = mlmc.estimator.Estimate.estimate_domain(quantity, sample_storage,
                                                                     quantile=quantile)
        moments_fn = Legendre(n_moments, estimated_domain)
        # Create estimator for your quantity
        return mlmc.estimator.Estimate(quantity=quantity, sample_storage=sample_storage,
                                              moments_fn=moments_fn)

    def process_results(self, sampler, n_levels):
        n_moments = 10
        sample_storage = sampler.sample_storage
        # Load result format from sample storage
        result_format = sample_storage.load_result_format()
        # Create quantity instance representing your real quantity of interest
        root_quantity = make_root_quantity(sample_storage, result_format)

        # You can access item of quantity according to result format
        target = root_quantity['target']
        time = target[10]  # times: [1]
        position = time['0']  # locations: ['0']
        x_quantity_value = position[0]
        y_quantity_value = position[1]

        quantile = 0.001
        x_estimator = self.create_estimator(x_quantity_value, sample_storage, quantile=quantile)
        y_estimator = self.create_estimator(y_quantity_value, sample_storage, quantile=quantile)

        root_quantity_estimated_domain = mlmc.estimator.Estimate.estimate_domain(root_quantity, sample_storage, quantile=quantile)
        root_quantity_moments_fn = Legendre(n_moments, root_quantity_estimated_domain)

        # There is another possible approach to calculating all moments at once and then choose quantity
        moments_quantity = moments(root_quantity, moments_fn=root_quantity_moments_fn, mom_at_bottom=True)
        moments_mean = estimate_mean(moments_quantity)
        target_mean = moments_mean['target']
        time_mean = target_mean[10]  # times: [1]
        location_mean = time_mean['0']  # locations: ['0']
        value_mean = location_mean[0]  # result shape: (1,)

        self.approx_distribution(x_estimator, n_levels, tol=1e-8)
        self.approx_distribution(y_estimator, n_levels, tol=1e-8)

    def approx_distribution(self, estimator, n_levels, tol=1.95):
        """
        Construct approximation of the density using given moment functions.
        :param estimator: mlmc.estimator.Estimate instance, it contains quantity for which the density is reconstructed
        :param tol: Tolerance of the fitting problem, with account for variances in moments
        :return: None
        """
        distr_obj, result, _, _ = estimator.construct_density(tol=tol)
        distr_plot = mlmc.plot.plots.Distribution(title="distributions", error_plot=None)
        distr_plot.add_distribution(distr_obj)

        if n_levels == 1:
            samples = estimator.get_level_samples(level_id=0)[..., 0]
            distr_plot.add_raw_samples(np.squeeze(samples))
        distr_plot.show(None)
        distr_plot.reset()

    def create_sampler(self, level_parameters):
        """
        Simulation dependent configuration
        :return: mlmc.sampler instance
        """
        # Create Pbs sampling pool
        sampling_pool = OneProcessPool()

        simulation_config = {
            "start_position": np.array([0, 0]),
            "start_velocity": np.array([10, 0]),
            "area_borders":  np.array([-100, 200, -300, 400]),
            "max_time": 10,
            "complexity": 2,  # used for initial estimate of number of operations per sample
            'fields_params': dict(model='gauss', dim=1, sigma=1, corr_length=0.1),
        }

        # Create simulation factory
        simulation_factory = ShootingSimulation2D(config=simulation_config)
        # Create HDF sample storage
        sample_storage = Memory()
        # Create sampler, it manages sample scheduling and so on
        sampler = Sampler(sample_storage=sample_storage, sampling_pool=sampling_pool, sim_factory=simulation_factory,
                          level_parameters=level_parameters)

        return sampler

    def generate_jobs(self, sampler, n_samples=None, renew=False, target_var=None):
        """
        Generate level samples
        :param n_samples: None or list, number of samples for each level
        :param renew: rerun failed samples with same random seed (= same sample id)
        :return: None
        """
        if n_samples is not None:
            sampler.set_initial_n_samples(n_samples)
        else:
            sampler.set_initial_n_samples()
        sampler.schedule_samples()
        sampler.ask_sampling_pool_for_samples()

    def all_collect(self, sampler):
        """
        Collect samples
        :param sampler: mlmc.Sampler object
        :return: None
        """
        running = 1
        while running > 0:
            running = 0
            running += sampler.ask_sampling_pool_for_samples()
            print("N running: ", running)

    @staticmethod
    def determine_level_parameters(n_levels, step_range):
        """
        Determine level parameters,
        In this case, a step of fine simulation at each level
        :param n_levels: number of MLMC levels
        :param step_range: simulation step range
        :return: List of lists
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
