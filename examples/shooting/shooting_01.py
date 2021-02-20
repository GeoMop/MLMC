import numpy as np
import mlmc.estimator
from mlmc.sampler import Sampler
from mlmc.sample_storage import Memory
from mlmc.sampling_pool import OneProcessPool
from examples.shooting.new_simulation_shooting import ShootingSimulation
from mlmc.quantity import make_root_quantity
from mlmc.quantity_estimate import moments, estimate_mean
from mlmc.moments import Legendre

# Simple example with OneProcessPool, MultiProcessPool, also show usage of different storages - Memory(), HDFStorage()


class ProcessShooting:

    def __init__(self):
        n_levels = 2
        # Number of MLMC levels
        step_range = [0.05, 0.005]
        # step_range [simulation step at the coarsest level, simulation step at the finest level]
        # Determine level parameters at each level (In this case, simulation step at each level) are set automatically
        level_parameters = ProcessShooting.determine_level_parameters(n_levels, step_range)
        # Determine number of samples at each level

        """
        Run MLMC
        """
        # Create sampler (mlmc.Sampler instance) - crucial class which actually schedule samples
        sampler = self.create_sampler(level_parameters=level_parameters)
        # Schedule samples
        self.generate_jobs(sampler, n_samples=None, target_var=1e-5)
        #self.generate_jobs(sampler, n_samples=[500, 500], renew=renew, target_var=1e-5)
        self.all_collect(sampler)  # Check if all samples are finished

        print("sampler self._n_scheduled_samples ", sampler._n_scheduled_samples)
        print("sampler.n_finished_samples ", sampler.n_finished_samples)

        self.process_results(sampler)

    def process_results(self, sampler):

        n_moments = 25

        sample_storage = sampler.sample_storage
        # Load result format from sample storage
        result_format = sample_storage.load_result_format()
        # Create quantity instance representing your real quantity of interest
        root_quantity = make_root_quantity(sample_storage, result_format)

        # You can access item of quantity according to result format
        conductivity = root_quantity['position']
        time = conductivity[10]  # times: [1]
        location = time['y']  # locations: ['0']
        q_value = location[0]

        print("q value ", q_value)

        # Compute moments
        quantile = 0.001
        estimated_domain = mlmc.estimator.Estimate.estimate_domain(q_value, sample_storage, quantile=quantile)
        moments_fn = Legendre(n_moments, estimated_domain)

        # Create estimator for your quantity
        estimator = mlmc.estimator.Estimate(quantity=q_value, sample_storage=sample_storage, moments_fn=moments_fn)
        means, vars = estimator.estimate_moments(moments_fn)

        print("means ", means)
        print("vars ", vars)

        # There is another possible approach to calculating all moments at once and then choose quantity
        moments_quantity = moments(root_quantity, moments_fn=moments_fn, mom_at_bottom=True)
        moments_mean = estimate_mean(moments_quantity)
        conductivity_mean = moments_mean['position']
        time_mean = conductivity_mean[10]  # times: [1]
        location_mean = time_mean['y']  # locations: ['0']
        value_mean = location_mean[0]  # result shape: (1,)
        assert value_mean.mean[0] == 1

        # true_domain = [-10, 10]  # keep all values on the original domain
        # central_moments = Monomial(self.n_moments, true_domain, ref_domain=true_domain, mean=means())
        # central_moments_quantity = moments(root_quantity, moments_fn=central_moments, mom_at_bottom=True)
        # central_moments_mean = estimate_mean(central_moments_quantity)

        # estimator.sub_subselect(sample_vector=[10000])

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
        simulation_factory = ShootingSimulation(config=simulation_config)

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
