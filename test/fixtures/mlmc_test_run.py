import os.path
import numpy as np
from mlmc.sampler import Sampler
from mlmc.sampling_pool import OneProcessPool
from mlmc.sample_storage_hdf import SampleStorageHDF
from mlmc import moments
import mlmc.tool.plot
import mlmc.estimator
import mlmc.archive.estimate
from mlmc.sim.synth_simulation import SynthSimulation


class MLMCTest:
    def __init__(self, n_levels, n_moments, distr, is_log=False, sim_method=None, quantile=None,
                 moments_class=moments.Legendre, domain=None):
        """
        Create TestMLMC object instance
        :param n_levels: number of levels
        :param n_moments: number of _moments_fn
        :param distr: distribution object
        :param is_log: use logarithm of _moments_fn
        :param sim_method: name of simulation method
        :param quantile: quantiles of domain determination
        :param moments_class: moments_fn class
        :param domain: distr domain
        """
        print("\n")
        print("L: {} R: {} distr: {} sim: {}".format(n_levels, n_moments, distr.dist.__class__.__name__ if 'dist' in distr.__dict__ else '',
                                                     sim_method))

        self.distr = distr
        self.n_levels = n_levels
        self.n_moments = n_moments
        self.is_log = is_log
        self.estimator = None

        step_range = [0.8, 0.01]
        level_parameters = mlmc.estimator.calc_level_params(step_range, n_levels)

        # All levels simulations objects and MLMC object
        self.sampler, self.sim_factory = self.create_sampler(level_parameters, sim_method)

        if domain is not None:
            true_domain = domain
        else:
            if quantile is not None:
                if quantile == 0:
                    if hasattr(distr, "domain"):
                        true_domain = distr.domain
                    else:
                        X = distr.rvs(size=1000)
                        true_domain = (np.min(X), np.max(X))
                else:
                    true_domain = distr.ppf([quantile, 1 - quantile])
            else:
                true_domain = distr.ppf([0.0001, 0.9999])

        self.true_domain = true_domain
        self.moments_fn = moments_class(n_moments, true_domain, log=is_log)

    def result_format(self):
        return self.sim_factory.result_format()

    def set_moments_fn(self, moments_class):
        self.moments_fn = moments_class(self.n_moments, self.true_domain, self.is_log)

    def set_estimator(self, quantity):
        self.estimator = mlmc.estimator.Estimate(quantity=quantity, sample_storage=self.sampler.sample_storage,
                                                 moments_fn=self.moments_fn)

    def create_sampler(self, level_parameters, sim_method=None):
        """
        Create sampler with HDF storage
        :param level_parameters: simulation params for each level
        :param sim_method: simulation method name
        :return: mlmc.sampler.Sampler
        """
        simulation_config = dict(distr=self.distr, complexity=2, nan_fraction=0, sim_method=sim_method)
        simulation_factory = SynthSimulation(simulation_config)
        output_dir = os.path.dirname(os.path.realpath(__file__))

        hdf5_file_name = "mlmc_{}.hdf5".format(self.n_levels)

        if os.path.exists(os.path.join(output_dir, hdf5_file_name)):
            os.remove(os.path.join(output_dir, hdf5_file_name))

        # Create sample storages
        sample_storage = SampleStorageHDF(file_path=os.path.join(output_dir, hdf5_file_name))
        # Create sampling pools
        sampling_pool = OneProcessPool()
        # sampling_pool_dir = OneProcessPool(work_dir=work_dir)
        sampler = Sampler(sample_storage=sample_storage, sampling_pool=sampling_pool, sim_factory=simulation_factory,
                          level_parameters=level_parameters)

        sampler.set_initial_n_samples()
        sampler.schedule_samples()
        sampler.ask_sampling_pool_for_samples()
        return sampler, simulation_factory

    def generate_samples(self, sample_vec=None, target_var=None):
        """
        Generate samples
        :param sample_vec: list, number of samples at each level
        :param target_var: target variance
        :return:
        """
        sample_vec = mlmc.estimator.determine_sample_vec(self.sampler.sample_storage.get_n_collected(),
                                                        self.sampler.n_levels, sample_vector=sample_vec)
        # generate samples
        self.sampler.set_initial_n_samples(sample_vec)
        self.sampler.schedule_samples()
        self.sampler.ask_sampling_pool_for_samples()

        if target_var is not None:
            if self.estimator is not None:
                # New estimation according to already finished samples
                variances, n_ops = self.estimator.estimate_diff_vars_regression(self.sampler._n_scheduled_samples)
                n_estimated = mlmc.estimator.estimate_n_samples_for_target_variance(target_var, variances, n_ops,
                                                                                    n_levels=self.sampler.n_levels)

                # Loop until number of estimated samples is greater than the number of scheduled samples
                while not self.sampler.process_adding_samples(n_estimated):
                    # New estimation according to already finished samples
                    variances, n_ops = self.estimator.estimate_diff_vars_regression(self.sampler._n_scheduled_samples)
                    n_estimated = mlmc.estimator.estimate_n_samples_for_target_variance(target_var, variances, n_ops,
                                                                                        n_levels=self.sampler.n_levels)
            else:
                print("Set estimator first")

    def clear_subsamples(self):
        for level in self.mc.levels:
            level.sample_indices = None
