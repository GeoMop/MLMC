import os
import sys
import shutil
import numpy as np
from scipy import stats

from mlmc.synth_simulation import SynthSimulation, SynthSimulationWorkspace
from mlmc.sampler import Sampler
from mlmc.sample_storage_hdf import SampleStorageHDF
from mlmc.sampling_pool import OneProcessPool, ProcessPool
from mlmc.sampling_pool_pbs import SamplingPoolPBS
from mlmc.moments import Legendre
from mlmc.quantity_estimate import QuantityEstimate


class ProcessPBS:

    def __init__(self):
        args = self.get_arguments(sys.argv[1:])

        print("args ", args)

        self.step_range = (1, 0.01)

        self.work_dir = args.work_dir
        self.force = args.force
        self.append = False

        if args.command == 'run':
            print("---RUN")
            self.run()
        elif args.command == 'collect':
            print("---COLLECT")
            self.append = True
            self.force = False
            self.run()
        elif args.command == 'resurrect':
            print("---RESURRECT")
            self.force = False
            self.append = True
            self.run(resurrect=True)

    def get_arguments(self, arguments):
        """
        Getting arguments from console
        :param arguments: list of arguments
        :return: namespace
        """
        import argparse
        parser = argparse.ArgumentParser()

        parser.add_argument('command', choices=['run', 'collect', 'resurrect'], help='run - create new execution,'
                                                                                     'collect - keep collected'
                                                                                     'resurrect - resurrect failed samples')
        parser.add_argument('work_dir', help='Work directory')
        parser.add_argument("-f", "--force", default=False, action='store_true', help="Clean before run")
        parser.add_argument("-c", "--clean", default=False, action='store_true', help="Clean auxiliary structures")

        args = parser.parse_args(arguments)

        return args

    def run(self, resurrect=False):
        np.random.seed(3)
        n_moments = 5
        distr = stats.norm(loc=1, scale=2)
        step_range = [0.01, 0.001]

        # Set work dir
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        shutil.copyfile('synth_sim_config.yaml', os.path.join(self.work_dir, 'synth_sim_config.yaml'))

        simulation_config = {"config_yaml": os.path.join(self.work_dir, 'synth_sim_config.yaml')}
        simulation_factory = SynthSimulationWorkspace(simulation_config)

        if self.force:
            file_path = os.path.join(self.work_dir, "mlmc_{}.hdf5".format(len(step_range)))
            if os.path.exists(file_path):
                os.remove(os.path.join(self.work_dir, "mlmc_{}.hdf5".format(len(step_range))))

        sample_storage = SampleStorageHDF(file_path=os.path.join(self.work_dir, "mlmc_{}.hdf5".format(len(step_range))),
                                          append=self.append)
        sampling_pool = SamplingPoolPBS(job_weight=20000000, work_dir=self.work_dir, force=self.force)

        pbs_config = dict(
            n_cores=1,
            n_nodes=1,
            select_flags=['cgroups=cpuacct'],
            mem='128mb',
            queue='charon_2h',
            home_dir='/storage/liberec3-tul/home/martin_spetlik/',
            pbs_process_file_dir='/auto/liberec3-tul/home/martin_spetlik/MLMC_new_design/src/mlmc',
            python='python3',
            modules=['module use /storage/praha1/home/jan-hybs/modules',
                     'module load python36-modules-gcc',
                     'module list']
        )

        sampling_pool.pbs_common_setting(flow_3=True, **pbs_config)

        # Plan and compute samples
        sampler = Sampler(sample_storage=sample_storage, sampling_pool=sampling_pool, sim_factory=simulation_factory,
                          step_range=step_range)

        true_domain = distr.ppf([0.0001, 0.9999])
        moments_fn = Legendre(n_moments, true_domain)

        if resurrect:
            sampler.ask_sampling_pool_for_samples()
            sampler.resurrect_failed()
            sampler.ask_sampling_pool_for_samples()
        else:
            sampler.set_initial_n_samples([12, 6])
            # sampler.set_initial_n_samples([1000])
            sampler.schedule_samples()
            sampler.ask_sampling_pool_for_samples()

        q_estimator = QuantityEstimate(sample_storage=sample_storage, moments_fn=moments_fn, sim_steps=step_range)

        # target_var = 1e-3
        # sleep = 0
        # add_coef = 0.1
        #
        # # @TODO: test
        # # New estimation according to already finished samples
        # variances, n_ops = q_estimator.estimate_diff_vars_regression(sampler._n_scheduled_samples)
        # n_estimated = new_estimator.estimate_n_samples_for_target_variance(target_var, variances, n_ops,
        #                                                                    n_levels=sampler.n_levels)
        # # Loop until number of estimated samples is greater than the number of scheduled samples
        # while not sampler.process_adding_samples(n_estimated, sleep, add_coef):
        #     # New estimation according to already finished samples
        #     variances, n_ops = q_estimator.estimate_diff_vars_regression(sampler._n_scheduled_samples)
        #     n_estimated = new_estimator.estimate_n_samples_for_target_variance(target_var, variances, n_ops,
        #                                                                        n_levels=sampler.n_levels)

        # print("collected samples ", sampler._n_created_samples)
        means, vars = q_estimator.estimate_moments(moments_fn)

        print("means ", means)
        print("vars ", vars)
        # assert means[0] == 1
        # assert np.isclose(means[1], 0, atol=1e-2)
        # assert vars[0] == 0


class Process:

    def __init__(self):
        args = self.get_arguments(sys.argv[1:])

        print("args ", args)

        self.step_range = (1, 0.01)

        self.work_dir = args.work_dir
        self.force = True
        self.append = False

        if args.command == 'run':
            print("---RUN")
            self.run()
        elif args.command == 'collect':
            print("---COLLECT")
            self.append = True
            self.force = False
            self.run()
        elif args.command == 'resurrect':
            print("---RESURRECT")
            self.force = False
            self.append = True
            self.run(resurrect=True)

    def get_arguments(self, arguments):
        """
        Getting arguments from console
        :param arguments: list of arguments
        :return: namespace
        """
        import argparse
        parser = argparse.ArgumentParser()

        parser.add_argument('command', choices=['run', 'collect', 'resurrect'], help='run - create new execution,'
                                                                                     'collect - keep collected'
                                                                                     'resurrect - resurrect failed samples')
        parser.add_argument('work_dir', help='Work directory')
        parser.add_argument("-f", "--force", default=False, action='store_true', help="Clean before run")
        parser.add_argument("-c", "--clean", default=False, action='store_true', help="Clean auxiliary structures")

        args = parser.parse_args(arguments)

        return args

    def run(self, resurrect=False):
        np.random.seed(3)
        n_moments = 5
        failed_fraction = 0.1

        # work_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '_test_tmp')
        # if os.path.exists(work_dir):
        #     shutil.rmtree(work_dir)
        # os.makedirs(work_dir)

        distr = stats.norm()
        step_range = [0.1, 0.001]

        # User configure and create simulation instance
        simulation_config = dict(distr=distr, complexity=2, nan_fraction=failed_fraction, sim_method='_sample_fn')
        #simulation_config = {"config_yaml": 'synth_sim_config.yaml'}
        simulation_factory = SynthSimulation(simulation_config)

        if self.force:
            os.remove(os.path.join(self.work_dir, "mlmc_{}.hdf5".format(len(step_range))))

        sample_storage = SampleStorageHDF(file_path=os.path.join(self.work_dir, "mlmc_{}.hdf5".format(len(step_range))),
                                          append=self.append)
        sampling_pool = OneProcessPool()

        # Plan and compute samples
        sampler = Sampler(sample_storage=sample_storage, sampling_pool=sampling_pool, sim_factory=simulation_factory,
                          step_range=step_range)

        true_domain = distr.ppf([0.0001, 0.9999])
        moments_fn = Legendre(n_moments, true_domain)
        # moments_fn = Monomial(n_moments, true_domain)

        if resurrect:
            sampler.ask_sampling_pool_for_samples()
            sampler.resurrect_failed()
            sampler.ask_sampling_pool_for_samples()
        else:
            sampler.set_initial_n_samples([12, 6])
            # sampler.set_initial_n_samples([1000])
            sampler.schedule_samples()
            sampler.ask_sampling_pool_for_samples()

        q_estimator = QuantityEstimate(sample_storage=sample_storage, moments_fn=moments_fn, sim_steps=step_range)
        #
        target_var = 1e-4
        sleep = 0
        add_coef = 0.1

        # # @TODO: test
        # # New estimation according to already finished samples
        # variances, n_ops = q_estimator.estimate_diff_vars_regression(sampler._n_scheduled_samples)
        # n_estimated = new_estimator.estimate_n_samples_for_target_variance(target_var, variances, n_ops,
        #                                                                    n_levels=sampler.n_levels)
        #
        # # Loop until number of estimated samples is greater than the number of scheduled samples
        # while not sampler.process_adding_samples(n_estimated, sleep, add_coef):
        #     # New estimation according to already finished samples
        #     variances, n_ops = q_estimator.estimate_diff_vars_regression(sampler._n_scheduled_samples)
        #     n_estimated = new_estimator.estimate_n_samples_for_target_variance(target_var, variances, n_ops,
        #                                                                        n_levels=sampler.n_levels)

        print("collected samples ", sampler._n_scheduled_samples)
        means, vars = q_estimator.estimate_moments(moments_fn)

        print("means ", means)
        print("vars ", vars)
        assert means[0] == 1
        assert np.isclose(means[1], 0, atol=1e-2)
        assert vars[0] == 0
        # sampler.schedule_samples()
        # sampler.ask_sampling_pool_for_samples()
        #
        # storage = sampler.sample_storage
        # results = storage.sample_pairs()


if __name__ == "__main__":
    process = ProcessPBS()
