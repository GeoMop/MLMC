import os
import shutil
import unittest
import numpy as np
import random
from scipy import stats
from mlmc.sim.simulation import QuantitySpec
from mlmc.sample_storage import Memory
from mlmc.sample_storage_hdf import SampleStorageHDF
from mlmc.quantity_concept import make_root_quantity, estimate_mean, apply, estimate_moment, estimate_moments
from mlmc.sampler import Sampler
from mlmc.moments import Legendre
from mlmc.quantity_estimate import QuantityEstimate
from mlmc.sampling_pool import OneProcessPool, ProcessPool
from mlmc.sim.synth_simulation import SynthSimulationWorkspace
from test.synth_sim_for_tests import SynthSimulationForTests


def _prepare_work_dir():
    work_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '_test_tmp')
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir)

    return work_dir


class QuantityTests(unittest.TestCase):
    def test_basics(self):
        work_dir = _prepare_work_dir()
        sample_storage = SampleStorageHDF(file_path=os.path.join(work_dir, "mlmc.hdf5"))
        result_format, sizes = self.fill_sample_storage(sample_storage)
        root_quantity = make_root_quantity(sample_storage, result_format)

        # results = sample_storage.sample_pairs()
        # print("results ", results)

        means = estimate_mean(root_quantity)
        self.assertEqual(len(means), np.sum(sizes))

        quantity_add = root_quantity + root_quantity
        means_add = estimate_mean(quantity_add)
        assert np.allclose((means + means), means_add)

        length = root_quantity['length']
        means_length = estimate_mean(length)
        assert np.allclose((means[sizes[0]:sizes[0]+sizes[1]]).tolist(), means_length.tolist())

        length_add = quantity_add['length']
        means_length_add = estimate_mean(length_add)
        assert np.allclose(means_length_add, means_length*2)

        depth = root_quantity['depth']
        means_depth = estimate_mean(depth)
        assert np.allclose((means[:sizes[0]]).tolist(), means_depth.tolist())

        # Interpolation in time
        locations = length.time_interpolation(2.5)
        mean_interp_value = estimate_mean(locations)

        # Select position
        position = locations['10']
        mean_position_1 = estimate_mean(position)
        assert np.allclose(mean_interp_value[:len(mean_interp_value)//2], mean_position_1)

        position = locations['20']
        mean_position_2 = estimate_mean(position)
        assert np.allclose(mean_interp_value[len(mean_interp_value)//2:], mean_position_2)

        width = root_quantity['width']
        width_locations = width.time_interpolation(1.2)
        mean_width_interp_value = estimate_mean(width_locations)

        # Select position
        position = width_locations['30']
        mean_position_1 = estimate_mean(position)
        assert np.allclose(mean_width_interp_value[:len(mean_width_interp_value)//2], mean_position_1)

        position = width_locations['40']
        mean_position_2 = estimate_mean(position)
        assert np.allclose(mean_width_interp_value[len(mean_width_interp_value)//2:], mean_position_2)

        quantity_add = root_quantity + root_quantity
        means_add = estimate_mean(quantity_add)
        assert np.allclose((means + means), means_add)
        #
        length = quantity_add['length']
        means_length = estimate_mean(length)
        assert np.allclose((means_add[sizes[0]:sizes[0]+sizes[1]]).tolist(), means_length.tolist())

        width = quantity_add['width']
        means_width = estimate_mean(width)
        assert np.allclose((means_add[sizes[0] + sizes[1]:sizes[0] + sizes[1] + sizes[2]]).tolist(), means_width.tolist())

        const = 5
        const_mult_quantity = const * root_quantity
        const_mult_mean = estimate_mean(const_mult_quantity)
        assert np.allclose((const * means).tolist(), const_mult_mean.tolist())

    def test_binary_operations(self):
        sample_storage = Memory()
        result_format, sizes = self.fill_sample_storage(sample_storage)
        root_quantity = make_root_quantity(sample_storage, result_format)

        means = estimate_mean(root_quantity)
        self.assertEqual(len(means), np.sum(sizes))

        quantity_add = root_quantity + root_quantity
        means_add = estimate_mean(quantity_add)
        assert np.allclose((means + means), means_add)

        quantity_add = root_quantity + root_quantity * 2
        means_add = estimate_mean(quantity_add)
        assert np.allclose((means + means * 2), means_add)

        quantity_add_mult = root_quantity + root_quantity * root_quantity
        means_add = estimate_mean(quantity_add_mult)

        quantity_add = root_quantity + root_quantity + root_quantity
        means_add = estimate_mean(quantity_add)
        assert np.allclose((means + means + means), means_add)

        length = quantity_add['length']
        means_length = estimate_mean(length)
        assert np.allclose((means_add[sizes[0]:sizes[0]+sizes[1]]).tolist(), means_length.tolist())

        width = quantity_add['width']
        means_width = estimate_mean(width)
        assert np.allclose((means_add[sizes[0] + sizes[1]:sizes[0] + sizes[1] + sizes[2]]).tolist(), means_width.tolist())

        const = 5
        const_mult_quantity = const * root_quantity
        const_mult_mean = estimate_mean(const_mult_quantity)
        means = estimate_mean(root_quantity)

        assert np.allclose((const * means).tolist(), const_mult_mean.tolist())

    def test_condition(self):
        sample_storage = Memory()
        result_format, size = self.fill_sample_storage(sample_storage)
        root_quantity = make_root_quantity(sample_storage, result_format)

        # results = sample_storage.sample_pairs()
        # print("results ", results)

        selected_quantity = root_quantity.select(root_quantity < 5)
        selected_quantity_mean = estimate_mean(selected_quantity)
        self.assertEqual(len(selected_quantity_mean), 0)

        #bound root quantity result - select the ones which meet conditions
        q_bounded = root_quantity.select((0 < root_quantity) < 10)
        mean_q_bounded = estimate_mean(q_bounded)

        quantity_add = root_quantity + root_quantity
        q_add_bounded = quantity_add.select((0 < quantity_add) < 20)
        means_add_bounded = estimate_mean(q_add_bounded)
        assert np.allclose((means_add_bounded), mean_q_bounded*2)

        q_bounded = root_quantity.select((10 < root_quantity) < 20)
        mean_q_bounded = estimate_mean(q_bounded)

        quantity_add = root_quantity + root_quantity
        q_add_bounded = quantity_add.select((20 < quantity_add) < 40)
        means_add_bounded = estimate_mean(q_add_bounded)
        assert np.allclose((means_add_bounded), mean_q_bounded * 2)

        length = root_quantity['length']
        mean_length = estimate_mean(length)
        #assert len(mean_q_bounded) == len(mean_length) * 2

        quantity_lt = length.select(length < 10)  # use just first sample
        means_lt = estimate_mean(quantity_lt)
        assert len(mean_length) == len(means_lt)

        quantity_le = length.select(length <= 9)  # use just first sample
        means_le = estimate_mean(quantity_le)
        assert len(mean_length) == len(means_le)

        quantity_lt = length.select(length < 1)  # no sample matches condition
        means_lt = estimate_mean(quantity_lt)
        assert len(means_lt) == 0

        quantity_lt_gt = length.select((9 < length) < 20)  # one sample matches condition
        means_lt_gt = estimate_mean(quantity_lt_gt)
        print("means length ", mean_length)
        print("means lt gt ", means_lt_gt)
        assert len(mean_length) == len(means_lt_gt)

        quantity_gt = length.select(100 < length) # no sample matches condition
        means_gt = estimate_mean(quantity_gt)
        assert len(means_gt) == 0

        quantity_ge = length.select(100 <= length)  # no sample matches condition
        means_ge = estimate_mean(quantity_ge)
        assert len(means_ge) == 0

        quantity_eq = length.select(1 == length)
        means_eq = estimate_mean(quantity_eq)
        assert len(means_eq) == 0

        quantity_ne = length.select(-1 != length)
        means_ne = estimate_mean(quantity_ne)
        assert np.allclose((means_ne).tolist(), mean_length.tolist())

        # Quantity sampling
        root_quantity_subsamples = root_quantity.select(root_quantity.sampling(size=2))
        means_eq = estimate_mean(root_quantity_subsamples)

        root_quantity_subsamples = root_quantity.select(root_quantity.sampling(size=10))
        means_eq = estimate_mean(root_quantity_subsamples)

    def test_functions(self):
        work_dir = _prepare_work_dir()
        sample_storage = SampleStorageHDF(file_path=os.path.join(work_dir, "mlmc.hdf5"))
        result_format, sizes = self.fill_sample_storage(sample_storage)
        root_quantity = make_root_quantity(sample_storage, result_format)

        # results = sample_storage.sample_pairs()
        # print("results ", results)

        root_quantity_means = estimate_mean(root_quantity)

        sin_root_quantity = apply([root_quantity], np.sin)
        sin_means = estimate_mean(sin_root_quantity)
        self.assertEqual(len(sin_means), np.sum(sizes))

        add_root_quantity = apply([root_quantity, root_quantity], np.add)  # Add arguments element-wise.
        add_root_quantity_means = estimate_mean(add_root_quantity)
        assert np.allclose(add_root_quantity_means.tolist(), (root_quantity_means * 2).tolist())

        max_root_quantity = apply([root_quantity, root_quantity], np.maximum)  # Element-wise maximum of array elements.
        max_root_quantity_means = estimate_mean(max_root_quantity)
        assert np.allclose(max_root_quantity_means.tolist(), (root_quantity_means).tolist())

        # matmul_root_quantity = apply([root_quantity, root_quantity], numpy_matmul)  # Return the cross product of two (arrays of) vectors.
        # matmul_root_quantity_means = estimate_mean(matmul_root_quantity)
        # print("matmul_root_quantity_means ", matmul_root_quantity_means)
        # print("matmul_root_quantity_means.shape ", matmul_root_quantity_means.shape)
        #assert np.allclose(matmul_root_quantity_means.tolist(), (root_quantity_means).tolist())

        # cross_root_quantity = apply([root_quantity, root_quantity], np.cross)  # Return the cross product of two (arrays of) vectors.
        # cross_root_quantity_means = estimate_mean(cross_root_quantity)
        # print("cross_root_quantity_means ", cross_root_quantity_means)
        # print("cross_root_quantity_means.shape ", cross_root_quantity_means.shape)
        # assert np.allclose(cross_root_quantity_means.tolist(), (root_quantity_means).tolist())

        length = root_quantity['length']
        sin_length = apply([length], np.sin)
        sin_means_length = estimate_mean(sin_length)
        assert np.allclose((sin_means[sizes[0]:sizes[0]+sizes[1]]).tolist(), sin_means_length.tolist())

        # Root quantity and length have different structure
        self.assertRaises(AssertionError, apply, [root_quantity, length], np.add)

    def fill_sample_storage(self, sample_storage):
        np.random.seed(123)
        n_levels = 3
        res_length = 3
        result_format = [
            QuantitySpec(name="depth", unit="mm", shape=(2, res_length - 2+1), times=[1, 2, 3], locations=['30', '40']),
            QuantitySpec(name="length", unit="m", shape=(2, res_length - 2+2), times=[1, 2, 3], locations=['10', '20']),
            QuantitySpec(name="width", unit="mm", shape=(2, res_length - 2+3), times=[1, 2, 3], locations=['30', '40'])
        ]

        sample_storage.save_global_data(result_format=result_format, level_parameters=np.ones(n_levels))

        successful_samples = {}
        failed_samples = {}
        n_ops = {}
        n_successful = 5
        sizes = []
        for l_id in range(n_levels):
            sizes = []
            for quantity_spec in result_format:
                sizes.append(np.prod(quantity_spec.shape) * len(quantity_spec.times) * len(quantity_spec.locations))

            # Dict[level_id, List[Tuple[sample_id:str, Tuple[fine_result: ndarray, coarse_result: ndarray]]]]
            successful_samples[l_id] = []
            for sample_id in range(n_successful):
                fine_result = np.random.randint(5 + 5*sample_id, high=5+5*(1+sample_id),
                                                size=(np.sum(sizes),))

                if l_id == 0:
                    coarse_result = (np.zeros((np.sum(sizes),)))
                else:
                    coarse_result = (np.random.randint(5 + 5*sample_id, high=5+5*(1+sample_id),
                                                       size=(np.sum(sizes),)))

                successful_samples[l_id].append((str(sample_id), (fine_result, coarse_result)))

            n_ops[l_id] = [random.random(), n_successful]

            sample_storage.save_scheduled_samples(l_id, samples=["S{:07d}".format(i) for i in range(n_successful)])

        sample_storage.save_samples(successful_samples, failed_samples)
        sample_storage.save_n_ops(list(n_ops.items()))

        return result_format, sizes

    def _create_sampler(self, step_range):
        # Set work dir
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        work_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '_test_tmp')
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)
        os.makedirs(work_dir)

        # Create simulations
        failed_fraction = 0.1
        distr = stats.norm()
        simulation_config = dict(distr=distr, complexity=2, nan_fraction=failed_fraction, sim_method='_sample_fn')
        simulation_factory = SynthSimulationForTests(simulation_config)
        # shutil.copyfile('synth_sim_config.yaml', os.path.join(work_dir, 'synth_sim_config.yaml'))
        # simulation_config = {"config_yaml": os.path.join(work_dir, 'synth_sim_config.yaml')}
        # simulation_workspace = SynthSimulationWorkspace(simulation_config)

        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        work_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '_test_tmp')
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)
        os.makedirs(work_dir)

        # Create sample storages
        sample_storage = SampleStorageHDF(file_path=os.path.join(work_dir, "mlmc_test.hdf5"))

        # Create sampling pools
        sampling_pool = OneProcessPool()
        # sampling_pool_dir = OneProcessPool(work_dir=work_dir)

        if sampling_pool._output_dir is not None:
            if os.path.exists(work_dir):
                shutil.rmtree(work_dir)
            os.makedirs(work_dir)
        if simulation_factory.need_workspace:
            os.chdir(os.path.dirname(os.path.realpath(__file__)))
            shutil.copyfile('synth_sim_config.yaml', os.path.join(work_dir, 'synth_sim_config.yaml'))

        sampler = Sampler(sample_storage=sample_storage, sampling_pool=sampling_pool, sim_factory=simulation_factory,
                          level_parameters=step_range)

        return sampler, simulation_factory

    # def test_simple_moments(self):
    #     sample_storage = SampleStorageHDF(file_path=os.path.join(work_dir, "mlmc.hdf5"))
    #     result_format, sizes = self.fill_sample_storage(sample_storage)
    #     root_quantity = make_root_quantity(sample_storage, result_format)
    #
    #     results = sample_storage.sample_pairs()
    #     print("results ", results)
    #
    #     means = estimate_mean(root_quantity)
    #     self.assertEqual(len(means), np.sum(sizes))
    #
    #     n_moments = 5
    #     distr = stats.norm()
    #     true_domain = distr.ppf([0.0001, 0.9999])
    #     moments_fn = Legendre(n_moments, true_domain)
    #
    #     moment_0_quantity = estimate_moment(root_quantity, moments_fn=moments_fn, i=0)
    #     moments_mean = estimate_mean(moment_0_quantity)
    #     print("moments mean ", moments_mean)

    def test_moments(self):
        np.random.seed(1234)
        n_moments = 5
        step_range = [[0.1], [0.001]]
        #step_range = [[0.1]]

        sampler, simulation_factory = self._create_sampler(step_range)

        distr = stats.norm()
        true_domain = distr.ppf([0.0001, 0.9999])
        moments_fn = Legendre(n_moments, true_domain)
        # moments_fn = Monomial(n_moments, true_domain)

        sampler.set_initial_n_samples([5, 5])
        sampler.schedule_samples()
        sampler.ask_sampling_pool_for_samples()

        # results = sampler.sample_storage.sample_pairs()
        # print("results ", results)

        q_estimator = QuantityEstimate(sample_storage=sampler.sample_storage, moments_fn=moments_fn,
                                       sim_steps=step_range)
        means, vars = q_estimator.estimate_moments(moments_fn)

        root_quantity = make_root_quantity(storage=sampler.sample_storage, q_specs=simulation_factory.result_format())

        moment_quantity = estimate_moment(root_quantity, moments_fn=moments_fn, i=0)
        moment_mean = estimate_mean(moment_quantity)

        moments_quantity = estimate_moments(root_quantity, moments_fn=moments_fn)
        moments_mean = estimate_mean(moments_quantity)

        new_moments = moments_quantity + moments_quantity
        new_moments_mean = estimate_mean(new_moments)

        assert np.allclose(moments_mean + moments_mean, new_moments_mean)
        assert np.allclose(moment_mean, moments_mean[0], np.ones((len(moment_mean), )))
        assert np.allclose(means, moments_mean[:, 0])

        selected_new_moments = new_moments.select(new_moments > -10)
        selected_new_moments_mean = estimate_mean(selected_new_moments)
        assert np.allclose(new_moments_mean, selected_new_moments_mean)


if __name__ == '__main__':
    unittest.main()
