import os
import shutil
import unittest
import numpy as np
import random
from scipy import stats
from mlmc.quantity.quantity_spec import QuantitySpec
from mlmc.sample_storage import Memory
from mlmc.sample_storage_hdf import SampleStorageHDF
from mlmc.quantity.quantity import make_root_quantity
from mlmc.quantity.quantity_estimate import estimate_mean, moment, moments, covariance, cache_clear
from mlmc import Quantity, QuantityConst
from mlmc import ScalarType
from mlmc.sampler import Sampler
from mlmc.moments import Monomial
from mlmc.sampling_pool import OneProcessPool
from test.synth_sim_for_tests import SynthSimulationForTests
import mlmc.estimator


def _prepare_work_dir():
    work_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '_test_tmp')
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir)

    return work_dir


class QuantityTests(unittest.TestCase):

    def test_basics(self):
        """
        Test basic quantity properties, especially indexing
        """
        work_dir = _prepare_work_dir()
        sample_storage = SampleStorageHDF(file_path=os.path.join(work_dir, "mlmc.hdf5"))
        result_format, sizes = self.fill_sample_storage(sample_storage)
        root_quantity = make_root_quantity(sample_storage, result_format)

        means = estimate_mean(root_quantity)
        self.assertEqual(len(means.mean), np.sum(sizes))

        quantity_add = root_quantity + root_quantity
        means_add = estimate_mean(quantity_add)
        assert np.allclose((means.mean + means.mean), means_add.mean)

        length = root_quantity['length']
        means_length = estimate_mean(length)
        assert np.allclose((means.mean[sizes[0]:sizes[0] + sizes[1]]).tolist(), means_length.mean.tolist())

        length_add = quantity_add['length']
        means_length_add = estimate_mean(length_add)
        assert np.allclose(means_length_add.mean, means_length.mean * 2)

        depth = root_quantity['depth']
        means_depth = estimate_mean(depth)
        assert np.allclose((means.mean[:sizes[0]]), means_depth.mean)

        # Interpolation in time
        locations = length.time_interpolation(2.5)
        mean_interp_value = estimate_mean(locations)

        # Select position
        position = locations['10']
        mean_position_1 = estimate_mean(position)
        assert np.allclose(mean_interp_value.mean[:len(mean_interp_value.mean) // 2], mean_position_1.mean.flatten())

        # Array indexing tests
        values = position
        values_mean = estimate_mean(values)
        assert values_mean[1:2].mean.shape == (1, 3)

        values = position
        values_mean = estimate_mean(values)
        assert values_mean[1].mean.shape == (3,)

        values = position[:, 2]
        values_mean = estimate_mean(values)
        assert len(values_mean.mean) == 2

        y = position[1, 2]
        y_mean = estimate_mean(y)
        assert len(y_mean.mean) == 1

        y = position[:, :]
        y_mean = estimate_mean(y)
        assert np.allclose(y_mean.mean, mean_position_1.mean)

        y = position[:1, 1:2]
        y_mean = estimate_mean(y)
        assert len(y_mean.mean) == 1

        y = position[:2, ...]
        y_mean = estimate_mean(y)
        assert len(y_mean.mean.flatten()) == 6

        value = values[1]
        value_mean = estimate_mean(value)
        assert values_mean.mean[1] == value_mean.mean

        value = values[0]
        value_mean = estimate_mean(value)
        assert values_mean.mean[0] == value_mean.mean

        position = locations['20']
        mean_position_2 = estimate_mean(position)
        assert np.allclose(mean_interp_value.mean[len(mean_interp_value.mean) // 2:], mean_position_2.mean.flatten())

        width = root_quantity['width']
        width_locations = width.time_interpolation(1.2)
        mean_width_interp_value = estimate_mean(width_locations)

        # Select position
        position = width_locations['30']
        mean_position_1 = estimate_mean(position)
        assert np.allclose(mean_width_interp_value.mean[:len(mean_width_interp_value.mean) // 2],
                           mean_position_1.mean.flatten())

        position = width_locations['40']
        mean_position_2 = estimate_mean(position)
        assert np.allclose(mean_width_interp_value.mean[len(mean_width_interp_value.mean) // 2:],
                           mean_position_2.mean.flatten())

        quantity_add = root_quantity + root_quantity
        means_add = estimate_mean(quantity_add)
        assert np.allclose((means.mean + means.mean), means_add.mean)

        length = quantity_add['length']
        means_length = estimate_mean(length)
        assert np.allclose((means_add.mean[sizes[0]:sizes[0] + sizes[1]]).tolist(), means_length.mean.tolist())

        width = quantity_add['width']
        means_width = estimate_mean(width)
        assert np.allclose((means_add.mean[sizes[0] + sizes[1]:sizes[0] + sizes[1] + sizes[2]]).tolist(),
                           means_width.mean.tolist())

        # Concatenate quantities
        quantity_dict = Quantity.QDict([("depth", depth), ("length", length)])
        quantity_dict_mean = estimate_mean(quantity_dict)
        assert np.allclose(quantity_dict_mean.mean, np.concatenate((means_depth.mean, means_length.mean)))

        length_concat = quantity_dict['length']
        means_length_concat = estimate_mean(length_concat)
        assert np.allclose(means_length_concat.mean, means_length.mean)
        locations = length_concat.time_interpolation(2.5)
        mean_interp_value = estimate_mean(locations)
        position = locations['10']
        mean_position_1 = estimate_mean(position)
        assert np.allclose(mean_interp_value.mean[:len(mean_interp_value.mean) // 2], mean_position_1.mean.flatten())
        values = position[:, 2]
        values_mean = estimate_mean(values)
        assert len(values_mean.mean) == 2
        y = position[1, 2]
        y_mean = estimate_mean(y)
        assert len(y_mean.mean) == 1
        y_add = np.add(5, y)
        y_add_mean = estimate_mean(y_add)
        assert np.allclose(y_add_mean.mean, y_mean.mean + 5)
        depth = quantity_dict['depth']
        means_depth_concat = estimate_mean(depth)
        assert np.allclose((means.mean[:sizes[0]]), means_depth_concat.mean)

        quantity_array = Quantity.QArray([[length, length], [length, length]])
        quantity_array_mean = estimate_mean(quantity_array)
        assert np.allclose(quantity_array_mean.mean.flatten(), np.concatenate((means_length.mean, means_length.mean,
                                                                            means_length.mean, means_length.mean)))

        quantity_timeseries = Quantity.QTimeSeries([(0, locations), (1, locations)])
        quantity_timeseries_mean = estimate_mean(quantity_timeseries)
        assert np.allclose(quantity_timeseries_mean.mean, np.concatenate((mean_interp_value.mean, mean_interp_value.mean)))

        quantity_field = Quantity.QField([("f1", length), ("f2", length)])
        quantity_field_mean = estimate_mean(quantity_field)
        assert np.allclose(quantity_field_mean.mean, np.concatenate((means_length.mean, means_length.mean)))

    def test_binary_operations(self):
        """
        Test quantity binary operations
        """
        work_dir = _prepare_work_dir()
        sample_storage = SampleStorageHDF(file_path=os.path.join(work_dir, "mlmc.hdf5"))
        result_format, sizes = self.fill_sample_storage(sample_storage)
        root_quantity = make_root_quantity(sample_storage, result_format)
        const = 5

        means = estimate_mean(root_quantity)
        self.assertEqual(len(means.mean), np.sum(sizes))

        # Addition
        quantity_add = root_quantity + root_quantity
        means_add = estimate_mean(quantity_add)
        assert np.allclose((means.mean + means.mean), means_add.mean)

        quantity_add_const = root_quantity + const
        means_add_const = estimate_mean(quantity_add_const)
        means_add_const.mean

        quantity_add = root_quantity + root_quantity + root_quantity
        means_add = estimate_mean(quantity_add)
        assert np.allclose((means.mean + means.mean + means.mean), means_add.mean)

        # Subtraction
        quantity_sub_const = root_quantity - const
        means_sub_const = estimate_mean(quantity_sub_const)
        means_sub_const.mean

        # Multiplication
        const_mult_quantity = root_quantity * const
        const_mult_mean = estimate_mean(const_mult_quantity)
        assert np.allclose((const * means.mean).tolist(), const_mult_mean.mean.tolist())

        # True division
        const_div_quantity = root_quantity / const
        const_div_mean = estimate_mean(const_div_quantity)
        assert np.allclose((means.mean/const).tolist(), const_div_mean.mean.tolist())

        # Mod
        const_mod_quantity = root_quantity % const
        const_mod_mean = estimate_mean(const_mod_quantity)
        const_mod_mean.mean

        # Further tests
        length = quantity_add['length']
        means_length = estimate_mean(length)
        assert np.allclose(means_add.mean[sizes[0]:sizes[0] + sizes[1]], means_length.mean)

        width = quantity_add['width']
        means_width = estimate_mean(width)
        assert np.allclose(means_add.mean[sizes[0] + sizes[1]:sizes[0] + sizes[1] + sizes[2]], means_width.mean)

        quantity_add = root_quantity + root_quantity * const
        means_add = estimate_mean(quantity_add)
        assert np.allclose((means.mean + means.mean * const), means_add.mean)

        quantity_add_mult = root_quantity + root_quantity * root_quantity
        means_add = estimate_mean(quantity_add_mult)

        #### right operators ####
        # Addition
        const_add_quantity = const + root_quantity
        const_add_means = estimate_mean(const_add_quantity)
        assert np.allclose(means_add_const.mean, const_add_means.mean)

        # Subtraction
        const_sub_quantity = const - root_quantity
        const_sub_means = estimate_mean(const_sub_quantity)
        assert np.allclose(means_sub_const.mean, -const_sub_means.mean)

        # Multiplication
        const_mult_quantity = const * root_quantity
        const_mult_mean = estimate_mean(const_mult_quantity)
        assert np.allclose((const * means.mean), const_mult_mean.mean)

        # True division
        const_div_quantity = const / root_quantity
        const_div_mean = estimate_mean(const_div_quantity)
        assert len(const_div_mean.mean) == len(means.mean)

        # Mod
        const_mod_quantity = const % root_quantity
        const_mod_mean = estimate_mean(const_mod_quantity)
        assert len(const_mod_mean.mean) == len(means.mean)

    def test_condition(self):
        """
        Test select method
        """
        sample_storage = Memory()
        result_format, size = self.fill_sample_storage(sample_storage)
        root_quantity = make_root_quantity(sample_storage, result_format)

        root_quantity_mean = estimate_mean(root_quantity)

        all_root_quantity = root_quantity.select(np.logical_or(0 < root_quantity, root_quantity < 10))
        all_root_quantity_mean = estimate_mean(all_root_quantity)
        assert np.allclose(root_quantity_mean.mean, all_root_quantity_mean.mean)

        selected_quantity = root_quantity.select(root_quantity < 0)
        with self.assertRaises(Exception):
            estimate_mean(selected_quantity)

        all_root_quantity = root_quantity.select(0 < root_quantity)
        all_root_quantity_mean = estimate_mean(all_root_quantity)
        assert np.allclose(root_quantity_mean.mean, all_root_quantity_mean.mean)

        root_quantity_comp = root_quantity.select(root_quantity == root_quantity)
        root_quantity_comp_mean = estimate_mean(root_quantity_comp)
        assert np.allclose(root_quantity_mean.mean, root_quantity_comp_mean.mean)

        root_quantity_comp = root_quantity.select(root_quantity < root_quantity)
        with self.assertRaises(Exception):
            estimate_mean(root_quantity_comp)

        #new_quantity = selected_quantity + root_quantity
        #self.assertRaises(AssertionError, (selected_quantity + root_quantity))

        # bound root quantity result - select the ones which meet conditions
        mask = np.logical_and(0 < root_quantity, root_quantity < 10)
        q_bounded = root_quantity.select(mask)
        mean_q_bounded = estimate_mean(q_bounded)

        q_bounded_2 = root_quantity.select(0 < root_quantity, root_quantity < 10)
        mean_q_bounded_2 = estimate_mean(q_bounded_2)
        assert np.allclose(mean_q_bounded.mean, mean_q_bounded.mean)

        quantity_add = root_quantity + root_quantity
        q_add_bounded = quantity_add.select(0 < quantity_add, quantity_add < 20)
        means_add_bounded = estimate_mean(q_add_bounded)
        assert np.allclose((means_add_bounded.mean), mean_q_bounded_2.mean * 2)

        q_bounded = root_quantity.select(10 < root_quantity, root_quantity < 20)
        mean_q_bounded = estimate_mean(q_bounded)

        q_add_bounded = quantity_add.select(20 < quantity_add, quantity_add < 40)
        means_add_bounded_2 = estimate_mean(q_add_bounded)
        assert np.allclose((means_add_bounded_2.mean), mean_q_bounded.mean * 2)

        q_add_bounded_3 = quantity_add.select(root_quantity < quantity_add)
        means_add_bounded_3 = estimate_mean(q_add_bounded_3)
        assert len(means_add_bounded_3.mean) == len(root_quantity_mean.mean)

        q_add_bounded_4 = quantity_add.select(root_quantity > quantity_add)
        with self.assertRaises(Exception):
            estimate_mean(q_add_bounded_4)

        q_add_bounded_5 = quantity_add.select(root_quantity < quantity_add, root_quantity < 10)
        means_add_bounded_5 = estimate_mean(q_add_bounded_5)
        assert len(means_add_bounded_5.mean) == len(mean_q_bounded.mean)

        length = root_quantity['length']
        mean_length = estimate_mean(length)
        quantity_lt = length.select(length < 10)  # use just first sample
        means_lt = estimate_mean(quantity_lt)
        assert len(mean_length.mean) == len(means_lt.mean)

        q_add_bounded_6 = quantity_add.select(root_quantity < quantity_add, length < 1)
        with self.assertRaises(Exception):
            estimate_mean(q_add_bounded_6)

        q_add_bounded_7 = quantity_add.select(root_quantity < quantity_add, length < 10)
        means_add_bounded_7 = estimate_mean(q_add_bounded_7)
        assert np.allclose(means_add_bounded_7.mean, means_add_bounded.mean)

        quantity_le = length.select(length <= 9)  # use just first sample
        means_le = estimate_mean(quantity_le)
        assert len(mean_length.mean) == len(means_le.mean)

        quantity_lt = length.select(length < 1)  # no sample matches condition
        with self.assertRaises(Exception):
            estimate_mean(quantity_lt)

        quantity_lt_gt = length.select(9 < length, length < 20)  # one sample matches condition
        means_lt_gt = estimate_mean(quantity_lt_gt)
        assert len(mean_length.mean) == len(means_lt_gt.mean)

        quantity_gt = length.select(10**5 < length)  # no sample matches condition
        with self.assertRaises(Exception):
            estimate_mean(quantity_gt)

        quantity_ge = length.select(10**5 <= length)  # no sample matches condition
        with self.assertRaises(Exception):
            estimate_mean(quantity_ge)

        quantity_eq = length.select(1 == length)
        with self.assertRaises(Exception):
             estimate_mean(quantity_eq)

        quantity_ne = length.select(-1 != length)
        means_ne = estimate_mean(quantity_ne)
        assert np.allclose((means_ne.mean).tolist(), mean_length.mean.tolist())

    def test_functions(self):
        """
        Test numpy functions
        """
        sample_storage = Memory()
        result_format, sizes = self.fill_sample_storage(sample_storage)
        root_quantity = make_root_quantity(sample_storage, result_format)

        root_quantity_means = estimate_mean(root_quantity)

        max_root_quantity = np.max(root_quantity, axis=0, keepdims=True)
        max_means = estimate_mean(max_root_quantity)
        assert len(max_means.mean) == 1

        sin_root_quantity = np.sin(root_quantity)
        sin_means = estimate_mean(sin_root_quantity)
        assert len(sin_means.mean) == np.sum(sizes)

        round_root_quantity = np.sum(root_quantity, axis=0, keepdims=True)
        round_means = estimate_mean(round_root_quantity)
        assert len(round_means.mean) == 1

        add_root_quantity = np.add(root_quantity, root_quantity)  # Add arguments element-wise.
        add_root_quantity_means = estimate_mean(add_root_quantity)
        assert np.allclose(add_root_quantity_means.mean.flatten(), (root_quantity_means.mean * 2))

        x = np.ones(108)
        add_one_root_quantity = np.add(x, root_quantity)  # Add arguments element-wise.
        add_one_root_quantity_means = estimate_mean(add_one_root_quantity)
        assert np.allclose(root_quantity_means.mean + np.ones((108,)), add_one_root_quantity_means.mean.flatten())

        x = np.ones(108)
        divide_one_root_quantity = np.divide(x, root_quantity)  # Add arguments element-wise.
        divide_one_root_quantity_means = estimate_mean(divide_one_root_quantity)
        assert np.all(divide_one_root_quantity_means.mean < 1)

        # Test broadcasting
        x = np.ones(108)
        arctan2_one_root_quantity = np.arctan2(x, root_quantity)  # Add arguments element-wise.
        arctan2_one_root_quantity_means = estimate_mean(arctan2_one_root_quantity)
        assert np.all(arctan2_one_root_quantity_means.mean < 1)

        max_root_quantity = np.maximum(root_quantity, root_quantity)  # Element-wise maximum of array elements.
        max_root_quantity_means = estimate_mean(max_root_quantity)
        assert np.allclose(max_root_quantity_means.mean.flatten(), root_quantity_means.mean)

        length = root_quantity['length']
        sin_length = np.sin(length)
        sin_means_length = estimate_mean(sin_length)
        assert np.allclose((sin_means.mean[sizes[0]:sizes[0]+sizes[1]]).tolist(), sin_means_length.mean.tolist())

        q_and = np.logical_and(True, root_quantity)
        self.assertRaises(TypeError, estimate_mean, q_and)

        cache_clear()
        x = np.ones((108, 5, 2))
        self.assertRaises(ValueError, np.add, x, root_quantity)

        x = np.ones((108, 5, 2))
        self.assertRaises(ValueError, np.divide, x, root_quantity)

    def test_quantity_const(self):
        x = QuantityConst(ScalarType(), 5)
        y = QuantityConst(ScalarType(), 10)
        z = x + y
        assert isinstance(z, QuantityConst)

    def fill_sample_storage(self, sample_storage):
        np.random.seed(123)
        n_levels = 3
        res_length = 3
        result_format = [
            QuantitySpec(name="depth", unit="mm", shape=(2, res_length - 2+1), times=[1, 2, 3], locations=['30', '40']),
            QuantitySpec(name="length", unit="m", shape=(2, res_length - 2+2), times=[1, 2, 3], locations=['10', '20']),
            QuantitySpec(name="width", unit="mm", shape=(2, res_length - 2+3), times=[1, 2, 3], locations=['30', '40'])]

        sample_storage.save_global_data(result_format=result_format, level_parameters=np.ones(n_levels))

        successful_samples = {}
        failed_samples = {}
        n_ops = {}
        n_successful = 150
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

    def _create_sampler(self, step_range, clean=False, memory=False):
        # Set work dir
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        work_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '_test_tmp')
        if clean:
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

        # Create sample storages
        if memory:
            sample_storage = Memory()
        else:
            sample_storage = SampleStorageHDF(file_path=os.path.join(work_dir, "mlmc_test.hdf5"))
        # Create sampling pools
        sampling_pool = OneProcessPool()
        # sampling_pool_dir = OneProcessPool(work_dir=work_dir)

        if clean:
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

    def test_moments(self):
        """
        Moments estimation
        """
        np.random.seed(1234)
        n_moments = 3
        step_range = [0.5, 0.01]
        n_levels = 3
        clean = True

        level_parameters = mlmc.estimator.determine_level_parameters(n_levels=n_levels, step_range=step_range)
        sampler, simulation_factory = self._create_sampler(level_parameters, clean=clean, memory=False)

        distr = stats.norm()
        true_domain = distr.ppf([0.0001, 0.9999])
        # moments_fn = Legendre(n_moments, true_domain)
        moments_fn = Monomial(n_moments, true_domain)

        sampler.set_initial_n_samples([100, 60, 15])
        sampler.schedule_samples()
        sampler.ask_sampling_pool_for_samples()

        root_quantity = make_root_quantity(storage=sampler.sample_storage, q_specs=simulation_factory.result_format())
        root_quantity_mean = estimate_mean(root_quantity)

        estimator = mlmc.estimator.Estimate(root_quantity, sample_storage=sampler.sample_storage, moments_fn=moments_fn)

        target_var = 1e-2
        sleep = 0
        add_coef = 0.1

        # New estimation according to already finished samples
        variances, n_ops = estimator.estimate_diff_vars_regression(sampler._n_scheduled_samples)
        n_estimated = mlmc.estimator.estimate_n_samples_for_target_variance(target_var, variances, n_ops,
                                                                            n_levels=sampler.n_levels)

        # Loop until number of estimated samples is greater than the number of scheduled samples
        while not sampler.process_adding_samples(n_estimated, sleep, add_coef):
            # New estimation according to already finished samples
            variances, n_ops = estimator.estimate_diff_vars_regression(sampler._n_scheduled_samples)
            n_estimated = mlmc.estimator.estimate_n_samples_for_target_variance(target_var, variances, n_ops,
                                                                                n_levels=sampler.n_levels)

        # Moments values are at the bottom
        moments_quantity = moments(root_quantity, moments_fn=moments_fn, mom_at_bottom=True)
        moments_mean = estimate_mean(moments_quantity)
        length_mean = moments_mean['length']
        time_mean = length_mean[1]
        location_mean = time_mean['10']
        values_mean = location_mean[0]

        assert np.allclose(values_mean.mean[:2], [1, 0.5], atol=1e-2)
        assert np.all(values_mean.var < target_var)

        new_moments = moments_quantity + moments_quantity
        new_moments_mean = estimate_mean(new_moments)
        assert np.allclose(moments_mean.mean + moments_mean.mean, new_moments_mean.mean)

        # Moments values are on the surface
        moments_quantity_2 = moments(root_quantity, moments_fn=moments_fn, mom_at_bottom=False)
        moments_mean = estimate_mean(moments_quantity_2)
        first_moment = moments_mean[0]
        second_moment = moments_mean[1]
        third_moment = moments_mean[2]
        assert np.allclose(values_mean.mean, [first_moment.mean[0], second_moment.mean[0], third_moment.mean[0]], atol=1e-4)

        # Central moments
        central_root_quantity = root_quantity - root_quantity_mean.mean
        monomial_mom_fn = Monomial(n_moments, domain=true_domain, ref_domain=true_domain)
        central_moments_quantity = moments(central_root_quantity, moments_fn=monomial_mom_fn, mom_at_bottom=True)

        central_moments_mean = estimate_mean(central_moments_quantity)
        length_mean = central_moments_mean['length']
        time_mean = length_mean[1]
        location_mean = time_mean['10']
        central_value_mean = location_mean[0]

        assert np.isclose(central_value_mean.mean[0], 1, atol=1e-10)
        assert np.isclose(central_value_mean.mean[1], 0, atol=1e-2)

        # Covariance
        covariance_quantity = covariance(root_quantity, moments_fn=moments_fn, cov_at_bottom=True)
        cov_mean = estimate_mean(covariance_quantity)
        length_mean = cov_mean['length']
        time_mean = length_mean[1]
        location_mean = time_mean['10']
        cov_mean = location_mean[0]
        assert np.allclose(values_mean.mean, cov_mean.mean[:, 0])

        # Single moment
        moment_quantity = moment(root_quantity, moments_fn=moments_fn, i=0)
        moment_mean = estimate_mean(moment_quantity)
        length_mean = moment_mean['length']
        time_mean = length_mean[1]
        location_mean = time_mean['10']
        value_mean = location_mean[0]
        assert len(value_mean.mean) == 1

        iter = 1000
        chunks_means = []
        chunks_vars = []
        chunks_subsamples = []
        rm_samples = []

        for i in range(iter):
            sample_vec = [30, 15, 10]
            root_quantity_subsamples = root_quantity.subsample(sample_vec)  # out of [100, 80, 50, 30, 10]
            moments_quantity = moments(root_quantity_subsamples, moments_fn=moments_fn, mom_at_bottom=True)
            mult_chunks_moments_mean = estimate_mean(moments_quantity)
            mult_chunks_length_mean = mult_chunks_moments_mean['length']
            mult_chunks_time_mean = mult_chunks_length_mean[1]
            mult_chunks_location_mean = mult_chunks_time_mean['10']
            mult_chunks_value_mean =mult_chunks_location_mean[0]

            chunks_means.append(mult_chunks_value_mean.mean)
            chunks_vars.append(mult_chunks_value_mean.var)
            chunks_subsamples.append(mult_chunks_value_mean.n_samples)

            rm_samples.append(mult_chunks_value_mean.n_rm_samples)

        assert np.allclose(np.mean(chunks_subsamples, axis=0), sample_vec, rtol=0.5)
        assert np.allclose(np.mean(chunks_means, axis=0), values_mean.mean, atol=1e-2)
        assert np.allclose(np.mean(chunks_vars, axis=0) / iter, values_mean.var, atol=1e-3)

    def dev_memory_usage_test(self):
        work_dir = "/home/martin/Documents/MLMC_quantity"
        sample_storage = SampleStorageHDF(file_path=os.path.join(work_dir, "mlmc_quantity_2.hdf5"))
        sample_storage.chunk_size = 1e6
        result_format = sample_storage.load_result_format()
        root_quantity = make_root_quantity(sample_storage, result_format)
        mean_root_quantity = estimate_mean(root_quantity)
