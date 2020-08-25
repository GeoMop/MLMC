import os
import shutil
import unittest
import numpy as np
import random
from mlmc.sim.simulation import QuantitySpec
from mlmc.sample_storage import Memory
from mlmc.sample_storage_hdf import SampleStorageHDF
from mlmc.quantity_concept import make_root_quantity, estimate_mean

work_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '_test_tmp')
if os.path.exists(work_dir):
    shutil.rmtree(work_dir)
os.makedirs(work_dir)


class QuantityTests(unittest.TestCase):
    def test_basics(self):
        sample_storage = SampleStorageHDF(file_path=os.path.join(work_dir, "mlmc.hdf5"))
        result_format, size = self.fill_sample_storage(sample_storage)
        root_quantity = make_root_quantity(sample_storage, result_format)

        # results = sample_storage.sample_pairs()
        # print("results ", results)

        means = estimate_mean(root_quantity)
        print("means ", means)
        self.assertEqual(len(means), size)

        length = root_quantity['length']
        means_length = estimate_mean(length)
        assert np.allclose((means[:len(means) // 2]).tolist(), means_length.tolist())

        # Interpolation in time
        locations = length.time_interpolation(2.5)
        mean_interp_value = estimate_mean(locations)
        #
        # Select position
        position = locations['10']
        mean_position_1 = estimate_mean(position)
        assert np.allclose(mean_interp_value[:2], mean_position_1)
        #
        position = locations['20']
        mean_position_2 = estimate_mean(position)
        assert np.allclose(mean_interp_value[2:], mean_position_2)

        width = root_quantity['width']
        width_locations = width.time_interpolation(1.2)
        mean_width_interp_value = estimate_mean(width_locations)

        # Select position
        position = width_locations['30']
        mean_position_1 = estimate_mean(position)
        assert np.allclose(mean_width_interp_value[:2], mean_position_1)

        position = width_locations['40']
        mean_position_2 = estimate_mean(position)
        assert np.allclose(mean_width_interp_value[2:], mean_position_2)

        quantity_add = root_quantity + root_quantity
        means_add = estimate_mean(quantity_add)
        assert np.allclose((means + means), means_add)

        length = quantity_add['length']
        means_length = estimate_mean(length)
        assert np.allclose((means_add[:len(means_add)//2]).tolist(), means_length.tolist())

        width = quantity_add['width']
        means_width = estimate_mean(width)
        assert np.allclose((means_add[len(means_add)//2:]).tolist(), means_width.tolist())

        const = 5
        const_mult_quantity = const * root_quantity
        const_mult_mean = estimate_mean(const_mult_quantity)
        assert np.allclose((const * means).tolist(), const_mult_mean.tolist())

    def test_binary_operations(self):
        sample_storage = Memory()
        result_format, size = self.fill_sample_storage(sample_storage)
        root_quantity = make_root_quantity(sample_storage, result_format)

        means = estimate_mean(root_quantity)
        self.assertEqual(len(means), size)

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
        assert np.allclose((means_add[:len(means_add)//2]).tolist(), means_length.tolist())

        width = quantity_add['width']
        means_width = estimate_mean(width)
        assert np.allclose((means_add[len(means_add)//2:]).tolist(), means_width.tolist())

        const = 5
        const_mult_quantity = const * root_quantity
        const_mult_mean = estimate_mean(const_mult_quantity)
        means = estimate_mean(root_quantity)

        assert np.allclose((const * means).tolist(), const_mult_mean.tolist())

    def test_condition(self):
        sample_storage = Memory()
        result_format, size = self.fill_sample_storage(sample_storage)
        root_quantity = make_root_quantity(sample_storage, result_format)

        selected_quantity = root_quantity.select(root_quantity < 5)
        selected_quantity_mean = estimate_mean(selected_quantity)
        print("selected quantity mean ", selected_quantity_mean)

        #bound root quantity result - select the ones which meet conditions
        q_bounded = root_quantity.select(0 < root_quantity < 10)
        mean_q_bounded = estimate_mean(q_bounded)

        quantity_add = root_quantity + root_quantity
        q_add_bounded = quantity_add.select(0 < quantity_add < 20)
        means_add_bounded = estimate_mean(q_add_bounded)
        assert np.allclose((means_add_bounded), mean_q_bounded*2)

        length = root_quantity['length']
        mean_length = estimate_mean(length)
        assert len(mean_q_bounded) == len(mean_length) * 2

        quantity_lt = length.select(length < 10)  # use just first sample
        #length.select(quantity_lt)
        # length.select(length < 10)
        # root_quantity.select(length < 10)
        # root_quantity.select(mask - zase to bude quantity, kde se napriklad nageneruje nahodny pocet jednicek) subsamples
        means_lt = estimate_mean(quantity_lt)
        assert len(mean_length) == len(means_lt)

        quantity_le = length.select(length <= 9)  # use just first sample
        means_le = estimate_mean(quantity_le)
        assert len(mean_length) == len(means_le)

        quantity_lt = length.select(length < 1) # no sample matches condition
        means_lt = estimate_mean(quantity_lt)
        assert len(means_lt) == 0

        quantity_lt_gt = length.select(9 < length < 20)  # one sample matches condition
        means_lt_gt = estimate_mean(quantity_lt_gt)
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
        print("means eq ", means_eq)

        root_quantity_subsamples = root_quantity.select(root_quantity.sampling(size=10))
        means_eq = estimate_mean(root_quantity_subsamples)
        print("means eq ", means_eq)

    def fill_sample_storage(self, sample_storage):
        np.random.seed(123)
        n_levels = 3
        res_length = 3
        result_format = [
            QuantitySpec(name="length", unit="m", shape=(2, res_length - 2), times=[1, 2, 3], locations=['10', '20']),
            QuantitySpec(name="width", unit="mm", shape=(2, res_length - 2), times=[1, 2, 3], locations=['30', '40'])
        ]

        sample_storage.save_global_data(result_format=result_format, level_parameters=np.ones(n_levels))

        successful_samples = {}
        failed_samples = {}
        n_ops = {}
        n_successful = 5
        q1_size = 0
        q2_size = 0
        for l_id in range(n_levels):
            q1_size = np.prod(result_format[0].shape) * len(result_format[0].times) * len(result_format[0].locations)
            q2_size = np.prod(result_format[1].shape) * len(result_format[1].times) * len(result_format[1].locations)

            # Dict[level_id, List[Tuple[sample_id:str, Tuple[fine_result: ndarray, coarse_result: ndarray]]]]
            successful_samples[l_id] = [(str(sample_id), (np.random.randint(5 + 5*sample_id, high=5+5*(1+sample_id),
                                                                            size=(q1_size + q2_size,)),
                                                          np.random.randint(5 + 5*sample_id, high=5+5*(1+sample_id),
                                                                            size=(q1_size + q2_size,))))
                                        for sample_id in range(n_successful)]
            n_ops[l_id] = [random.random(), n_successful]

            sample_storage.save_scheduled_samples(l_id, samples=["S{:07d}".format(i) for i in range(n_successful)])

        sample_storage.save_samples(successful_samples, failed_samples)
        sample_storage.save_n_ops(n_ops)

        return result_format, q1_size + q2_size


if __name__ == '__main__':
    unittest.main()
