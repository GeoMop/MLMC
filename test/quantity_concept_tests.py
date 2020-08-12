import unittest
import numpy as np
import random
from mlmc.sim.simulation import QuantitySpec
from mlmc.sample_storage import Memory
from mlmc.quantity_concept import make_root_quantity
from mlmc.quantity_concept_2 import make_root_quantity, estimate_mean


class MyTestCase(unittest.TestCase):
    def test_something(self):
        sample_storage = Memory()
        result_format, size = self.fill_sample_storage(sample_storage)
        root_quantity = make_root_quantity(sample_storage, result_format)

        means = estimate_mean(root_quantity)
        print("means ", means)

        self.assertEqual(len(means), size)

        length = root_quantity['length']
        #length_mean = estimate_mean(length)

        add_quantity = root_quantity + root_quantity
        add_mean = estimate_mean(add_quantity)
        print("add mean ", add_mean)

        self.assertEqual((means + means).tolist(), add_mean.tolist())

    def fill_sample_storage(self, sample_storage):
        np.random.seed(123)
        n_levels = 3
        res_length = 5
        result_format = [
            QuantitySpec(name="length", unit="m", shape=(2, res_length - 2), times=[1, 2, 3], locations=['10', '20']),
            QuantitySpec(name="width", unit="mm", shape=(2, res_length - 2), times=[1, 2, 3], locations=['30', '40'])
        ]

        sample_storage.save_global_data(result_format=result_format, level_parameters=np.ones(n_levels))

        successful_samples = {}
        failed_samples = {}
        n_ops = {}
        n_successful = 3
        q1_size = 0
        q2_size = 0
        for l_id in range(n_levels):
            q1_size = np.prod(result_format[0].shape) * len(result_format[0].times) * len(result_format[0].locations)
            q2_size = np.prod(result_format[1].shape) * len(result_format[1].times) * len(result_format[1].locations)

            # Dict[level_id, List[Tuple[sample_id:str, Tuple[fine_result: ndarray, coarse_result: ndarray]]]]
            successful_samples[l_id] = [(str(sample_id), (np.random.randint(5, size=(q1_size + q2_size,)),
                                                          np.random.randint(5, size=(q1_size + q2_size,))))
                                        for sample_id in range(n_successful)]
            n_ops[l_id] = [random.random(), n_successful]

            sample_storage.save_scheduled_samples(l_id, samples=["S{:07d}".format(i) for i in range(n_successful)])

        sample_storage.save_samples(successful_samples, failed_samples)
        sample_storage.save_n_ops(n_ops)

        return result_format, q1_size + q2_size


if __name__ == '__main__':
    unittest.main()
