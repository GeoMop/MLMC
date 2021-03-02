import random
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from mlmc.tool import plot
import mlmc.tool.simple_distribution
import mlmc.estimator
import mlmc.quantity_estimate as qe
from mlmc.sample_storage import Memory
from mlmc.quantity_spec import QuantitySpec, ChunkSpec
import numpy as np
from mlmc.sample_storage_hdf import SampleStorageHDF
from mlmc.moments import Legendre, Monomial
from mlmc.quantity import make_root_quantity
import mlmc.tool.simple_distribution


def plot_loss(train_loss, val_loss):
    plt.plot(train_loss, label='loss')
    plt.plot(val_loss, label='val_loss')
    #plt.ylim([0, 10])
    plt.yscale("log")
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    plt.show()


def analyze_results(target, predictions):
    statistics, pvalue = ks_2samp(target, predictions)

    print("Target mean: {}, var: {}, Q25: {}, Q50: {}, Q75: {}".format(np.mean(target),
                                                                       np.var(target),
                                                                       np.quantile(target, 0.25),
                                                                       np.quantile(target, 0.5),
                                                                       np.quantile(target, 0.75)))
    print("Predic mean: {}, var: {}, Q25: {}, Q50: {}, Q75: {}".format(np.mean(predictions),
                                                                       np.var(predictions),
                                                                       np.quantile(predictions, 0.25),
                                                                       np.quantile(predictions, 0.5),
                                                                       np.quantile(predictions, 0.75)))

    print("KS statistics: {}, pvalue: {}".format(statistics, pvalue))
    # The closer KS statistic is to 0 the more likely it is that the two samples were drawn from the same distribution

    plt.hist(target,  alpha=0.5, label='target', density=True)
    plt.hist(predictions, alpha=0.5, label='predictions', density=True)
    plt.legend(loc='upper right')
    plt.show()


def estimate_density(values, title="Density"):
    sample_storage = Memory()
    n_levels = 1
    n_moments = 25
    distr_accuracy = 1e-7

    distr_plot = plot.Distribution(title=title,
                                   log_density=True)

    result_format = [QuantitySpec(name="flow", unit="m", shape=(1,), times=[0], locations=['0'])]

    sample_storage.save_global_data(result_format=result_format, level_parameters=np.ones(n_levels))

    successful_samples = {}
    failed_samples = {}
    n_ops = {}
    n_successful = len(values)
    for l_id in range(n_levels):
        sizes = []
        for quantity_spec in result_format:
            sizes.append(np.prod(quantity_spec.shape) * len(quantity_spec.times) * len(quantity_spec.locations))

        # Dict[level_id, List[Tuple[sample_id:str, Tuple[fine_result: ndarray, coarse_result: ndarray]]]]
        successful_samples[l_id] = []
        for sample_id in range(len(values)):
            successful_samples[l_id].append((str(sample_id), (values[sample_id], 0)))

        n_ops[l_id] = [random.random(), n_successful]

        sample_storage.save_scheduled_samples(l_id, samples=["S{:07d}".format(i) for i in range(n_successful)])

    sample_storage.save_samples(successful_samples, failed_samples)
    sample_storage.save_n_ops(list(n_ops.items()))

    quantity = make_root_quantity(storage=sample_storage, q_specs=result_format)
    length = quantity['flow']
    time = length[0]
    location = time['0']
    value_quantity = location[0]

    quantile = 0.001
    true_domain = mlmc.estimator.Estimate.estimate_domain(value_quantity, sample_storage, quantile=quantile)
    print("true domain ", true_domain)
    moments_fn = Legendre(n_moments, true_domain)

    estimator = mlmc.estimator.Estimate(quantity=value_quantity, sample_storage=sample_storage, moments_fn=moments_fn)

    reg_param = 0
    target_var = 1e-4
    distr_obj, info, result, moments_fn = estimator.construct_density(
        tol=distr_accuracy,
        reg_param=reg_param,
        orth_moments_tol=target_var)

    samples = value_quantity.samples(ChunkSpec(level_id=0, n_samples=sample_storage.get_n_collected()[0]))[..., 0]

    distr_plot.add_raw_samples(np.squeeze(samples))

    distr_plot.add_distribution(distr_obj, label="")

    # kl = mlmc.tool.simple_distribution.KL_divergence(self.cut_distr.pdf, distr_obj.density,
    #                                                  self.cut_distr.domain[0], self.cut_distr.domain[1])
    #kl_divergences.append(kl)

    distr_plot.show(file=None)


    return estimator.estimate_moments()


def create_quantity(target, predictions):
    sample_storage = Memory()
    n_levels = 2

    result_format = [QuantitySpec(name="flow", unit="m", shape=(1,), times=[0], locations=['0'])]

    sample_storage.save_global_data(result_format=result_format, level_parameters=np.ones(n_levels))

    successful_samples = {}
    failed_samples = {}
    n_ops = {}
    n_successful = len(target)
    for l_id in range(n_levels):
        sizes = []
        for quantity_spec in result_format:
            sizes.append(np.prod(quantity_spec.shape) * len(quantity_spec.times) * len(quantity_spec.locations))

        successful_samples[l_id] = []
        for sample_id in range(n_successful):
            if l_id == 0:
                fine_result = predictions[sample_id]
                coarse_result = (np.zeros((np.sum(sizes),)))
            else:
                fine_result = target[sample_id]
                coarse_result = predictions[sample_id]

            successful_samples[l_id].append((str(sample_id), (fine_result, coarse_result)))

        n_ops[l_id] = [random.random(), n_successful]
        sample_storage.save_scheduled_samples(l_id, samples=["S{:07d}".format(i) for i in range(n_successful)])

    sample_storage.save_samples(successful_samples, failed_samples)
    sample_storage.save_n_ops(list(n_ops.items()))

    quantity = make_root_quantity(storage=sample_storage, q_specs=result_format)
    length = quantity['flow']
    time = length[0]
    location = time['0']
    value_quantity = location[0]

    return value_quantity, sample_storage


def diff_moments(target, predictions):
    n_moments = 25
    quantity, target_sample_storage = create_quantity(target, predictions)

    quantile = 0.001
    true_domain = mlmc.estimator.Estimate.estimate_domain(quantity, target_sample_storage, quantile=quantile)

    moments_fn = Legendre(n_moments, true_domain)

    quantity_moments = qe.moments(quantity, moments_fn)


    moments_mean = qe.estimate_mean(quantity_moments)

    print("moments l means ", moments_mean.l_means)
    print("moments l vars ", moments_mean.l_vars)

    print("np.max values mean l vars ", np.max(moments_mean.l_vars, axis=1))

    print("moments mean ", moments_mean.mean)
    print("moments var ", moments_mean.var)


def create_quantity_mlmc(data):
    sample_storage = Memory()
    n_levels = len(data)

    result_format = [QuantitySpec(name="conductivity", unit="m", shape=(1, 1), times=[1], locations=['0'])]

    sample_storage.save_global_data(result_format=result_format, level_parameters=np.ones(n_levels))

    successful_samples = {}
    failed_samples = {}
    n_ops = {}
    n_successful = 15
    sizes = []
    for l_id in range(n_levels):
        n_successful = data[l_id].shape[1]
        sizes = []
        for quantity_spec in result_format:
            sizes.append(np.prod(quantity_spec.shape) * len(quantity_spec.times) * len(quantity_spec.locations))

        # Dict[level_id, List[Tuple[sample_id:str, Tuple[fine_result: ndarray, coarse_result: ndarray]]]]
        successful_samples[l_id] = []
        for sample_id in range(n_successful):

            fine_result = data[l_id][:, sample_id, 0]
            if l_id == 0:
                coarse_result = (np.zeros((np.sum(sizes),)))
            else:
                coarse_result = data[l_id][:, sample_id, 1]
            successful_samples[l_id].append((str(sample_id), (fine_result, coarse_result)))

        n_ops[l_id] = [random.random(), n_successful]

        sample_storage.save_scheduled_samples(l_id, samples=["S{:07d}".format(i) for i in range(n_successful)])

    sample_storage.save_samples(successful_samples, failed_samples)
    sample_storage.save_n_ops(list(n_ops.items()))

    return sample_storage


def estimate_moments(sample_storage):
    n_moments = 25
    result_format = sample_storage.load_result_format()
    root_quantity = make_root_quantity(sample_storage, result_format)

    conductivity = root_quantity['conductivity']
    time = conductivity[1]  # times: [1]
    location = time['0']  # locations: ['0']
    q_value = location[0, 0]

    # @TODO: How to estimate true_domain?
    quantile = 0.001
    true_domain = mlmc.estimator.Estimate.estimate_domain(q_value, sample_storage, quantile=quantile)
    print("true domain ", true_domain)
    moments_fn = Legendre(n_moments, true_domain)

    estimator = mlmc.estimator.Estimate(quantity=q_value, sample_storage=sample_storage, moments_fn=moments_fn)
    means, vars = estimator.estimate_moments(moments_fn)

    # moments_quantity = moments(root_quantity, moments_fn=moments_fn, mom_at_bottom=True)
    # moments_mean = estimate_mean(moments_quantity)
    # conductivity_mean = moments_mean['conductivity']
    # time_mean = conductivity_mean[1]  # times: [1]
    # location_mean = time_mean['0']  # locations: ['0']
    # values_mean = location_mean[0]  # result shape: (1,)
    #
    # print("values_mean.n_samples ", values_mean.n_samples)
    # print("values_mean.l_means ", values_mean.l_means)
    #
    # print("l_means / n samples ", values_mean.l_means / values_mean.n_samples[:, None])
    # print("values mean. l_vars ", values_mean.l_vars)
    #
    # print("np.max values mean l vars ", np.max(values_mean.l_vars, axis=1))
    #
    # print("values_mean mean ", values_mean.mean)
    # print("values_mean var ", values_mean.var)

    return means, vars, estimator


# def construct_density(estimator):
#
#     distr_obj, info, result, moments_fn = estimator.construct_density(
#         tol=distr_accuracy,
#         reg_param=reg_param,
#         orth_moments_tol=target_var)
#
#     samples = estimator.quantity.samples(ChunkSpec(level_id=0, n_samples=estimator._sample_storage.get_n_collected()[0]))[..., 0]
#
#     return distr_obj, samples


def compare_densities(estimator_1, estimator_2, label_1="", label_2=""):

    distr_plot = plot.ArticleDistribution(title="densities", log_density=True)
    tol = 1e-10
    reg_param = 0

    distr_obj, result, _, _ = estimator_2.construct_density(tol=tol, reg_param=reg_param)
    #distr_plot.add_raw_samples(np.squeeze(samples))
    distr_plot.add_distribution(distr_obj, label=label_1, color="blue")

    distr_obj, result, _, _ = estimator_2.construct_density(tol=tol, reg_param=reg_param)
    #distr_plot.add_raw_samples(np.squeeze(samples))
    distr_plot.add_distribution(distr_obj, label=label_2, color="red")
    #

    distr_plot.show(file=None)


def get_quantity_estimator(sample_storage):
    n_moments = 25
    result_format = sample_storage.load_result_format()
    root_quantity = make_root_quantity(sample_storage, result_format)
    conductivity = root_quantity['conductivity']
    time = conductivity[1]  # times: [1]
    location = time['0']  # locations: ['0']
    quantity = location[0, 0]

    quantile = 0.001
    true_domain = mlmc.estimator.Estimate.estimate_domain(quantity, sample_storage, quantile=quantile)
    moments_fn = Legendre(n_moments, true_domain)

    return mlmc.estimator.Estimate(quantity=quantity, sample_storage=sample_storage, moments_fn=moments_fn)


def process_mlmc(targets, predictions, train_targets, val_targets):
    n_levels = 5
    mlmc_file = "/home/martin/Documents/metamodels/data/cl_0_3_s_4/L5/mlmc_5.hdf5"

    sample_storage = SampleStorageHDF(file_path=mlmc_file)
    original_means, original_vars, estimator = estimate_moments(sample_storage)

    # Test storage creation
    data = []
    for l_id in range(n_levels):
        level_samples = estimator.get_level_samples(level_id=l_id)
        data.append(level_samples)
    sample_storage_2 = create_quantity_mlmc(data)
    means_2, vars_2, estimator_2 = estimate_moments(sample_storage_2)
    assert np.allclose(original_means, means_2)
    assert np.allclose(original_vars, vars_2)

    data = []
    for l_id in range(n_levels):
        level_samples = estimator.get_level_samples(level_id=l_id)
        if l_id == 0:
            level_samples = np.concatenate((train_targets.reshape(1, len(train_targets), 1),
                                            targets.reshape(1, len(targets), 1)), axis=1)
        data.append(level_samples)
    sample_storage_nn = create_quantity_mlmc(data)
    means_nn, vars_nn, estimator_nn = estimate_moments(sample_storage_nn)


    print("original means ", original_means)
    print("means nn ", means_nn)

    print("original vars ", original_vars)
    print("vars nn ", vars_nn)
    assert np.allclose(original_means, means_nn, atol=1e-3)
    assert np.allclose(original_vars, vars_nn, atol=1e-3)

    original_q_estimator = get_quantity_estimator(sample_storage)
    no_val_samples_q_estimator = get_quantity_estimator(sample_storage_nn)

    compare_densities(estimator, no_val_samples_q_estimator, label_1="original", label_2="without validation values")

