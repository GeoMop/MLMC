import os
import random
import copy
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
#from mlmc.tool import plot
#import mlmc.tool.simple_distribution
import mlmc.estimator
from mlmc.tool import gmsh_io
import mlmc.quantity_estimate as qe
from mlmc.sample_storage import Memory
from mlmc.quantity_spec import QuantitySpec, ChunkSpec
import numpy as np
from mlmc.sample_storage_hdf import SampleStorageHDF
from mlmc.moments import Legendre, Monomial
from mlmc.quantity import make_root_quantity
from mlmc.metamodel.create_graph import extract_mesh_gmsh_io
#import mlmc.tool.simple_distribution

QUANTILE = 0.01


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
    #statistics, pvalue = ks_2samp(target, predictions)

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

    #print("KS statistics: {}, pvalue: {}".format(statistics, pvalue))
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

    quantile = QUANTILE
    true_domain = mlmc.estimator.Estimate.estimate_domain(value_quantity, sample_storage, quantile=quantile)
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

    result_format = [QuantitySpec(name="conductivity", unit="m", shape=(1, 1), times=[1], locations=['0'])]

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

    quantile = QUANTILE
    true_domain = mlmc.estimator.Estimate.estimate_domain(quantity, target_sample_storage, quantile=quantile)

    moments_fn = Legendre(n_moments, true_domain)

    quantity_moments = qe.moments(quantity, moments_fn)


    moments_mean = qe.estimate_mean(quantity_moments)

    print("moments l means ", moments_mean.l_means)
    print("moments l vars ", moments_mean.l_vars)

    print("np.max values mean l vars ", np.max(moments_mean.l_vars, axis=1))

    print("moments mean ", moments_mean.mean)
    print("moments var ", moments_mean.var)


def create_quantity_mlmc(data, level_parameters, num_ops=None):
    sample_storage = Memory()
    n_levels = len(data)

    result_format = [QuantitySpec(name="conductivity", unit="m", shape=(1, 1), times=[1], locations=['0'])]
    sample_storage.save_global_data(result_format=result_format, level_parameters=level_parameters)

    successful_samples = {}
    failed_samples = {}
    n_ops = {}
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

        # if num_ops is not None:
        #     n_ops[l_id] = [num_ops[l_id], n_successful]
        # else:
        n_ops[l_id] = [random.random(), n_successful]

        sample_storage.save_scheduled_samples(l_id, samples=["S{:07d}".format(i) for i in range(n_successful)])

    #
    # print("successful samples ")
    # print("l 0", successful_samples[0][:10])
    # print("l 1", successful_samples[1][:10])
    # print("l 2", successful_samples[2][:10])
    # print("l 3", successful_samples[3][:10])

    sample_storage.save_samples(successful_samples, failed_samples)
    sample_storage.save_n_ops(list(n_ops.items()))

    return sample_storage


def estimate_moments(sample_storage, true_domain=None):
    n_moments = 15
    result_format = sample_storage.load_result_format()
    root_quantity = make_root_quantity(sample_storage, result_format)

    conductivity = root_quantity['conductivity']
    time = conductivity[1]  # times: [1]
    location = time['0']  # locations: ['0']
    q_value = location[0, 0]

    if true_domain is None:
        quantile = QUANTILE
        true_domain = mlmc.estimator.Estimate.estimate_domain(q_value, sample_storage, quantile=quantile)
    moments_fn = Legendre(n_moments, true_domain)

    estimator = mlmc.estimator.Estimate(quantity=q_value, sample_storage=sample_storage, moments_fn=moments_fn)
    #means, vars = estimator.estimate_moments(moments_fn)

    moments_mean = qe.estimate_mean(qe.moments(q_value, moments_fn))
    return moments_mean, estimator, true_domain, q_value


def ref_storage(mlmc_file):
    sample_storage = SampleStorageHDF(file_path=mlmc_file)
    return sample_storage


def get_largest_domain(storages):
    true_domains = []
    for storage in storages:
        result_format = storage.load_result_format()
        root_quantity = make_root_quantity(storage, result_format)

        conductivity = root_quantity['conductivity']
        time = conductivity[1]  # times: [1]
        location = time['0']  # locations: ['0']
        q_value = location[0, 0]

        # @TODO: How to estimate true_domain?
        quantile = QUANTILE
        domain = mlmc.estimator.Estimate.estimate_domain(q_value, storage, quantile=quantile)

        true_domains.append([domain[0], domain[1]])

    true_domains = np.array(true_domains)

    #print("true domains ", true_domains)
    true_domain = [np.min(true_domains[:, 0]), np.max(true_domains[:, 1])]
    #true_domain = [np.max(true_domains[:, 0]), np.min(true_domains[:, 1])]
    #true_domain = [np.mean(true_domains[:, 0]), np.mean(true_domains[:, 1])]


    #true_domain = true_domain[-1]

    return true_domain


def compare_moments(original_q_estimator, predict_q_estimator, ref_estimator):
    original_q_estimator.estimate_moments()
    orig_moments_mean = original_q_estimator.moments_mean

    predict_q_estimator.estimate_moments()
    predict_moments_mean = predict_q_estimator.moments_mean

    ref_estimator.estimate_moments()
    ref_moments_mean = ref_estimator.moments_mean

    #print("ref moments mean ", ref_moments_mean.mean)
    print("orig moments mean ", orig_moments_mean.mean)
    print("predict moments mean ", predict_moments_mean.mean)

    print("ref orig mean SSE ", np.sum((ref_moments_mean.mean - orig_moments_mean.mean)**2))
    print("ref predict mean SSE ", np.sum((ref_moments_mean.mean - predict_moments_mean.mean) ** 2))
    #
    print("ref orig mean SE ", np.sum(np.abs((ref_moments_mean.mean - orig_moments_mean.mean))))
    print("ref predict mean SE ", np.sum(np.abs((ref_moments_mean.mean - predict_moments_mean.mean))))

    #print("ref moments var ", ref_moments_mean.var)
    print("orig moments var ", orig_moments_mean.var)
    print("predict moments var ", predict_moments_mean.var)

    # print("MAX orig moments var ", np.max(orig_moments_mean.l_vars, axis=1))
    # print("MAX predict moments var ", np.max(predict_moments_mean.l_vars, axis=1))

    print("ref orig var SSE ", np.sum((ref_moments_mean.var - orig_moments_mean.var) ** 2))
    print("ref predict var SSE ", np.sum((ref_moments_mean.var - predict_moments_mean.var) ** 2))
    #
    print("ref orig var SE ", np.sum(np.abs((ref_moments_mean.var - orig_moments_mean.var))))
    print("ref predict var SE ", np.sum(np.abs((ref_moments_mean.var - predict_moments_mean.var))))

    return orig_moments_mean, predict_moments_mean, ref_moments_mean

    # l_0_samples = predict_q_estimator.get_level_samples(level_id=0)
    # l_1_samples = predict_q_estimator.get_level_samples(level_id=1)
    # l_2_samples = predict_q_estimator.get_level_samples(level_id=2)
    #
    # print("l 0 samples shape ", np.squeeze(l_0_samples).shape)
    # print("l 1 samples shape ", np.squeeze(l_1_samples[..., 0]).shape)
    #
    # print("l_0_samples.var ", np.var(np.squeeze(l_0_samples)))
    # print("l_1_samples ", l_1_samples)
    #
    # diff = l_1_samples[..., 0] - l_1_samples[..., 1]
    #
    # print("l1 diff ", diff)
    # print("var l1 diff ", np.var(diff))
    # print("fine l_1_samples.var ", np.var(np.squeeze(l_1_samples[..., 0])))
    # print("fine l_2_samples.var ", np.var(np.squeeze(l_2_samples[..., 0])))


def compare_densities(estimator_1, estimator_2, ref_estimator, label_1="", label_2=""):

    distr_plot = plot.ArticleDistribution(title="densities", log_density=True)
    tol = 1e-10
    reg_param = 0

    print("orig estimator")
    distr_obj_1, result, _, _ = estimator_1.construct_density(tol=tol, reg_param=reg_param)
    #distr_plot.add_distribution(distr_obj_1, label=label_1, color="blue")

    print("predict estimator")
    distr_obj_2, result, _, _ = estimator_2.construct_density(tol=tol, reg_param=reg_param)
    #distr_plot.add_distribution(distr_obj_2, label=label_2, color="red", line_style="--")

    print("Ref estimator")
    ref_distr_obj, result, _, _ = ref_estimator.construct_density(tol=tol, reg_param=reg_param)
    #distr_plot.add_distribution(ref_distr_obj, label="MC reference", color="black", line_style=":")

    kl_div_ref_mlmc = mlmc.tool.simple_distribution.KL_divergence(ref_distr_obj.density, distr_obj_1.density, ref_distr_obj.domain[0], ref_distr_obj.domain[1])

    print("KL div ref|mlmc: {}".format(kl_div_ref_mlmc))

    # domain = [np.min([ref_distr_obj.domain[0], distr_obj_2.domain[0]]),
    #           np.max([ref_distr_obj.domain[1], distr_obj_2.domain[1]])]
    kl_div_ref_gnn = mlmc.tool.simple_distribution.KL_divergence(ref_distr_obj.density, distr_obj_2.density, ref_distr_obj.domain[0],
                                                         ref_distr_obj.domain[1])

    print("KL div ref|mlmc prediction: {}".format(kl_div_ref_gnn))

    distr_plot.add_distribution(distr_obj_1, label=label_1 + ", KL(ref|orig):{:0.4g}".format(kl_div_ref_mlmc), color="blue")
    distr_plot.add_distribution(distr_obj_2, label=label_2 + ", KL(ref|gnn):{:0.4g}".format(kl_div_ref_gnn), color="red", line_style="--")
    distr_plot.add_distribution(ref_distr_obj, label="MC reference", color="black", line_style=":")

    distr_plot.show(file=None)


def get_quantity_estimator(sample_storage, true_domain=None):
    n_moments = 25
    result_format = sample_storage.load_result_format()
    root_quantity = make_root_quantity(sample_storage, result_format)
    conductivity = root_quantity['conductivity']
    time = conductivity[1]  # times: [1]
    location = time['0']  # locations: ['0']
    quantity = location[0, 0]

    if true_domain is None:
        quantile = QUANTILE
        true_domain = mlmc.estimator.Estimate.estimate_domain(quantity, sample_storage, quantile=quantile)

    moments_fn = Legendre(n_moments, true_domain)

    return mlmc.estimator.Estimate(quantity=quantity, sample_storage=sample_storage, moments_fn=moments_fn)


def get_n_estimated(sample_storage, estimator, n_ops=None):
    target_var = 5e-5
    #moments, estimator, _, quantity = estimate_moments(sample_storage, true_domain=true_domain)

    n_level_samples = sample_storage.get_n_collected()
    # New estimation according to already finished samples

    print("n level samples ", n_level_samples)
    variances, n_samples = estimator.estimate_diff_vars()
    #variances, est_n_ops = estimator.estimate_diff_vars_regression(n_level_samples)

    if n_ops is None:
        n_ops = n_samples
    print("get n estimated n ops ", n_ops)
    n_estimated = mlmc.estimator.estimate_n_samples_for_target_variance(target_var, variances, n_ops,
                                                                        n_levels=len(n_level_samples))
    return n_estimated, variances, n_samples


def get_storage_info(sample_storage):
    moments, estimator, _, _ = estimate_moments(sample_storage)
    n_collected = sample_storage.get_n_collected()
    max_vars = np.max(np.array(moments.l_vars) / np.array(sample_storage.get_n_collected())[:, np.newaxis], axis=1)
    print("n collected ", n_collected)
    print("moments.l_vars max ", max_vars)
    return n_collected, max_vars


def cut_samples(data, sample_storage, new_n_collected, new_l_0=0):
    new_data = []
    for l_id, (d, n_est) in enumerate(zip(data, new_n_collected)):
        if n_est > 0:
            if l_id == new_l_0:
                print(d.shape)
                print("np.min(d.shape[1], n_est) ", np.min([d.shape[1], n_est]))
                fine_samples = d[:, :np.min([d.shape[1], n_est]), 0].reshape(1, np.min([d.shape[1], n_est]), 1)
                coarse_samples = np.zeros(fine_samples.shape)
                new_data.append(np.concatenate((fine_samples, coarse_samples), axis=2))
            else:
                new_data.append(d[:, :np.min([d.shape[1], n_est]), :])
    sample_storage = create_quantity_mlmc(new_data, level_parameters=sample_storage.get_level_parameters())

    return sample_storage


def plot_progress(conv_layers, flatten_output, dense_layers, mesh_file=None):

    if mesh_file is not None:
        #mesh = gmsh_io.GmshIO(fields_mesh)
        mesh_data = extract_mesh_gmsh_io(mesh_file, get_points=True)
        points = mesh_data['points']
        X = points[:, 0]
        Y = points[:, 1]

    for inputs, weights, outputs in conv_layers:

        plt.matshow(weights[-1][0])
        plt.show()
        # Note: weights have different shape than the mesh

        for index, input in enumerate(inputs[::10]):
            if mesh_file:
                fig, ax = plt.subplots(1, 1, figsize=(15, 10))

                cont = ax.tricontourf(X, Y, input[0].ravel(), levels=16)
                fig.colorbar(cont)
                plt.title("input")
                plt.show()

                print("range(outputs[index][0].shape[1]) ", range(outputs[index][0].shape[1]))
                for i in range(outputs[index][0].shape[1]):
                    channel_output = outputs[index][0][:, i]
                    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
                    cont = ax.tricontourf(X, Y, channel_output, levels=16)
                    fig.colorbar(cont)
                    plt.title("channel {}".format(i))

                    plt.show()

            else:
                plt.matshow(input[0])
                plt.show()
                plt.matshow(outputs[index][0])
                plt.show()

            # print("shape ", c_layer._outputs[index][0].shape)
            plt.matshow(np.sum(outputs[index][0], axis=0, keepdims=True))
            plt.show()

            # plt.matshow(self._inputs[-1][0])
            # plt.show()
            # plt.matshow(self._outputs[-1][0])
            # plt.show()

            # print("output flatten ", self._output_flatten)
            # print("final output ", final_output)

            plt.matshow([flatten_output[-1]])
            plt.show()
            # plt.matshow(final_output[-1])
            # plt.show()

    for inputs, weights, outputs in dense_layers:
        plt.matshow(inputs[-1])
        plt.show()
        plt.matshow(outputs[-1])
        plt.show()


def process_mlmc(mlmc_file, sampling_info_path, ref_mlmc_file, targets, predictions, train_targets, train_predictions,
                 val_targets, l_0_targets=None, l_0_predictions=None,
                 l1_sample_time=None, l0_sample_time=None, nn_level=0, replace_level=False, stats=False):
    #level_zero = False

    if not stats:
        print("nn_level ", nn_level)
        print("replace level ", replace_level)

    if not stats:
        # print("nn_level ", nn_level)
        # print("replace level ", replace_level)

        # targets = np.exp(targets)
        # predictions = np.exp(predictions)
        # l_0_predictions = np.exp(l_0_predictions)
        # l_0_targets = np.exp(l_0_targets)

        # print("targets ", targets)
        # print("predictions ", predictions)
        plt.hist(targets, bins=50, alpha=0.5, label='target', density=True)
        plt.hist(predictions, bins=50, alpha=0.5, label='predictions', density=True)

        #plt.hist(targets - predictions, bins=50, alpha=0.5, label='predictions', density=True)
        plt.legend(loc='upper right')
        #plt.xlim(-0.5, 1000)
        plt.yscale('log')
        plt.show()

        plt.hist(l_0_targets, bins=50, alpha=0.5, label='l_0_target', density=True)
        plt.hist(l_0_predictions, bins=50, alpha=0.5, label='l_0_predictions', density=True)

        # plt.hist(targets - predictions, bins=50, alpha=0.5, label='predictions', density=True)
        plt.legend(loc='upper right')
        # plt.xlim(-0.5, 1000)
        plt.yscale('log')
        plt.show()

    # targets = np.exp(targets)
    # predictions = np.exp(predictions)

    # # Creating plot
    # plt.boxplot(targets)
    # plt.boxplot(predictions)
    # # show plot
    # plt.show()
    #exit()

    #######
    ### Create storage fromm original MLMC data
    ######
    sample_storage = SampleStorageHDF(file_path=mlmc_file)
    print("get n collected ", sample_storage.get_n_collected())
    n_levels = len(sample_storage.get_level_ids())
    original_moments, estimator, original_true_domain, _ = estimate_moments(sample_storage)

    orig_max_vars = np.max(original_moments.l_vars, axis=1)
    print("orig max vars ", orig_max_vars)

    ######
    ### Get n ops
    ######
    n_ops, field_times, coarse_flow, fine_flow = get_sample_times_mlmc(mlmc_file)
    # n_ops, _, _ = get_sample_times(sampling_info_path)
    # n_ops = n_ops[n_levels-1:]

    if n_ops is None:
        n_ops = sample_storage.get_n_ops()
        field_times = np.zeros(len(n_ops))
        flow_times = np.zeros(len(n_ops))
    # Test storage creation
    data_mlmc = []
    for l_id in range(n_levels):
        level_samples = estimator.get_level_samples(level_id=l_id)
        data_mlmc.append(level_samples)

    print("original level params", sample_storage.get_level_parameters())
    sample_storage = create_quantity_mlmc(data_mlmc, level_parameters=sample_storage.get_level_parameters())
    print("Original storage")
    orig_storage_n_collected, orig_storage_max_vars = get_storage_info(sample_storage)

    if replace_level:
        n_lev = n_levels
        level_params = [*sample_storage.get_level_parameters()]
    else:
        n_lev = n_levels - nn_level + 1
        level_params = [sample_storage.get_level_parameters()[nn_level], *sample_storage.get_level_parameters()[nn_level:]]

    # Use predicted data as zero level results and level one coarse results
    data_nn = []
    n_ops_predict = []

    nn_lev_sim_time = 0
    for l_id in range(nn_level):
        nn_lev_sim_time = n_ops[l_id] - nn_lev_sim_time

    for l_id in range(n_lev):
        if l_id == 0:
            level_samples = l_0_predictions.reshape(1, len(l_0_predictions), 1)
            #level_samples = np.ones((1, len(l_0_predictions), 1))
            ft_index = nn_level
            if nn_level > 0:
                ft_index = nn_level - 1
            n_ops_predict.append(l0_sample_time)# + field_times[ft_index] / 2)
            #level_samples = level_samples[:, :50000, :]
        else:
            if replace_level:
                level_id = l_id
                level_samples = estimator.get_level_samples(level_id=level_id)
                n_ops_predict.append(n_ops[level_id])
            else:
                if l_id == 1:
                    # print("l id replaced level ", l_id)
                    # print("len(predictions ) ", len(predictions))

                    coarse_level_samples = predictions.reshape(1, len(predictions), 1)
                    #coarse_level_samples = np.ones((1, len(predictions), 1))
                    fine_level_samples = targets.reshape(1, len(targets), 1)

                    # coarse_level_samples = np.concatenate((coarse_level_samples,
                    #                 train_predictions.reshape(1, len(train_predictions), 1)), axis=1)
                    #
                    # fine_level_samples = np.concatenate((fine_level_samples,
                    #                                      train_targets.reshape(1, len(train_targets), 1)), axis=1)

                    # coarse_level_samples = coarse_level_samples[:, :len(predictions)//2, :]
                    # fine_level_samples = fine_level_samples[:, :len(predictions)//2, :]
                    level_samples = np.concatenate((fine_level_samples, coarse_level_samples), axis=2)

                    n_ops_predict.append(n_ops[nn_level] - nn_lev_sim_time
                                         + l1_sample_time)
                else:
                    if replace_level:
                        level_id = l_id #- 1
                    else:
                        level_id = l_id + nn_level - 1
                    #print("level id ", level_id)
                    level_samples = estimator.get_level_samples(level_id=level_id)
                    n_ops_predict.append(n_ops[level_id])

        data_nn.append(level_samples)

    print("level params ", level_params)
    sample_storage_predict = create_quantity_mlmc(data_nn, level_parameters=level_params)
    print("n ops predict ", n_ops_predict)
    print("Storage predict info")
    predict_storage_n_collected, predict_storage_max_vars = get_storage_info(sample_storage_predict)

    # if stats:
    #     return orig_storage_max_vars, predict_storage_max_vars

    ######
    ### Create estimators
    ######
    ref_sample_storage = ref_storage(ref_mlmc_file)
    domain = get_largest_domain([sample_storage, sample_storage_predict, ref_sample_storage])
    original_q_estimator = get_quantity_estimator(sample_storage, true_domain=domain)
    predict_q_estimator = get_quantity_estimator(sample_storage_predict, true_domain=domain)
    #ref_estimator = get_quantity_estimator(ref_sample_storage, true_domain=domain)

    #######
    ### Calculate N estimated samples
    #######
    print("n ops ", n_ops)
    print("n ops predict ", n_ops_predict)

    # # # remove levels
    # # # #@TODO: remove asap
    # new_n_samples = [0, 0, sample_storage.get_n_collected()[-3], sample_storage.get_n_collected()[-2], sample_storage.get_n_collected()[-1]]
    # sample_storage = cut_samples(data_mlmc, sample_storage, new_n_samples, new_l_0=2)
    # print("Cut storage info")
    # get_storage_info(sample_storage)
    # original_q_estimator = get_quantity_estimator(sample_storage, true_domain=domain)
    # n_ops = n_ops[2:]
    # # # #####

    #### Original data
    n_ops_est = copy.deepcopy(n_ops)
    #n_ops_est[0] = n_ops_est[0] / 1000
    n_estimated_orig, l_vars_orig, n_samples_orig = get_n_estimated(sample_storage, original_q_estimator, n_ops=n_ops_est)
    print("n estimated orig ", n_estimated_orig)
    #print("l vars orig ", np.array(l_vars_orig) / np.array(sample_storage.get_n_collected())[:, np.newaxis])

    sample_storage = cut_samples(data_mlmc, sample_storage, n_estimated_orig)
    #
    # sample_storage_predict = cut_samples(data_nn, sample_storage_predict, [sample_storage_predict.get_n_collected()[0],
    #                                                                          *n_estimated_orig])
    #
    # predict_q_estimator = get_quantity_estimator(sample_storage_predict, true_domain=domain)
    #
    # print("sample storage predict n collected ", sample_storage_predict.get_n_collected())
    # print("sample storage n collected ", sample_storage.get_n_collected())

    n_ops_predict_orig = copy.deepcopy(n_ops_predict)
    #n_ops_predict_orig[0] = n_ops_predict_orig[0] /5
    #n_ops_predict = np.array(n_ops_predict)**2
    #n_ops_predict[0] = n_ops_predict[0] / 1000
    print("n ops predict for estimate ", n_ops_predict)
    # n_samples = [10000, 2000, 500, 150, 40, 11]

    n_estimated_nn, l_var_nn, n_samples_nn = get_n_estimated(sample_storage_predict, predict_q_estimator,
                                                             n_ops=n_ops_predict)
    #n_estimated_nn = [50000, 10000, 850]
    sample_storage_predict = cut_samples(data_nn, sample_storage_predict, n_estimated_nn)

    #######
    ## Estimate total time
    #######
    print("NN estimated ", n_estimated_nn)
    print("MLMC estimated ", n_estimated_orig)
    NN_time_levels = n_ops_predict_orig * np.array(n_estimated_nn)
    n_collected_times = n_ops * np.array(n_estimated_orig)
    print("NN time levels ", NN_time_levels)
    print("MLMC time levels", n_collected_times)
    nn_total_time = np.sum(NN_time_levels)
    print("NN total time ", nn_total_time)
    mlmc_total_time = np.sum(n_collected_times)
    print("MLMC total time ", mlmc_total_time)

    #original_moments, estimator, original_true_domain, _ = estimate_moments(sample_storage)

    # # Use train and test data without validation data
    # data = []
    # for l_id in range(n_levels):
    #     level_samples = estimator.get_level_samples(level_id=l_id)
    #     if l_id == 0:
    #         level_samples = np.concatenate((train_targets.reshape(1, len(train_targets), 1),
    #                                         targets.reshape(1, len(targets), 1)), axis=1)
    #     data.append(level_samples)
    # sample_storage_nn = create_quantity_mlmc(data)
    # moments_nn, estimator_nn, _, _ = estimate_moments(sample_storage_nn, true_domain=original_true_domain)



        # n0 = 100
        # nL = 10
        # num_levels = n_levels + 1
        # initial_n_samples = np.round(np.exp2(np.linspace(np.log2(n0), np.log2(nL), num_levels))).astype(int)
        # if len(initial_n_samples) == len(data):
        #     for i in range(len(data)):
        #         print(data[i].shape)
        #         data[i] = data[i][:, :initial_n_samples[i], :]
        #         print("data[i].shape ", data[i].shape)

        #level_params = [sample_storage.get_level_parameters()[0], *sample_storage.get_level_parameters()]


        # print("means nn ", moments_nn.mean)
        # print("means_predict ", moments_predict.mean)
        #
        # print("means nn - means predict ", moments_nn.mean - moments_predict.mean)
        # print("abs means nn - means predict ", np.abs(moments_nn.mean - moments_predict.mean))
        #
        # print("vars nn ", moments_nn.var)
        # print("vars predict ", moments_predict.var)
        #
        # print("moments_nn.l_means ", moments_nn.l_means[0])
        # print("moments_predict.l_means ", moments_predict.l_means[0])
        #
        # print("moments nn n samples ", moments_nn.n_samples)
        # print("moments nn n removed samples ", moments_predict.n_rm_samples)
        # print("moments predict n samples ", moments_predict.n_samples)
        # print("moments predict n removed samples ", moments_predict.n_rm_samples)
        #
        # for l_id, (l_mom, l_mom_pred) in enumerate(zip(moments_nn.l_means, moments_predict.l_means)):
        #     print("L id: {}, mom diff: {}".format(l_id, l_mom - l_mom_pred))

    domain = get_largest_domain([sample_storage, sample_storage_predict, ref_sample_storage])
    original_q_estimator = get_quantity_estimator(sample_storage, true_domain=domain)
    predict_q_estimator = get_quantity_estimator(sample_storage_predict, true_domain=domain)
    ref_estimator = get_quantity_estimator(ref_sample_storage, true_domain=domain)

    orig_moments_mean, predict_moments_mean, ref_moments_mean = compare_moments(original_q_estimator, predict_q_estimator, ref_estimator)

    kl_mlmc, kl_nn = -1, -1
    # kl_mlmc, kl_nn = compare_densities(original_q_estimator, predict_q_estimator, ref_estimator,
    #                   label_1="orig N: {}".format(sample_storage.get_n_collected()),
    #                   label_2="gnn N: {}".format(sample_storage_predict.get_n_collected()))

    if stats:
        return sample_storage.get_n_collected(), sample_storage_predict.get_n_collected(), n_ops, n_ops_predict, orig_moments_mean,\
               predict_moments_mean, ref_moments_mean, sample_storage.get_level_parameters(), sample_storage_predict.get_level_parameters()

    plot_moments({"ref": ref_estimator, "orig": original_q_estimator, "nn": predict_q_estimator})



def plot_moments(mlmc_estimators):
    n_moments = 25
    moments_plot = mlmc.tool.plot.MomentsPlots(
        title="Legendre {} moments".format(n_moments))

    # moments_plot = mlmc.tool.plot.PlotMoments(
    #     title="Monomial {} moments".format(self.n_moments), log_mean_y=False)

    for nl, estimator in mlmc_estimators.items():

        moments_mean = qe.estimate_mean(qe.moments(estimator._quantity, estimator._moments_fn))
        est_moments = moments_mean.mean
        est_vars = moments_mean.var

        n_collected = [str(n_c) for n_c in estimator._sample_storage.get_n_collected()]
        moments_plot.add_moments((moments_mean.mean, moments_mean.var), label="#{} N:".format(nl) + ", ".join(n_collected))

        # print("moments level means ", moments_mean.l_means)
        # print("moments level vars ", moments_mean.l_vars)
        # print("moments level max vars ", np.max(moments_mean.l_vars, axis=1))
        # print("est moments ", est_moments)
        # print("est_vars ", est_vars)
        # print("np.max(est_vars) ", np.max(est_vars))

    moments_plot.show(None)
    #moments_plot.show(file=os.path.join(self.work_dir, "{}_moments".format(self.n_moments)))
    moments_plot.reset()


def analyze_mlmc_data():
    n_levels = 5
    # mlmc_file = "/home/martin/Documents/metamodels/data/cl_0_3_s_4/L5/mlmc_5.hdf5"
    mlmc_file = "/home/martin/Documents/metamodels/data/cl_0_1_s_1/L5/mlmc_5.hdf5"

    sample_storage = SampleStorageHDF(file_path=mlmc_file)
    original_moments, estimator, original_true_domain = estimate_moments(sample_storage)

    # Test storage creation
    data = []
    for l_id in range(n_levels):
        level_samples = estimator.get_level_samples(level_id=l_id)
        l_fine = np.squeeze(level_samples[..., 0])

        print("mean l_fine ", np.mean(l_fine))
        plt.hist(l_fine, alpha=0.5, label='{}'.format(l_id), density=True)
        data.append(level_samples)

    plt.legend(loc='upper right')
    plt.show()
    sample_storage_2 = create_quantity_mlmc(data)
    moments_2, estimator_2, _ = estimate_moments(sample_storage_2)
    assert np.allclose(original_moments.mean, moments_2.mean)
    assert np.allclose(original_moments.var, moments_2.var)


def get_sample_times_mlmc(mlmc_file):
    sample_storage = SampleStorageHDF(file_path=mlmc_file)

    n_ops = sample_storage.get_n_ops()
    generate_rnd = sample_storage.get_generate_rnd_times()
    extract_mesh = sample_storage.get_extract_mesh_times()
    make_fields = sample_storage.get_make_field_times()
    coarse_flow = sample_storage.get_coarse_flow_times()
    fine_flow = sample_storage.get_fine_flow_times()

    def time_for_sample_func(data):
        new_n_ops = []
        for nop in data:
            nop = np.squeeze(nop)
            if len(nop) > 0:
                new_n_ops.append(nop[0] / nop[1])
        return new_n_ops

    print("generated rnd ", generate_rnd)

    generate_rnd = time_for_sample_func(generate_rnd)
    extract_mesh = time_for_sample_func(extract_mesh)
    make_fields = time_for_sample_func(make_fields)
    coarse_flow = time_for_sample_func(coarse_flow)
    fine_flow = time_for_sample_func(fine_flow)

    field_times = generate_rnd + extract_mesh + make_fields

    print("n ops ", n_ops)
    print("field times ", field_times)
    print("coarse flow ", coarse_flow)
    print("fine flow ", fine_flow)

    return n_ops, field_times, coarse_flow, fine_flow


def get_sample_times(sampling_info_path):
    n_levels = [5]
    for nl in n_levels:
        variances = []
        n_ops = []
        times = []

        times_scheduled_samples = []
        running_times = []
        flow_running_times = []

        for i in range(0, 100):
            sampling_info_path_iter = os.path.join(sampling_info_path, str(i))
            if os.path.isdir(sampling_info_path_iter):
                variances.append(np.load(os.path.join(sampling_info_path_iter, "variances.npy")))
                n_ops.append(np.load(os.path.join(sampling_info_path_iter, "n_ops.npy")))
                times.append(np.load(os.path.join(sampling_info_path_iter, "time.npy")))

                running_times.append(np.load(os.path.join(sampling_info_path_iter, "running_times.npy")))
                flow_running_times.append(np.load(os.path.join(sampling_info_path_iter, "flow_running_times.npy")))
                if os.path.exists(os.path.join(sampling_info_path_iter, "scheduled_samples_time.npy")):
                    times_scheduled_samples.append(
                        np.load(os.path.join(sampling_info_path_iter, "scheduled_samples_time.npy")))
            else:
                break

        def time_for_sample_func(data):
            new_n_ops = []
            for nop in data:
                nop = np.squeeze(nop)
                if len(nop) > 0:
                    new_n_ops.append(nop[:, 0]/nop[:, 1])
            return new_n_ops

        n_ops = time_for_sample_func(n_ops)
        running_times = time_for_sample_func(running_times)
        flow_running_times = time_for_sample_func(flow_running_times)


        field_times = np.mean(np.array(running_times) - np.array(flow_running_times) - np.array(flow_running_times),
                              axis=0)

        flow_times = np.mean(np.array(flow_running_times), axis=0)
        n_ops = np.mean(n_ops, axis=0)

        # print("n ops ", n_ops)
        # print("running times ", np.mean(running_times, axis=0))
        # print("flow running times ", flow_times)
        # exit()

        #n_ops = np.mean(running_times, axis=0)  # CPU time of simulation (fields + flow for both coarse and fine sample)

        print("field times ", field_times)

        print("n ops ", n_ops)
        print("type n ops ", type(n_ops))

        if np.isnan(np.all(n_ops)):
            n_ops = None

        return n_ops, field_times, flow_times


def plot_data(data, label):
    plt.hist(data, alpha=0.5, label=label, density=True)

    #plt.hist(predictions, bins=50, alpha=0.5, label='predictions', density=True)
    # plt.hist(targets - predictions, bins=50, alpha=0.5, label='predictions', density=True)
    plt.legend(loc='upper right')
    # plt.xlim(-0.5, 1000)
    #plt.yscale('log')
    plt.show()


if __name__ == "__main__":
    analyze_mlmc_data()
