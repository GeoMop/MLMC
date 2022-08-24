import os
import random
import copy
import matplotlib.pyplot as plt
import mlmc.estimator
from mlmc.tool import gmsh_io
import mlmc.quantity.quantity_estimate as qe
from mlmc.sample_storage import Memory
from mlmc.quantity.quantity_spec import QuantitySpec, ChunkSpec
import numpy as np
from mlmc.sample_storage_hdf import SampleStorageHDF
from mlmc.moments import Legendre, Monomial
from mlmc.quantity.quantity import make_root_quantity
from mlmc.metamodel.create_graph import extract_mesh_gmsh_io
from mlmc.plot import plots


QUANTILE = 1e-6
# QUANTILE = 0.001
N_MOMENTS = 20
TARGET_VAR = 1e-5


def use_levels_from_mlmc(sample_storage, use_levels):
    print("mlmc sample storage get N collected ", sample_storage.get_n_collected())
    n_levels = len(sample_storage.get_level_ids())
    original_moments, estimator, original_true_domain, _ = estimate_moments(sample_storage)
    data_mlmc = []
    new_coarse_data = []
    level_parameters = sample_storage.get_level_parameters()
    new_level_parameters = []

    mlmc_n_collected = estimator._sample_storage.get_n_collected()
    for l_id, l_n_collected in zip(range(n_levels), mlmc_n_collected):
        print("l id ", l_id)
        if l_id == np.max(use_levels):
            new_coarse_data = estimator.get_level_samples(level_id=1, n_samples=mlmc_n_collected[1])[:, :, 1]

        if l_id in use_levels:
            level_samples = estimator.get_level_samples(level_id=l_id, n_samples=l_n_collected)
            #fine_samples = level_samples[:, :, 0]
            #coarse_samples = level_samples[:, :, 1]
            print("level samples shape ", level_samples.shape)
        else:
            continue

        print("level samples shape ", level_samples.shape)


        # if len(data_mlmc) == 0:
        #     level_samples[:, :, 1] = 0
        if len(new_coarse_data) > 0:
            print("new coarse data len ", len(new_coarse_data[0]))
            print("new_coarse_data[:, :level_samples.shape[1]] ", new_coarse_data[:, :level_samples.shape[1]])
            level_samples[:, :, 1] = new_coarse_data[:, :level_samples.shape[1]]

        print("level samples ", level_samples)
        new_level_parameters.append(level_parameters[l_id])
        data_mlmc.append(level_samples)

    exit()

    final_sample_storage = create_quantity_mlmc(data_mlmc, level_parameters=np.array(new_level_parameters))

    return final_sample_storage


def remove_level(mlmc_hdf_file, rm_level_id=0, use_levels=[]):
    sample_storage = SampleStorageHDF(file_path=mlmc_hdf_file)

    print("mlmc sample storage get N collected ", sample_storage.get_n_collected())
    n_levels = len(sample_storage.get_level_ids())
    original_moments, estimator, original_true_domain, _ = estimate_moments(sample_storage)

    n_ops, field_times, coarse_flow, fine_flow = get_sample_times_mlmc(mlmc_hdf_file)
    n_ops_est = n_ops

    if len(use_levels) > 0:
        final_sample_storage = use_levels_from_mlmc(sample_storage, use_levels)
    else:
        data_mlmc = []
        mlmc_n_collected = estimator._sample_storage.get_n_collected()
        for l_id, l_n_collected in zip(range(n_levels), mlmc_n_collected):
            if l_id <= rm_level_id:
                continue
            print("l id ", l_id)
            if rm_level_id + 1 == l_id:
                level_samples = estimator.get_level_samples(level_id=l_id, n_samples=l_n_collected)
                level_samples[:, :, 1] = 0
                print("level sampels shape ", level_samples.shape)

            else:
                level_samples = estimator.get_level_samples(level_id=l_id, n_samples=l_n_collected)
            print("level samples ", level_samples)
            data_mlmc.append(level_samples)

        final_sample_storage = create_quantity_mlmc(data_mlmc, level_parameters=sample_storage.get_level_parameters()[rm_level_id+1:])

    return final_sample_storage


def cut_original_test(mlmc_hdf_file, n_levels=None):
    print("mlmc hdf file ", mlmc_hdf_file)
    sample_storage = SampleStorageHDF(file_path=mlmc_hdf_file)

    print("mlmc sample storage get N collected ", sample_storage.get_n_collected())

    n_levels = len(sample_storage.get_level_ids())
    original_moments, estimator, original_true_domain, _ = estimate_moments(sample_storage)

    n_ops, field_times, coarse_flow, fine_flow = get_sample_times_mlmc(mlmc_hdf_file)
    n_ops_est = n_ops

    data_mlmc = []
    mlmc_n_collected = estimator._sample_storage.get_n_collected()
    for l_id, l_n_collected in zip(range(n_levels), mlmc_n_collected):
        level_samples = estimator.get_level_samples(level_id=l_id, n_samples=l_n_collected)
        data_mlmc.append(level_samples)

    print("original level params", sample_storage.get_level_parameters())
    sample_storage = create_quantity_mlmc(data_mlmc, level_parameters=sample_storage.get_level_parameters())

    n0 = 2000
    nL = 100
    n_levels = sample_storage.get_n_levels()
    n_samples = np.round(np.exp2(np.linspace(np.log2(n0), np.log2(nL), n_levels))).astype(int)
    print('n samples ', n_samples)

    sample_storage_for_estimated = cut_samples(data_mlmc, sample_storage, n_samples)  # [2300])

    original_q_estimator_est = get_quantity_estimator(sample_storage_for_estimated)

    n_estimated_orig, l_vars_orig, n_samples_orig = get_n_estimated(sample_storage_for_estimated,
                                                                    original_q_estimator_est, n_ops=n_ops_est)
    print("n estimated orig ", n_estimated_orig)

    sample_storage_for_estimated = cut_samples(data_mlmc, sample_storage_for_estimated, n_estimated_orig,
                                               bootstrap=True)

def analyze_output(targets, mult_coef=1,dataset_config=None):

    if dataset_config.get('output_scale', False):
        print("targets[:10] ", targets[:10])
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        ax.hist(targets, bins=50, alpha=0.5, label='target', density=True)
        # fig.colorbar(cont)
        plt.title("training targets")
        plt.show()

        mean_targets = dataset_config.get('mean_output', False)
        var_targets = dataset_config.get('var_output', False)

        # mean_targets = np.mean(targets)
        print("mean targets ", mean_targets)
        # print("mean targets axis=0 " , np.mean(targets, axis=0))
        # var_targets = np.var(targets)

        targets = var_targets * targets + mean_targets

        targets_orig = np.exp(targets)
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))

        ax.hist(targets_orig, bins=50, alpha=0.5, label='targets orig', density=True)
        # fig.colorbar(cont)
        plt.title("targets orig")
        plt.show()

        new_mult_coef = 1 / np.mean(targets_orig)
        print("np.log(targets_orig * new_mult_coef)[:10] ", np.log(targets_orig * new_mult_coef)[:10])
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        ax.hist(np.log(targets_orig * new_mult_coef), bins=50, alpha=0.5, label='targets orig', density=True)
        # fig.colorbar(cont)
        plt.title("log(targets orig * mult coef)")
        plt.show()

        new_mult_coef = 1 / np.mean(np.log(targets_orig))
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        ax.hist(np.log(targets_orig) * new_mult_coef, bins=50, alpha=0.5, label='targets orig', density=True)
        # fig.colorbar(cont)
        plt.title("log(targets orig) * mult coef")
        plt.show()

        print("mean log targets orig ", np.mean(np.log(targets_orig)))

        log_scaled_targets_orig = (np.log(targets_orig) - np.mean(np.log(targets_orig))) / np.var(np.log(targets_orig))
        print("log scaled targets orig[:10] ", log_scaled_targets_orig[:10])
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        ax.hist(log_scaled_targets_orig, bins=50, alpha=0.5, label='targets orig', density=True)
        # fig.colorbar(cont)
        plt.title("log scaled target orig")
        plt.show()

        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        ax.hist(np.log(targets_orig), bins=50, alpha=0.5, label='target orig log', density=True)
        # fig.colorbar(cont)
        plt.title("log targets orig")
        plt.show()


    else:
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        ax.hist(targets, bins=50, alpha=0.5, label='target', density=True)
        #fig.colorbar(cont)
        plt.title("training targets")
        plt.show()

        targets_orig = np.exp(targets) / mult_coef
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))

        ax.hist(targets_orig, bins=50, alpha=0.5, label='targets orig', density=True)
        # fig.colorbar(cont)
        plt.title("targets orig")
        plt.show()

        new_mult_coef = 1/np.mean(targets_orig)
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        ax.hist(np.log(targets_orig*new_mult_coef), bins=50, alpha=0.5, label='targets orig', density=True)
        # fig.colorbar(cont)
        plt.title("log(targets orig * mult coef)")
        plt.show()

        new_mult_coef = 1 / np.mean(np.log(targets_orig))
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        ax.hist(np.log(targets_orig) * new_mult_coef, bins=50, alpha=0.5, label='targets orig', density=True)
        # fig.colorbar(cont)
        plt.title("log(targets orig) * mult coef")
        plt.show()

        log_scaled_targets_orig = (np.log(targets_orig) - np.mean(np.log(targets_orig))) / np.var(np.log(targets_orig))
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        ax.hist(log_scaled_targets_orig, bins=50, alpha=0.5, label='targets orig', density=True)
        # fig.colorbar(cont)
        plt.title("log scaled target orig")
        plt.show()



        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        ax.hist(np.log(targets_orig), bins=50, alpha=0.5, label='target orig log', density=True)
        # fig.colorbar(cont)
        plt.title("log targets orig")
        plt.show()

    exit()


def _variance_cost_analysis(l_var_nn, n_ops_predict):

    print("np.max(l_var_nn, axis=0) ", np.max(l_var_nn, axis=1))
    vl_cl = np.max(l_var_nn, axis=1) * n_ops_predict

    print("VlCl ", vl_cl)

    beta = {}
    gamma = {}
    beta_1 = {}
    gamma_1 = {}
    moments_m = [1, 5, 10, 15, 20]
    #n_collected = estimator._sample_storage.get_n_collected()
    # cost_levels = n_collected * np.array(new_n_ops)
    cost_levels = np.array(n_ops_predict)

    max_level_var = True
    beta_levels = []
    gamma_levels = []
    beta_levels_2 = []
    gamma_levels_2 = []

    level_vars = np.max(l_var_nn, axis=1)

    level_costs = n_ops_predict

    print("cost levels ", cost_levels)
    for l_id, l_vars in enumerate(l_var_nn, start=1):
        # print("l_vars ", l_vars)
        # print("l id ", l_id)
        if max_level_var is True:
            if l_id < len(level_vars):
                beta_levels.append(-1 * np.log(level_vars[l_id]/level_vars[l_id-1]))
                gamma_levels.append(np.log2(level_costs[l_id] / level_costs[l_id - 1]))

            if l_id == 1:
                beta_levels_2.append(-1 * (np.log(level_vars[l_id - 1])))
                gamma_levels_2.append(np.log2(level_costs[l_id-1]))
            else:
                beta_levels_2.append(-1 * (np.log(level_vars[l_id-1])/(l_id-1)))
                gamma_levels_2.append(np.log2(level_costs[l_id - 1])/(l_id-1))
        else:
            for moment in moments_m:
                if moment not in beta:
                    beta[moment] = {}
                    beta_1[moment] = {}

                if l_id not in beta[moment]:
                    beta[moment][l_id] = []
                    beta_1[moment][l_id] = []

                if l_id not in gamma:
                    gamma[l_id] = []
                    gamma_1[l_id] = []

                # print("l_id ", l_id)
                # print("len l vars ", len(l_vars))
                # print("moment ", moment)
                # print("beta ", beta)
                beta[moment][l_id].append(-1 * np.log2(l_vars[moment]) / l_id)

                # print("level l vars ", l_vars)
                # print("moments_mean.l_vars[l_id-2] ", moments_mean.l_vars[l_id-1])
                if l_id < len(cost_levels):
                    beta_1[moment][l_id].append(-1 * np.log2(l_var_nn[l_id][moment] / l_vars[moment]))

            gamma[l_id].append(np.log2(cost_levels[l_id - 1]) / l_id)

            if l_id < len(cost_levels):
                # print("cost levles ", cost_levels)
                # print("l id ", l_id)
                gamma_1[l_id].append(np.log2(cost_levels[l_id] / cost_levels[l_id - 1]))

    print("beta levels ", beta_levels)
    print("gamma levels ", gamma_levels)

    print("beta levels 2", beta_levels_2)
    print("gamma levels 2", gamma_levels_2)

    beta_curve = []
    gamma_curve = []

    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    ax.plot(level_vars, label="vars")
    ax.plot(level_costs, label="costs")

    for b, g in zip(beta_levels, gamma_levels):
        beta_curve = []
        gamma_curve = []
        for l in range(len(level_vars)):
            beta_curve.append(2**(-1*b*l))
            gamma_curve.append(2 ** (g * l))

        ax.plot(beta_curve, label="beta: {}".format(b))
        ax.plot(gamma_curve, label="gamma: {}".format(g))

    # fig.colorbar(cont)
    plt.title("levels - var, cost")
    plt.yscale("log")
    plt.legend()
    plt.show()


    print("beta: {}, beta_1:{}".format(beta, beta_1))
    print("gamma: {}, gamma_1:{}".format(gamma, gamma_1))

    exit()


def process_mlmc(nn_hdf_file, sampling_info_path, ref_mlmc_file, targets, predictions, train_targets, train_predictions,
                 val_targets, l_0_targets=None, l_0_predictions=None,
                 l1_sample_time=None, l0_sample_time=None, nn_level=0, replace_level=False, stats=False, mlmc_hdf_file=None,
                 learning_time=0, dataset_config={}, targets_to_est=None, predictions_to_est=None):
    """
    :param l1_sample_time: preprocess_time / len(data) + learning_time / len(data),
                           preprocess_time includes graph creation time and FlowDataset creation time
    """
    # level_zero = False
    cut_est = True
    all_samples = False
    n0, nL = 2000, 100
    domain_largest = True  # If False than domain=None - domain is determined given the simulation samples
    distr_domain_largest = False#False
    replace_level = False#False

    rm_level = None#0
    rm_level_nn = None#None
    use_levels = []#[0, 2]


    #cut_original_test(nn_hdf_file)
    # exit()

    # output_mult_factor = dataset_config.get('output_mult_factor', 1)
    # #print("output mult factor ", output_mult_factor)
    # #analyze_output(targets, output_mult_factor, dataset_config=dataset_config)
    # plt.hist(targets, bins=50, alpha=0.5, label='target', density=True)
    # plt.hist(predictions, bins=50, alpha=0.5, label='predictions', density=True)
    #
    # # plt.hist(targets - predictions, bins=50, alpha=0.5, label='predictions', density=True)
    # plt.legend(loc='upper right')
    # # plt.xlim(-0.5, 1000)
    # plt.yscale('log')
    # plt.show()
    #
    # plt.hist(l_0_targets, bins=50, alpha=0.5, label='l_0_target', density=True)
    # plt.hist(l_0_predictions, bins=50, alpha=0.5, label='l_0_predictions', density=True)
    #
    # # print("lo targets ", l_0_targets)
    # # print("l0 predictions ", l_0_predictions)
    # # exit()
    #
    # # # plt.hist(targets - predictions, bins=50, alpha=0.5, label='predictions', density=True)
    # plt.legend(loc='upper right')
    # # plt.xlim(-0.5, 1000)
    # plt.yscale('log')
    # plt.show()
    #
    #
    # plt.hist(targets, bins=50, alpha=0.5, label='target', density=True)
    # plt.hist(predictions, bins=50, alpha=0.5, label='predictions', density=True)
    #
    # plt.legend(loc='upper right')
    # # plt.xlim(-0.5, 1000)
    # plt.yscale('log')
    #
    # plt.show()
    # #
    # plt.hist(targets_to_est, bins=50, alpha=0.5, label='target to est', density=True)
    # plt.hist(predictions_to_est, bins=50, alpha=0.5, label='predictions to est', density=True)
    #
    # plt.legend(loc='upper right')
    # # plt.xlim(-0.5, 1000)
    # plt.yscale('log')
    # plt.show()

    if targets_to_est is not None and predictions_to_est is not None:
        print("len targets_to_est ", len(targets_to_est))
        targets = np.concatenate((targets, targets_to_est), axis=0)
        predictions = np.concatenate((predictions, predictions_to_est), axis=0)
        # targets = np.concatenate((targets, targets_to_est), axis=0)
        # predictions = np.concatenate((predictions, predictions_to_est), axis=0)

        # l_0_targets = np.concatenate((l_0_targets, targets_to_est), axis=0)
        # l_0_predictions = np.concatenate((l_0_predictions, predictions_to_est), axis=0)

        # targets = np.concatenate((targets, l_0_targets), axis=0)
        # predictions = np.concatenate((predictions, l_0_predictions), axis=0)

        # targets = np.concatenate((targets, targets_to_est[:5000]), axis=0)
        # predictions = np.concatenate((predictions, predictions_to_est[:5000]), axis=0)

        # l_0_targets = np.concatenate((l_0_targets, targets_to_est), axis=0)
        # l_0_predictions = np.concatenate((l_0_predictions, predictions_to_est), axis=0)

        # targets = np.concatenate((targets, l_0_targets[-1000:]), axis=0)
        # predictions = np.concatenate((predictions, l_0_predictions[-1000:]), axis=0)

        # print("len targets to est", len(targets_to_est))
        # print("len predictions to est ", len(predictions_to_est))
        # exit()


    print("len(targets ", len(targets))
    print("len l0 targets ", len(l_0_targets))


    if dataset_config.get('output_normalization', False):
        min_out = dataset_config.get('min_output')
        max_out = dataset_config.get('max_output')

        targets = targets * (max_out - min_out) + min_out
        predictions = predictions * (max_out - min_out) + min_out
        l_0_targets = l_0_targets * (max_out - min_out) + min_out
        l_0_predictions = l_0_predictions * (max_out - min_out) + min_out

    if dataset_config.get('output_scale', False):
        # mean_targets = np.mean(targets)
        # var_targets = np.var(targets)

        mean_targets = dataset_config.get('mean_output', False)
        var_targets = dataset_config.get('var_output', False)

        targets = var_targets * targets + mean_targets
        predictions = var_targets * predictions + mean_targets

        # mean_l_0_targets = mean_targets
        # var_l_0_targets = var_targets

        l_0_targets = var_targets * l_0_targets + mean_targets
        l_0_predictions = var_targets * l_0_predictions + mean_targets

    if dataset_config.get('output_log', False):
        targets = np.exp(targets)
        predictions = np.exp(predictions)
        l_0_predictions = np.exp(l_0_predictions)
        l_0_targets = np.exp(l_0_targets)

    if dataset_config.get('first_log_output', False):
        targets = np.exp(targets)
        predictions = np.exp(predictions)
        l_0_predictions = np.exp(l_0_predictions)
        l_0_targets = np.exp(l_0_targets)

    print("targets ", targets)
    print("len targets ", len(targets))

    # if not stats:
    # print("nn_level ", nn_level)
    # print("replace level ", replace_level)

    # targets = np.exp(targets)
    # predictions = np.exp(predictions)
    # l_0_predictions = np.exp(l_0_predictions)
    # l_0_targets = np.exp(l_0_targets)

    # print("targets ", targets)
    # print("predictions ", predictions)
    # print("targets ", targets)
    # print("predictions ", predictions)
    # plt.hist(targets, bins=50, alpha=0.5, label='target', density=True)
    # plt.hist(predictions, bins=50, alpha=0.5, label='predictions', density=True)
    #
    # # plt.hist(targets - predictions, bins=50, alpha=0.5, label='predictions', density=True)
    # plt.legend(loc='upper right')
    # # plt.xlim(-0.5, 1000)
    # plt.yscale('log')
    # plt.show()
    #
    # plt.hist(l_0_targets, bins=50, alpha=0.5, label='l_0_target', density=True)
    # plt.hist(l_0_predictions, bins=50, alpha=0.5, label='l_0_predictions', density=True)
    #
    # # print("lo targets ", l_0_targets)
    # # print("l0 predictions ", l_0_predictions)
    # # exit()
    #
    # # plt.hist(targets - predictions, bins=50, alpha=0.5, label='predictions', density=True)
    # plt.legend(loc='upper right')
    # # plt.xlim(-0.5, 1000)
    # plt.yscale('log')
    # plt.show()
    #

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
    if mlmc_hdf_file is None:
        mlmc_hdf_file = nn_hdf_file

    if rm_level is not None:
        sample_storage = remove_level(mlmc_hdf_file, rm_level_id=rm_level, use_levels=use_levels)
    else:
        sample_storage = SampleStorageHDF(file_path=mlmc_hdf_file)

    print("mlmc sample storage get N collected ", sample_storage.get_n_collected())

    n_levels = len(sample_storage.get_level_ids())
    original_moments, estimator, original_true_domain, _ = estimate_moments(sample_storage)

    print("moments.mean ", original_moments.mean)
    print("moments.var ", original_moments.var)

    if rm_level_nn is not None:
        sample_storage_nn = remove_level(nn_hdf_file, rm_level_id=rm_level_nn, use_levels=use_levels)
    else:
        sample_storage_nn = SampleStorageHDF(file_path=nn_hdf_file)


    original_moments_nn, estimator_nn, original_true_domain_nn, _ = estimate_moments(sample_storage_nn)
    print("nn moments.mean ", original_moments_nn.mean)
    print("nn moments.var ", original_moments_nn.var)

    orig_max_vars = np.max(original_moments.l_vars, axis=1)
    # print("orig max vars ", orig_max_vars)

    ######
    ### Get n ops
    ######
    n_ops, field_times, coarse_flow, fine_flow = get_sample_times_mlmc(mlmc_hdf_file)
    #n_ops = [20.279451930908973, 87.91808330548963, 216.172210888505, 892.2780022583306, 2646.912581985272]
    # n_ops, _, _ = get_sample_times(sampling_info_path)
    print("n ops ", n_ops)

    n_ops = n_ops[-sample_storage.get_n_levels():]
    # field_times = field_times[-sample_storage.get_n_levels():]
    # coarse_flow = coarse_flow[-sample_storage.get_n_levels():]
    # fine_flow = fine_flow[-sample_storage.get_n_levels():]

    if n_ops is None:
        n_ops = sample_storage.get_n_ops()
        field_times = np.zeros(len(n_ops))
        flow_times = np.zeros(len(n_ops))
    # Test storage creation
    data_mlmc = []
    mlmc_n_collected = estimator._sample_storage.get_n_collected()
    for l_id, l_n_collected in zip(range(n_levels), mlmc_n_collected):
        level_samples = estimator.get_level_samples(level_id=l_id, n_samples=l_n_collected)
        data_mlmc.append(level_samples)

    print("original level params", sample_storage.get_level_parameters())
    sample_storage = create_quantity_mlmc(data_mlmc, level_parameters=sample_storage.get_level_parameters())
    if cut_est:
        # n0 = 2000
        # nL = 100
        n_levels = sample_storage.get_n_levels()

        if all_samples:
            n_samples = sample_storage.get_n_collected()
        else:
            n_samples = np.round(np.exp2(np.linspace(np.log2(n0), np.log2(nL), n_levels))).astype(int)

        print('cut n samples for sample storgae estimate', n_samples)
        sample_storage_for_estimated = cut_samples(data_mlmc, sample_storage, n_samples)  # [2300])

    print("Original storage")
    orig_storage_n_collected, orig_storage_max_vars = get_storage_info(sample_storage)
    print("orig storage max vars ", orig_storage_max_vars)

    mlmc_nn_diff_level = 1
    if replace_level:
        n_lev = n_levels
        level_params = [*sample_storage.get_level_parameters()]
    else:
        n_lev = n_levels - nn_level + 1
        level_params = [sample_storage.get_level_parameters()[nn_level],
                        *sample_storage.get_level_parameters()[mlmc_nn_diff_level - 1:]]

    # Use predicted data as zero level results and level one coarse results
    data_nn = []
    n_ops_predict = []

    print("l0_sample_time ", l0_sample_time)
    print("l1_sample_time ", l1_sample_time)

    ############################
    ### Calculate First level (follows meta-level) n_ops
    ############################
    print("learning time ", learning_time)
    print("l1_sample_time ", l1_sample_time)


    #learning_time = learning_time / 4
    C_1_cost = 0
    C_2_cost = 0

    #len_data = 50000  # 50000#80000
    len_data = 4000 # case with independent samples
    len_train_data = 2000
    len_data_preprocess = len_data + 50000
    l0_predict_time = 1e-3
    preprocess_time = l1_sample_time * len_data - learning_time
    preprocess_time_per_sample = preprocess_time / len_data_preprocess

    learning_time = 310 # cl_0_1_s_1: L1 - 180, L2 - 220, L3 - 310, L4 - 820, L5 - 3320.261721275
    learning_time = 2840 # 02 conc cond, L1 - 400, L2 - 400 , L3 - 450, L4 - 550 , L5 - 2840

    #n_ops_train = preprocess_time_per_sample + (learning_time / len_train_data) + l0_predict_time

    C_1_cost = l0_sample_time  # time for generating random field
    C_2_cost = preprocess_time_per_sample + l0_predict_time + n_ops[0]


    print("C1 cost: {}, C2 cost: {}".format(C_1_cost, C_2_cost))


    ####
    # Notes:
    #  l1_sample_time = preprocess_time / len(data) + learning_time / len(data)
    #
    #  n_ops_train includes:
    #                       preprocess time per sample
    #                       learning time per training sample
    #                       l0_prediction_time - the time necessary to predict neural network outcome for given already preprocessed data
    #l1_sample_time = n_ops_train
    # New l1_sample_time corresponds to the time necessary for Level 1 samples (fine + coarse)
    #
    #################################

    #print("L1 sample time ", l1_sample_time)

    print("learning time ", learning_time)
    print("preprocess time ", preprocess_time)
    print("preprocess time per sample ", preprocess_time_per_sample)

    #print("new l1 sample time ", l1_sample_time)


    # print("n ops train ", n_ops_train)
    # print("n ops test ", n_ops_test)

    # n_ops = n_ops_0 + (n_ops_train * (len_train_data / nn_estimated) + n_ops_test * (len_test_data / nn_estimated)) / (
    #     nn_estimated)
    # n_ops += predict_l0_time

    # l0_sample_time, l1_sample_time = l1_sample_time, l0_sample_time
    nn_n_collected = estimator_nn._sample_storage.get_n_collected()
    print("nn n collected ", nn_n_collected)

    nn_lev_sim_time = 0
    for l_id in range(nn_level):
        nn_lev_sim_time = n_ops[l_id] - nn_lev_sim_time

    for l_id in range(n_lev):
        if l_id == 0:
            level_samples = l_0_predictions.reshape(1, len(l_0_predictions), 1)
            # level_samples = np.ones((1, len(l_0_predictions), 1))
            ft_index = nn_level

            # if output_mult_factor != 1:
            #     level_samples /= output_mult_factor

            if nn_level > 0:
                ft_index = nn_level - 1
            n_ops_predict.append(l0_sample_time)  # + field_times[ft_index] / 2)

            print("L0 n ops predict ", n_ops_predict)

            # print("l0_sample_time ", l0_sample_time)
            # print("len l0 predictions ", len(l_0_predictions))
            # exit()
        else:
            if replace_level:
                level_id = l_id
                level_samples = estimator_nn.get_level_samples(level_id=level_id, n_samples=nn_n_collected[level_id])

                n_ops_predict.append(n_ops[level_id])
                print("replace level n ops ", n_ops_predict)
            else:
                if l_id < mlmc_nn_diff_level:
                    continue
                if l_id == mlmc_nn_diff_level:
                    coarse_level_samples = predictions.reshape(1, len(predictions), 1)
                    fine_level_samples = targets.reshape(1, len(targets), 1)

                    # print("coarse level samples ", coarse_level_samples)
                    # print("fine level sampels ", fine_level_samples)
                    # exit()

                    # coarse_level_samples = np.concatenate((coarse_level_samples,
                    #                 train_predictions.reshape(1, len(train_predictions), 1)), axis=1)
                    #
                    # fine_level_samples = np.concatenate((fine_level_samples,
                    #                                      train_targets.reshape(1, len(train_targets), 1)), axis=1)

                    # coarse_level_samples = coarse_level_samples[:, :len(predictions)//2, :]
                    # fine_level_samples = fine_level_samples[:, :len(predictions)//2, :]
                    level_samples = np.concatenate((fine_level_samples, coarse_level_samples), axis=2)
                    print("nn level ", nn_level)

                    # if output_mult_factor != 1:
                    #     level_samples /= output_mult_factor
                    print("level sampels corase and fine shape", fine_level_samples.shape)

                    print("level sampels ", level_samples)
                    print("fine - coarse ", fine_level_samples - coarse_level_samples)
                    print("fine - coarse ", np.var(fine_level_samples - coarse_level_samples))

                    # if output_mult_factor != 1:
                    #     level_samples /= output_mult_factor

                    n_ops_predict.append(n_ops[mlmc_nn_diff_level - 1] - nn_lev_sim_time + l1_sample_time)
                else:
                    if replace_level:
                        level_id = l_id  # - 1
                    else:
                        level_id = l_id + nn_level - 1
                    if level_id >= len(nn_n_collected):
                        level_samples = []
                        n_ops_predict[-1] = n_ops[level_id]
                        continue
                    else:
                        level_samples = estimator_nn.get_level_samples(level_id=level_id,
                                                                       n_samples=nn_n_collected[level_id])
                        print('n ops ', n_ops)
                        print("level id ", level_id)

                        n_ops_predict.append(n_ops[level_id])
                    # print("n ops predict append", n_ops_predict)

                    # if output_mult_factor != 1:
                    #     level_samples /= output_mult_factor

        print("level samples rescaled ", level_samples)

        #level_samples = np.log(level_samples)
        #level_samples /= output_mult_factor

        print("leel samples exp ", level_samples)

        #print("level samples exp ", level_samples)

        data_nn.append(level_samples)

    print("n ops predict ", n_ops_predict)

    # n_ops_predict[1] += n_ops_predict[1]*0.2

    print("level params ", level_params)
    sample_storage_predict = create_quantity_mlmc(data_nn, level_parameters=level_params)
    if cut_est:
        # n0 = 2000
        # nL = 100
        n_levels = sample_storage_predict.get_n_levels()

        #n_samples = np.round(np.exp2(np.linspace(np.log2(n0), np.log2(nL), n_levels))).astype(int)

        if all_samples:
            n_samples = sample_storage.get_n_collected()
        else:
            n_samples = np.round(np.exp2(np.linspace(np.log2(n0), np.log2(nL), n_levels))).astype(int)

        print('cut n samples for MC+NN estimate', n_samples)
        sample_storage_predict_for_estimate = cut_samples(data_nn, sample_storage_predict, n_samples)

        # print("n ops predict ", n_ops_predict)
        print("Storage predict info")
        predict_storage_n_collected, predict_storage_max_vars = get_storage_info(sample_storage_predict_for_estimate)
        print("predict storage n collected ", predict_storage_n_collected)
        print("predict storage max vars ", predict_storage_max_vars)

    # if stats:
    #     return orig_storage_max_vars, predict_storage_max_vars

    ######
    ### Create estimators
    ######
    ref_sample_storage = ref_storage(ref_mlmc_file)
    if domain_largest:
        domain = get_largest_domain([sample_storage, sample_storage_predict, ref_sample_storage])
    else:
        domain = None
    original_q_estimator = get_quantity_estimator(sample_storage, true_domain=domain)
    predict_q_estimator = get_quantity_estimator(sample_storage_predict, true_domain=domain)

    if cut_est:
        if domain_largest:
            domain = get_largest_domain([sample_storage_for_estimated, sample_storage_predict_for_estimate, ref_sample_storage])
        else:
            domain = None
        original_q_estimator_est = get_quantity_estimator(sample_storage_for_estimated, true_domain=domain)
        predict_q_estimator_est = get_quantity_estimator(sample_storage_predict_for_estimate, true_domain=domain)
    # ref_estimator = get_quantity_estimator(ref_sample_storage, true_domain=domain)

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

    # print("n ops ", n_ops)
    # print("n ops predict ", n_ops_predict)
    # exit()
    # n_ops = [30.1978988484516, 2321.096786450282]
    # n_ops_predict = [0.33790084, 30.91978988484516, 2321.096786450282]

    #### Original data
    n_ops_est = copy.deepcopy(n_ops)
    if not cut_est:
        # n_ops_est[0] = n_ops_est[0] / 1000
        print("sample_storage.get_n_collected() ", sample_storage.get_n_collected())
        n_estimated_orig, l_vars_orig, n_samples_orig = get_n_estimated(sample_storage, original_q_estimator,
                                                                        n_ops=n_ops_est)

        print("n estimated orig ", n_estimated_orig)
        print("n ops est ", n_ops_est)
        #print("sample storage for estimated n collected ", sample_storage_for_estimated.get_n_collected())

    ######
    ## initial N guess
    if cut_est:
        print("n ops est ", n_ops_est)
        n_estimated_orig, l_vars_orig, n_samples_orig = get_n_estimated(sample_storage_for_estimated,
                                                                        original_q_estimator_est, n_ops=n_ops_est)
        print("n estimated orig ", n_estimated_orig)

        sample_storage_for_estimated = cut_samples(data_mlmc, sample_storage, n_estimated_orig,
                                                   bootstrap=False)

        # Another estimate simulate adding samples algo
        original_q_estimator_est_2 = get_quantity_estimator(sample_storage_for_estimated, true_domain=domain)
        n_estimated_orig, l_vars_orig, n_samples_orig = get_n_estimated(sample_storage_for_estimated,
                                                                        original_q_estimator_est_2, n_ops=n_ops_est)

    print("new n estimated orig ", n_estimated_orig)

    sample_storage = cut_samples(data_mlmc, sample_storage, n_estimated_orig, bootstrap=False)


    # exit()

    n_estimated_nn, l_var_nn, n_samples_nn = get_n_estimated(sample_storage_predict_for_estimate, predict_q_estimator_est,
                                                             n_ops=n_ops_predict)

    # print("l var nn ", l_var_nn)
    # print("n ops predict ", n_ops_predict)
    # print("n estimated nn ", n_estimated_nn)
    #
    # print("n estimated orig ", n_estimated_orig)

    # new_n_estimated_nn = []
    # for n_est in n_estimated_nn:
    #     new_n_estimated_nn.append(int(n_est + n_est * 0.2))
    # n_estimated_nn = new_n_estimated_nn

    # ("new n estimated nn ", n_estimated_nn)
    # exit()
    # n_estimated_nn = [50000, 10000, 850]
    # sample_storage_predict_2 = cut_samples(data_nn, sample_storage_predict, n_estimated_nn)
    # predict_q_estimator_2 = get_quantity_estimator(sample_storage_predict_2, true_domain=domain)
    #
    #
    # new_n_estimated_nn, new_l_var_nn, new_n_samples_nn = get_n_estimated(sample_storage_predict_2, predict_q_estimator_2,
    #                                                          n_ops=n_ops_predict)

    # print("new n estimated nn", new_n_estimated_nn)

    # n_estimated_nn = new_n_estimated_nn
    sample_storage_predict_0 = copy.deepcopy(sample_storage_predict)

    # predict_q_estimator.quantity = predict_q_estimator.quantity.subsample(sample_vec=n_estimated_nn)

    print("n estiamted nn ", n_estimated_nn)
    print("n ops predict ", n_ops_predict)

    ###############
    ## Recalculate first level (level of metamodel and simulation difference) n ops
    ###############
    # Use NN as new MC
    if len(n_estimated_nn) == 1:
        cost_tr = C_2_cost * len_train_data
        if n_estimated_nn[0] > len_train_data:
            cost_te = C_1_cost * (n_estimated_nn[0] - len_train_data)
        else:
            cost_te = 0

        # n_ops_predict_C_2 = [C_2_cost]
        # print("l1 sample time ", l1_sample_time)
        # #orig_n_ops = n_ops_predict[0] - l1_sample_time
        # cost_tr = (l1_sample_time + n_ops[0]) * (len_train_data)
        # n_ops_test = (preprocess_time_per_sample + l0_predict_time)
        # cost_te = n_ops_test * (n_estimated_nn[0] - len_train_data)
        #
        # print("cost tr ", cost_tr)
        # print("cost te ", cost_te)

        #n_ops_predict[1] = orig_n_ops + ((cost_tr + cost_te) / n_estimated_nn[1])

        n_ops_predict[0] = ((cost_tr + cost_te) / n_estimated_nn[0])

        print("n ops predict final", n_ops_predict)
        print("n ops ", n_ops)
    else:
        # n_ops_test = (preprocess_time_per_sample + l0_predict_time)  # * (n_estimated_nn[1] - len_train_data)
        # if n_estimated_nn[1] > len_train_data:
        #     orig_n_ops = n_ops_predict[1] - l1_sample_time
        #     cost_tr = l1_sample_time * (len_train_data)  # / n_estimated_nn[1])
        #     cost_te = n_ops_test * (n_estimated_nn[1] - len_train_data)  # / n_estimated_nn[1])
        #
        #     # note L1 sample time is n_ops_train
        #
        #     n_ops_predict[1] = orig_n_ops + ((cost_tr + cost_te) / n_estimated_nn[1])
        #
        #     print("preprocess time per sample", preprocess_time_per_sample)
        #     print("orig n ops ", orig_n_ops)
        #     print("cost_tr ", cost_tr)
        #     print("cost te ", cost_te)

        n_ops_predict[0] = C_1_cost
        n_ops_predict[1] = C_2_cost

    #n_ops_predict = [0.33790084, 30.91978988484516, 2321.096786450282]
    n_ops_predict_orig = n_ops_predict

    ######
    ## initial N guess
    if cut_est:
        n_estimated_nn, l_var_nn, n_samples_nn = get_n_estimated(sample_storage_predict_for_estimate,
                                                                 predict_q_estimator_est, n_ops=n_ops_predict)

        print("NN FIRST n estimated nn ", n_estimated_nn)
        all_n_estimated_nn = []
        for i in range(10):

            sample_storage_predict_for_est_2 = cut_samples(data_nn, sample_storage_predict, n_estimated_nn, bootstrap=True)

            predict_q_estimator_est_2 = get_quantity_estimator(sample_storage_predict_for_est_2, true_domain=domain)

            n_estimated_nn, l_var_nn, n_samples_nn = get_n_estimated(sample_storage_predict_for_est_2,
                                                                     predict_q_estimator_est_2,
                                                                     n_ops=n_ops_predict)

            all_n_estimated_nn.append(n_estimated_nn)

        print("all n estimated nn ", all_n_estimated_nn)
        n_estimated_nn = np.mean(all_n_estimated_nn, axis=0, dtype=np.int32)
        print("mean n estimated nn ", n_estimated_nn)


    # n_estimated_nn, l_var_nn, n_samples_nn = get_n_estimated(sample_storage_predict_for_est_2, predict_q_estimator_est_2,
    #                                                          n_ops=n_ops_predict)

    print("NN SECOND n collected before estimate ", sample_storage_predict_for_est_2.get_n_collected())
    print("NN SECOND n estimated nn ", n_estimated_nn)

    # n_estimated_nn, l_var_nn, n_samples_nn = get_n_estimated(sample_storage_predict,
    #                                                          predict_q_estimator,
    #                                                          n_ops=n_ops_predict)

    # n_estimated_nn_C1_C2, _, _ = get_n_estimated(sample_storage_predict, predict_q_estimator,
    #                                                          n_ops=n_ops_predict)

    print("n ops ", n_ops)
    print("new n ops predict ", n_ops_predict)
    print("new n estimated nn ", n_estimated_nn)
    #print("n ops predict C1 C2", n_ops_predict_C1_C2)
    #print("new n estimated nn C1 C2 ", n_estimated_nn_C1_C2)
    # exit()

    sample_storage_predict = cut_samples(data_nn, sample_storage_predict, n_estimated_nn)

    moms, _, _, _ = estimate_moments(sample_storage_predict)
    print("moms.vars ", moms.var)
    # exit()

    #sample_storage_predict = sample_storage_predict_for_estimate

    # predict_q_estimator = get_quantity_estimator(sample_storage_predict_0, true_domain=domain)
    # predict_q_estimator.quantity = predict_q_estimator.quantity.subsample(sample_vec=n_estimated_nn)

    print("new n estimated nn ", n_estimated_nn)
    print("new l var nn ", l_var_nn)

    ###############################
    ###  Variance cost relation ###
    ###############################
    #_variance_cost_analysis(l_var_nn, n_ops_predict)


    # predict_moments = compute_moments(sample_storage_predict)
    # print("predict moments var ", predict_moments.var)

    #######
    ## Estimate total time
    #######
    print("NN estimated ", n_estimated_nn)
    print("MLMC estimated ", n_estimated_orig)
    print("n ops predict_orig ", n_ops_predict_orig)
    print("n ops ", n_ops)
    #print("n estimated nn[1] ", n_estimated_nn[1])
    # n_ops_predict_orig = n_ops_predict
    # n_ops_test = preprocess_time / (n_estimated_nn[1] - len_train_data)
    # if n_estimated_nn[1] > len_train_data:
    #     n_ops_predict_orig[1] = n_ops_predict_orig[1] - l1_sample_time + ((l1_sample_time * (len_train_data/n_estimated_nn[1]) + \
    #                              n_ops_test * ((n_estimated_nn[1] - len_train_data)/n_estimated_nn[1])) / n_estimated_nn[1])
    # n_ops_predict = n_ops_predict_orig
    # # print("n ops predict ", n_ops_predict)

    NN_time_levels = n_ops_predict_orig * np.array(n_estimated_nn)
    n_collected_times = n_ops * np.array(n_estimated_orig)

    print("NN time levels ", NN_time_levels)
    print("MLMC time levels", n_collected_times)
    nn_total_time = np.sum(NN_time_levels) + learning_time
    print("NN total time ", nn_total_time)
    mlmc_total_time = np.sum(n_collected_times)
    print("MLMC total time ", mlmc_total_time)
    # nn_total_time = np.sum(n_ops_predict_C1_C2 * np.array(n_estimated_nn_C1_C2)) + learning_time
    # print("NN total time + learning time ", np.sum(n_ops_predict_C1_C2 * np.array(n_estimated_nn_C1_C2)) + learning_time)

    # original_moments, estimator, original_true_domain, _ = estimate_moments(sample_storage)

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

    # level_params = [sample_storage.get_level_parameters()[0], *sample_storage.get_level_parameters()]

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

    # if domain_largest:
    #     domain = get_largest_domain([sample_storage, sample_storage_predict, ref_sample_storage])
    #     common_domain = get_largest_domain([sample_storage, sample_storage_predict, ref_sample_storage])
    # else:
    #     domain = None

    #domain = get_largest_domain([sample_storage, sample_storage_predict, ref_sample_storage])
    original_q_estimator = get_quantity_estimator(sample_storage, true_domain=domain)
    predict_q_estimator = get_quantity_estimator(sample_storage_predict, true_domain=domain)

    final_sample_storage_moments, _, _, _ = estimate_moments(sample_storage)
    print("final_sample_storage_moments.mean ", final_sample_storage_moments.mean)
    print("final_sample_storage_moments.var ", final_sample_storage_moments.var)

    final_sample_storage_predict_moments, _, _, _ = estimate_moments(sample_storage_predict)
    print("final_sample_storage_predict_moments.mean ", final_sample_storage_predict_moments.mean)
    print("final_sample_storage_predict_moments.var ", final_sample_storage_predict_moments.var)

    # if cut_est:
    #     original_q_estimator = get_quantity_estimator(sample_storage_for_estimated, true_domain=domain)
    #     predict_q_estimator = get_quantity_estimator(sample_storage_predict_for_estimate, true_domain=domain)

    print("ref samples ", ref_sample_storage)
    print("domain ", domain)

    if distr_domain_largest:
        domain = get_largest_domain([sample_storage, sample_storage_predict, ref_sample_storage])
        common_domain = get_largest_domain([sample_storage, sample_storage_predict, ref_sample_storage])
        original_q_estimator = get_quantity_estimator(sample_storage, true_domain=domain)
        predict_q_estimator = get_quantity_estimator(sample_storage_predict, true_domain=domain)
    else:
        domain = None

    ref_estimator = get_quantity_estimator(ref_sample_storage, true_domain=domain)
    #ref_estimator = None
    orig_moments_mean, predict_moments_mean, ref_moments_mean = compare_moments(original_q_estimator,
                                                                                predict_q_estimator, ref_estimator)

    ref_orig_moments, ref_predict_moments, mlmc_predict_moments = compare_moments_2(sample_storage, sample_storage_predict, ref_sample_storage)

    #level_kurtosis(original_q_estimator, predict_q_estimator)

    kl_mlmc, kl_nn = -1, -1
    orig_orth_moments, predict_orth_moments, ref_orth_moments = None, None, None
    kl_mlmc, kl_nn, orig_orth_moments, predict_orth_moments, ref_orth_moments = compare_densities(original_q_estimator, predict_q_estimator, ref_estimator,
                      label_1="orig N: {}".format(n_estimated_orig),
                      label_2="gnn N: {}".format(n_estimated_nn))

    if stats:
        return n_estimated_orig, n_estimated_nn, n_ops, n_ops_predict, orig_moments_mean, \
               predict_moments_mean, ref_moments_mean, sample_storage.get_level_parameters(), \
               sample_storage_predict.get_level_parameters(), kl_mlmc, kl_nn, TARGET_VAR, \
               orig_orth_moments, predict_orth_moments, ref_orth_moments,\
               ref_orig_moments, ref_predict_moments, mlmc_predict_moments, learning_time

    plot_moments({"ref": ref_estimator, "orig": original_q_estimator, "nn": predict_q_estimator})


def plot_loss(train_loss, val_loss, train_acc=None):
    plt.plot(train_loss, label='loss')
    if train_acc is not None:
        plt.plot(train_acc, label='train acc')
    plt.plot(val_loss, label='val_loss')

    print("len train loss ", len(train_loss))
    print("len val loss ", len(val_loss))
    print("final train loss ", train_loss[-1])
    print("final val loss ", val_loss[-1])


    print("val loss min: {}, corresponding train loss: {}".format(np.min(val_loss), train_loss[np.argmin(val_loss)]))
    print("min val loss position: {}".format(np.argmin(val_loss)))
    print("train loss min: {}, corresponding val loss: {}".format(np.min(train_loss), val_loss[np.argmin(train_loss)]))
    print("min train loss position: {}".format(np.argmin(train_loss)))

    print("min train after min val: {}".format(np.min(train_loss[np.argmin(val_loss):])))
    print("min train after min val position: {}".format(np.argmin(val_loss) + np.argmin(train_loss[np.argmin(val_loss):])))

    #plt.ylim([1, 3])
    plt.yscale("log")
    #plt.axhline(np.min(train_loss))
    #plt.axhline(np.min(val_loss))
    plt.axvline(x=np.argmin(train_loss), color="blue")
    plt.axvline(x=np.argmin(val_loss), color="green")
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_learning_rate(learning_rates):
    plt.plot(learning_rates, label='loss')
    #plt.ylim([0, 8])
    #plt.yscale("log")
    plt.xlabel('Epoch')
    plt.ylabel('Learning rate')
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
    n_moments = N_MOMENTS
    distr_accuracy = 1e-7

    distr_plot = plots.Distribution(title=title,
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
    target_var = TARGET_VAR
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
    n_moments = N_MOMENTS
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
    print("true domain ", true_domain)

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

    print("true domains ", true_domains)

    # print("true domains ", true_domains[0])
    #true_domain = true_domains[-1] # ref


    #true_domain = [np.min(true_domains[:, 0]), np.max(true_domains[:, 1])]
    #true_domain = [np.max(true_domains[:, 0]), np.min(true_domains[:, 1])]
    #true_domain = [np.mean(true_domains[:, 0]), np.mean(true_domains[:, 1])]

    #true_domain = true_domains[-1]  # ref domain
    true_domain = true_domains[0]  # MC domain
    return true_domain


def compare_moments_2(sample_storage, sample_storage_predict, ref_sample_storage):
    true_domains = []
    for storage in [sample_storage, sample_storage_predict, ref_sample_storage]:
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

    mlmc_ref_domain = [np.max([true_domains[0][0], true_domains[-1][0]]),
              np.min([true_domains[0][1], true_domains[-1][1]])]

    print("true domain ", true_domains)

    # just mlmc domain
    mlmc_ref_domain = true_domains[-1]  # ref domain
    mlmc_ref_domain = true_domains[0]  # mlmc domain


    nn_ref_domain = [np.max([true_domains[1][0], true_domains[-1][0]]),
                       np.min([true_domains[1][1], true_domains[-1][1]])]

    # just nn domain
    nn_ref_domain = true_domains[-1]  # ref domain
    nn_ref_domain = true_domains[1]  # mlmc domain


    print("mlmc ref domain ", mlmc_ref_domain)
    print("nn ref domain ", nn_ref_domain)

    print("############################   COMPARE MOMENTS  2   ####################################")
    ####
    ## MLMC vs REF
    ###
    ref_estimator = get_quantity_estimator(ref_sample_storage, true_domain=mlmc_ref_domain)
    mlmc_estimator = get_quantity_estimator(sample_storage, true_domain=mlmc_ref_domain)

    ref_estimator.estimate_moments()
    ref_moments_mean = ref_estimator.moments_mean

    mlmc_estimator.estimate_moments()
    orig_moments_mean = mlmc_estimator.moments_mean

    # print("ref moments mean ", ref_moments_mean.mean)
    # print("orig moments mean ", orig_moments_mean.mean)

    ref_orig_moments = [ref_moments_mean, orig_moments_mean]

    print("ref orig mean SSE ", np.sum((ref_moments_mean.mean - orig_moments_mean.mean) ** 2))
    print("ref orig mean SE ", np.sum(np.abs((ref_moments_mean.mean - orig_moments_mean.mean))))

    print("ref orig var SSE ", np.sum((ref_moments_mean.var - orig_moments_mean.var) ** 2))
    print("ref orig var SE ", np.sum(np.abs((ref_moments_mean.var - orig_moments_mean.var))))

    ##############################
    ##############################
    ###  NN PREDICT vs REF
    ###
    ref_estimator = get_quantity_estimator(ref_sample_storage, true_domain=nn_ref_domain)
    predict_estimator = get_quantity_estimator(sample_storage_predict, true_domain=nn_ref_domain)

    ref_estimator.estimate_moments()
    ref_moments_mean = ref_estimator.moments_mean

    predict_estimator.estimate_moments()
    predict_moments_mean = predict_estimator.moments_mean

    ref_predict_moments = [ref_moments_mean, predict_moments_mean]

    print("ref predict mean SSE ", np.sum((ref_moments_mean.mean - predict_moments_mean.mean) ** 2))
    # print("predict moments mean ", predict_moments_mean.mean)
    print("ref predict mean SE ", np.sum(np.abs((ref_moments_mean.mean - predict_moments_mean.mean))))

    print("ref predict var SSE ", np.sum((ref_moments_mean.var - predict_moments_mean.var) ** 2))
    print("ref predict var SE ", np.sum(np.abs((ref_moments_mean.var - predict_moments_mean.var))))


    ##############################
    ###############################place
    ### MLMC vs MC +NN
    mlmc_estimator = get_quantity_estimator(sample_storage, true_domain=true_domains[0])
    predict_estimator = get_quantity_estimator(sample_storage_predict, true_domain=true_domains[0])

    mlmc_estimator.estimate_moments()
    mlmc_moments_mean = mlmc_estimator.moments_mean

    predict_estimator.estimate_moments()
    predict_moments_mean = predict_estimator.moments_mean

    mlmc_predict_moments = [mlmc_moments_mean, predict_moments_mean]


    # print("ref moments var ", ref_moments_mean.var)
    #print("orig moments var ", orig_moments_mean.var)
    #print("predict moments var ", predict_moments_mean.var)

    # print("MAX orig moments var ", np.max(orig_moments_mean.l_vars, axis=1))
    # print("MAX predict moments var ", np.max(predict_moments_mean.l_vars, axis=1))

    print("MC moments ", mlmc_moments_mean.mean)
    print("MC + NN moments ", predict_moments_mean.mean)

    print("mlmc predict mean SSE ", np.sum((mlmc_moments_mean.mean - predict_moments_mean.mean) ** 2))
    # print("predict moments mean ", predict_moments_mean.mean)
    print("mlmc predict mean SE ", np.sum(np.abs((mlmc_moments_mean.mean - predict_moments_mean.mean))))

    print("mlmc predict var SSE ", np.sum((mlmc_moments_mean.var - predict_moments_mean.var) ** 2))
    print("mlmc predict var SE ", np.sum(np.abs((mlmc_moments_mean.var - predict_moments_mean.var))))



    print("##############################################################")

    return ref_orig_moments, ref_predict_moments, mlmc_predict_moments


def compare_moments(original_q_estimator, predict_q_estimator, ref_estimator=None):
    print("############################ COMPARE MOMENTS ####################################")
    original_q_estimator.estimate_moments()
    orig_moments_mean = original_q_estimator.moments_mean

    predict_q_estimator.estimate_moments()
    predict_moments_mean = predict_q_estimator.moments_mean

    ref_moments_mean = None
    if ref_estimator is not None:
        ref_estimator.estimate_moments()
        ref_moments_mean = ref_estimator.moments_mean

        print("ref moments mean ", ref_moments_mean.mean)
        print("orig moments mean ", orig_moments_mean.mean)
        print("predict moments mean ", predict_moments_mean.mean)

        print("ref orig mean SSE ", np.sum((ref_moments_mean.mean - orig_moments_mean.mean)**2))
        print("ref predict mean SSE ", np.sum((ref_moments_mean.mean - predict_moments_mean.mean) ** 2))
        #
        print("ref orig mean SE ", np.sum(np.abs((ref_moments_mean.mean - orig_moments_mean.mean))))
        print("ref predict mean SE ", np.sum(np.abs((ref_moments_mean.mean - predict_moments_mean.mean))))

        orig_diff = ref_moments_mean.mean - orig_moments_mean.mean
        predict_diff = ref_moments_mean.mean - predict_moments_mean.mean
        orig_diff[0] = 1
        predict_diff[0] = 1

        # print("np.abs((ref_moments_mean.mean - orig_moments_mean.mean))/orig_diff ", np.abs((ref_moments_mean.mean - orig_moments_mean.mean))/orig_diff)
        # print("np.sum(np.abs((ref_moments_mean.mean - predict_moments_mean.mean))/predict_diff ",  np.abs((ref_moments_mean.mean - predict_moments_mean.mean))/predict_diff)
        #
        # print("ref orig mean SE relative", np.sum(np.abs((ref_moments_mean.mean - orig_moments_mean.mean))/orig_diff))
        # print("ref predict mean SE relative", np.sum(np.abs((ref_moments_mean.mean - predict_moments_mean.mean))/predict_diff))

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

    print("##############################################################")

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

def level_kurtosis(original_q_estimator, predict_q_estimator):
    original_q_estimator.kurtosis_check()


def compare_densities(estimator_1, estimator_2, ref_estimator, label_1="", label_2=""):
    distr_plot = plots.ArticleDistributionPDF(title="densities", log_density=True, set_x_lim=False, quantity_name="$c [kgm^{-3}]$")
    tol = 1e-7
    reg_param = 0

    print("orig estimator")
    distr_obj_1, _, result, _, orig_orth_moments = estimator_1.construct_density(tol=tol, reg_param=reg_param,
                                                              orth_moments_tol=TARGET_VAR)
    #distr_plot.add_distribution(distr_obj_1, label=label_1, color="blue")

    print("predict estimator")
    distr_obj_2, _, result, _, predict_orth_moments = estimator_2.construct_density(tol=tol, reg_param=reg_param,  orth_moments_tol=TARGET_VAR)
    #distr_plot.add_distribution(distr_obj_2, label=label_2, color="red", line_style="--")

    print("Ref estimator")
    ref_distr_obj, _, result, _, ref_orth_moments = ref_estimator.construct_density(tol=tol, reg_param=reg_param,  orth_moments_tol=TARGET_VAR)
    #distr_plot.add_distribution(ref_distr_obj, label="MC reference", color="black", line_style=":")

    ref_estimator_pdf = get_quantity_estimator(ref_estimator._sample_storage, true_domain=None, n_moments=N_MOMENTS)
    ref_distr_obj, _, result, _, ref_orth_moments_pdf = ref_estimator_pdf.construct_density(tol=tol,
                                                                                            reg_param=reg_param,
                                                                                            orth_moments_tol=TARGET_VAR)

    # domain = [np.max([ref_distr_obj.domain[0], distr_obj_1.domain[0], distr_obj_2.domain[0]]),
    #           np.min([ref_distr_obj.domain[1], distr_obj_1.domain[1], distr_obj_2.domain[1]])]
    domain = [np.max([ref_distr_obj.domain[0], distr_obj_1.domain[0]]),
              np.min([ref_distr_obj.domain[1], distr_obj_1.domain[1]])]
    kl_div_ref_mlmc = mlmc.tool.simple_distribution.KL_divergence(ref_distr_obj.density, distr_obj_1.density, domain[0], domain[1])

    print("KL div ref|mlmc: {}".format(kl_div_ref_mlmc))

    domain = [np.max([ref_distr_obj.domain[0], distr_obj_2.domain[0]]),
              np.min([ref_distr_obj.domain[1], distr_obj_2.domain[1]])]
    kl_div_ref_gnn = mlmc.tool.simple_distribution.KL_divergence(ref_distr_obj.density, distr_obj_2.density, domain[0],
                                                         domain[1])

    print("KL div ref|mlmc prediction: {}".format(kl_div_ref_gnn))

    #distr_plot.add_distribution(distr_obj_1, label=label_1 + ", KL(ref|orig):{:0.4g}".format(kl_div_ref_mlmc), color="blue")
    #distr_plot.add_distribution(distr_obj_2, label=label_2 + ", KL(ref|gnn):{:0.4g}".format(kl_div_ref_gnn), color="red", line_style="--")
    distr_plot.add_distribution(distr_obj_1, label=r"$D_{3LMC}:$" + "{:0.4g}".format(kl_div_ref_mlmc), color="blue")
    distr_plot.add_distribution(distr_obj_2, label=r"$D_{3LMC-M}:$" + "{:0.4g}".format(kl_div_ref_gnn), color="red", line_style="--")
    distr_plot.add_distribution(ref_distr_obj, label="MC ref", color="black", line_style=":")

    distr_plot.show(file=None)
    distr_plot.show(file="densities.pdf")


    return kl_div_ref_mlmc, kl_div_ref_gnn, orig_orth_moments, predict_orth_moments, ref_orth_moments


def get_quantity_estimator(sample_storage, true_domain=None, quantity=None, n_moments=None):
    if n_moments is None:
        n_moments = N_MOMENTS
    result_format = sample_storage.load_result_format()
    if quantity is None:
        root_quantity = make_root_quantity(sample_storage, result_format)
        conductivity = root_quantity['conductivity']
        time = conductivity[1]  # times: [1]
        location = time['0']  # locations: ['0']
        quantity = location[0, 0]

    if true_domain is None:
        quantile = QUANTILE
        true_domain = mlmc.estimator.Estimate.estimate_domain(quantity, sample_storage, quantile=quantile)

    print("true domain")
    moments_fn = Legendre(n_moments, true_domain)
    #moments_fn = Monomial(n_moments, true_domain)

    return mlmc.estimator.Estimate(quantity=quantity, sample_storage=sample_storage, moments_fn=moments_fn)


def get_n_estimated(sample_storage, estimator, n_ops=None):
    target_var = TARGET_VAR
    #moments, estimator, _, quantity = estimate_moments(sample_storage, true_domain=true_domain)

    n_level_samples = sample_storage.get_n_collected()
    # New estimation according to already finished samples

    #print("n level samples ", n_level_samples)
    variances, n_samples = estimator.estimate_diff_vars()
    #print("n samples ", n_samples)
    #variances, est_n_ops = estimator.estimate_diff_vars_regression(n_level_samples)

    if n_ops is None:
        n_ops = n_samples
    # print("get n estimated n ops ", n_ops)
    # print("variances ", variances)
    n_estimated = mlmc.estimator.estimate_n_samples_for_target_variance(target_var, variances, n_ops,
                                                                        n_levels=len(n_level_samples))
    return n_estimated, variances, n_samples


def get_storage_info(sample_storage):
    moments, estimator, _, _ = estimate_moments(sample_storage)
    n_collected = sample_storage.get_n_collected()
    max_vars = np.max(np.array(moments.l_vars) / np.array(sample_storage.get_n_collected())[:, np.newaxis], axis=1)
    print("n collected ", n_collected)
    print("moments.l_vars max ", max_vars)


    print('moments l vars ', moments.l_vars)
    return n_collected, max_vars


def cut_samples(data, sample_storage, new_n_collected, new_l_0=0, bootstrap=False):
    new_data = []
    for l_id, (d, n_est) in enumerate(zip(data, new_n_collected)):
        # print("len d :", d.shape[1])
        #print("cut samples n est", n_est)
        if n_est > 0:
            if l_id == new_l_0:
                if bootstrap:
                    sample_idx = np.random.choice(list(range(0, d.shape[1]-1)), size=n_est, replace=True)
                    if len(sample_idx) > d.shape[1]:
                        fine_samples = d[:, :np.min([d.shape[1], n_est]), 0].reshape(1, np.min([d.shape[1], n_est]), 1)
                    else:
                        fine_samples = d[:, sample_idx, 0].reshape(1, np.min([d.shape[1], len(sample_idx)]), 1)
                else:
                    fine_samples = d[:, :np.min([d.shape[1], n_est]), 0].reshape(1, np.min([d.shape[1], n_est]), 1)

                coarse_samples = np.zeros(fine_samples.shape)
                new_data.append(np.concatenate((fine_samples, coarse_samples), axis=2))
            else:
                if bootstrap:
                    sample_idx = np.random.choice(list(range(0, d.shape[1] - 1)), size=n_est, replace=True)
                    if len(sample_idx) > d.shape[1]:
                        new_data.append(d[:, sample_idx, :])
                    else:
                        new_data.append(d[:, :np.min([d.shape[1], n_est]), :])
                else:
                    new_data.append(d[:, :np.min([d.shape[1], n_est]), :])

    # print("new data ", new_data)
    # print("new data shape ", np.array(new_data).shape)
    #
    # print("var new data ", np.var(new_data, axis=-2))

    sample_storage = create_quantity_mlmc(new_data, level_parameters=sample_storage.get_level_parameters())

    return sample_storage


def plot_progress(conv_layers, dense_layers, output_flatten, mesh_file=None, n_samples=5):

    if mesh_file is not None:
        #mesh = gmsh_io.GmshIO(fields_mesh)
        mesh_data = extract_mesh_gmsh_io(mesh_file, get_points=True)
        points = mesh_data['points']
        X = points[:, 0]
        Y = points[:, 1]

    for idx, conv_layer in conv_layers.items():
        inputs, weights, outputs = conv_layer[0], conv_layer[1], conv_layer[2]
        plt.matshow(weights[-1])
        plt.show()
        # Note: weights have different shape than the mesh

        print("inputs ", inputs)
        print("weights ", weights)
        print("outputs ", outputs)

        for index, input in enumerate(inputs[:n_samples]):
            if mesh_file:
                for i in range(inputs[index].shape[1]):
                    input_feature = inputs[index][:, i]
                    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
                    print("inputs.shape ",  input_feature.shape)
                    print("X.shape ", X.shape)
                    print("Y.shape ", Y.shape)
                    print("input.ravel().shape ",  input_feature.ravel().shape)
                    cont = ax.tricontourf(X, Y,  input_feature.ravel(), levels=16)
                    fig.colorbar(cont)
                    plt.title("input feature {}".format(i))
                    plt.show()

                for i in range(outputs[index].shape[1]):
                    channel_output = outputs[index][:, i]
                    print("channel output shape ", channel_output.shape)
                    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
                    cont = ax.tricontourf(X, Y, channel_output, levels=16)
                    fig.colorbar(cont)
                    plt.title("output channel {}".format(i))

                    plt.show()

            else:
                plt.matshow(input[0])
                plt.show()
                plt.matshow(outputs[index][0])
                plt.show()

            # print("shape ", c_layer._outputs[index][0].shape)
            plt.matshow(np.sum(outputs[index], axis=0, keepdims=True))
            plt.title("flatten")
            plt.show()

            fig, ax = plt.subplots(1, 1, figsize=(15, 10))
            cont = ax.tricontourf(X, Y, np.sum(outputs[index], axis=1), levels=16)
            fig.colorbar(cont)
            plt.title("sum channels")
            plt.show()

            # plt.matshow(self._inputs[-1][0])
            # plt.show()
            # plt.matshow(self._outputs[-1][0])
            # plt.show()

            # print("output flatten ", self._output_flatten)
            # print("final output ", final_output)

            if len(output_flatten) > 0:
                plt.matshow([output_flatten[-1]])
                plt.title("flatten")
                plt.show()

    for idx, dense_layer in dense_layers.items():
        inputs, weights, outputs = dense_layer[0], dense_layer[1], dense_layer[2]

        plt.matshow([inputs[-1]])
        plt.show()
        plt.matshow([outputs[-1]])
        plt.show()


    exit()


def plot_moments(mlmc_estimators):
    n_moments = N_MOMENTS
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


def load_sim_data(sim_data_file):
    """
    Load saved simulation data
    :return: dict
    """
    import json

    with open(sim_data_file, 'r') as file:
        # First we load existing data into a dict.
        data = json.load(file)

    generate_rnd = []
    extract_mesh = []
    make_fields = []
    coarse_flow = []
    fine_flow = []
    for level_data in data.values():
        generate_rnd_level = np.zeros(2)
        extract_mesh_level = np.zeros(2)
        make_fields_level = np.zeros(2)
        coarse_flow_level = np.zeros(2)
        fine_flow_level = np.zeros(2)
        for d in level_data:
            generate_rnd_level[0] += d["generate_rnd"]
            generate_rnd_level[1] += 1

            extract_mesh_level[0] += d["extract_mesh"]
            extract_mesh_level[1] += 1

            make_fields_level[0] += d["make_field"]
            make_fields_level[1] += 1

            coarse_flow_level[0] += d["coarse_flow"]
            coarse_flow_level[1] += 1

            fine_flow_level[0] += d["fine_flow"]
            fine_flow_level[1] += 1

    generate_rnd.append(generate_rnd_level)
    extract_mesh.append(extract_mesh_level)
    make_fields.append(make_fields_level)
    coarse_flow.append(coarse_flow_level)
    fine_flow.append(fine_flow_level)

    return generate_rnd, extract_mesh, make_fields, coarse_flow, fine_flow



def get_sample_times_mlmc(mlmc_file, sample_storage=None):
    if sample_storage is None:
        sample_storage = SampleStorageHDF(file_path=mlmc_file)

    sim_data_file = os.path.join(os.path.dirname(mlmc_file), 'sim_data.json')

    n_ops = sample_storage.get_n_ops()
    if os.path.exists(sim_data_file):
        generate_rnd, extract_mesh, make_fields, coarse_flow, fine_flow = load_sim_data(sim_data_file)
    else:
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
