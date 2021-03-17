import os
import numpy as np
import time
import random
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Run on CPU only
from mlmc.metamodel.create_graph import graph_creator
from mlmc.metamodel.flow_dataset import FlowDataset

from mlmc.metamodel.own_cheb_conv import OwnChebConv
# Make numpy printouts easier to read.

# np.set_printoptions(precision=9, suppress=True)
import tensorflow as tf
from scipy.stats import ks_2samp
import sklearn.model_selection
from mlmc.metamodel.custom_methods import abs_activation, var_loss_function, total_loss_function
from mlmc.metamodel.postprocessing import analyze_results, plot_loss, estimate_density, diff_moments, process_mlmc
from mlmc.metamodel.flow_task_NN import DNN
from mlmc.metamodel.flow_task_CNN import CNN
from mlmc.metamodel.graph_models import NetGCN

from mlmc.metamodel.flow_task_GNN_2 import GNN
from spektral.layers import GCNConv, GlobalSumPool, ChebConv, GraphSageConv, ARMAConv, GATConv, APPNPConv, GINConv
from tensorflow.keras.losses import MeanSquaredError, KLDivergence
from tensorflow.keras.regularizers import l2
from spektral.data import MixedLoader
from spektral.layers.ops import sp_matrix_to_sp_tensor

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

epochs = 100


def prepare_data(data):
    data = np.squeeze(np.stack(data.to_numpy(), axis=0))
    return np.asarray(data).astype('float64')


def split_dataset(dataset):
    # Load data
    dataset = dataset.dropna()
    train_x, test_x, train_y, test_y = sklearn.model_selection.train_test_split(dataset.x, dataset.y,
                                                                                test_size=0.2, random_state=123)

    train_x = prepare_data(train_x)
    train_y = prepare_data(train_y)

    test_x = prepare_data(test_x)
    test_y = prepare_data(test_y)

    return train_x, train_y, test_x, test_y


def run():
    # Parameters
    loss = "mean_squared_error"
    optimizer = tf.optimizers.Adam(learning_rate=0.001)

    data = FlowDataset()
    # dataset = data.dataset[:10000]
    # test_dataset = data.dataset[10000:50000]

    dataset = data.dataset[:50000]
    test_dataset = data.dataset[50000:]

    train_input = prepare_data(dataset.x)
    train_output = prepare_data(dataset.y)

    # train_input, train_output, test__input, test_output = split_dataset(dataset)
    # print("len test(output) ", len(test_output))

    dnn = DNN(loss=loss, optimizer=optimizer, output_activation=abs_activation, hidden_activation='relu', epochs=150)
    dnn.fit(train_input, train_output)

    test_input = prepare_data(test_dataset.x)
    test_output = prepare_data(test_dataset.y)

    predictions = dnn.predict(test_input)
    predictions = np.squeeze(predictions)

    print("len(predictions) ", len(predictions))

    plot_loss(dnn.history.history['loss'], dnn.history.history['val_loss'])
    analyze_results(test_output, predictions)

    estimate_density(test_output)
    estimate_density(predictions)


def run_CNN(output_dir, hdf_path, l_0_output_dir, l_0_hdf_path, save_path, mesh, level):
    # Parameters
    loss = "mean_squared_error"
    optimizer = tf.optimizers.Adam(learning_rate=0.01)

    data = FlowDataset(output_dir=output_dir, level=level)
    dataset = data.dataset[:]

    train_input, train_output, test_input, test_output = split_dataset(dataset)

    train_input = train_input[:2000]
    train_output = train_output[:2000]

    print("len test(output) ", len(test_output))

    train_input = np.expand_dims(train_input, axis=-1)
    test_input = np.expand_dims(test_input, axis=-1)

    print("train input shape ", train_input.shape)

    dnn = CNN(loss=loss, optimizer=optimizer, output_activation=abs_activation, hidden_activation='relu')

    dnn.fit(train_input, train_output)

    test_dataset = data.dataset[2000:]
    test_input = prepare_data(test_dataset.x)
    test_input = np.expand_dims(test_input, axis=-1)
    print("test input shape ", test_input.shape)
    test_output = prepare_data(test_dataset.y)

    predictions = dnn.predict(test_input)
    predictions = np.squeeze(predictions)

    plot_loss(dnn.history.history['loss'], dnn.history.history['val_loss'])

    analyze_results(test_output, predictions)

    # estimate_density(test_output)
    # estimate_density(predictions)


def bootstrap():
    loss = "mean_absolute_error"
    optimizer = tf.optimizers.Adam(learning_rate=0.001)
    n_subsamples = 10
    size = 10000

    train_losses = []
    val_losses = []
    all_test_outputs = []
    all_predictions = []
    ks_statistics = []
    ks_p_values = []

    data = FlowDataset()
    dataset = data.dataset.dropna()

    for i in range(n_subsamples):
        dset = dataset.sample(size, replace=True)
        train_input, train_output, test_input, test_output = split_dataset(dset)

        print("Size TRAIN in: {}, out: {}, TEST in: {}, out: {}".format(len(train_input), len(train_output),
                                                                        len(test_input), len(test_output)))

        dnn = DNN(loss=loss, optimizer=optimizer, output_activation=abs_activation, hidden_activation='relu')
        dnn.fit(train_input, train_output)

        predictions = dnn.predict(test_input)
        predictions = np.squeeze(predictions)

        train_losses.append(dnn.history.history['loss'])
        val_losses.append(dnn.history.history['val_loss'])

        all_test_outputs.append(test_output)
        all_predictions.append(predictions)

        statistics, pvalue = ks_2samp(test_output, predictions)
        ks_statistics.append(statistics)
        ks_p_values.append(pvalue)

    analyze_results(np.mean(all_test_outputs, axis=0), np.mean(all_predictions, axis=0))
    analyze_results(np.var(all_test_outputs, axis=0), np.var(all_predictions, axis=0))
    #
    # estimate_density(np.mean(all_test_outputs, axis=0), title="Test outputs")
    # estimate_density(np.mean(all_predictions, axis=0), title="Predictions")


def statistics(run_method, output_dir, hdf_path, l_0_output_dir, l_0_hdf_path, save_path, mesh, level):
    n_subsamples = 3
    train_losses = []
    val_losses = []
    test_losses = []
    all_test_outputs = []
    all_predictions = []
    learning_times = []

    # Parameters
    # conv_layer = GCNConv
    conv_layer = ChebConv  # Seems better than GCNConv, good distribution of predictions
    # conv_layer = GraphSageConv  # Seems better than ChebConv, good loss but very narrow distribution of predictions
    # # conv_layer = ARMAConv  # Seems worse than GraphSageConv
    # conv_layer = GATConv  # Slow and not better than GraphSageConv
    # # conv_layer = APPNPConv  # Not bad but worse than GraphSageConv
    # # conv_layer = GINConv  # it is comparable to APPNPConv
    # act_func = "relu"  # "tanh"#"elu"  # ReLU keep predictions above zero

    save_path = os.path.join(save_path, conv_layer.__name__)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    else:
        print("dir exists {}".format(save_path))
        exit()

    for i in range(n_subsamples):
        gnn, targets, predictions, learning_time = run_method(output_dir, hdf_path, l_0_output_dir, l_0_hdf_path,
                                                              save_path, mesh, level=nn_level, stats=True)

        # save_times(save_path, False, (preprocess_time, len(data)), learning_time, (predict_l_0_time, len(l_0_targets)))
        # save_load_data(save_path, False, targets, predictions, train_targets, val_targets, l_0_targets, l_0_predictions)

        all_test_outputs.append(targets)
        all_predictions.append(predictions)

        train_losses.append(gnn._train_loss[-1])
        val_losses.append(gnn._val_loss[-1])
        test_losses.append(gnn._test_loss[-1])
        learning_times.append(learning_time)

    for i in range(len(train_losses)):
        print("train loss ", train_losses[i])
        print("test loss ", test_losses[i])
        analyze_results(all_test_outputs[i], all_predictions[i])
        print("learning time ", learning_times[i])
        print("##################################################")

    print("worse train loss ", np.max(train_losses))
    print("worse test loss ", np.max(test_losses))

    save_statistics(save_path, all_test_outputs, all_predictions, train_losses, val_losses, test_losses, learning_times)

    # plot_loss(train_losses, val_losses)
    # analyze_results(np.mean(all_test_outputs, axis=0), np.mean(all_predictions, axis=0))
    # analyze_results(np.var(all_test_outputs, axis=0), np.var(all_predictions, axis=0))
    #
    # estimate_density(np.mean(all_test_outputs, axis=0), title="Test outputs")
    # estimate_density(np.mean(all_predictions, axis=0), title="Predictions")


def save_statistics(path, all_test_outputs, all_predictions, train_losses, val_losses, test_losses, learning_times):
    np.save(os.path.join(path, "all_test_outputs"), all_test_outputs)
    np.save(os.path.join(path, "all_predictions"), all_predictions)
    np.save(os.path.join(path, "train_losses"), train_losses)
    np.save(os.path.join(path, "val_losses"), val_losses)
    np.save(os.path.join(path, "test_losses"), test_losses)
    np.save(os.path.join(path, "learning_times"), learning_times)


def load_statistics(path):
    all_test_outputs = np.load(os.path.join(path, "all_test_outputs.npy"))
    all_predictions = np.load(os.path.join(path, "all_predictions.npy"))
    train_losses = np.load(os.path.join(path, "train_losses.npy"))
    val_losses = np.load(os.path.join(path, "val_losses.npy"))
    test_losses = np.load(os.path.join(path, "test_losses.npy"))
    learning_times = np.load(os.path.join(path, "learning_times.npy"))

    return all_test_outputs, all_predictions, train_losses, val_losses, test_losses, learning_times


def analyze_statistics(save_path):
    conv_layer = GCNConv
    conv_layer = ChebConv  # Seems better than GCNConv, good distribution of predictions
    # conv_layer = GraphSageConv  # Seems better than ChebConv, good loss but very narrow distribution of predictions
    # # conv_layer = ARMAConv  # Seems worse than GraphSageConv
    # conv_layer = GATConv  # Slow and not better than GraphSageConv
    # # conv_layer = APPNPConv  # Not bad but worse than GraphSageConv
    # # conv_layer = GINConv  # it is comparable to APPNPConv
    # act_func = "relu"  # "tanh"#"elu"  # ReLU keep predictions above zero
    save_path = os.path.join(save_path, conv_layer.__name__)
    if not os.path.isdir(save_path):
        print("dir exists")
        exit()

    all_test_outputs, all_predictions, train_losses, val_losses, test_losses, learning_times = load_bootstrap(save_path)

    print("learning times ", learning_times)
    print("test losses ", test_losses)

    for targets, predictions in zip(all_test_outputs, all_predictions):
        print("mean targets ", np.mean(targets))
        print("mean predictions ", np.mean(predictions))

        print("var targets ", np.var(targets))
        print("var predictions ", np.var(predictions))


def run_GNN(output_dir, hdf_path, l_0_output_dir, l_0_hdf_path, save_path, mesh, level, conv_layer=None, stats=False,
            gnn=None, model=None):
    # Parameters
    if conv_layer is None:
        conv_layer = GCNConv
        conv_layer = ChebConv  # Seems better than GCNConv, good distribution of predictions
        # conv_layer = GraphSageConv  # Seems better than ChebConv, good loss but very narrow distribution of predictions
        # # conv_layer = ARMAConv  # Seems worse than GraphSageConv
        # conv_layer = GATConv  # Slow and not better than GraphSageConv
        # # conv_layer = APPNPConv  # Not bad but worse than GraphSageConv
        # # conv_layer = GINConv  # it is comparable to APPNPConv
        # act_func = "relu"  # "tanh"#"elu"  # ReLU keep predictions above zero

    loss = MeanSquaredError()  # var_loss_function#
    #loss = KLDivergence()
    # loss = total_loss_function
    optimizer = tf.optimizers.Adam(learning_rate=0.001)
    batch_size = 2000#2000
    epochs = 500
    hidden_regularization = None  # l2(2e-10)

    preprocess_start_time = time.process_time()
    #graph_creator(output_dir, hdf_path, mesh, level=level)

    # Load data
    data = FlowDataset(output_dir=output_dir, level=level)
    data = data#[:15000]
    print("lev data ", len(data))

    preprocess_time = time.process_time() - preprocess_start_time

    learning_time_start = time.process_time()
    # data.a = conv_layer.preprocess(data.a)
    data.a = sp_matrix_to_sp_tensor(data.a)

    train_data_len = int(len(data) * 0.8)
    train_data_len = 2000
    print("train data len ", train_data_len)

    # Train/valid/test split
    data_tr, data_te = data[:train_data_len], data[train_data_len:],
    np.random.shuffle(data_tr)

    val_data_len = int(len(data_tr) * 0.2)
    data_tr, data_va = data_tr[:-val_data_len], data_tr[-val_data_len:]

    print("data_tr len ", len(data_tr))
    print("data_va len ", len(data_va))
    print("data_te len ", len(data_te))

    # We use a MixedLoader since the dataset is in mixed mode
    loader_tr = MixedLoader(data_tr, batch_size=batch_size, epochs=epochs)
    loader_va = MixedLoader(data_va, batch_size=batch_size)
    loader_te = MixedLoader(data_te, batch_size=batch_size)

    if gnn is None:
        gnn = GNN(loss=loss, optimizer=optimizer, conv_layer=conv_layer, output_activation=abs_activation,
                  hidden_activation='relu', patience=150, hidden_reqularizer=hidden_regularization,
                  model=model)  # tanh takes to much time
        # ideally patience = 150
        # batch_size 500, ideally 500 epochs, patience 35

        # gnn.run_eagerly = True
        train_targets = gnn.fit(loader_tr, loader_va, loader_te)

    train_targets, train_predictions = gnn.predict(loader_tr)
    train_predictions = np.squeeze(train_predictions)

    val_targets = gnn.val_targets

    targets, predictions = gnn.predict(loader_te)
    predictions = np.squeeze(predictions)
    learning_time = time.process_time() - learning_time_start

    # print("np.var(target-predictions) ", np.var(targets - predictions))

    if stats is True:
        return gnn, targets, predictions, learning_time

    plot_loss(gnn._train_loss, gnn._val_loss)
    analyze_results(targets, predictions)

    # print("np.max(targets) ", np.max(targets))
    # print("np.min(targets) ", np.min(targets))
    #
    # print("np.max(predictions) ", np.max(predictions))
    # print("np.min(predictions) ", np.min(predictions))
    #
    # print("data.max_output ", data.max_output)
    # print("data.min_output ", data.min_output)


    # rescaled_targets = (data.max_output - data.min_output)/(np.max(targets) - np.min(targets))*(targets - np.max(targets)) + data.min_output
    #
    #
    # rescaled_predictions = (data.max_output - data.min_output) / (np.max(predictions) - np.min(predictions)) * (
    #             targets - np.max(predictions)) + data.min_output
    #
    # analyze_results(rescaled_targets, rescaled_predictions)

    # target_means, target_vars = estimate_density(targets, title="Test outputs")
    # pred_means, pred_vars = estimate_density(predictions, title="Predictions")
    # #
    # print("target means ", target_means)
    # print("predic means ", pred_means)
    # #
    # print("target vars ", target_vars)
    # print("predic vars ", pred_vars)

    #
    # diff_means, diff_vars = estimate_density(targets - predictions, title="diff")
    # print("diff_means ", diff_means)
    # print("diff vars ", diff_vars)

    # diff_moments(targets, predictions)

    predict_l_0_start_time = time.process_time()
    l_0_targets, l_0_predictions = predict_level_zero(gnn, l_0_output_dir, l_0_hdf_path, mesh, batch_size)
    predict_l_0_time = time.process_time() - predict_l_0_start_time

    save_times(save_path, False, (preprocess_time, len(data)), learning_time, (predict_l_0_time, len(l_0_targets)))
    save_load_data(save_path, False, targets, predictions, train_targets, train_predictions, val_targets, l_0_targets,
                   l_0_predictions)


def predict_level_zero(nn, output_dir, hdf_path, mesh, batch_size=1000, ):
    # graph_creator(output_dir, hdf_path, mesh, level=0)

    # Load data
    data = FlowDataset(output_dir=output_dir)
    data = data  # [:10000]
    # data.a = conv_layer.preprocess(data.a)
    data.a = sp_matrix_to_sp_tensor(data.a)

    loader_te = MixedLoader(data, batch_size=batch_size)

    targets, predictions = nn.predict(loader_te)
    predictions = np.squeeze(predictions)
    # analyze_results(targets, predictions)
    return targets, predictions


def save_times(path, load=False, preprocess=None, learning_time=None, predict_l_0=None):
    if load:
        preprocess_time = None
        preprocess_n = None
        predict_time = None
        predict_n = None
        if os.path.exists(os.path.join(path, "preprocess_time.npy")):
            preprocess_time = np.load(os.path.join(path, "preprocess_time.npy"))
        if os.path.exists(os.path.join(path, "preprocess_n.npy")):
            preprocess_n = np.load(os.path.join(path, "preprocess_n.npy"))
        if os.path.exists(os.path.join(path, "learning_time.npy")):
            learning_time = np.load(os.path.join(path, "learning_time.npy"))
        if os.path.exists(os.path.join(path, "predict_l_0_time.npy")):
            predict_time = np.load(os.path.join(path, "predict_l_0_time.npy"))
        if os.path.exists(os.path.join(path, "predict_l_0_n.npy")):
            predict_n = np.load(os.path.join(path, "predict_l_0_n.npy"))
        return preprocess_time, preprocess_n, learning_time, predict_time, predict_n
    else:
        if preprocess is not None:
            np.save(os.path.join(path, "preprocess_time"), preprocess[0])
            np.save(os.path.join(path, "preprocess_n"), preprocess[1])
        if learning_time is not None:
            np.save(os.path.join(path, "learning_time"), learning_time)
        if preprocess is not None:
            np.save(os.path.join(path, "predict_l_0_time"), predict_l_0[0])
            np.save(os.path.join(path, "predict_l_0_n"), predict_l_0[1])


def save_load_data(path, load=False, targets=None, predictions=None, train_targets=None, train_predictions=None,
                   val_targets=None, l_0_targets=None,
                   l_0_predictions=None):
    if load:
        if os.path.exists(os.path.join(path, "targets.npy")):
            targets = np.load(os.path.join(path, "targets.npy"))
        if os.path.exists(os.path.join(path, "predictions.npy")):
            predictions = np.load(os.path.join(path, "predictions.npy"))
        if os.path.exists(os.path.join(path, "train_targets.npy")):
            train_targets = np.load(os.path.join(path, "train_targets.npy"))
        if os.path.exists(os.path.join(path, "train_predictions.npy")):
            train_predictions = np.load(os.path.join(path, "train_predictions.npy"))
        if os.path.exists(os.path.join(path, "val_targets.npy")):
            val_targets = np.load(os.path.join(path, "val_targets.npy"))
        if os.path.exists(os.path.join(path, "l_0_targets.npy")):
            l_0_targets = np.load(os.path.join(path, "l_0_targets.npy"))
        if os.path.exists(os.path.join(path, "l_0_predictions.npy")):
            l_0_predictions = np.load(os.path.join(path, "l_0_predictions.npy"))
        return targets, predictions, train_targets, train_predictions, val_targets, l_0_targets, l_0_predictions
    else:
        if targets is not None:
            np.save(os.path.join(path, "targets"), targets)
        if predictions is not None:
            np.save(os.path.join(path, "predictions"), predictions)
        if train_targets is not None:
            np.save(os.path.join(path, "train_targets"), train_targets)
        if train_predictions is not None:
            np.save(os.path.join(path, "train_predictions"), train_predictions)
        if val_targets is not None:
            np.save(os.path.join(path, "val_targets"), val_targets)
        if l_0_targets is not None:
            np.save(os.path.join(path, "l_0_targets"), l_0_targets)
        if l_0_predictions is not None:
            np.save(os.path.join(path, "l_0_predictions"), l_0_predictions)


def process_results(hdf_path, sampling_info_path, ref_mlmc_file, save_path, nn_level, replace_level):
    targets, predictions, train_targets, train_predictions, val_targets, l_0_targets, l_0_predictions = save_load_data(
        save_path, load=True)
    preprocess_time, preprocess_n, learning_time, predict_l_0_time, predict_l_0_n = save_times(save_path, load=True)

    l1_sample_time = preprocess_time / preprocess_n + learning_time / preprocess_n
    l0_sample_time = predict_l_0_time / predict_l_0_n

    print("preprocess_time ", preprocess_time)
    print("preprocess_n ", preprocess_n)
    print("learning_time ", learning_time)
    print("predict_l_0_time ", predict_l_0_time)
    print("predict_l_0_n ", predict_l_0_n)

    print("l1 sample time ", l1_sample_time)
    print("l0 sample time ", l0_sample_time)

    print("len targets ", len(targets))
    print("len predictions ", len(predictions))

    print("len train targets ", len(train_targets))
    print("len val targets ", len(val_targets))

    process_mlmc(hdf_path, sampling_info_path, ref_mlmc_file, targets, predictions, train_targets, train_predictions,
                 val_targets, l_0_targets,
                 l_0_predictions, l1_sample_time, l0_sample_time, nn_level=nn_level, replace_level=replace_level)


def get_config(case=0):
    if case == 0:
        cl = "cl_0_3_s_4"
        nn_level = 0
        replace_level = False
        mesh = "/home/martin/Documents/metamodels/data/L1/test/01_cond_field/l_step_0.055_common_files/mesh.msh"
        # mesh = "/home/martin/Documents/metamodels/data/1000_ele/cl_0_1_s_1/L2/l_step_0.027624156655057155_common_files/mesh.msh"
        output_dir = "/home/martin/Documents/metamodels/data/1000_ele/{}/L5/test/01_cond_field/output/".format(cl)
        hdf_path = "/home/martin/Documents/metamodels/data/1000_ele/{}/L5/mlmc_5.hdf5".format(cl)
        save_path = "/home/martin/Documents/metamodels/data/1000_ele/{}".format(cl)
        l_0_output_dir = "/home/martin/Documents/metamodels/data/1000_ele/{}/L{}/test/01_cond_field/output/".format(cl, nn_level + 1)
        l_0_hdf_path = "/home/martin/Documents/metamodels/data/1000_ele/{}/L{}/mlmc_{}.hdf5".format(cl, nn_level + 1, nn_level + 1)
        sampling_info_path = "/home/martin/Documents/metamodels/data/1000_ele/{}/sampling_info".format(cl)
        ref_mlmc_file = "/home/martin/Documents/metamodels/data/1000_ele/{}/L1_benchmark/mlmc_1.hdf5".format(cl)
    elif case == 1:
        cl = "cl_0_3_s_4"
        nn_level = 1
        replace_level = False
        mesh = "/home/martin/Documents/metamodels/data/L1/test/01_cond_field/l_step_0.055_common_files/mesh.msh"
        #mesh = "/home/martin/Documents/metamodels/data/1000_ele/cl_0_1_s_1/L2/l_step_0.027624156655057155_common_files/mesh.msh"
        output_dir = "/home/martin/Documents/metamodels/data/1000_ele/{}/L5/test/01_cond_field/output/".format(cl)
        hdf_path = "/home/martin/Documents/metamodels/data/1000_ele/{}/L5/mlmc_5.hdf5".format(cl)
        save_path = "/home/martin/Documents/metamodels/data/1000_ele/{}".format(cl)
        l_0_output_dir = "/home/martin/Documents/metamodels/data/1000_ele/{}/L{}/test/01_cond_field/output/".format(cl, nn_level + 1)
        l_0_hdf_path = "/home/martin/Documents/metamodels/data/1000_ele/{}/L{}/mlmc_{}.hdf5".format(cl, nn_level + 1, nn_level + 1)
        sampling_info_path = "/home/martin/Documents/metamodels/data/1000_ele/{}/sampling_info".format(cl)
        ref_mlmc_file = "/home/martin/Documents/metamodels/data/1000_ele/{}/L1_benchmark/mlmc_1.hdf5".format(cl)

    elif case == 2:
        cl = "cl_0_1_s_1"
        nn_level = 3
        replace_level = False
        # mesh = "/home/martin/Documents/metamodels/data/L1/test/01_cond_field/l_step_0.055_common_files/mesh.msh"
        mesh = "/home/martin/Documents/metamodels/data/5_ele/cl_0_1_s_1/L5/l_step_0.020196309484414757_common_files/mesh.msh"
        output_dir = "/home/martin/Documents/metamodels/data/5_ele/{}/L5/test/01_cond_field/output/".format(cl)
        hdf_path = "/home/martin/Documents/metamodels/data/5_ele/{}/L5/mlmc_5.hdf5".format(cl)
        save_path = "/home/martin/Documents/metamodels/data/5_ele/{}".format(cl)
        l_0_output_dir = "/home/martin/Documents/metamodels/data/5_ele/{}/L1_{}/test/01_cond_field/output/".format(cl, nn_level)
        l_0_hdf_path = "/home/martin/Documents/metamodels/data/5_ele/{}/L1_{}/mlmc_1.hdf5".format(cl, nn_level)
        sampling_info_path = "/home/martin/Documents/metamodels/data/5_ele/{}/sampling_info".format(cl)
        ref_mlmc_file = "/home/martin/Documents/metamodels/data/5_ele/{}/L1_3/mlmc_1.hdf5".format(cl)

    elif case == 3 or case == 4:
        cl = "cl_0_1_s_1"
        if case == 4:
            cl = "cl_0_3_s_4"
        nn_level = 3
        replace_level = False
        # mesh = "/home/martin/Documents/metamodels/data/L1/test/01_cond_field/l_step_0.055_common_files/mesh.msh"
        mesh = "/home/martin/Documents/metamodels/data/5_ele/cl_0_1_s_1/L5/l_step_0.020196309484414757_common_files/mesh.msh"
        output_dir = "/home/martin/Documents/metamodels/data/5_ele/{}/L5/test/01_cond_field/output/".format(cl)
        hdf_path = "/home/martin/Documents/metamodels/data/5_ele/{}/L5/mlmc_5.hdf5".format(cl)
        save_path = "/home/martin/Documents/metamodels/data/5_ele/{}".format(cl)
        l_0_output_dir = "/home/martin/Documents/metamodels/data/5_ele/{}/L1_{}/test/01_cond_field/output/".format(cl,nn_level)
        l_0_hdf_path = "/home/martin/Documents/metamodels/data/5_ele/{}/L1_{}/mlmc_1.hdf5".format(cl, nn_level)
        sampling_info_path = "/home/martin/Documents/metamodels/data/5_ele/{}/sampling_info".format(cl)
        ref_mlmc_file = "/home/martin/Documents/metamodels/data/5_ele/{}/L1_3/mlmc_1.hdf5".format(cl)

    elif case == 5:
        cl = "cl_0_3_s_4"
        nn_level = 0
        replace_level = False
        # mesh = "/home/martin/Documents/metamodels/data/L1/test/01_cond_field/l_step_0.055_common_files/mesh.msh"
        mesh = "/home/martin/Documents/metamodels/data/5_ele/cl_0_1_s_1/L5/l_step_0.020196309484414757_common_files/mesh.msh"
        output_dir = "/home/martin/Documents/metamodels/data/5_ele/{}/L1_3/test/01_cond_field/output/".format(cl)
        hdf_path = "/home/martin/Documents/metamodels/data/5_ele/{}/L1_3/mlmc_1.hdf5".format(cl)
        save_path = "/home/martin/Documents/metamodels/data/5_ele/{}".format(cl)
        l_0_output_dir = "/home/martin/Documents/metamodels/data/5_ele/{}/L1_{}/test/01_cond_field/output/".format(cl,nn_level)
        l_0_hdf_path = "/home/martin/Documents/metamodels/data/5_ele/{}/L1_{}/mlmc_1.hdf5".format(cl, nn_level)
        sampling_info_path = "/home/martin/Documents/metamodels/data/5_ele/{}/sampling_info".format(cl)
        ref_mlmc_file = "/home/martin/Documents/metamodels/data/5_ele/{}/L1_3/mlmc_1.hdf5".format(cl)

    return output_dir, hdf_path, l_0_output_dir, l_0_hdf_path, save_path, mesh, sampling_info_path, ref_mlmc_file, replace_level, nn_level,


if __name__ == "__main__":
    case = 4
    output_dir, hdf_path, l_0_output_dir, l_0_hdf_path, save_path, mesh, sampling_info_path, ref_mlmc_file, replace_level, nn_level = get_config(
        case)

    # import cProfile
    # import pstats
    # pr = cProfile.Profile()
    # pr.enable()
    #
    # my_result = run_GNN(output_dir, hdf_path, l_0_output_dir, l_0_hdf_path, save_path, mesh, level=nn_level)
    #
    # pr.disable()
    # ps = pstats.Stats(pr).sort_stats('cumtime')
    # ps.print_stats()

    # gnn, _, _, _ = run_GNN(output_dir, hdf_path, l_0_output_dir, l_0_hdf_path, save_path, mesh, level=nn_level, stats=True)
    #
    # case = 4
    # output_dir, hdf_path, l_0_output_dir, l_0_hdf_path, save_path, mesh, sampling_info_path, ref_mlmc_file, replace_level, nn_level = get_config(case)

    #run_GNN(output_dir, hdf_path, l_0_output_dir, l_0_hdf_path, save_path, mesh, level=nn_level, conv_layer=GCNConv, model=NetGCN)  # , gnn=gnn)

    #run_GNN(output_dir, hdf_path, l_0_output_dir, l_0_hdf_path, save_path, mesh, level=nn_level)  # , gnn=gnn)

    # run_CNN(output_dir, hdf_path, l_0_output_dir, l_0_hdf_path, save_path, mesh, level=nn_level)
    process_results(hdf_path, sampling_info_path, ref_mlmc_file, save_path, nn_level, replace_level)

    # analyze_bootstrap(save_path)
    # statistics(run_GNN, output_dir, hdf_path, l_0_output_dir, l_0_hdf_path, save_path, mesh, level=nn_level)
    # bootstrap_GNN()
    # run()
    # run_CNN()
    # bootstrap()

    # gnn, targets, predictions, learning_time = run_GNN(output_dir, hdf_path, l_0_output_dir, l_0_hdf_path, save_path, mesh, level=nn_level, stats=True)
