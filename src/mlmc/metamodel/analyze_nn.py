import os
import numpy as np
import time
import random
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Run on CPU only
from mlmc.metamodel.create_graph import graph_creator
from mlmc.metamodel.flow_dataset import FlowDataset
# Make numpy printouts easier to read.

#np.set_printoptions(precision=9, suppress=True)
import tensorflow as tf
from scipy.stats import ks_2samp
import sklearn.model_selection
from mlmc.metamodel.custom_methods import abs_activation
from mlmc.metamodel.postprocessing import analyze_results, plot_loss, estimate_density, diff_moments, process_mlmc
from mlmc.metamodel.flow_task_NN import DNN
from mlmc.metamodel.flow_task_CNN import CNN


from mlmc.metamodel.flow_task_GNN_2 import GNN
from spektral.layers import GCNConv, GlobalSumPool, ChebConv, GraphSageConv, ARMAConv, GATConv, APPNPConv, GINConv
from tensorflow.keras.losses import MeanSquaredError
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

    #train_input, train_output, test__input, test_output = split_dataset(dataset)
    #print("len test(output) ", len(test_output))

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


def run_CNN():
    # Parameters
    loss = "mean_squared_error"
    optimizer = tf.optimizers.Adam(learning_rate=0.001)

    data = FlowDataset()
    dataset = data.dataset[:10000]


    train_input, train_output, test_input, test_output = split_dataset(dataset)

    print("len test(output) ", len(test_output))

    train_input = np.expand_dims(train_input, axis=-1)
    test_input = np.expand_dims(test_input, axis=-1)

    print("train input shape ", train_input.shape)

    dnn = CNN(loss=loss, optimizer=optimizer, output_activation=abs_activation, hidden_activation='relu')

    dnn.fit(train_input, train_output)

    test_dataset = data.dataset[10000:]
    test_input = prepare_data(test_dataset.x)
    test_output = prepare_data(test_dataset.y)

    predictions = dnn.predict(test_input)
    predictions = np.squeeze(predictions)

    plot_loss(dnn.history.history['loss'], dnn.history.history['val_loss'])

    analyze_results(test_output, predictions)

    estimate_density(test_output)
    estimate_density(predictions)


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


def bootstrap_GNN():
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
        # Parameters
        # conv_layer = GCNConv
        conv_layer = ChebConv  # Seems better than GCNConv, good distribution of predictions
        # # conv_layer = GraphSageConv  # Seems better than ChebConv, good loss but very narrow distribution of predictions
        # # conv_layer = ARMAConv  # Seems worse than GraphSageConv
        # # conv_layer = GATConv  # Slow and not better than GraphSageConv
        # # conv_layer = APPNPConv  # Not bad but worse than GraphSageConv
        # # conv_layer = GINConv  # it is comparable to APPNPConv
        # act_func = "relu"  # "tanh"#"elu"  # ReLU keep predictions above zero
        loss = MeanSquaredError()
        optimizer = tf.optimizers.Adam(learning_rate=0.001)
        batch_size = 1000
        epochs = 100

        # Load data
        data = FlowDataset()
        data = data  # [:10000]
        # data.a = conv_layer.preprocess(data.a)
        data.a = sp_matrix_to_sp_tensor(data.a)

        train_data_len = int(len(data) * 0.8)
        train_data_len = 10000

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

        gnn = GNN(loss=loss, optimizer=optimizer, conv_layer=conv_layer, output_activation=abs_activation,
                  hidden_activation='relu', patience=10)
        gnn.fit(loader_tr, loader_va, loader_te)

        targets, predictions = gnn.predict(loader_te)
        predictions = np.squeeze(predictions)

        all_test_outputs.append(targets)
        all_predictions(predictions)

        train_losses.append(gnn._train_loss[-1])
        val_losses.append(gnn._val_loss[-1])


    plot_loss(train_losses, val_losses)
    analyze_results(np.mean(all_test_outputs, axis=0), np.mean(all_predictions, axis=0))
    analyze_results(np.var(all_test_outputs, axis=0), np.var(all_predictions, axis=0))

    estimate_density(np.mean(all_test_outputs, axis=0), title="Test outputs")
    estimate_density(np.mean(all_predictions, axis=0), title="Predictions")


def run_GNN(output_dir, hdf_path, l_0_output_dir, l_0_hdf_path, save_path):
    # Parameters
    conv_layer = GCNConv
    conv_layer = ChebConv  # Seems better than GCNConv, good distribution of predictions
    # # conv_layer = GraphSageConv  # Seems better than ChebConv, good loss but very narrow distribution of predictions
    # # conv_layer = ARMAConv  # Seems worse than GraphSageConv
    # conv_layer = GATConv  # Slow and not better than GraphSageConv
    # # conv_layer = APPNPConv  # Not bad but worse than GraphSageConv
    # # conv_layer = GINConv  # it is comparable to APPNPConv
    # act_func = "relu"  # "tanh"#"elu"  # ReLU keep predictions above zero
    loss = MeanSquaredError()
    optimizer = tf.optimizers.Adam(learning_rate=0.01)
    batch_size = 1000
    epochs = 100
    hidden_regularization = None#l2(2e-10)

    preprocess_start_time = time.process_time()
    graph_creator(output_dir, hdf_path)

    # Load data
    data = FlowDataset(output_dir=output_dir)
    data = data  # [:10000]
    preprocess_time = time.process_time() - preprocess_start_time

    learning_time_start = time.process_time()
    # data.a = conv_layer.preprocess(data.a)
    data.a = sp_matrix_to_sp_tensor(data.a)

    train_data_len = int(len(data) * 0.8)
    train_data_len = 2000

    # Train/valid/test split
    data_tr, data_te = data[:train_data_len], data[train_data_len:],
    np.random.shuffle(data_tr)

    val_data_len = int(len(data_tr) * 0.2)
    data_tr, data_va = data_tr[:-val_data_len], data_tr[-val_data_len:]

    # print("data_tr len ", len(data_tr))
    # print("data_va len ", len(data_va))
    # print("data_te len ", len(data_te))

    # We use a MixedLoader since the dataset is in mixed mode
    loader_tr = MixedLoader(data_tr, batch_size=batch_size, epochs=epochs)
    loader_va = MixedLoader(data_va, batch_size=batch_size)
    loader_te = MixedLoader(data_te, batch_size=batch_size)

    gnn = GNN(loss=loss, optimizer=optimizer, conv_layer=conv_layer, output_activation=abs_activation,
              hidden_activation='relu', patience=20, hidden_reqularizer=hidden_regularization)
    train_targets = gnn.fit(loader_tr, loader_va, loader_te)

    val_targets = gnn.val_targets

    targets, predictions = gnn.predict(loader_te)
    predictions = np.squeeze(predictions)
    learning_time = time.process_time() - learning_time_start

    #print("np.var(target-predictions) ", np.var(targets - predictions))

    #plot_loss(gnn._train_loss, gnn._val_loss)
    #analyze_results(targets, predictions)

    # target_means, target_vars = estimate_density(targets, title="Test outputs")
    # pred_means, pred_vars = estimate_density(predictions, title="Predictions")
    #
    # print("target means ", target_means)
    # print("predic means ", pred_means)
    #
    # print("target vars ", target_vars)
    # print("predic vars ", pred_vars)
    #
    # diff_means, diff_vars = estimate_density(targets - predictions, title="diff")
    # print("diff_means ", diff_means)
    # print("diff vars ", diff_vars)

    #diff_moments(targets, predictions)

    predict_l_0_start_time = time.process_time()
    l_0_targets, l_0_predictions = predict_level_zero(gnn, l_0_output_dir, l_0_hdf_path, batch_size)
    predict_l_0_time = time.process_time() - predict_l_0_start_time

    save_times(save_path, False, (preprocess_time, len(data)), learning_time, (predict_l_0_time, len(l_0_targets)))
    save_load_data(save_path, False, targets, predictions, train_targets, val_targets, l_0_targets, l_0_predictions)


def predict_level_zero(nn, output_dir, hdf_path, batch_size=1000):
    graph_creator(output_dir, hdf_path)

    # Load data
    data = FlowDataset(output_dir=output_dir)
    data = data  # [:10000]
    # data.a = conv_layer.preprocess(data.a)
    data.a = sp_matrix_to_sp_tensor(data.a)

    loader_te = MixedLoader(data, batch_size=batch_size)

    targets, predictions = nn.predict(loader_te)
    predictions = np.squeeze(predictions)
    #analyze_results(targets, predictions)
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


def save_load_data(path, load=False, targets=None, predictions=None, train_targets=None, val_targets=None, l_0_targets=None,
                   l_0_predictions=None):

    if load:
        if os.path.exists(os.path.join(path, "targets.npy")):
            targets = np.load(os.path.join(path, "targets.npy"))
        if os.path.exists(os.path.join(path, "predictions.npy")):
            predictions = np.load(os.path.join(path, "predictions.npy"))
        if os.path.exists(os.path.join(path, "train_targets.npy")):
            train_targets = np.load(os.path.join(path, "train_targets.npy"))
        if os.path.exists(os.path.join(path, "val_targets.npy")):
            val_targets = np.load(os.path.join(path, "val_targets.npy"))
        if os.path.exists(os.path.join(path, "l_0_targets.npy")):
            l_0_targets = np.load(os.path.join(path, "l_0_targets.npy"))
        if os.path.exists(os.path.join(path, "l_0_predictions.npy")):
            l_0_predictions = np.load(os.path.join(path, "l_0_predictions.npy"))
        return targets, predictions, train_targets, val_targets, l_0_targets, l_0_predictions
    else:
        if targets is not None:
            np.save(os.path.join(path, "targets"), targets)
        if predictions is not None:
            np.save(os.path.join(path, "predictions"), predictions)
        if train_targets is not None:
            np.save(os.path.join(path, "train_targets"), train_targets)
        if val_targets is not None:
            np.save(os.path.join(path, "val_targets"), val_targets)
        if l_0_targets is not None:
            np.save(os.path.join(path, "l_0_targets"), l_0_targets)
        if l_0_predictions is not None:
            np.save(os.path.join(path, "l_0_predictions"), l_0_predictions)


def process_results(hdf_path, sampling_info_path, save_path):
    targets, predictions, train_targets, val_targets, l_0_targets, l_0_predictions = save_load_data(save_path, load=True)
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

    process_mlmc(hdf_path, sampling_info_path, targets, predictions, train_targets, val_targets, l_0_targets, l_0_predictions, l1_sample_time, l0_sample_time)


if __name__ == "__main__":
    output_dir = "/home/martin/Documents/metamodels/data/1000_ele/cl_0_1_s_1/L5/test/01_cond_field/output/"
    hdf_path = "/home/martin/Documents/metamodels/data/1000_ele/cl_0_1_s_1/L5/mlmc_5.hdf5"

    save_path = "/home/martin/Documents/metamodels/data/1000_ele/cl_0_1_s_1"

    l_0_output_dir = "/home/martin/Documents/metamodels/data/1000_ele/cl_0_1_s_1/L1/test/01_cond_field/output/"
    l_0_hdf_path = "/home/martin/Documents/metamodels/data/1000_ele/cl_0_1_s_1/L1/mlmc_1.hdf5"

    sampling_info_path = "/home/martin/Documents/metamodels/data/1000_ele/cl_0_1_s_1/sampling_info"

    # import cProfile
    # import pstats
    # pr = cProfile.Profile()
    # pr.enable()
    #
    # my_result = run_GNN()
    #
    # pr.disable()
    # ps = pstats.Stats(pr).sort_stats('cumtime')
    # ps.print_stats()

    #run_GNN(output_dir, hdf_path, l_0_output_dir, l_0_hdf_path, save_path)
    #
    process_results(hdf_path, sampling_info_path, save_path)
    #bootstrap_GNN()
    #run()

    #run_CNN()

    #bootstrap()