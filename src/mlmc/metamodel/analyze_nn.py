import os
import numpy as np
import random
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Run on CPU only
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


def run_GNN():
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

    # Parameters
    #conv_layer = GCNConv
    conv_layer = ChebConv  # Seems better than GCNConv, good distribution of predictions
    # # conv_layer = GraphSageConv  # Seems better than ChebConv, good loss but very narrow distribution of predictions
    # # conv_layer = ARMAConv  # Seems worse than GraphSageConv
    # conv_layer = GATConv  # Slow and not better than GraphSageConv
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
              hidden_activation='relu', patience=20)
    train_targets = gnn.fit(loader_tr, loader_va, loader_te)

    val_targets = gnn.val_targets

    targets, predictions = gnn.predict(loader_te)
    predictions = np.squeeze(predictions)

    print("np.var(target-predictions) ", np.var(targets - predictions))

    plot_loss(gnn._train_loss, gnn._val_loss)
    analyze_results(targets, predictions)

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

    diff_moments(targets, predictions)

    save_load_data(False, targets, predictions, train_targets, val_targets)


def save_load_data(load=False, targets=None, predictions=None, train_targets=None, val_targets=None):
    path = "/home/martin/Documents/metamodels/data/"

    if load:
        targets = None
        predictions = None
        if os.path.exists(os.path.join(path, "targets.npy")):
            targets = np.load(os.path.join(path, "targets.npy"))
        if os.path.exists(os.path.join(path, "predictions.npy")):
            predictions = np.load(os.path.join(path, "predictions.npy"))
        if os.path.exists(os.path.join(path, "train_targets.npy")):
            train_targets = np.load(os.path.join(path, "train_targets.npy"))
        if os.path.exists(os.path.join(path, "val_targets.npy")):
            val_targets = np.load(os.path.join(path, "val_targets.npy"))
        return targets, predictions, train_targets, val_targets
    else:
        if targets is not None:
            np.save(os.path.join(path, "targets"), targets)
        if predictions is not None:
            np.save(os.path.join(path, "predictions"), predictions)
        if train_targets is not None:
            np.save(os.path.join(path, "train_targets"), train_targets)
        if val_targets is not None:
            np.save(os.path.join(path, "val_targets"), val_targets)


def process_results():
    targets, predictions, train_targets, val_targets = save_load_data(load=True)

    print("len targets ", len(targets))
    print("len predictions ", len(predictions))

    print("len train targets ", len(train_targets))
    print("len val targets ", len(val_targets))

    process_mlmc(targets, predictions, train_targets, val_targets)


if __name__ == "__main__":

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

    #run_GNN()

    process_results()
    #bootstrap_GNN()
    #run()

    #run_CNN()

    #bootstrap()