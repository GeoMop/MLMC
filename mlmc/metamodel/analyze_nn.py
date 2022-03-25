import warnings
import os
import logging
logging.getLogger('tensorflow').disabled = True
logging.getLogger('absl').disabled = True

import numpy as np
import time
import glob
import copy
import pickle
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Run on CPU only
from mlmc.metamodel.flow_dataset import FlowDataset
from mlmc.metamodel.create_graph import graph_creator
from mlmc.moments import Legendre_tf, Monomial
from mlmc.metamodel.random_field_time import corr_field_sample_time
from mlmc.plot import plots
import matplotlib.pyplot as plt
# Make numpy printouts easier to read.
from scipy import stats
# np.set_printoptions(precision=9, suppress=True)
import tensorflow as tf

from tensorflow import keras
from scipy.stats import ks_2samp
import sklearn.model_selection
from mlmc.metamodel.custom_methods import abs_activation, MSE_moments
from mlmc.metamodel.postprocessing import analyze_results, plot_loss, estimate_density, process_mlmc, plot_progress, plot_learning_rate
from mlmc.metamodel.flow_task_NN import DNN
from mlmc.metamodel.flow_task_CNN import CNN
import keras.backend as K
from mlmc.metamodel.flow_task_GNN_2 import GNN
from tensorflow.keras.losses import MeanSquaredError
from spektral.data import MixedLoader
from spektral.utils.sparse import sp_matrix_to_sp_tensor

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

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


def run_CNN(output_dir, hdf_path, l_0_output_dir, l_0_hdf_path, save_path, mesh, level, log):
    # Parameters
    loss = "mean_squared_error"
    optimizer = tf.optimizers.Adam(learning_rate=0.01)

    data = FlowDataset(output_dir=output_dir, level=level, log=log)
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


def run_SVR(config, stats=True, train=True, log=False, seed=1234):
    from sklearn.svm import SVR
    print("seed ", seed)

    batch_size = 200
    epochs = 1000
    hidden_regularization = None  # l2(2e-10)
    graph_creation_time = config['graph_creation_time']
    if graph_creation_time == 0:
        graph_creator_preproces_time = time.process_time()
        graph_creator(config['output_dir'], config['hdf_path'], config['mesh'], level=config['level'])
        graph_creation_time = time.process_time() - graph_creator_preproces_time
        print("graph creation time ", graph_creation_time)
        exit()

    preprocess_start_time = time.process_time()
    # Load data
    data = FlowDataset(output_dir=config['output_dir'], level=config['level'], log=log)
    data.shuffle(seed=seed)

    dataset = data.dataset
    dataset = dataset.sample(frac=1)

    train = dataset[:config['n_train_samples']]
    test = dataset[config['n_train_samples']:]

    train_input, train_output = train.x, train.y
    test_input, test_output = test.x, test.y

    train_input = prepare_data(train_input)
    train_output = prepare_data(train_output)

    test_input = prepare_data(test_input)
    test_output = prepare_data(test_output)

    # sc_X = StandardScaler()
    # sc_y = StandardScaler()
    # train_input = sc_X.fit_transform(train_input)
    # train_output = sc_y.fit_transform(train_output.reshape(-1,1))
    # test_input = sc_X.fit_transform(test_input)
    # test_output = sc_y.fit_transform(test_output.reshape(-1,1))
    #train_input, train_output, test_input, test_output = split_dataset(dataset)

    preprocess_time = time.process_time() - preprocess_start_time
    preprocess_time = preprocess_time + graph_creation_time
    learning_time_start = time.process_time()

    print("train input ", train_input.shape)
    print("train output ", train_output.shape)

    svr_rbf = SVR(kernel='rbf', verbose=True)  # 'linear' kernel fitting is never-ending and 'poly' kernel gives very bad score (e.g. -2450), sigmoid gives also bad score (e.g. -125)
    svr_rbf.fit(train_input, train_output)
    train_error = svr_rbf.score(train_input, train_output)

    #print("svr_rbf.get_params() ", svr_rbf.get_params())
    total_steps = 0

    #test_input = sc_X.fit_transform(test_input)
    test_error = svr_rbf.score(test_input, test_output)

    targets = test_output
    train_targets = train_output

    # test_y = sc_y.fit_transform(test.y.to_numpy().reshape(-1,1))

    predictions = svr_rbf.predict(test_input)

    print("train error ", train_error)
    print("test error ", test_error)

    train_predictions = svr_rbf.predict(train_input)
    #train_predictions = np.squeeze(train_predictions)

    learning_time = time.process_time() - learning_time_start
    print("learning time ", learning_time)

    val_targets = []

    orig_targets = targets
    orig_predictions = predictions
    print("MSE ", np.mean((predictions - targets) ** 2))

    if log:
        targets = np.exp(targets)
        predictions = np.exp(predictions)

    print("np.var(target-predictions) ", np.var(targets - predictions))

    if not stats:
        #plot_loss(gnn._train_loss, gnn._val_loss)
        analyze_results(targets, predictions)

        import matplotlib.pyplot as plt

        # plt.hist(train_output, bins=50, alpha=0.5, label='train target', density=True)
        # plt.hist(train_predictions, bins=50, alpha=0.5, label='train predictions', density=True)
        #
        # # plt.hist(targets - predictions, bins=50, alpha=0.5, label='predictions', density=True)
        # plt.legend(loc='upper right')
        # # plt.xlim(-0.5, 1000)
        # plt.yscale('log')
        # plt.show()

        plt.hist(targets, bins=50, alpha=0.5, label='target', density=True)
        plt.hist(predictions, bins=50, alpha=0.5, label='predictions', density=True)

        # plt.hist(targets - predictions, bins=50, alpha=0.5, label='predictions', density=True)
        plt.legend(loc='upper right')
        # plt.xlim(-0.5, 1000)
        plt.yscale('log')
        plt.show()

    #predict_l_0_start_time = time.process_time()
    l_0_targets, l_0_predictions, predict_l_0_time = predict_level_zero_SVR(svr_rbf, config['l_0_output_dir'],
                                                                            config['l_0_hdf_path'],
                                                                            config['mesh'], batch_size, log,
                                                                            stats=stats,
                                                                            corr_field_config=config['corr_field_config'],
                                                                            seed=seed)

    val_predictions = []

    if stats:
        l1_sample_time = preprocess_time / len(data) + learning_time / len(data)
        l0_sample_time = predict_l_0_time / len(l_0_targets)

        # print("targets ", targets)
        # print("predictions ", predictions)

        # orig_max_vars, predict_max_vars = process_mlmc(hdf_path, sampling_info_path, ref_mlmc_file, targets, predictions, train_targets,
        #              train_predictions,
        #              val_targets, l_0_targets,
        #              l_0_predictions, l1_sample_time, l0_sample_time, nn_level=level, replace_level=replace_level,
        #                                                stats=stats)

        return svr_rbf, targets, predictions, learning_time, train_targets, train_predictions, \
               val_targets, val_predictions, l_0_targets, l_0_predictions, l1_sample_time, l0_sample_time, total_steps, None, None, None

    save_times(config['save_path'], False, (preprocess_time, len(data)), learning_time, (predict_l_0_time, len(l_0_targets)))
    save_load_data(config['save_path'], False, targets, predictions, train_targets, train_predictions, val_targets, l_0_targets,
                   l_0_predictions)


def predict_level_zero_SVR(nn, output_dir, hdf_path, mesh, batch_size=1000, log=False, stats=False, corr_field_config=None, seed=1234):
    #graph_creator(output_dir, hdf_path, mesh, level=0)
    sample_time = 0
    if corr_field_config:
        sample_time = corr_field_sample_time(mesh, corr_field_config)

    # Load data
    data = FlowDataset(output_dir=output_dir, log=log)
    data.shuffle(seed=seed)
    dataset = data.dataset[:]

    predict_time_start = time.process_time()
    test_input = prepare_data(dataset.x)
    targets = prepare_data(dataset.y)
    #print("data prepared")

    predictions = []
    for i in range(0, len(test_input), batch_size):
        predictions.extend(nn.predict(test_input[i:i + batch_size]))
    predictions = np.array(predictions)
    #print("predictison shape ", predictions.shape)
    predictions = np.squeeze(predictions)

    if log:
        targets = np.exp(targets)
        predictions = np.exp(predictions)
    # analyze_results(targets, predictions)
    predict_time = time.process_time() - predict_time_start
    return targets, predictions, predict_time + sample_time * len(data)


def statistics(config):
    n_subsamples = 25

    model_title, mch_l_model, log = config['machine_learning_model']
    model_data = {}
    model_data["log"] = log


    # seeds = []
    # for i in range(n_subsamples):
    #     seeds.append(i * 125)

    if not os.path.isdir(config['save_path']):
        os.makedirs(config['save_path'])

        if os.path.exists(os.path.join(config['save_path'], "dataset_config.pkl")):
            os.remove(os.path.join(config['save_path'], "dataset_config.pkl"))

            # create a binary pickle file
        with open(os.path.join(config['save_path'], "dataset_config.pkl"), "wb") as writer:
            pickle.dump(config["dataset_config"], writer)

    else:
        print("dir exists {}".format(config['save_path']))
        exit()

    for i in range(n_subsamples):
        iter_dir = os.path.join(config['save_path'], "{}".format(i))
        if not os.path.isdir(iter_dir):
            os.makedirs(iter_dir)

            config['iter_dir'] = iter_dir

            gnn, targets, predictions, learning_time, train_targets, train_predictions, \
            val_targets, val_predictions, l_0_targets, l_0_predictions, l1_sample_time, l0_sample_time, total_steps = \
                mch_l_model(config, stats=True, train=config.get('train_model', True), log=log, seed=i)

            if config['save_model']:
                model_data["model"] = gnn._model
                model_data["train_loss"] = gnn._train_loss
                model_data["train_acc"] = gnn._train_acc
                model_data["val_loss"] = gnn._val_loss
                model_data["test_loss"] = gnn._test_loss
                model_data["learning_rates"] = gnn._learning_rates
            model_data["test_targets"] = targets
            model_data["test_predictions"] = predictions
            model_data["train_targets"] = train_targets
            model_data["train_predictions"] = train_predictions
            model_data["val_targets"] = val_targets
            model_data["val_predictions"] = val_predictions
            model_data["l_0_targets"] = l_0_targets
            model_data["l_0_predictions"] = l_0_predictions
            model_data["l1_sample_time"] = l1_sample_time
            model_data["l0_sample_time"] = l0_sample_time
            model_data["total_steps"] = total_steps
            model_data["learning_times"] = learning_time

            save_statistics(iter_dir, model_data)

            # save_times(save_path, False, (preprocess_time, len(data)), learning_time, (predict_l_0_time, len(l_0_targets)))
            # save_load_data(save_path, False, targets, predictions, train_targets, val_targets, l_0_targets, l_0_predictions)

    # for i in range(len(train_losses)):
    #     print("train loss ", train_losses[i])
    #     print("test loss ", test_losses[i])
    #     analyze_results(all_targets[i], all_predictions[i])
    #     print("learning time ", learning_times[i])
    #     print("##################################################")

    return analyze_statistics(config)

    # plot_loss(train_losses, val_losses)
    # analyze_results(np.mean(all_test_outputs, axis=0), np.mean(all_predictions, axis=0))
    # analyze_results(np.var(all_test_outputs, axis=0), np.var(all_predictions, axis=0))
    #
    # estimate_density(np.mean(all_test_outputs, axis=0), title="Test outputs")
    # estimate_density(np.mean(all_predictions, axis=0), title="Predictions")


def save_statistics(save_dir_path, model_data):
    for file_name, data in model_data.items():
        if file_name == "model" and data is not None:
            data.save(os.path.join(save_dir_path, file_name))
        else:
            np.save(os.path.join(save_dir_path, file_name), data)


def load_statistics(dir_path):
    models_data = {}
    models_data["model"] = []
    models_data["train_loss"] = []
    models_data["train_acc"] = []
    models_data["val_loss"] = []
    models_data["test_loss"] = []
    models_data["learning_rates"] = []
    models_data["test_targets"] = []
    models_data["test_predictions"] = []
    models_data["train_targets"] = []
    models_data["train_predictions"] = []
    models_data["val_targets"] = []
    models_data["val_predictions"] = []
    models_data["l_0_targets"] = []
    models_data["l_0_predictions"] = []
    models_data["l1_sample_time"] = []
    models_data["l0_sample_time"] = []
    models_data["total_steps"] = []
    models_data["learning_times"] = []
    models_data["log"] = []
    models_data["dataset_config"] = []


    #dirs = (os.path.split(dir_path)[-1]).split("_")
    n_iters = 25
    for i in range(n_iters):
        data_dir_path = os.path.join(dir_path, str(i))
        if not os.path.isdir(data_dir_path):
            print("data dir not exists {}".format(data_dir_path))
            break
        if os.path.exists(os.path.join(data_dir_path, 'model')):
            models_data['model'].append(keras.models.load_model(os.path.join(data_dir_path, 'model')))
        for file in glob.glob(os.path.join(data_dir_path, "*.npy")):
            file_name = os.path.split(file)[-1]
            file_name = file_name.split(".")[0]
            # if file_name not in models_data:
            #     print("file name ", file_name)
            #     models_data[file_name] = []
            #     print("np.load(file, allow_pickle=True) ", np.load(file, allow_pickle=True))
            #     exit()
            models_data[file_name].append(np.load(file, allow_pickle=True))

        if os.path.exists(os.path.join(data_dir_path, "dataset_config.pkl")):
            # Save config to Pickle
            import pickle
            # create a binary pickle file
            with open(os.path.join(data_dir_path, "dataset_config.pkl"), "rb") as reader:
                dataset_config = pickle.load(reader)
                models_data["dataset_config"].append(dataset_config)

    return models_data


def plot_sse(data_nn, data_mlmc, x_label="ith moment", y_label="MSE", title=""):
    import matplotlib
    #matplotlib.rcParams.update({'font.size': 38})
    matplotlib.rcParams.update({'lines.markersize': 14})
    fig, axes = plt.subplots(1, 1, figsize=(22, 10))
    data = np.array(data_nn)
    x = range(data.shape[1])
    axes.set_title(title)
    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)
    axes.errorbar(x, np.mean(data_nn, axis=0), yerr=np.sqrt(np.var(data_nn, axis=0)), fmt='o', label="NN MLMC", color="red")
    axes.errorbar(x, np.mean(data_mlmc, axis=0), yerr=np.sqrt(np.var(data_mlmc, axis=0)), fmt='o', label="MLMC", color="blue")
    fig.legend()
    fig.savefig("{}.pdf".format(title))
    fig.show()


def compare_models(model_1, model_2, config):
    check_loss(config, model_1, dataset_config=config["dataset_config"])
    check_loss(config, model_2, dataset_config=config["dataset_config"])

    exit()


def check_loss(config, model, log=True, dataset_config={}):
    if model is None:
        return
    batch_size = config['batch_size']

    config['dataset_config'] = dataset_config

    print("config ", config)
    print("dataset config ", config["dataset_config"])

    data = FlowDataset(output_dir=config['output_dir'], level=config['level'], log=log, config=config, index=None)
    data = data  # [:10000]

    data.a = config['conv_layer'].preprocess(data.a)
    data.a = sp_matrix_to_sp_tensor(data.a)

    train_data_len = config["n_train_samples"]

    idx = 0
    data_tr = data[idx * train_data_len: idx * train_data_len + train_data_len]
    data_te = data.get_test_data(idx, train_data_len)

    print("len(datate) ", len(data_te))
    print("batch size ", batch_size)

    loader_tr = MixedLoader(data_tr, batch_size=batch_size)
    loader_te = MixedLoader(data_te, batch_size=batch_size)

    train_targets, train_predictions = model_predict(model, loader_tr)
    train_predictions = np.squeeze(train_predictions)

    test_targets, test_predictions = model_predict(model, loader_te)
    test_predictions = np.squeeze(test_predictions)

    #print("(train_predictions - train_targets)  ", (train_predictions - train_targets))

    train_MSE = np.mean((train_predictions - train_targets) ** 2)
    train_bias = np.mean((train_targets - np.mean(train_predictions))**2)
    train_variance = np.mean((train_predictions - np.mean(train_predictions))**2)
    train_variance_2 = np.var(train_predictions)

    test_MSE = np.mean((test_predictions - test_targets) ** 2)
    test_bias = np.mean((test_targets - np.mean(test_predictions)) ** 2)
    test_variance = np.mean((test_predictions - np.mean(test_predictions)) ** 2)
    test_variance_2 = np.var(test_predictions)

    # print("test targets ", np.sort(test_targets)[:10])
    # print("test predictions ", test_predictions)

    print("train MSE: {}, test MSE: {}".format(train_MSE, test_MSE))

    print("train MSE: {}, bias: {}, variance: {}, var2: {}".format(train_MSE, train_bias, train_variance, train_variance_2))
    print("test MSE: {}, bias: {}, variance: {}, var2: {}".format(test_MSE, test_bias, test_variance, test_variance_2))

    exit()

    conv_layers = {}
    dense_layers = {}
    flatten_input = []
    flatten_output = []


def model_predict(model, loader):
    targets = []
    predictions = []
    step = 0
    for batch in loader:
        step += 1
        inputs, target = batch
        targets.extend(target)
        predictions.extend(model(inputs, training=False))

        if step == loader.steps_per_epoch:
            return targets, predictions

    return targets, predictions


def predict_data(config, model, mesh_file, log=True):
    if model is None:
        return
    batch_size = config['batch_size']
    data = FlowDataset(output_dir=config['output_dir'], level=config['level'], log=log, config=config, index=0,
                       predict=True)
    data = data  # [:10000]

    data.a = config['conv_layer'].preprocess(data.a)
    data.a = sp_matrix_to_sp_tensor(data.a)

    # idx = 0
    # data_te = data.get_test_data(idx, train_data_len)
    data_te = data[-1:]

    print("len(datate) ", len(data_te))
    print("batch size ", batch_size)

    # We use a MixedLoader since the dataset is in mixed mode
    loader = MixedLoader(data_te, batch_size=batch_size)

    conv_layers = {}
    dense_layers = {}
    flatten_input = []
    flatten_output = []

    step = 0
    for batch in loader:
        if step == loader.steps_per_epoch:
            break
        inputs, target = batch
        x, a = inputs

        print("x ", x)

        for conv_index, conv_layer in enumerate(model._conv_layers):
            if conv_index not in conv_layers:
                conv_layers[conv_index] = [[], [], []]
            conv_layers[conv_index][0].extend(x)  # inputs
            print("conv_layer.kernel.numpy().shape", conv_layer.kernel.numpy().shape)
            conv_layers[conv_index][1].extend(conv_layer.kernel.numpy())  # weights (kernel)
            conv_out = conv_layer([x, a])

            print("conv out ", conv_out)
            conv_layers[conv_index][2].extend(conv_out)  # outputs

        flatten_input = conv_layers[conv_index][2][-1]
        # flatten_output = model.flatten(conv_out)
        #
        # print("flatten output ", flatten_output)

        prev_layer_input = conv_out
        prev_layer = model.flatten

        print("flatten output ", flatten_output)
        print("model._dense_layers ", model._dense_layers)

        for index, dense_layer in enumerate(model._dense_layers):
            # if index == 1:
            #     break

            if index not in dense_layers:
                dense_layers[index] = [[], [], []]

            if prev_layer is None:
                prev_layer = model._dense_layers[index - 1]

            print("dense layer ", dense_layer)
            print("dense layer ", dense_layer.weights)
            print("prev layer ", prev_layer)
            print("prev layer ", prev_layer.weights)

            #
            # print("prev layer ", prev_layer.weights)

            #print("dense layer kernel", dense_layer.kernel)
            #print("model.flatten(conv_out) ", model.flatten(conv_out))

            print("prev layer input ", prev_layer_input)

            dense_layers[index][0].extend(prev_layer(prev_layer_input))  # inputs
            dense_layers[index][1].extend(dense_layer.weights)  # weights (kernel)
            dense_layers[index][2].extend(dense_layer(prev_layer(prev_layer_input)))  # outputs

            prev_layer_input = prev_layer(prev_layer_input)

            prev_layer = None

        step += 1

    plot_progress(conv_layers, dense_layers, flatten_output, mesh_file=mesh_file)


def remove_empty(data_1, data_2):
    new_data_1 = []
    new_data_2 = []

    # print("data 1 ", data_1)
    # print("data 2 ", data_2)

    for d1, d2 in zip(data_1, data_2):
        if len(d1) > 0 and len(d2) > 0:
            new_data_1.append(d1)
            new_data_2.append(d2)

    # print("new data ", new_data_1)
    # print("new data ", new_data_2)
    return np.array(new_data_1), np.array(new_data_2)


def remove_outliers(data, limit):
    new_data = []
    for d in data:
        if d < limit:
            new_data.append(d)
    return new_data


def process_data(data_dict):
    new_dict = data_dict
    for tag, d in data_dict.items():
        print("tag ", tag)
        print("d ", d)

        if tag in ["train_predictions", "train_targets", "test_predictions", "test_targets"]:
            dt = []

            min_length = 10**9
            for item in data_dict[tag]:
                if len(item) < min_length and len(item) > 0:
                    min_length = len(item)

            for item in data_dict[tag]:
                dt.append(item[:min_length])
            # print("dt " ,dt)
            # print("array dt shape ", np.array(dt).shape)
            # print("tag ", tag)

            new_dict[tag] = np.array(dt)

    return new_dict


def analyze_statistics(config, get_model=True):
    if not os.path.isdir(config['save_path']):
        print("dir not exists")
        exit()

    data_dict = load_statistics(config['save_path'])

    data_dict = process_data(data_dict)

    # print("train predictions type ", type(data_dict["train_predictions"]))
    # print("train predictions type ", type(data_dict["train_predictions"][0]))
    # print("train predictions shape ", np.array(data_dict["train_predictions"]).shape)
    # print("train predictions ", data_dict["train_predictions"])
    # print("train predictions as matrix shape", np.asmatrix(np.array(data_dict["train_predictions"])).shape)

    # print("data dict ", data_dict)

    # for key, data_dict in models_data.items():
    #     print("model: {}".format(key))

    mlmc_n_collected_all = []
    nn_n_collected_all = []
    n_ops_all = []
    n_ops_predict_all = []

    all_mlmc_moments_mean = []
    all_nn_moments_mean = []

    all_mlmc_moments_var = []
    all_nn_moments_var = []

    mlmc_times = []
    nn_times = []
    mlmc_times_levels = []
    nn_times_levels = []

    mlmc_l_vars = []
    mlmc_vars = []
    nn_vars = []
    nn_l_vars = []
    mlmc_l_means = []
    nn_l_means = []
    mlmc_vars_mse = []
    nn_vars_mse = []
    mlmc_means_mse = []
    nn_means_mse = []
    kl_mlmc_all = []
    kl_nn_all = []

    orth_mlmc_means_mse = []
    orth_nn_means_mse = []

    train_MSE_list = []
    train_bias = []
    train_variance = []
    test_MSE_list = []
    test_bias = []
    test_variance = []

    limit = 1e10  # 0.008#0.01#0.0009
    #limit = 0.37

    for i in range(len(data_dict["test_targets"])):
        # print("index i ", i)
        # if i == 4:
        #     continue

        # if i == 1:
        #     continue

        #print("index ", i)

        # if i not in [0,2]:
        #     continue

        # if i in [2, 11, 12]:
        #     continue

        # if i in [7, 13, 14]:
        #     continue

        # if i not in [13]:
        #     continue

        predictions = data_dict["test_predictions"][i]
        targets = data_dict["test_targets"][i]
        train_predictions = data_dict["train_predictions"][i]
        train_targets = data_dict["train_targets"][i]
        val_predictions = data_dict["val_predictions"][i]
        val_targets = data_dict["val_targets"][i]
        l_0_predictions = data_dict["l_0_predictions"][i]
        l_0_targets = data_dict["l_0_targets"][i]
        l1_sample_time = data_dict["l1_sample_time"][i]
        l0_sample_time = data_dict["l0_sample_time"][i]
        total_steps = data_dict["total_steps"][i]
        learning_time = data_dict["learning_times"][i]
        print("learning time ", learning_time)

        try:
            model = data_dict["model"][i]
            model_train_loss = data_dict["train_loss"][i]
            if "train_acc" in data_dict and len(data_dict["train_acc"]) > 0:
                model_train_acc = data_dict["train_acc"][i]
            else:
                model_train_acc = None
            model_val_loss = data_dict["val_loss"][i]
            model_test_loss = data_dict["test_loss"][i]
            model_learning_rates = data_dict["learning_rates"][i]
        except:
            model = None

        if model is not None:
            plot_loss(model_train_loss, model_val_loss, model_train_acc)
            plot_learning_rate(model_learning_rates)
            print("model learning rates ", model_learning_rates)

            print("model ", model)
            print("dir(model.optimizer) ", dir(model.optimizer))
            #print("model weights ", model.weights)
            print("model.optimizer", model.optimizer)
            # print("model.optimizer", K.eval(model.optimizer.lr))
            # exit()

        iter_test_MSE = np.mean((predictions - targets) ** 2)
        iter_test_bias = np.sqrt(np.mean((targets - np.mean(predictions)) ** 2))
        iter_test_variance = np.mean((predictions - np.mean(predictions)) ** 2)

        iter_train_MSE = np.mean((train_predictions - train_targets) ** 2)
        iter_train_bias = np.sqrt(np.mean((train_targets - np.mean(train_predictions)) ** 2))
        iter_train_variance = np.mean((train_predictions - np.mean(train_predictions)) ** 2)

        train_MSE_list.append(iter_train_MSE)
        train_bias.append(iter_train_bias)
        train_variance.append(iter_train_variance)
        test_MSE_list.append(iter_test_MSE)
        test_bias.append(iter_test_bias)
        test_variance.append(iter_test_variance)

        if iter_test_MSE > limit:
            continue

        print("iter test MSE: {}, bias: {}, variance:{} ".format(iter_test_MSE, iter_test_bias, iter_test_variance))
        print("iter train MSE: {}, bias: {}, variance:{} ".format(iter_train_MSE, iter_train_bias, iter_train_variance))


        # if "current_patience" in data_dict:
        #     current_patience = data_dict["current_patience"][i]
        #     print("current patience ", current_patience)

        dataset_config = {}
        if 'dataset_config' in data_dict and len(data_dict.get("dataset_config")) > 0:
            dataset_config = data_dict.get("dataset_config")[i]
        else:
            if os.path.exists(os.path.join(config['save_path'], "dataset_config.pkl")):
                # Save config to Pickle
                import pickle
                # create a binary pickle file
                with open(os.path.join(config['save_path'], "dataset_config.pkl"), "rb") as reader:
                    dataset_config = pickle.load(reader)

        config['dataset_config'] = dataset_config

        #check_loss(config, model, dataset_config=config["dataset_config"])

        #predict_data(config, model, mesh_file=config["mesh"])
        #exit()

        print("total steps ", total_steps)
        try:
            mlmc_n_collected, nn_mlmc_n_collected, n_ops, n_ops_predict, orig_moments_mean, predict_moments_mean, \
            ref_moments_mean, orig_level_params, nn_level_params, kl_mlmc, kl_nn, target_variance, \
            orig_orth_moments, predict_orth_moments, ref_orth_moments = process_mlmc(config['hdf_path'],
                                                                                     config['sampling_info_path'],
                                                                                     config['ref_mlmc_file'], targets,
                                                                                     predictions, train_targets,
                                                                                     train_predictions,
                                                                                     val_targets, l_0_targets,
                                                                                     l_0_predictions, l1_sample_time,
                                                                                     l0_sample_time,
                                                                                     nn_level=config['level'],
                                                                                     replace_level=config['replace_level'],
                                                                                     mlmc_hdf_file=config['mlmc_hdf_path'],
                                                                                     stats=True,
                                                                                     learning_time=learning_time,
                                                                                     dataset_config=dataset_config)
        except:
             continue

        mlmc_n_collected_all.append(mlmc_n_collected)
        nn_n_collected_all.append(nn_mlmc_n_collected)
        n_ops_all.append(n_ops)
        n_ops_predict_all.append(n_ops_predict)
        mlmc_times_levels.append(np.array(mlmc_n_collected) * np.array(n_ops))
        mlmc_times.append(np.sum(np.array(mlmc_n_collected) * np.array(n_ops)))
        nn_times.append(np.sum(np.array(nn_mlmc_n_collected) * np.array(n_ops_predict)))
        nn_times_levels.append(np.array(nn_mlmc_n_collected) * np.array(n_ops_predict))

        mlmc_l_vars.append(orig_moments_mean.l_vars)
        nn_l_vars.append(predict_moments_mean.l_vars)

        mlmc_vars.append(orig_moments_mean.var)
        nn_vars.append(predict_moments_mean.var)

        mlmc_l_means.append(orig_moments_mean.l_means)
        nn_l_means.append(predict_moments_mean.l_means)

        mlmc_vars_mse.append((ref_moments_mean.var - orig_moments_mean.var) ** 2)
        nn_vars_mse.append((ref_moments_mean.var - predict_moments_mean.var) ** 2)

        mlmc_means_mse.append((ref_moments_mean.mean - orig_moments_mean.mean) ** 2)
        nn_means_mse.append((ref_moments_mean.mean) ** 2 - predict_moments_mean.mean)

        # print("np.min(len(ref_orth_moments.mean), len(orig_orth_moments.mean)) ", np.min(len(ref_orth_moments.mean), len(orig_orth_moments.mean)))
        # print("ref_orth_moments.mean[:np.min(len(ref_orth_moments.mean), len(orig_orth_moments.mean))] ", ref_orth_moments.mean[:np.min(len(ref_orth_moments.mean), len(orig_orth_moments.mean))])
        if ref_orth_moments is not None:
            orth_mlmc_means_mse.append(
                (ref_orth_moments.mean[:np.min((len(ref_orth_moments.mean), len(orig_orth_moments.mean)))] -
                 orig_orth_moments.mean[:np.min((len(ref_orth_moments.mean), len(orig_orth_moments.mean)))]) ** 2)
            orth_nn_means_mse.append(
                (ref_orth_moments.mean[:np.min((len(ref_orth_moments.mean), len(predict_orth_moments.mean)))]
                 - predict_orth_moments.mean[
                   :np.min((len(ref_orth_moments.mean), len(predict_orth_moments.mean)))]) ** 2)

        print("orig moments mean ", orig_moments_mean.mean)
        all_mlmc_moments_mean.append(orig_moments_mean.mean)
        all_nn_moments_mean.append(predict_moments_mean.mean)

        all_mlmc_moments_var.append(orig_moments_mean.var)
        all_nn_moments_var.append(predict_moments_mean.var)

        kl_mlmc_all.append(kl_mlmc)
        kl_nn_all.append(kl_nn)

    moments_plot = plots.MomentsPlots(log_var_y=True)

    print("all mlmc moments mean ", all_mlmc_moments_mean)
    print("np.mean(all_mlmc_moments_var, axis=0) ", np.mean(all_mlmc_moments_var, axis=0))
    moments_plot.add_moments((np.mean(all_mlmc_moments_mean, axis=0),
                              np.mean(all_mlmc_moments_var, axis=0)), label="mlmc moments")
    moments_plot.add_moments((np.mean(all_nn_moments_mean, axis=0),
                              np.mean(all_nn_moments_var, axis=0)), label="nn moments")
    moments_plot.add_moments((orig_moments_mean.mean,
                              orig_moments_mean.var), label="orig moments")
    moments_plot.show(None)

    display_vars(mlmc_vars, nn_vars, target_variance=target_variance)


    print("mlmc l vars list ", mlmc_l_vars)

    print("mlmc l vars ", np.mean(mlmc_l_vars, axis=0))
    print("nn l vars ", np.mean(nn_l_vars, axis=0))

    print("var mlmc l vars ", np.var(mlmc_l_vars, axis=0))
    print("var nn l vars ", np.var(nn_l_vars, axis=0))

    print("MAX mlmc l vars ", np.max(np.mean(mlmc_l_vars, axis=0), axis=1))
    print("MAX nn l vars ", np.max(np.mean(nn_l_vars, axis=0), axis=1))
    #
    print("mlmc means MSE ", np.mean(mlmc_means_mse, axis=0))
    print("nn means MSE ", np.mean(nn_means_mse, axis=0))

    print("mlmc times ", mlmc_times)
    print("nn times ", nn_times)

    print("mlmc times levels ", mlmc_times_levels)
    print("nn times levels ", nn_times_levels)

    print("n ops all ", n_ops_all)
    print("n ops predict all ", n_ops_predict_all)
    print("len(nn times) ", len(nn_times))

    mlmc_total_time = np.mean(mlmc_times)
    nn_total_time = np.mean(nn_times)

    print("#############################")
    print("mlmc total time ", mlmc_total_time)
    print("nn total time ", nn_total_time)
    print("#############################")
    print("KL mlmc ", np.mean(kl_mlmc_all))
    print("KL nn ", np.mean(kl_nn_all))

    print("MC: to 10: {}, above: {}".format(np.sum(np.mean(mlmc_means_mse, axis=0)[:10]),
                                            np.sum(np.mean(mlmc_means_mse, axis=0)[10:])))
    print("NN: to 10: {}, above: {}".format(np.sum(np.mean(nn_means_mse, axis=0)[:10]),
                                            np.sum(np.mean(nn_means_mse, axis=0)[10:])))

    n_ops_mlmc_mean = np.mean(n_ops_all, axis=0)
    n_ops_nn_mean = np.mean(n_ops_predict_all, axis=0)

    # print("n ops all ", n_ops_all)
    # print("n ops predict all ", n_ops_predict_all)
    #
    # print("n ops mlmc mean ", n_ops_mlmc_mean)
    # print("n ops nn mean ", n_ops_nn_mean)

    mlmc_n_collected = np.mean(mlmc_n_collected_all, axis=0)
    nn_n_collected = np.mean(nn_n_collected_all, axis=0)

    # print("mlmc n collected ", mlmc_n_collected_all)
    # print("nn n collected all ", nn_n_collected_all)
    # print("mlmc n collected ", mlmc_n_collected)
    # print("nn n collected ", nn_n_collected)

    plt_var = plots.VarianceNN()
    plt_var.set_n_ops(np.mean(n_ops_predict_all, axis=0))
    l_vars = np.mean(mlmc_l_vars, axis=0)
    # print("np.squeeze(orig_level_params) ", orig_level_params)
    # print("l vars ", l_vars)
    # print("np.squeeze(orig_level_params) shape", orig_level_params.shape)
    # print("l vars shape", l_vars.shape)
    print("orig level params ", orig_level_params)
    # plt_var.add_level_variances(np.squeeze(orig_level_params), l_vars)
    plt_var.add_level_variances(orig_level_params, l_vars)

    # plt_var.show(None)
    # plt_var.show("mlmc_vars")
    #
    # plt_var = plot.Variance()
    l_vars = np.mean(nn_l_vars, axis=0)
    # print("nn l vars  ", l_vars)
    # print("nn level parsm ", nn_level_params)
    level_params = np.squeeze(nn_level_params)
    level_params[0] *= 2
    plt_var.add_level_variances_nn(level_params, l_vars)
    plt_var.show("nn_vars")
    plt_var.show(None)

    plot_sse(nn_vars_mse, mlmc_vars_mse, title="moments_var")
    plot_sse(nn_means_mse, mlmc_means_mse, title="moments_mean")
    plot_sse(mlmc_means_mse, mlmc_means_mse, title="mlmc moments_mean")

    # if ref_orth_moments is not None:
    #     print("orth nn means mse ", orth_nn_means_mse)
    #     print("orth mlmc means mse ", orth_mlmc_means_mse)
    #     plot_sse(orth_nn_means_mse, orth_mlmc_means_mse, title="orthogonal moments_mean")

    data_dict["test_targets"] = np.array(data_dict["test_targets"])
    data_dict["test_predictions"] = np.array(data_dict["test_predictions"])
    data_dict["train_targets"] = np.array(data_dict["train_targets"])
    data_dict["train_predictions"] = np.array(data_dict["train_predictions"])
    data_dict["val_targets"] = np.array(data_dict["val_targets"])
    data_dict["val_predictions"] = np.array(data_dict["val_predictions"])

    print("data dict train predictions ", data_dict["train_predictions"])

    # if data_dict["log"][0]:
    #     # print("np.exp(10)", np.exp(10))
    #     print("test targets ", data_dict["test_targets"])
    #     print("type test targets ", type(data_dict["test_targets"]))
    #
    #     data_dict["test_targets"], data_dict["test_predictions"] = exp_values(data_dict["test_targets"], data_dict["test_predictions"])
    #     data_dict["train_targets"], data_dict["train_predictions"] = exp_values(data_dict["train_targets"], data_dict["train_predictions"])
    #
    #     # print("test targets ", data_dict["test_targets"])
    # print("test predictions ", data_dict["test_predictions"])
    #
    # print("orig max vars ", data_dict["orig_max_vars"])
    # print("predict max vars ", data_dict["predict_max_vars"])

    # mean_orig_vars = np.mean(data_dict["orig_max_vars"], axis=0)
    # mean_predict_vars = np.mean(data_dict["predict_max_vars"], axis=0)
    total_steps = np.mean(data_dict["total_steps"])

    # print("mean orig vars ", mean_orig_vars)
    # print("mean predict vars ", mean_predict_vars)
    print("total steps ", total_steps)

    # print("test targets ",  data_dict["test_targets"])
    # print("test predictions ", data_dict["test_predictions"])
    # print("test diff ", data_dict["test_predictions"] - data_dict["test_targets"])
    # print("test diff squared ", (data_dict["test_predictions"] - data_dict["test_targets"])**2)

    # print("(test_predictions - test_targets)**2 ", (data_dict["test_predictions"] - data_dict["test_targets"])**2)

    # print("test targets shape ", data_dict["test_targets"].shape)

    test_MSE = np.mean((data_dict["test_predictions"] - data_dict["test_targets"]) ** 2, axis=1)
    print("val predictions ", data_dict["val_predictions"])
    print("val targets ", data_dict["val_targets"])
    #val_MSE = np.mean((data_dict["val_predictions"] - data_dict["val_targets"]) ** 2, axis=1)
    test_RMSE = np.sqrt(test_MSE)

    test_MAE = np.mean(np.abs(data_dict["test_predictions"] - data_dict["test_targets"]), axis=1)

    #print("val MSE ", val_MSE)
    print("test MSE ", test_MSE)

    all_test_RSE = []
    for index, t_targets in enumerate(data_dict["test_targets"]):
        if test_MSE[index] > limit:
            continue
        mean_t = np.mean(t_targets)
        RSE = np.sum((data_dict["test_predictions"][index] - t_targets) ** 2) / np.sum((t_targets - mean_t) ** 2)
        all_test_RSE.append(RSE)

    all_train_RSE = []
    try:
        for index, t_targets in enumerate(data_dict["train_targets"]):
            if test_MSE[index] > limit:
                continue
            mean_t = np.mean(t_targets)
            # print("train predictions index ", data_dict["train_predictions"][index])
            RSE = np.sum((data_dict["train_predictions"][index] - t_targets) ** 2) / np.sum((t_targets - mean_t) ** 2)
            all_train_RSE.append(RSE)
    except:
        pass

    # print("all test RSE ", np.mean(all_test_RSE))

    # Relative squared error
    test_RSE = np.sum((data_dict["test_predictions"] - data_dict["test_targets"]) ** 2) / \
               np.sum((data_dict["test_targets"] - np.mean(data_dict["test_targets"])) ** 2)

    print("test RSE ", test_RSE)

    test_RAE = np.sqrt(np.sum((data_dict["test_predictions"] - data_dict["test_targets"]) ** 2)) / \
               np.sqrt(np.sum((data_dict["test_targets"]) ** 2))

    print("test MSE / mean targets", np.mean(test_MSE) / np.mean(data_dict["test_targets"]))

    print("test RSE ", test_RSE)
    print("test RAE ", test_RAE)
    print("test_MSE ", test_MSE)

    t_mse_sum = []
    for t_mse in test_MSE:
        # Note: je mozne odstranit vetsi hodnoty MSE pro L4, protoze by slo dosahnout mensich hodnot pokud by se navysil pocet iteraci nebo by se vysledek pro nejlepsi train + val MSE a ne posledni vysledek
        if t_mse > limit:  # 0.009:
            continue
        t_mse_sum.append(t_mse)

    print("t mse ", t_mse_sum)
    print("LEN t mse ", len(t_mse_sum))
    print("T MSE sum ", np.mean(t_mse_sum))

    print("train_predictions ", np.array(data_dict["train_predictions"]).shape)
    print("train_targets ", data_dict["train_targets"])

    data_dict["train_predictions"], data_dict["train_targets"] = remove_empty(data_dict["train_predictions"],
                                                                              data_dict["train_targets"])

    print("remove empty train targets ", data_dict["train_targets"])

    # data_dict["train_predictions"] = np.squeeze(data_dict["train_predictions"])

    print("train_predictions - train_targets ", data_dict["train_predictions"] - data_dict["train_targets"])

    train_MSE = np.mean((data_dict["train_predictions"] - data_dict["train_targets"]) ** 2, axis=1)
    train_RMSE = np.sqrt(train_MSE)
    train_MAE = np.mean(np.abs(data_dict["train_predictions"] - data_dict["train_targets"]), axis=1)
    learning_times = data_dict["learning_times"]

    # Relative squared error
    train_RSE = np.sum((data_dict["train_predictions"] - data_dict["train_targets"]) ** 2) / \
                np.sum((data_dict["train_targets"] - np.mean(data_dict["train_targets"])) ** 2)

    # Relative absolute error
    train_RAE = np.sqrt(np.sum((data_dict["train_predictions"] - data_dict["train_targets"]) ** 2)) / \
                np.sqrt(np.sum((data_dict["train_targets"]) ** 2))

    print("train RSE ", train_RSE)
    print("train REA ", train_RAE)

    # plot_data(test_MSE, label="test MSE")
    # plot_data(test_MAE, label="test MAE")

    print("NN moments MSE sum ", np.sum(np.mean(nn_means_mse, axis=0)))

    print("mean test MSE ", np.mean(test_MSE))
    # print("mean test RSE ", np.mean(test_RSE))
    # print("mean test RMSE ", np.mean(test_RMSE))
    # print("mean test MAE ", np.mean(test_MAE))
    print("max test MSE ", np.max(test_MSE))
    # print("max test RMSE ", np.max(test_RMSE))
    # print("max test MAE ", np.max(test_MAE))

    print("train_MSE ", train_MSE)

    print("mean train MSE ", np.mean(train_MSE))

    print("test RSE ", np.mean(all_test_RSE))
    print("test RSE ", np.mean(all_train_RSE))
    # print("mean train RSE ", np.mean(train_RSE))
    # print("mean train RMSE ", np.mean(train_RMSE))
    # print("mean train MAE ", np.mean(train_MAE))
    print("max train MSE ", np.max(train_MSE))
    # print("max train RMSE ", np.max(train_RMSE))
    # print("max train MAE ", np.max(train_MAE))

    print("learning time ", learning_times)
    print("mean learning time ", np.mean(learning_times))
    print("max learning time ", np.max(learning_times))

    test_MSE = remove_outliers(test_MSE, limit)
    train_MSE = remove_outliers(train_MSE, limit)
    print("############# OUTPUT ################")
    print("len(train MSE) ", len(train_MSE))
    print("train MSE ", np.mean(train_MSE))
    # print("train MSE sqrt var", np.sqrt(np.var(train_MSE)))
    # print("train MSE std", np.std(train_MSE))

    # output_mult_factor = 1437603411
    # print("orig train MSE ", train_MSE)
    # train_MSE = np.array(train_MSE) * output_mult_factor
    # print("train MSE ", train_MSE)
    # test_MSE = np.array(test_MSE) * output_mult_factor

    print("train MSE ", train_MSE)
    print("stats.sem(train_MSE) ", stats.sem(train_MSE))
    print("test MSE ", np.mean(test_MSE))
    print("test MSE ", test_MSE)
    print("stats.sem(test_MSE) ", stats.sem(test_MSE))
    print("train MSE: {}, bias: {}, variance: {}".format(np.mean(train_MSE_list), np.mean(train_bias), np.mean(train_variance)))
    print("test MSE: {}, bias: {}, variance: {}".format(np.mean(test_MSE_list), np.mean(test_bias),
                                                         np.mean(test_variance)))
    # print("test MSE std", np.sqrt(np.var(test_MSE)))
    print("train RSE ", np.mean(all_train_RSE))
    print("test RSE ", np.mean(all_test_RSE))

    print("nn total time ", nn_total_time)
    print("mlmc total time ", mlmc_total_time)

    print("KL mlmc ", np.mean(kl_mlmc_all))
    print("KL nn ", np.mean(kl_nn_all))

    print("mean learning time ", np.mean(learning_times))
    print("max learning time ", np.max(learning_times))

    print("######################################")
    return train_MSE, test_MSE, all_train_RSE, all_test_RSE, nn_total_time, mlmc_total_time, kl_mlmc_all, kl_nn_all,\
           learning_times


def plot_sse(data_nn, data_mlmc, x_label="ith moment", y_label="MSE", title=""):
    import matplotlib
    matplotlib.rcParams.update({'font.size': 16})
    matplotlib.rcParams.update({'lines.markersize': 8})
    #fig, axes = plt.subplots(1, 1, figsize=(22, 10))
    fig, axes = plt.subplots(1, 1, figsize=(8, 5))
    data = np.array(data_nn)
    x = range(data.shape[1])
    axes.set_title(title)
    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)
    axes.errorbar(x, np.mean(data_mlmc, axis=0), yerr=np.sqrt(np.var(data_mlmc, axis=0)), fmt='o', label="MC",
                  color="blue")
    axes.errorbar(x, np.mean(data_nn, axis=0), yerr=np.sqrt(np.var(data_nn, axis=0)), fmt='o',
                  label="MLMC with meta-model", color="red")
    fig.legend()
    fig.savefig("{}.pdf".format(title))
    fig.show()


def plot_sse_scatter(data_nn, data_mlmc, x_label="ith moment", y_label="MSE", title=""):
    import matplotlib
    #matplotlib.rcParams.update({'font.size': 38})
    matplotlib.rcParams.update({'lines.markersize': 14})
    fig, axes = plt.subplots(1, 1, figsize=(22, 10))
    data = np.array(data_nn)

    print("data.shape ", data.shape)
    x = range(len(data[0]))
    axes.set_title(title)
    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)
    axes.scatter(x, np.mean(data_nn, axis=0), label="NN MLMC", color="red")
    axes.scatter(x, np.mean(data_mlmc, axis=0), label="MLMC", color="blue")
    fig.legend()
    fig.savefig("{}.pdf".format(title))
    fig.show()


def display_vars(mlmc_vars, nn_vars, target_variance, title=""):
    mlmc_mean_vars = np.mean(mlmc_vars, axis=0)
    nn_mean_vars = np.mean(nn_vars, axis=0)

    import matplotlib
    # matplotlib.rcParams.update({'font.size': 38})
    matplotlib.rcParams.update({'lines.markersize': 14})
    fig, axes = plt.subplots(1, 1, figsize=(22, 10))

    x = range(0, len(mlmc_vars[0]))
    axes.set_xlabel("moments")
    axes.set_ylabel("var")
    axes.set_yscale("log")

    axes.axhline(y=target_variance, label="target var: {}".format(target_variance))

    print("mlmc error bar ", np.sqrt(np.var(mlmc_vars, axis=0)))
    axes.errorbar(x, np.mean(mlmc_vars, axis=0), yerr=np.sqrt(np.var(mlmc_vars, axis=0)), fmt='o', label="MLMC vars",
                  color="blue")

    print("mlmc vars ", mlmc_vars)
    print("nn vars ", nn_vars)
    print("nn error bar ", np.sqrt(np.var(nn_vars, axis=0)))
    axes.errorbar(x, np.mean(nn_vars, axis=0), yerr=np.sqrt(np.var(nn_vars, axis=0)), fmt='o', label="NN vars",
                  color="red")

    fig.legend()
    fig.savefig("{}.pdf".format(title))
    fig.show()


def set_model_weights(new_model, old_model):
    for new_conv_layer, old_conv_layer in zip(new_model._conv_layers, old_model._conv_layers):
        new_conv_layer.kernel = old_conv_layer.kernel
        new_conv_layer.bias = old_conv_layer.bias

    # print("input data shape ", input_data.dataset[0].shape)
    #
    # print(old_conv_layer.kernel.numpy().shape)
    # input_imgs = Input(shape=(None, 108, 1))
    # print("old_model.flatten.weights", old_model.flatten().weights)

    for new_dense_layer, old_dense_layer in zip(new_model._dense_layers, old_model._dense_layers):

        # print("old_dense_layer.get_weights() shape ", old_dense_layer.get_weights()[0].shape)
        # print("old_dense_layer.get_weights() shape ", old_dense_layer.get_weights()[1].shape)
        # input_imgs = Input(shape=(None, 108, 1))
        # new_dense_layer(input_imgs)
        # # model = Model(inputs=input_imgs, outputs=encoded)
        # # dense_layer.set_weights(weights)
        #
        # print("new dense layer weights ", new_dense_layer.weights)
        new_dense_layer.set_weights(old_dense_layer.get_weights())

    # print("old_dense_layer.get_weights() ", old_dense_layer.get_weights())
    # print("new_model._dense_layers[-1].weights ", new_model._dense_layers[-1].weights)
    # exit()


def set_model_layers(new_model, old_model):
    for new_conv_layer, old_conv_layer in zip(new_model._conv_layers, old_model._conv_layers):
        new_conv_layer.kernel = old_conv_layer.kernel
        new_conv_layer.bias = old_conv_layer.bias

        # print("old conv layer get config ", old_conv_layer.get_config())
        # print("new conv layer get config ", new_conv_layer.get_config())

    #exit()

    # print("input data shape ", input_data.dataset[0].shape)
    #
    # print(old_conv_layer.kernel.numpy().shape)
    # input_imgs = Input(shape=(None, 108, 1))
    # print("old_model.flatten.weights", old_model.flatten().weights)

    for new_dense_layer, old_dense_layer in zip(new_model._dense_layers, old_model._dense_layers):
        # config = layer.get_config()
        # weights = layer.get_weights()
        # cloned_layer = type(layer).from_config(config)
        # cloned_layer.build(layer.input_shape)
        # cloned_layer.set_weights(weights)

        # print("old_dense_layer.get_weights() shape ", old_dense_layer.get_weights()[0].shape)
        # print("old_dense_layer.get_weights() shape ", old_dense_layer.get_weights()[1].shape)
        # input_imgs = Input(shape=(None, 108, 1))
        # new_dense_layer(input_imgs)
        # # model = Model(inputs=input_imgs, outputs=encoded)
        # # dense_layer.set_weights(weights)
        #
        # print("new dense layer weights ", new_dense_layer.weights)
        #new_dense_layer.set_weights(old_dense_layer.get_weights())

        new_dense_layer.set_weights(old_dense_layer.get_weights())

    # print("new_model._dense_layers[0].get_config() ", new_model._dense_layers[0].get_config())
    # print("old_model._dense_layers[0].get_config() ", old_model._dense_layers[0].get_config())
    # print("new_model._dense_layers[0].get_weights() ", new_model._dense_layers[0].get_weights())
    # print("old_model._dense_layers[0].get_weights() ", old_model._dense_layers[0].get_weights())


    # print("old_dense_layer.get_weights() ", old_dense_layer.get_weights())
    # print("new_model._dense_layers[-1].weights ", new_model._dense_layers[-1].weights)
    # exit()


def run_GNN(config, stats=True, train=True, log=False, seed=0):
    print("seed ", seed)

    loss = MeanSquaredError()  # var_loss_function#
    accuracy_func = MSE_moments
    # loss = MeanAbsoluteError()
    # loss = MeanSquaredLogarithmicError()
    #loss = KLDivergence()
    # loss = total_loss_function
    optimizer = tf.optimizers.Adam(learning_rate=config['learning_rate'])
    batch_size = config['batch_size']#2000
    epochs = config['epochs']#1000
    hidden_regularization = None  # l2(2e-10)

    graph_creation_time = config['graph_creation_time']
    if graph_creation_time == 0:
        graph_creator_preproces_time = time.process_time()

        graph_creator(config['output_dir'], config['hdf_path'], config['mesh'], level=config['level'],
                      feature_names=config.get('feature_names', [['conductivity']]))
        graph_creation_time = time.process_time() - graph_creator_preproces_time
        print("graph creation time ", graph_creation_time)
        exit()

    preprocess_start_time = time.process_time()
    # Load data
    data = FlowDataset(output_dir=config['output_dir'], level=config['level'], log=log, config=config, index=seed)
    data = data#[:10000]

    # Dataset preprocess config
    config['dataset_config'] = data._dataset_config

    # print("n node features ", data.graphs[0].n_node_features)
    # print("graph x", data.graphs[0].x)
    # print("graphs[0] ", repr(data.graphs[0]))
    # exit()

    #print("len data ", len(data))
    #data.shuffle(seed=seed)
    preprocess_time = time.process_time() - preprocess_start_time
    #print("preproces time ", preprocess_time)
    preprocess_time = preprocess_time + graph_creation_time
    #print("total preprocess time ", preprocess_time)

    learning_time_start = time.process_time()
    data.a = config['conv_layer'].preprocess(data.a)
    data.a = sp_matrix_to_sp_tensor(data.a)
    #train_data_len = int(len(data) * 0.8)
    train_data_len = config['n_train_samples']
    # Train/valid/test split
    #print("train data len ", train_data_len)

    if not train:
        data_tr = data
        data_te = data
    else:
        #data_tr = data[seed*train_data_len: seed*train_data_len + train_data_len]
        data_tr = data.get_train_data(seed, train_data_len)
        #print("data tr ", data_tr)
        data_te = data.get_test_data(seed, train_data_len)
    #data_tr, data_te = data[:train_data_len], data[train_data_len:]

    gnn = config['gnn'](**config['model_config'])

    # if hasattr(gnn._loss,'__name__') and gnn._loss.__name__ == "MSE_moments":
    #     tr_output = [g.y for g in data_tr]
    #     n_moments = 3
    #     quantile = 0.001
    #     domain = np.percentile(tr_output, [100 * quantile, 100 * (1 - quantile)])
    #     moments_fn = Legendre_tf(n_moments, domain)
    #     #accuracy_func = MSE_moments(moments_fn=moments_fn)
    #     gnn._loss = MSE_moments(moments_fn=moments_fn)

    #np.random.shuffle(data_tr)
    val_data_len = int(len(data_tr) * config['val_samples_ratio'])
    #print("val data len ", val_data_len)
    #data_tr, data_va = data_tr.split_val_train(val_data_len)
    data_tr, data_va = data_tr[:-val_data_len], data_tr[-val_data_len:]

    # print("data tr ", data_tr)
    # print("data va ", data_va)

    # print("data_tr len ", len(data_tr))
    # print("data_va len ", len(data_va))
    # print("data_te len ", len(data_te))

    # We use a MixedLoader since the dataset is in mixed mode
    loader_tr = MixedLoader(data_tr, batch_size=batch_size, epochs=epochs)
    loader_va = MixedLoader(data_va, batch_size=batch_size)
    loader_te = MixedLoader(data_te, batch_size=batch_size)

    if not train:
        gnn.fit(MixedLoader(data_tr[:10], batch_size=batch_size, epochs=epochs),
                MixedLoader(data_tr[10:20], batch_size=batch_size), MixedLoader(data_tr[20:30], batch_size=batch_size))
        set_model_weights(gnn._model, config["set_model"])

        #set_model_layers(gnn._model, config["set_model"])

        #gnn._model = config["set_model"]
        #compare_models(gnn._model, config["set_model"], config)

    #
    if gnn is None:
        gnn = GNN(loss=loss, optimizer=optimizer, conv_layer=config['conv_layer'], output_activation=abs_activation,
                  hidden_activation='relu', patience=150, hidden_reqularizer=hidden_regularization,
                  model=config['model'], accuracy_func=accuracy_func)  # tanh takes to much time
        # ideally patience = 150
        # batch_size 500, ideally 500 epochs, patience 35

    if train:
        print("gnn ", gnn)
        # gnn.run_eagerly = True
        train_targets = gnn.fit(loader_tr, loader_va, loader_te)

    learning_time = time.process_time() - learning_time_start

    states = gnn._states
    # print("states ", states)
    # for state in states.values():
    #     print("state._model", state._model)

    if len(states) > 0:
        min_key = np.min(list(states.keys()))
        gnn = states[min_key]

    # print("gnn._model.layers[min].get_weights() ", states[np.min(list(states.keys()))]._model.layers[0].get_weights())
    # print("gnn._model.layers[max].get_weights() ", states[np.max(list(states.keys()))]._model.layers[0].get_weights())

    train_targets, train_predictions = gnn.predict(MixedLoader(data_tr, batch_size=batch_size, epochs=1))
    train_predictions = np.squeeze(train_predictions)

    val_targets, val_predictions = gnn.predict(loader_va)
    val_predictions = np.squeeze(val_predictions)

    #val_targets = gnn.val_targets
    total_steps = gnn._total_n_steps

    targets, predictions = gnn.predict(loader_te)
    predictions = np.squeeze(predictions)

    #print("learning time ", learning_time)

    targets = np.array(targets)
    predictions = np.array(predictions)

    #print("MSE ", np.mean((predictions-targets)**2))

    if log:
        targets = np.exp(targets)
        predictions = np.exp(predictions)

    if not stats:
        analyze_results(targets, predictions)
        plot_loss(gnn._train_loss, gnn._val_loss)
        analyze_results(targets, predictions)

        import matplotlib.pyplot as plt

        plt.hist(targets, bins=50, alpha=0.5, label='target', density=True)
        plt.hist(predictions, bins=50, alpha=0.5, label='predictions', density=True)

        # plt.hist(targets - predictions, bins=50, alpha=0.5, label='predictions', density=True)
        plt.legend(loc='upper right')
        # plt.xlim(-0.5, 1000)
        plt.yscale('log')
        plt.show()

    #predict_l_0_start_time = time.process_time()
    l_0_targets, l_0_predictions, predict_l_0_time = predict_level_zero(gnn, config['l_0_output_dir'],
                                                                        config['l_0_hdf_path'], config['mesh'],
                                                                        config['conv_layer'], batch_size, log,
                                                                        stats=stats,
                                                                        corr_field_config=config['corr_field_config'],
                                                                        seed=seed,
                                                                        feature_names=config.get('feature_names', [['conductivity']]),
                                                                        config=config
                                                                        )
    #predict_l_0_time = time.process_time() - predict_l_0_start_time

    if stats:
        l1_sample_time = preprocess_time / len(data) + learning_time / len(data)
        l0_sample_time = predict_l_0_time / len(l_0_targets)

        # print("targets ", targets)
        # print("predictions ", predictions)

        # orig_max_vars, predict_max_vars = process_mlmc(hdf_path, sampling_info_path, ref_mlmc_file, targets, predictions, train_targets,
        #              train_predictions,
        #              val_targets, l_0_targets,
        #              l_0_predictions, l1_sample_time, l0_sample_time, nn_level=level, replace_level=replace_level,
        #                                                stats=stats)

        return gnn, targets, predictions, learning_time, train_targets, train_predictions,\
               val_targets, val_predictions, l_0_targets, l_0_predictions, l1_sample_time, l0_sample_time, total_steps

    save_times(config['save_path'], False, (preprocess_time, len(data)), learning_time, (predict_l_0_time, len(l_0_targets)))
    save_load_data(config['save_path'], False, targets, predictions, train_targets, train_predictions, val_targets, l_0_targets,
                   l_0_predictions)


def predict_level_zero(nn, output_dir, hdf_path, mesh, conv_layer, batch_size=1000, log=False, stats=False,
                       corr_field_config=None, seed=1234, feature_names=[], config=None):
    #graph_creator(output_dir, hdf_path, mesh, level=0, feature_names=feature_names)
    # Load data
    sample_time = 0
    if corr_field_config:
        sample_time = corr_field_sample_time(mesh, corr_field_config)
    else:
        raise Exception("No corr field config passed")

    data = FlowDataset(output_dir=output_dir, log=log, config=config, predict=True)#, mesh=mesh, corr_field_config=corr_field_config)
    #data = data  # [:10000]
    data.shuffle(seed=seed)
    
    # print("output_dir ", output_dir)
    # print("len(data) ", len(data))
    # print("data[0] ", data[0])
    
    predict_time_start = time.process_time()
    data.a = conv_layer.preprocess(data.a)
    #data.a = sp_matrix_to_sp_tensor(data.a)

    loader_te = MixedLoader(data, batch_size=batch_size)

    targets, predictions = nn.predict(loader_te)
    predictions = np.squeeze(predictions)

    if not stats:
        analyze_results(targets, predictions)

    if log:
        targets = np.exp(targets)
        predictions = np.exp(predictions)
    if not stats:
        analyze_results(targets, predictions)

    predict_time = time.process_time() - predict_time_start

    return targets, predictions, predict_time + sample_time * len(data)


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
