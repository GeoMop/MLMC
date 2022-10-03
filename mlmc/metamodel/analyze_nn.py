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
from mlmc.metamodel.image_flow_dataset_tf import ImageFlowDataset
#from mlmc.metamodel.image_flow_dataset_pytorch import ImageFlowDataset
from mlmc.metamodel.create_graph import graph_creator, image_creator
from mlmc.moments import Legendre_tf, Monomial
from mlmc.metamodel.random_field_time import corr_field_sample_time
from mlmc.plot import plots
import matplotlib.pyplot as plt
# Make numpy printouts easier to read.
from scipy import stats
# np.set_printoptions(precision=9, suppress=True)
import tensorflow as tf
#from torch.utils.data import DataLoader

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
from spektral.data import MixedLoader, BatchLoader
from spektral.utils.sparse import sp_matrix_to_sp_tensor
from sklearn.metrics import r2_score
import tensorflow_datasets as tfds

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


def run_CNN(config, stats=True, train=True, log=False, index=0):
    print("seed ", index)

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
        image_creator_preproces_time = time.process_time()

        image_creator(config['output_dir'], config['hdf_path'], config['mesh'], level=config['level'],
                      feature_names=config.get('feature_names', [['conductivity']]))
        graph_creation_time = time.process_time() - image_creator_preproces_time
        print("image creation time ", graph_creation_time)
        exit()

    preprocess_start_time = time.process_time()

    independent_samples = config.get("independent_samples", False)

    if independent_samples and train:

        dataset = ImageFlowDataset(data_dir=config['output_dir'], config=config, independent_samples=config["independent_samples"])

        data_tr = dataset.get_train_data(index=index, length=config['n_train_samples'])
        data_te = dataset.get_test_data(index=index, length=config['n_test_samples'])
        # data = ImageFlowDataset(output_dir=config['output_dir'], level=config['level'], log=log, config=config,
        #                    index=index, n_test_samples=100000)
        #
        # len_all_samples = len(data)
        #
        # last_train_sample = index * config['n_train_samples'] + config['n_train_samples']
        # last_test_sample = len_all_samples - (index * config['n_train_samples'] + config['n_train_samples'])
        #
        # print("last train sample ", last_train_sample)
        # print("last test sample ", last_test_sample)
        #
        # if last_train_sample > last_test_sample:
        #     return
        #
        # data_tr = ImageFlowDataset(output_dir=config['output_dir'], level=config['level'], log=log, config=config,
        #                       index=index, train_samples=True, independent_sample=True)
        #
        # print("len data tr ", len(data_tr))
        #
        # data_te = ImageFlowDataset(output_dir=config['output_dir'], level=config['level'], log=log, config=config,
        #                       index=index, predict=True, test_samples=True, independent_samples=True)
        #
        # print("len data te ", len(data_te))

    else:
        if train:
            dataset = ImageFlowDataset(data_dir=config['output_dir'], config=config)

            data_tr = dataset.get_train_data(index=index, length=config['n_train_samples'])
            data_te = dataset.get_test_data(index=index, length=config['n_train_samples'])
        else:

            # data = FlowDataset(output_dir=config['output_dir'], level=config['level'], log=log, config=config,
            #                    index=index)
            # print("len data")
            # data_tr = data
            # data_te = data
            config["n_train_samples"] = 20000

            dataset = ImageFlowDataset(data_dir=config['output_dir'], config=config)

            data_tr = dataset.get_train_data(index=index, length=config['n_train_samples'])
            data_te = dataset.get_test_data(index=index, length=config['n_test_samples'])

    # Dataset preprocess config
    config['dataset_config'] = data_tr.dataset_config()

    preprocess_time = time.process_time() - preprocess_start_time
    preprocess_time = preprocess_time + graph_creation_time
    print("preprocess time ", preprocess_time)

    data_te_predict = None
    if "predict_dir" in config and config["predict_dir"] is not None:
        # data_te_predict = ImageFlowDataset(output_dir=config['predict_dir'], level=config['level'], log=log, config=config,
        #                               index=index, n_test_samples=50000)

        data_te_predict = ImageFlowDataset(output_dir=config['predict_dir'], config=config, predict=True)

        # data_te_predict.a = config['conv_layer'].preprocess(data_te_predict.a)
        # data_te_predict.a = sp_matrix_to_sp_tensor(data_te_predict.a)

    learning_time_start = time.time()

    val_data_len = int(len(data_tr) * config['val_samples_ratio'])
    data_tr, data_va = data_tr[:-val_data_len], data_tr[-val_data_len:]

    features, target = data_tr[0]
    # print("features ", features.shape)
    # features = features[..., np.newaxis]
    # print("features.shape ", features.shape)
    # print("target ", target.shape)

    ot = (tf.float64, tf.float64)
    os = (tf.TensorShape([features.shape[0], features.shape[1], features.shape[2]]), tf.TensorShape([]))
    # ds = tf.data.Dataset.from_generator(dg,
    #                                     output_types=ot,
    #                                     output_shapes=os)

    ds_train = tf.data.Dataset.from_generator(data_tr._generate_examples, output_types=ot, output_shapes=os)# output_shapes=(512,512,1))
    ds_valid = tf.data.Dataset.from_generator(data_va._generate_examples,  output_types=ot, output_shapes=os)
    ds_test = tf.data.Dataset.from_generator(data_te._generate_examples,  output_types=ot, output_shapes=os)


    #print("ds_valid.shape ", ds_valid.output_shapes)

    #print("type ds train ", type(ds_train))

    # def count(stop):
    #     i = 0
    #     while i < stop:
    #         yield i
    #         i += 1
    #
    # for n in count(5):
    #     print(n)
    #
    # ds_counter = tf.data.Dataset.from_generator(count, args=[25], output_types=tf.int32, output_shapes=(), )
    #
    # for count_batch in ds_counter.batch(10):
    #     print(count_batch.numpy())
    #
    # exit()

    # for features, target in ds_train.batch(10):
    #     print("features ", features)
    #     print("target ", target)

    # exit()
    #
    #
    #
    print("len data tr ", len(data_tr))
    print("len data va ", len(data_va))
    print("len data te ", len(data_te))

    gnn = config['gnn'](**config['model_config'])

    # We use a MixedLoader since the dataset is in mixed mode
    loader_tr = ds_train#BatchLoader(data_tr, batch_size=batch_size, epochs=epochs)
    loader_va = ds_valid
    loader_te = ds_test

    # loader_tr = DataLoader(data_tr, batch_size=batch_size)
    # loader_va = DataLoader(data_va, batch_size=batch_size)
    # loader_te = DataLoader(data_te, batch_size=batch_size)

    if data_te_predict is not None:
        loader_te_predict = DataLoader(data_te_predict, batch_size=batch_size)

    if not train:
        gnn.fit(MixedLoader(data_tr[:10], batch_size=batch_size, epochs=epochs),
                MixedLoader(data_tr[10:20], batch_size=batch_size), MixedLoader(data_tr[20:30], batch_size=batch_size))
        set_model_weights(gnn._model, config["set_model"])

        # import visualkeras
        # model = gnn._model
        #
        # visualkeras.layered_view(model).show()  # display using your system viewer
        # visualkeras.layered_view(model, to_file='output.png')  # write to disk
        # visualkeras.layered_view(model, to_file='output.png').show()  # write and show

        #set_model_layers(gnn._model, config["set_model"])

        #gnn._model = config["set_model"]
        #compare_models(gnn._model, config["set_model"], config)

    #
    # if gnn is None:
    #     gnn = GNN(loss=loss, optimizer=optimizer, conv_layer=config['conv_layer'], output_activation=abs_activation,
    #               hidden_activation='relu', patience=150, hidden_reqularizer=hidden_regularization,
    #               model=config['model'], accuracy_func=accuracy_func)  # tanh takes to much time
        # ideally patience = 150
        # batch_size 500, ideally 500 epochs, patience 35

    if train:
        print("cnn ", gnn)
        # gnn.run_eagerly = True
        train_targets = gnn.fit(loader_tr, loader_va, loader_te, config)

    learning_time = time.time() - learning_time_start

    print("learning time ", learning_time)

    if train:
        states = gnn._states
        # for state in states.values():
        #     print("state._model", state._model)

        if len(states) > 0:
            min_key = np.min(list(states.keys()))
            gnn = states[min_key]

    # print("gnn._model.layers[min].get_weights() ", states[np.min(list(states.keys()))]._model.layers[0].get_weights())
    # print("gnn._model.layers[max].get_weights() ", states[np.max(list(states.keys()))]._model.layers[0].get_weights())

    train_targets, train_predictions = gnn.predict(ds_train)
    train_predictions = np.squeeze(train_predictions)

    val_targets, val_predictions = gnn.predict(loader_va)
    val_predictions = np.squeeze(val_predictions)

    #val_targets = gnn.val_targets
    total_steps = gnn._total_n_steps

    targets, predictions = gnn.predict(loader_te)
    predictions = np.squeeze(predictions)

    targets_to_est = []
    predictions_to_est = []

    if data_te_predict is not None:
        targets_to_est, predictions_to_est = gnn.predict(loader_te_predict)
        predictions_to_est = np.squeeze(predictions_to_est)
        targets_to_est = np.array(targets_to_est)
        predictions_to_est = np.array(predictions_to_est)

    #print("learning time ", learning_time)

    targets = np.array(targets)
    predictions = np.array(predictions)

    #print("MSE ", np.mean((predictions-targets)**2))

    # if log:
    #     targets = np.exp(targets)
    #     predictions = np.exp(predictions)
    #     target_to_est = np.exp(targets_to_est)
    #     predictions_to_est = np.exp(predictions_to_est)

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
    #l_0_targets, l_0_predictions, predict_l_0_time = [], [], []
    l_0_targets, l_0_predictions, predict_l_0_time = predict_level_zero_CNN(gnn, config['l_0_output_dir'],
                                                                            config['l_0_hdf_path'], config['mesh'],
                                                                            config['conv_layer'], batch_size, log,
                                                                            stats=stats,
                                                                            corr_field_config=config[
                                                                                'corr_field_config'],
                                                                            seed=index,
                                                                            feature_names=config.get('feature_names', [
                                                                                ['conductivity']]),
                                                                            config=config,
                                                                            mean_std_features=dataset.get_mean_std_features(),
                                                                            mean_std_target=dataset.get_mean_std_target()
                                                                            )
    #predict_l_0_time = time.process_time() - predict_l_0_start_time

    if stats:
        l1_sample_time = preprocess_time / (len(data_tr) + len(data_te)) + learning_time / (len(data_tr) + len(data_te))
        l0_sample_time = predict_l_0_time / len(l_0_targets)

        # print("targets ", targets)
        # print("predictions ", predictions)

        # orig_max_vars, predict_max_vars = process_mlmc(hdf_path, sampling_info_path, ref_mlmc_file, targets, predictions, train_targets,
        #              train_predictions,
        #              val_targets, l_0_targets,
        #              l_0_predictions, l1_sample_time, l0_sample_time, nn_level=level, replace_level=replace_level,
        #                                                stats=stats)

        return gnn, targets, predictions, learning_time, train_targets, train_predictions,\
               val_targets, val_predictions, l_0_targets, l_0_predictions, l1_sample_time, l0_sample_time, total_steps, targets_to_est, predictions_to_est

    save_times(config['save_path'], False, (preprocess_time, (len(data_tr) + len(data_te))), learning_time, (predict_l_0_time, len(l_0_targets)))
    save_load_data(config['save_path'], False, targets, predictions, train_targets, train_predictions, val_targets, l_0_targets,
                   l_0_predictions)


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
            val_targets, val_predictions, l_0_targets, l_0_predictions, l1_sample_time, l0_sample_time,\
            total_steps, targets_to_est, predictions_to_est = \
                mch_l_model(config, stats=True, train=config.get('train_model', True), log=log, index=i)

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
            model_data["targets_to_est"] = targets_to_est
            model_data["predictions_to_est"] = predictions_to_est

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
    models_data["targets_to_est"] = []
    models_data["predictions_to_est"] = []

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



        for conv_index, conv_layer in enumerate(model._conv_layers):
            print("x.shape ", x.shape)
            if conv_index not in conv_layers:
                conv_layers[conv_index] = [[], [], []]
            conv_layers[conv_index][0].extend(x)  # inputs
            print("conv_layer.kernel.numpy().shape", conv_layer.kernel.numpy().shape)
            conv_layers[conv_index][1].extend(conv_layer.kernel.numpy())  # weights (kernel)

            conv_out = conv_layer([x, a])

            print("conv out shape", conv_out.shape)
            conv_layers[conv_index][2].extend(conv_out)  # outputs

            x = np.array(conv_out)

            print("x " ,x)


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


def compute_beta_gamma(all_l_vars, cost_levels, n_collected):
    beta = {}
    gamma = {}
    beta_1 = {}
    gamma_1 = {}
    moments_m = [1, 5, 10, 15, 20, 25, 30]
    #n_collected = estimator._sample_storage.get_n_collected()
    # cost_levels = n_collected * np.array(new_n_ops)
    # cost_levels = np.array(new_n_ops)

    # print("cost levels ", cost_levels)
    # print("all l vars ", all_l_vars)
    #
    print("cost levels ", cost_levels)
    # print("all l vars ", all_l_vars)
    for l_id, l_vars in enumerate(all_l_vars, start=1):
        # print("l_vars ", l_vars)
        # print("l id ", l_id)

        if l_id not in beta:
            beta[l_id] = []
            beta_1[l_id] = []

        if l_id not in gamma:
            gamma[l_id] = []
            gamma_1[l_id] = []

        # print("l_id ", l_id)
        # print("len l vars ", len(l_vars))
        # print("moment ", moment)
        # print("beta ", beta)
        beta[l_id].append(-1 * np.log2(np.max(l_vars)) / l_id)

        # print("level l vars ", l_vars)
        # print("moments_mean.l_vars[l_id-2] ", moments_mean.l_vars[l_id-1])
        if l_id < len(cost_levels):
            beta_1[l_id].append(-1 * np.log2(np.max(all_l_vars[l_id]) / np.max(l_vars)))
        else:
            beta_1[l_id].append(0)

        gamma[l_id].append(np.log2(cost_levels[l_id - 1]) / l_id)

        if l_id < len(cost_levels):
            # print("cost levles ", cost_levels)
            # print("l id ", l_id)
            gamma_1[l_id].append(np.log2(cost_levels[l_id] / cost_levels[l_id - 1]))
        else:
            gamma_1[l_id].append(0)



    # for l_id, l_vars in enumerate(all_l_vars, start=1):
    #     print("l_vars ", l_vars)
    #     print("l id ", l_id)
    #     for moment in moments_m:
    #         if moment not in beta:
    #             beta[moment] = {}
    #             beta_1[moment] = {}
    #
    #         if l_id not in beta[moment]:
    #             beta[moment][l_id] = []
    #             beta_1[moment][l_id] = []
    #
    #         if l_id not in gamma:
    #             gamma[l_id] = []
    #             gamma_1[l_id] = []
    #
    #         # print("l_id ", l_id)
    #         # print("len l vars ", len(l_vars))
    #         # print("moment ", moment)
    #         # print("beta ", beta)
    #         beta[moment][l_id].append(-1 * np.log2(l_vars[moment]) / l_id)
    #
    #         # print("level l vars ", l_vars)
    #         # print("moments_mean.l_vars[l_id-2] ", moments_mean.l_vars[l_id-1])
    #         print("all_l_vars[l_id][moment] ", all_l_vars[l_id][moment])
    #         print("l_vars[moment] ", l_vars[moment])
    #         if l_id < len(cost_levels):
    #             beta_1[moment][l_id].append(-1 * np.log2(all_l_vars[l_id][moment] / l_vars[moment]))
    #
    #     gamma[l_id].append(np.log2(cost_levels[l_id - 1]) / l_id)
    #
    #     if l_id < len(cost_levels):
    #         # print("cost levles ", cost_levels)
    #         # print("l id ", l_id)
    #         gamma_1[l_id].append(np.log2(cost_levels[l_id] / cost_levels[l_id - 1]))

    print("beta ", beta)
    print("gamma ", gamma)

    print("beta 1 ", beta_1)
    print("gamma 1 ", gamma_1)
    #
    # print("n collected ", n_collected)
    # print("n ops ", cost_levels)
    #
    # print("cost ", n_collected * np.array(cost_levels))


    return beta.values(), beta_1.values(), gamma.values(), gamma_1.values()


def analyze_mlmc_properties(mlmc_l_vars, n_ops, mlmc_n_collected, nn_l_vars, n_ops_predict, nn_n_collected_all):
    mlmc_l_vars_mean = np.mean(mlmc_l_vars, axis=0)
    nn_l_vars_mean = np.mean(nn_l_vars, axis=0)

    mlmc_n_ops_mean = np.mean(n_ops, axis=0)
    nn_n_ops_mean = np.mean(n_ops_predict, axis=0)

    mlmc_n_collected_mean = np.mean(mlmc_n_collected, axis=0)
    nn_n_collected_mean = np.mean(nn_n_collected_all, axis=0)

    fig, axes = plt.subplots(1, 1, figsize=(8, 5))

    axes.set_xscale("log")
    axes.set_yscale("log")


    axes.plot(np.max(mlmc_l_vars_mean,axis=1), label="mlmc l vars")
    axes.plot(np.max(nn_l_vars_mean, axis=1), label="nn l vars")
    axes.plot(mlmc_n_ops_mean, label="mlmc n ops")
    axes.plot(nn_n_ops_mean, label="nn n ops")
    axes.plot(mlmc_n_collected_mean, label="mlmc_col")
    axes.plot(nn_n_collected_mean, label="nn_col")
    fig.legend()
    #fig.savefig("{}.pdf".format(title))
    fig.show()








def analyze_statistics(config, get_model=True):
    if not os.path.isdir(config['save_path']):
        print("dir not exists")
        exit()

    data_dict = load_statistics(config['save_path'])

    data_dict = process_data(data_dict)

    rescale_data = True

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
    mlmc_means_diff = []
    nn_means_diff = []

    mlmc_vars_mse_2 = []
    nn_vars_mse_2 = []
    mlmc_nn_vars_mse_2 = []
    mlmc_means_mse_2 = []
    nn_means_mse_2 = []
    mlmc_nn_means_mse_2 = []
    mlmc_means_diff_2 = []
    nn_means_diff_2 = []
    mlmc_nn_means_diff_2 = []

    mlmc_moments_mean = []
    mlmc_nn_moments_mean = []
    mlmc_moments_var = []
    mlmc_nn_moments_var = []

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
    train_RSE_list = []
    train_RMSE_list = []
    train_MAE_list = []
    train_relRMSE_list = []
    train_RMSEstd_list = []
    test_relRMSE_list = []
    train_reldiffRMSE_list = []
    test_reldiffRMSE_list = []
    train_iqrRMSE_list = []
    test_iqrRMSE_list = []
    test_RMSEstd_list = []

    all_train_samples = []
    all_test_samples = []

    all_beta = []
    all_beta_1 = []
    all_gamma = []
    all_gamma_1 = []

    nn_all_beta = []
    nn_all_beta_1 = []
    nn_all_gamma = []
    nn_all_gamma_1 = []

    test_RSE_list = []
    test_RMSE_list = []
    test_MAE_list = []

    MSE_to_est = []
    RMSE_to_est = []
    norm_RMSE_to_est = []

    limit = 5  # 0.008#0.01#0.0009
    limit = 10**10
    #limit = 0.7

    for i in range(len(data_dict["test_targets"])):
        # print("index i ", i)
        # if i == 4:
        #     continue

        # if i == 1:
        #     continue

        #print("index ", i)

        # if i not in [0, 1,2,3, 4, 7,8, 9, 10, 11, 12]:  L2
        #     continue

        # if i not in [0, 1,2,3, 4,5,6, 8, 9, 10,11, 12]:
        #     continue

        # if i not in [1, 2, 3, 4, 5, 6]:
        #     continue

        # if i in [7, 13, 14]:
        #     continue
        #
        # if i not in [0]:
        #     continue

        predictions = data_dict["test_predictions"][i]
        targets = data_dict["test_targets"][i]
        train_predictions = data_dict["train_predictions"][i]
        train_targets = data_dict["train_targets"][i]
        if "targets_to_est" in data_dict and len(data_dict["targets_to_est"]) > 0:
            targets_to_est = data_dict["targets_to_est"][i]
            predictions_to_est = data_dict["predictions_to_est"][i]
        else:
            targets_to_est = []
            predictions_to_est = []
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
            #plot_loss(model_train_loss, model_val_loss, model_train_acc)
            #plot_learning_rate(model_learning_rates)
            #print("model learning rates ", model_learning_rates)

            print("model ", model)
            print("dir(model.optimizer) ", dir(model.optimizer))
            #print("model weights ", model.weights)
            print("model.optimizer", model.optimizer)
            # print("model.optimizer", K.eval(model.optimizer.lr))
            # exit()

        if rescale_data:
            if "dataset_config" in data_dict:

                dataset_config = data_dict["dataset_config"][i]

                print("dataset config ", dataset_config)

                if dataset_config.get('output_normalization', False):
                    min_out = dataset_config.get('min_output')
                    max_out = dataset_config.get('max_output')

                    targets = targets * (max_out - min_out) + min_out
                    predictions = predictions * (max_out - min_out) + min_out

                    l_0_targets = l_0_targets * (max_out - min_out) + min_out
                    l_0_predictions = l_0_predictions * (max_out - min_out) + min_out

                    train_targets = train_targets * (max_out - min_out) + min_out
                    train_predictions = train_predictions * (max_out - min_out) + min_out

                    if len(targets_to_est) > 0:
                        targets_to_est = targets_to_est * (max_out - min_out) + min_out
                        predictions_to_est = predictions_to_est * (max_out - min_out) + min_out

                if dataset_config.get('output_scale', False):
                    # mean_targets = np.mean(targets)
                    # var_targets = np.var(targets)

                    mean_targets = dataset_config.get('mean_output', False)
                    var_targets = dataset_config.get('var_output', False)

                    targets = var_targets * targets + mean_targets
                    predictions = var_targets * predictions + mean_targets

                    l_0_targets = var_targets * l_0_targets + mean_targets
                    l_0_predictions = var_targets * l_0_predictions + mean_targets

                    if len(targets_to_est) > 0:
                        targets_to_est = var_targets * targets_to_est + mean_targets
                        predictions_to_est = var_targets * predictions_to_est + mean_targets

                    # mean_l_0_targets = mean_targets
                    # var_l_0_targets = var_targets

                    train_targets = var_targets * train_targets + mean_targets
                    train_predictions = var_targets * train_predictions + mean_targets

                if dataset_config.get('output_log', False):
                    targets = np.exp(targets)
                    predictions = np.exp(predictions)

                    l_0_targets = np.exp(l_0_targets)
                    l_0_predictions = np.exp(l_0_predictions)

                    if len(targets_to_est) > 0:
                        targets_to_est = np.exp(targets_to_est)
                        predictions_to_est = np.exp(predictions_to_est)
                    train_predictions = np.exp(train_predictions)
                    train_targets = np.exp(train_targets)

                if dataset_config.get('first_log_output', False):
                    targets = np.exp(targets)
                    predictions = np.exp(predictions)
                    l_0_targets = np.exp(l_0_targets)
                    l_0_predictions = np.exp(l_0_predictions)

                    if len(targets_to_est) > 0:
                        targets_to_est = np.exp(targets_to_est)
                        predictions_to_est = np.exp(predictions_to_est)
                    train_predictions = np.exp(train_predictions)
                    train_targets = np.exp(train_targets)

        iter_test_MSE = np.mean((predictions - targets) ** 2)

        iter_test_bias = np.sqrt(np.mean((targets - np.mean(predictions)) ** 2))
        iter_test_variance = np.mean((predictions - np.mean(predictions)) ** 2)

        iter_train_MSE = np.mean((train_predictions - train_targets) ** 2)

        iter_MSE_to_est = np.mean((l_0_predictions - l_0_targets) ** 2)
        #iter_MSE_to_est = np.mean((predictions_to_est - targets_to_est) ** 2)

        iter_train_bias = np.sqrt(np.mean((train_targets - np.mean(train_predictions)) ** 2))
        iter_train_variance = np.mean((train_predictions - np.mean(train_predictions)) ** 2)

        all_test_samples.append(targets)
        all_train_samples.append(train_targets)

        mean_t = np.mean(targets)
        iter_test_RSE = np.sum((predictions - targets) ** 2) / np.sum((targets - mean_t) ** 2)

        mean_tr = np.mean(train_targets)
        iter_train_RSE = np.sum((train_predictions - train_targets) ** 2) / np.sum((train_targets - mean_tr) ** 2)

        iter_test_MAE = np.abs((predictions - targets))
        iter_train_MAE = np.abs((train_predictions - train_targets))


        MSE_to_est.append(iter_MSE_to_est)
        RMSE_to_est.append(np.sqrt(iter_MSE_to_est))
        norm_RMSE_to_est.append(np.sqrt(iter_MSE_to_est)/np.mean(l_0_targets))#targets_to_est))

        train_RMSEstd_list.append(np.sqrt(iter_train_MSE)/np.std(train_targets))
        test_RMSEstd_list.append(np.sqrt(iter_MSE_to_est)/np.std(l_0_targets))

        train_MSE_list.append(iter_train_MSE)
        train_RSE_list.append(iter_train_RSE)
        train_RMSE_list.append(np.sqrt(iter_train_MSE))
        train_relRMSE_list.append(np.sqrt(iter_train_MSE)/np.mean(train_targets))
        train_reldiffRMSE_list.append(np.sqrt(iter_train_MSE) / (np.max(train_targets) - np.min(train_targets)))
        train_MAE_list.append(iter_train_MAE)

        q3, q1 = np.percentile(train_targets, [75, 25])
        train_iqrRMSE_list.append(np.sqrt(iter_train_MSE) / (q3 - q1))

        train_bias.append(iter_train_bias)
        train_variance.append(iter_train_variance)

        test_MSE_list.append(iter_test_MSE)
        test_RSE_list.append(iter_test_RSE)
        test_RMSE_list.append(np.sqrt(iter_test_MSE))
        test_relRMSE_list.append(np.sqrt(iter_test_MSE) / np.mean(targets))

        test_reldiffRMSE_list.append(np.sqrt(iter_test_MSE) / (np.max(targets) - np.min(targets)))

        q3, q1 = np.percentile(targets, [75, 25])
        test_iqrRMSE_list.append(np.sqrt(iter_test_MSE) / (q3 - q1))

        test_MAE_list.append(iter_test_MAE)

        test_bias.append(iter_test_bias)
        test_variance.append(iter_test_variance)

        print("iter test MSE ", iter_test_MSE)
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
        #try:
        mlmc_n_collected, nn_mlmc_n_collected, n_ops, n_ops_predict, orig_moments_mean, predict_moments_mean, \
        ref_moments_mean, orig_level_params, nn_level_params, kl_mlmc, kl_nn, target_variance, \
        orig_orth_moments, predict_orth_moments, ref_orth_moments,\
        ref_orig_moments, ref_predict_moments, mlmc_predict_moments, learning_time_post = process_mlmc(config['hdf_path'],
                                                                                 config['sampling_info_path'],
                                                                                 config['ref_mlmc_file'],
                                                                                 data_dict["test_targets"][i],
                                                                                 data_dict["test_predictions"][i],
                                                                                 data_dict["train_targets"][i],
                                                                                 data_dict["train_predictions"][i],
                                                                                 data_dict["val_targets"][i],
                                                                                 data_dict["l_0_targets"][i],
                                                                                 data_dict["l_0_predictions"][i],
                                                                                 l1_sample_time,
                                                                                 l0_sample_time,
                                                                                 nn_level=config['level'],
                                                                                 replace_level=config['replace_level'],
                                                                                 mlmc_hdf_file=config['mlmc_hdf_path'],
                                                                                 stats=True,
                                                                                 learning_time=learning_time,
                                                                                 dataset_config=dataset_config,
                                                                                 targets_to_est=data_dict["targets_to_est"][i],
                                                                                 predictions_to_est=data_dict["predictions_to_est"][i]
                                                                                   )
        # except:
            #      continue

        mlmc_n_collected_all.append(mlmc_n_collected)
        nn_n_collected_all.append(nn_mlmc_n_collected)
        n_ops_all.append(n_ops)
        n_ops_predict_all.append(n_ops_predict)
        mlmc_times_levels.append(np.array(mlmc_n_collected) * np.array(n_ops))
        mlmc_times.append(np.sum(np.array(mlmc_n_collected) * np.array(n_ops)))

        print("nn mlmc n collected ", nn_mlmc_n_collected)
        print("n ops predict ", n_ops_predict)

        nn_times.append(np.sum(np.array(nn_mlmc_n_collected) * np.array(n_ops_predict)) + learning_time_post)
        nn_times_levels.append(np.array(nn_mlmc_n_collected) * np.array(n_ops_predict))

        mlmc_l_vars.append(orig_moments_mean.l_vars)
        nn_l_vars.append(predict_moments_mean.l_vars)

        beta, beta_1, gamma, gamma_1 = compute_beta_gamma(orig_moments_mean.l_vars, n_ops, mlmc_n_collected)
        all_beta.append(list(beta))
        all_beta_1.append(list(beta_1))
        all_gamma.append(list(gamma))
        all_gamma_1.append(list(gamma_1))
        beta, beta_1, gamma, gamma_1 = compute_beta_gamma(predict_moments_mean.l_vars, n_ops_predict, nn_mlmc_n_collected)
        nn_all_beta.append(list(beta))
        nn_all_beta_1.append(list(beta_1))
        nn_all_gamma.append(list(gamma))
        nn_all_gamma_1.append(list(gamma_1))

        mlmc_vars.append(orig_moments_mean.var)
        nn_vars.append(predict_moments_mean.var)

        # mlmc_vars.append(ref_orig_moments[1].var)
        # nn_vars.append(ref_predict_moments[1].var)

        mlmc_l_means.append(orig_moments_mean.l_means)
        nn_l_means.append(predict_moments_mean.l_means)

        mlmc_vars_mse.append((ref_moments_mean.var - orig_moments_mean.var) ** 2)
        nn_vars_mse.append((ref_moments_mean.var - predict_moments_mean.var) ** 2)

        mlmc_means_mse.append((ref_moments_mean.mean - orig_moments_mean.mean) ** 2)
        nn_means_mse.append((ref_moments_mean.mean - predict_moments_mean.mean) ** 2)

        mlmc_means_diff.append(np.abs(ref_moments_mean.mean - orig_moments_mean.mean))
        nn_means_diff.append(np.abs(ref_moments_mean.mean - predict_moments_mean.mean))


        #######################################
        # Moments same domain for comparison  #
        #######################################

        mlmc_vars_mse_2.append((ref_orig_moments[0].var - ref_orig_moments[1].var) ** 2)
        nn_vars_mse_2.append((ref_predict_moments[0].var - ref_predict_moments[1].var) ** 2)

        mlmc_means_mse_2.append((ref_orig_moments[0].mean - ref_orig_moments[1].mean) ** 2)
        nn_means_mse_2.append((ref_predict_moments[0].mean - ref_predict_moments[1].mean) ** 2)

        mlmc_means_diff_2.append(np.abs(ref_orig_moments[0].mean - ref_orig_moments[1].mean))
        nn_means_diff_2.append(np.abs(ref_predict_moments[0].mean - ref_predict_moments[1].mean))

        mlmc_nn_vars_mse_2.append((mlmc_predict_moments[0].var - mlmc_predict_moments[1].var) ** 2)
        mlmc_nn_means_mse_2.append((mlmc_predict_moments[0].mean - mlmc_predict_moments[1].mean) ** 2)
        mlmc_nn_means_diff_2.append(np.abs(mlmc_predict_moments[0].mean - mlmc_predict_moments[1].mean))


        mlmc_moments_mean.append(mlmc_predict_moments[0].mean)
        mlmc_moments_var.append(mlmc_predict_moments[0].var)
        mlmc_nn_moments_mean.append(mlmc_predict_moments[1].mean)
        mlmc_nn_moments_var.append(mlmc_predict_moments[1].var)

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

    #analyze_mlmc_properties(mlmc_l_vars, n_ops_all, mlmc_n_collected_all, nn_l_vars, n_ops_predict_all, nn_n_collected_all)

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

    moments_plot_2 = plots.MomentsPlots(log_var_y=True, title="Moments MLMC domain")
    moments_plot_2.add_moments((np.mean(mlmc_moments_mean, axis=0),
                              np.mean(mlmc_moments_var, axis=0)), label="mlmc moments")
    moments_plot_2.add_moments((np.mean(mlmc_nn_moments_mean, axis=0),
                              np.mean(mlmc_nn_moments_var, axis=0)), label="mlmc nn moments")

    moments_plot_2.show(None)

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

    print("mlmc means diff ", np.mean(mlmc_means_diff, axis=0))
    print("nn means diff ", np.mean(nn_means_diff, axis=0))

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

    print("mlmc n collected ", mlmc_n_collected_all)
    print("nn n collected all ", nn_n_collected_all)
    print("mlmc n collected ", mlmc_n_collected)
    print("nn n collected ", nn_n_collected)

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

    plt_var_2 = plots.VarianceNN2()
    plt_var_2.set_n_ops(np.mean(n_ops_predict_all, axis=0))
    plt_var_2.add_level_variances(np.array([474, 2714, 18397]), l_vars)

    # plt_var.show(None)
    # plt_var.show("mlmc_vars")
    #
    # plt_var = plot.Variance()
    l_vars = np.mean(nn_l_vars, axis=0)
    # print("nn l vars  ", l_vars)
    # print("nn level parsm ", nn_level_params)
    if len(nn_level_params) > 1:
        level_params = np.squeeze(nn_level_params)
    else:
        level_params = nn_level_params[0]

    print("level params ", level_params)

    level_params[0] *= 2
    plt_var.add_level_variances_nn(level_params, l_vars)
    plt_var.show("nn_vars")
    plt_var.show(None)


    print("level params ", level_params.shape)
    print("l_vars.shape ", l_vars.shape)
    print("l_vars ", l_vars)


    # plt_var_2.add_level_variances_nn(np.array([300, 474, 2714, 18397]), l_vars)
    # plt_var_2.show("nn_vars_2")
    # plt_var_2.show(None)

    plot_sse(nn_vars_mse, mlmc_vars_mse, title="moments_var")
    plot_sse(nn_means_mse, mlmc_means_mse, title="moments_mean")
    plot_sse(mlmc_means_mse, mlmc_means_mse, title="mlmc moments_mean")

    plot_sse(nn_vars_mse_2, mlmc_vars_mse_2, title="moments_var same domain")
    plot_sse(nn_means_mse_2, mlmc_means_mse_2, title="moments_mean same domain")
    plot_sse(mlmc_means_mse_2, mlmc_means_mse_2, title="mlmc moments_mean same domain")


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


    for pred, target in zip(data_dict["test_predictions"], data_dict["test_targets"]):
        R_squared = r2_score(target, pred)
        print("R squared ", R_squared)

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
    print("mean test RSE ", np.mean(test_RSE))
    print("mean test RMSE ", np.mean(test_RMSE))
    print("mean test MAE ", np.mean(test_MAE))
    print("max test MSE ", np.max(test_MSE))
    # print("max test RMSE ", np.max(test_RMSE))
    # print("max test MAE ", np.max(test_MAE))

    print("train_MSE ", train_MSE)

    print("mean train MSE ", np.mean(train_MSE))

    print("test RSE ", np.mean(all_test_RSE))
    print("test RSE ", np.mean(all_train_RSE))
    print("mean train RSE ", np.mean(train_RSE))
    print("mean train RMSE ", np.mean(train_RMSE))
    print("mean train MAE ", np.mean(train_MAE))
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
    print("SEM train MSE: {}".format(stats.sem(train_MSE_list)))
    print("SEM test MSE: {}".format(stats.sem(test_MSE_list)))
    # print("test MSE std", np.sqrt(np.var(test_MSE)))
    print("mean train sample: {}".format(np.mean(all_train_samples)))
    print("mean test sample: {}".format(np.mean(all_test_samples)))
    #print("mean sample : {}".format(np.mean(np.array(all_train_samples).flatten() + np.array(all_test_samples).flatten())))
    print("RMSE train: {}, test: {}".format(np.mean(train_RMSE_list), np.mean(test_RMSE_list)))
    print("relative RMSE train: {}, test: {}".format(np.mean(train_RMSE_list)/np.mean(all_train_samples), np.mean(test_RMSE_list)/np.mean(all_test_samples)))
    print("iter relative RMSE train: {}, test: {}".format(np.mean(train_relRMSE_list),
                                                     np.mean(test_relRMSE_list)))
    print("iter relative diff RMSE train: {}, test: {}".format(np.mean(train_reldiffRMSE_list),
                                                          np.mean(test_reldiffRMSE_list)))
    print("iter iqr RMSE train: {}, test: {}".format(np.mean(train_iqrRMSE_list),
                                                               np.mean(test_iqrRMSE_list)))
    print("RSE train: {}, test: {}".format(np.mean(train_RSE_list), np.mean(test_RSE_list)))
    print("MAE train: {}, test: {}".format(np.mean(train_MAE_list), np.mean(test_MAE_list)))

    #########################
    ## targets to est data ##
    #########################
    print("test est")
    print("MSE: {}".format(np.mean(MSE_to_est)))
    print("RMSE: {}".format(np.mean(RMSE_to_est)))
    print("norm RMSE: {}".format(norm_RMSE_to_est))

    q75, q25 = np.percentile(norm_RMSE_to_est, [75, 25])
    intr_qr = q75 - q25
    max_test = q75 + (1.5 * intr_qr)
    min_test = q25 - (1.5 * intr_qr)

    #print("min test: {}, max test: {}".format(min_test, max_test))

    q75, q25 = np.percentile(train_relRMSE_list, [75, 25])
    intr_qr = q75 - q25
    max_train = q75 + (1.5 * intr_qr)
    min_train = q25 - (1.5 * intr_qr)

    rmse_to_mean_train = []
    rmse_to_mean_test = []
    print("norm_RMSE_to_est ", norm_RMSE_to_est)
    print("train_relRMSE_to_est ", train_relRMSE_list)

    # print("max train ", max_train)
    # print("min train ", min_train)

    for rmse_test, rmse_train in zip(norm_RMSE_to_est, train_relRMSE_list):
        if min_test < rmse_test < max_test and min_train < rmse_train < max_train:
            rmse_to_mean_train.append(rmse_train)
            rmse_to_mean_test.append(rmse_test)

    print("rmse to mean train ", rmse_to_mean_train)
    print("rmse to mean test ", rmse_to_mean_test)
    print("mean norm train RMSE: {}".format(np.mean(rmse_to_mean_train)))
    print("mean norm test RMSE: {}".format(np.mean(rmse_to_mean_test)))



    #####################
    #####################
    print("train_RMSEstd_list ", train_RMSEstd_list)
    print("test_RMSEstd_list ", test_RMSEstd_list)

    q75, q25 = np.percentile(test_RMSEstd_list, [75, 25])
    intr_qr = q75 - q25
    max_test = q75 + (1.5 * intr_qr)
    min_test = q25 - (1.5 * intr_qr)

    # print("min test: {}, max test: {}".format(min_test, max_test))

    q75, q25 = np.percentile(train_RMSEstd_list, [75, 25])
    intr_qr = q75 - q25
    max_train = q75 + (1.5 * intr_qr)
    min_train = q25 - (1.5 * intr_qr)

    rmse_std_train = []
    rmse_std_test = []

    for rmse_train, rmse_test in zip(train_RMSEstd_list, test_RMSEstd_list):
        if min_test < rmse_test < max_test and min_train < rmse_train < max_train:
            rmse_std_train.append(rmse_train)
            rmse_std_test.append(rmse_test)

    print("iter RMSE/std train: {}, test: {}".format(np.mean(rmse_std_train),
                                                     np.mean(rmse_std_test)))




    # print("train RSE ", np.mean(train_RSE))
    # print("test RSE ", np.mean(test_RMSE))
    #
    # print("train RMSE ", np.mean(train_RMSE))
    # print("test RMSE ", np.mean(test_RMSE))

    print("MC mom mean MSE: total: {:0.5g}, to 10: {:0.5g}, above: {:0.5g}".format(np.sum(np.mean(mlmc_means_mse, axis=0)), np.sum(np.mean(mlmc_means_mse, axis=0)[:10]),
                                            np.sum(np.mean(mlmc_means_mse, axis=0)[10:])))
    print("NN mom mean MSE: total: {:0.5g}, to 10: {:0.5g}, above: {:0.5g}".format(np.sum(np.mean(nn_means_mse, axis=0)), np.sum(np.mean(nn_means_mse, axis=0)[:10]),
                                            np.sum(np.mean(nn_means_mse, axis=0)[10:])))

    print("compare moments each couple has same domain (common domain)")
    print("MC mom mean MSE: total: {:0.5g}, to 10: {:0.5g}, above: {:0.5g}".format(
        np.sum(np.mean(mlmc_means_mse_2, axis=0)), np.sum(np.mean(mlmc_means_mse_2, axis=0)[:10]),
        np.sum(np.mean(mlmc_means_mse_2, axis=0)[10:])))
    print("NN mom mean MSE: total: {:0.5g}, to 10: {:0.5g}, above: {:0.5g}".format(np.sum(np.mean(nn_means_mse_2, axis=0)),
                                                                                 np.sum(np.mean(nn_means_mse_2, axis=0)[
                                                                                        :10]),
                                                                                 np.sum(np.mean(nn_means_mse_2, axis=0)[
                                                                                        10:])))

    #print("mlmc vs NN mom MSE: ", mlmc_nn_means_mse_2)
    mom_mean_MSE_realization = np.mean(mlmc_nn_means_mse_2, axis=1)
    #print("mlmc vs NN mom MSE for realization: ", np.mean(mlmc_nn_means_mse_2, axis=1))
    print("mlmc vs NN mom MSE for realization: ", list(mom_mean_MSE_realization))
    print("mlmc vs NN mom mean MSE for realization: ", np.mean(mom_mean_MSE_realization))
    # print("MLMC vs NN mom mean MSE: total: {:0.5g}, to 10: {:0.5g}, above: {:0.5g}".format(
    #     np.sum(np.mean(mom_mean_MSE_realization, axis=0)), np.sum(np.mean(mom_mean_MSE_realization, axis=0)[:10]),
    #     np.sum(np.mean(mom_mean_MSE_realization, axis=0)[10:])))
    print("MLMC vs NN mom var MSE: total: {:0.5g}, to 10: {:0.5g}, above: {:0.5g}".format(
        np.sum(np.mean(mlmc_nn_vars_mse_2, axis=0)),
        np.sum(np.mean(mlmc_nn_vars_mse_2, axis=0)[:10]),
        np.sum(np.mean(mlmc_nn_vars_mse_2, axis=0)[10:])))

    print("nn total time ", nn_total_time)
    print("mlmc total time ", mlmc_total_time)
    print("nn times ", nn_times)
    print("mlmc times ", mlmc_times)

    print("KL mlmc ", np.mean(kl_mlmc_all))
    print("KL nn ", np.mean(kl_nn_all))

    print("mean learning time ", np.mean(learning_times))
    print("max learning time ", np.max(learning_times))

    # print("beta mean ", np.mean(all_beta, axis=0))
    # print("beta 1  mean ", np.mean(all_beta_1, axis=0))
    #
    # print("gamma mean ", np.mean(all_gamma, axis=0))
    # print("gamma 1  mean ", np.mean(all_gamma_1, axis=0))
    #
    # print("nn beta mean ", np.mean(nn_all_beta, axis=0))
    # print("nn beta 1  mean ", np.mean(nn_all_beta_1, axis=0))
    #
    # print("nn gamma mean ", np.mean(nn_all_gamma, axis=0))
    # print("nn gamma 1  mean ", np.mean(nn_all_gamma_1, axis=0))

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


def run_GNN(config, stats=True, train=True, log=False, index=0):
    print("seed ", index)

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

    independent_samples = config.get("independent_samples", False)

    if independent_samples and train:
        data = FlowDataset(output_dir=config['output_dir'], level=config['level'], log=log, config=config,
                           index=index, n_test_samples=100000)
        len_all_samples = len(data)

        last_train_sample = index * config['n_train_samples'] + config['n_train_samples']
        last_test_sample = len_all_samples - (index * config['n_train_samples'] + config['n_train_samples'])

        print("last train sample ", last_train_sample)
        print("last test sample ", last_test_sample)

        if last_train_sample > last_test_sample:
            return

        data_tr = FlowDataset(output_dir=config['output_dir'], level=config['level'], log=log, config=config,
                              index=index, train_samples=True, independent_sample=True)

        print("len data tr ", len(data_tr))

        data_te = FlowDataset(output_dir=config['output_dir'], level=config['level'], log=log, config=config,
                              index=index, predict=True, test_samples=True, independent_samples=True)

        print("len data te ", len(data_te))

    else:
        if train:
            data_tr = FlowDataset(output_dir=config['output_dir'], level=config['level'], log=log, config=config,
                                  index=index, train_samples=True)
            data_te = FlowDataset(output_dir=config['output_dir'], level=config['level'], log=log, config=config,
                                  index=index, predict=True, test_samples=True)
        else:

            # data = FlowDataset(output_dir=config['output_dir'], level=config['level'], log=log, config=config,
            #                    index=index)
            # print("len data")
            # data_tr = data
            # data_te = data
            config["n_train_samples"] = 20000
            data_tr = FlowDataset(output_dir=config['output_dir'], level=config['level'], log=log, config=config,
                                  index=index, train_samples=True)
            data_te = FlowDataset(output_dir=config['output_dir'], level=config['level'], log=log, config=config,
                                  index=index, predict=True, test_samples=True)


    # Dataset preprocess config
    config['dataset_config'] = data_tr._dataset_config

    preprocess_time = time.process_time() - preprocess_start_time
    preprocess_time = preprocess_time + graph_creation_time
    print("preprocess time ", preprocess_time)

    data_te_predict = None
    if "predict_dir" in config and config["predict_dir"] is not None:
        # data_te_predict = FlowDataset(output_dir=config['predict_dir'], level=config['level'], log=log, config=config,
        #                               index=index, n_test_samples=50000)

        data_te_predict = FlowDataset(output_dir=config['predict_dir'], config=config, predict=True)

        data_te_predict.a = config['conv_layer'].preprocess(data_te_predict.a)
        data_te_predict.a = sp_matrix_to_sp_tensor(data_te_predict.a)

    learning_time_start = time.time()
    data_tr.a = sp_matrix_to_sp_tensor(config['conv_layer'].preprocess(data_tr.a))
    data_te.a = data_tr.a #sp_matrix_to_sp_tensor(config['conv_layer'].preprocess(data_te.a))

    val_data_len = int(len(data_tr) * config['val_samples_ratio'])
    # print("val data len ", val_data_len)
    # data_tr, data_va = data_tr.split_val_train(val_data_len)
    data_tr, data_va = data_tr[:-val_data_len], data_tr[-val_data_len:]

    print("len data tr ", len(data_tr))
    print("len data va ", len(data_va))
    print("len data te ", len(data_te))

    # #############################
    # #### OLD version
    # #############################
    # # Load data
    # data = FlowDataset(output_dir=config['output_dir'], level=config['level'], log=log, config=config, index=seed)
    # data = data#[:10000]
    #
    # # Dataset preprocess config
    # config['dataset_config'] = data._dataset_config
    #
    # #print("len data ", len(data))
    # #data.shuffle(seed=seed)
    # preprocess_time = time.process_time() - preprocess_start_time
    # #print("preproces time ", preprocess_time)
    # preprocess_time = preprocess_time + graph_creation_time
    # #print("total preprocess time ", preprocess_time)
    #
    # learning_time_start = time.process_time()
    # data.a = config['conv_layer'].preprocess(data.a)
    # data.a = sp_matrix_to_sp_tensor(data.a)
    # #train_data_len = int(len(data) * 0.8)
    # train_data_len = config['n_train_samples']
    # # Train/valid/test split
    # #print("train data len ", train_data_len)
    #
    # if not train:
    #     data_tr = data
    #     data_te = data
    # else:
    #     #data_tr = data[seed*train_data_len: seed*train_data_len + train_data_len]
    #     data_tr = data.get_train_data(seed, train_data_len)
    #     #print("data tr ", data_tr)
    #     data_te = data.get_test_data(seed, train_data_len)
    # #data_tr, data_te = data[:train_data_len], data[train_data_len:]
    #
    # #np.random.shuffle(data_tr)
    # val_data_len = int(len(data_tr) * config['val_samples_ratio'])
    # #print("val data len ", val_data_len)
    # #data_tr, data_va = data_tr.split_val_train(val_data_len)
    # data_tr, data_va = data_tr[:-val_data_len], data_tr[-val_data_len:]


    ###########################################
    ###########################################
    ###########################################
    ###########################################

    # print("data tr ", data_tr)
    # print("data va ", data_va)

    # print("data_tr len ", len(data_tr))
    # print("data_va len ", len(data_va))
    # print("data_te len ", len(data_te))

    gnn = config['gnn'](**config['model_config'])

    # We use a MixedLoader since the dataset is in mixed mode
    loader_tr = MixedLoader(data_tr, batch_size=batch_size, epochs=epochs)
    loader_va = MixedLoader(data_va, batch_size=batch_size)
    loader_te = MixedLoader(data_te, batch_size=batch_size)

    if data_te_predict is not None:
        loader_te_predict = MixedLoader(data_te_predict, batch_size=batch_size)

    if not train:
        gnn.fit(MixedLoader(data_tr[:10], batch_size=batch_size, epochs=epochs),
                MixedLoader(data_tr[10:20], batch_size=batch_size), MixedLoader(data_tr[20:30], batch_size=batch_size))
        set_model_weights(gnn._model, config["set_model"])

        # import visualkeras
        # model = gnn._model
        #
        # visualkeras.layered_view(model).show()  # display using your system viewer
        # visualkeras.layered_view(model, to_file='output.png')  # write to disk
        # visualkeras.layered_view(model, to_file='output.png').show()  # write and show

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

    learning_time = time.time() - learning_time_start

    print("learning time ", learning_time)

    if train:
        states = gnn._states
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

    targets_to_est = []
    predictions_to_est = []

    if data_te_predict is not None:
        targets_to_est, predictions_to_est = gnn.predict(loader_te_predict)
        predictions_to_est = np.squeeze(predictions_to_est)
        targets_to_est = np.array(targets_to_est)
        predictions_to_est = np.array(predictions_to_est)

    #print("learning time ", learning_time)

    targets = np.array(targets)
    predictions = np.array(predictions)

    #print("MSE ", np.mean((predictions-targets)**2))

    if log:
        targets = np.exp(targets)
        predictions = np.exp(predictions)
        target_to_est = np.exp(targets_to_est)
        predictions_to_est = np.exp(predictions_to_est)

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
    #l_0_targets, l_0_predictions, predict_l_0_time = [], [], []
    l_0_targets, l_0_predictions, predict_l_0_time = predict_level_zero(gnn, config['l_0_output_dir'],
                                                                        config['l_0_hdf_path'], config['mesh'],
                                                                        config['conv_layer'], batch_size, log,
                                                                        stats=stats,
                                                                        corr_field_config=config['corr_field_config'],
                                                                        seed=index,
                                                                        feature_names=config.get('feature_names', [['conductivity']]),
                                                                        config=config
                                                                        )
    #predict_l_0_time = time.process_time() - predict_l_0_start_time

    if stats:
        l1_sample_time = preprocess_time / (len(data_tr) + len(data_te)) + learning_time / (len(data_tr) + len(data_te))
        l0_sample_time = predict_l_0_time / len(l_0_targets)

        # print("targets ", targets)
        # print("predictions ", predictions)

        # orig_max_vars, predict_max_vars = process_mlmc(hdf_path, sampling_info_path, ref_mlmc_file, targets, predictions, train_targets,
        #              train_predictions,
        #              val_targets, l_0_targets,
        #              l_0_predictions, l1_sample_time, l0_sample_time, nn_level=level, replace_level=replace_level,
        #                                                stats=stats)

        return gnn, targets, predictions, learning_time, train_targets, train_predictions,\
               val_targets, val_predictions, l_0_targets, l_0_predictions, l1_sample_time, l0_sample_time, total_steps, targets_to_est, predictions_to_est

    save_times(config['save_path'], False, (preprocess_time, (len(data_tr) + len(data_te))), learning_time, (predict_l_0_time, len(l_0_targets)))
    save_load_data(config['save_path'], False, targets, predictions, train_targets, train_predictions, val_targets, l_0_targets,
                   l_0_predictions)


def run_DeepMap(config, stats=True, train=True, log=False, index=0):
    print("seed ", index)

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

    independent_samples = config.get("independent_samples", False)

    if independent_samples and train:
        data = FlowDataset(output_dir=config['output_dir'], level=config['level'], log=log, config=config,
                           index=index, n_test_samples=100000)
        len_all_samples = len(data)

        last_train_sample = index * config['n_train_samples'] + config['n_train_samples']
        last_test_sample = len_all_samples - (index * config['n_train_samples'] + config['n_train_samples'])

        print("last train sample ", last_train_sample)
        print("last test sample ", last_test_sample)

        if last_train_sample > last_test_sample:
            return

        data_tr = FlowDataset(output_dir=config['output_dir'], level=config['level'], log=log, config=config,
                              index=index, train_samples=True, independent_sample=True)

        print("len data tr ", len(data_tr))

        data_te = FlowDataset(output_dir=config['output_dir'], level=config['level'], log=log, config=config,
                              index=index, predict=True, test_samples=True, independent_samples=True)

        print("len data te ", len(data_te))

    else:
        if train:
            data_tr = FlowDataset(output_dir=config['output_dir'], level=config['level'], log=log, config=config,
                                  index=index, train_samples=True)
            data_te = FlowDataset(output_dir=config['output_dir'], level=config['level'], log=log, config=config,
                                  index=index, predict=True, test_samples=True)
        else:

            # data = FlowDataset(output_dir=config['output_dir'], level=config['level'], log=log, config=config,
            #                    index=index)
            # print("len data")
            # data_tr = data
            # data_te = data
            config["n_train_samples"] = 20000
            data_tr = FlowDataset(output_dir=config['output_dir'], level=config['level'], log=log, config=config,
                                  index=index, train_samples=True)
            data_te = FlowDataset(output_dir=config['output_dir'], level=config['level'], log=log, config=config,
                                  index=index, predict=True, test_samples=True)


    # Dataset preprocess config
    config['dataset_config'] = data_tr._dataset_config

    preprocess_time = time.process_time() - preprocess_start_time
    preprocess_time = preprocess_time + graph_creation_time
    print("preprocess time ", preprocess_time)

    data_te_predict = None
    if "predict_dir" in config and config["predict_dir"] is not None:
        # data_te_predict = FlowDataset(output_dir=config['predict_dir'], level=config['level'], log=log, config=config,
        #                               index=index, n_test_samples=50000)

        data_te_predict = FlowDataset(output_dir=config['predict_dir'], config=config, predict=True)

        data_te_predict.a = config['conv_layer'].preprocess(data_te_predict.a)
        data_te_predict.a = sp_matrix_to_sp_tensor(data_te_predict.a)

    learning_time_start = time.time()
    data_tr.a = sp_matrix_to_sp_tensor(config['conv_layer'].preprocess(data_tr.a))
    data_te.a = data_tr.a #sp_matrix_to_sp_tensor(config['conv_layer'].preprocess(data_te.a))

    val_data_len = int(len(data_tr) * config['val_samples_ratio'])
    # print("val data len ", val_data_len)
    # data_tr, data_va = data_tr.split_val_train(val_data_len)
    data_tr, data_va = data_tr[:-val_data_len], data_tr[-val_data_len:]

    print("len data tr ", len(data_tr))
    print("len data va ", len(data_va))
    print("len data te ", len(data_te))

    gnn = config['gnn'](**config['model_config'])

    # We use a MixedLoader since the dataset is in mixed mode
    loader_tr = MixedLoader(data_tr, batch_size=batch_size, epochs=epochs)
    loader_va = MixedLoader(data_va, batch_size=batch_size)
    loader_te = MixedLoader(data_te, batch_size=batch_size)

    if data_te_predict is not None:
        loader_te_predict = MixedLoader(data_te_predict, batch_size=batch_size)

    if not train:
        gnn.fit(MixedLoader(data_tr[:10], batch_size=batch_size, epochs=epochs),
                MixedLoader(data_tr[10:20], batch_size=batch_size), MixedLoader(data_tr[20:30], batch_size=batch_size))
        set_model_weights(gnn._model, config["set_model"])

        # import visualkeras
        # model = gnn._model
        #
        # visualkeras.layered_view(model).show()  # display using your system viewer
        # visualkeras.layered_view(model, to_file='output.png')  # write to disk
        # visualkeras.layered_view(model, to_file='output.png').show()  # write and show

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

    learning_time = time.time() - learning_time_start

    print("learning time ", learning_time)

    if train:
        states = gnn._states
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

    targets_to_est = []
    predictions_to_est = []

    if data_te_predict is not None:
        targets_to_est, predictions_to_est = gnn.predict(loader_te_predict)
        predictions_to_est = np.squeeze(predictions_to_est)
        targets_to_est = np.array(targets_to_est)
        predictions_to_est = np.array(predictions_to_est)

    #print("learning time ", learning_time)

    targets = np.array(targets)
    predictions = np.array(predictions)

    #print("MSE ", np.mean((predictions-targets)**2))

    if log:
        targets = np.exp(targets)
        predictions = np.exp(predictions)
        target_to_est = np.exp(targets_to_est)
        predictions_to_est = np.exp(predictions_to_est)

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
    #l_0_targets, l_0_predictions, predict_l_0_time = [], [], []
    l_0_targets, l_0_predictions, predict_l_0_time = predict_level_zero(gnn, config['l_0_output_dir'],
                                                                        config['l_0_hdf_path'], config['mesh'],
                                                                        config['conv_layer'], batch_size, log,
                                                                        stats=stats,
                                                                        corr_field_config=config['corr_field_config'],
                                                                        seed=index,
                                                                        feature_names=config.get('feature_names', [['conductivity']]),
                                                                        config=config
                                                                        )
    #predict_l_0_time = time.process_time() - predict_l_0_start_time

    if stats:
        l1_sample_time = preprocess_time / (len(data_tr) + len(data_te)) + learning_time / (len(data_tr) + len(data_te))
        l0_sample_time = predict_l_0_time / len(l_0_targets)

        # print("targets ", targets)
        # print("predictions ", predictions)

        # orig_max_vars, predict_max_vars = process_mlmc(hdf_path, sampling_info_path, ref_mlmc_file, targets, predictions, train_targets,
        #              train_predictions,
        #              val_targets, l_0_targets,
        #              l_0_predictions, l1_sample_time, l0_sample_time, nn_level=level, replace_level=replace_level,
        #                                                stats=stats)

        return gnn, targets, predictions, learning_time, train_targets, train_predictions,\
               val_targets, val_predictions, l_0_targets, l_0_predictions, l1_sample_time, l0_sample_time, total_steps, targets_to_est, predictions_to_est

    save_times(config['save_path'], False, (preprocess_time, (len(data_tr) + len(data_te))), learning_time, (predict_l_0_time, len(l_0_targets)))
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

    #
    #config["n_train_samples"] = 2000
    #data = FlowDataset(output_dir=output_dir, log=log, config=config)
    data = FlowDataset(output_dir=output_dir, log=log, config=config, predict=True, n_test_samples=2000)#, mesh=mesh, corr_field_config=corr_field_config)
    #data = data  # [:10000]
    data.shuffle(seed=seed)
    
    # print("output_dir ", output_dir)
    # print("len(data) ", len(data))
    # print("data[0] ", data[0])
    
    predict_time_start = time.process_time()
    data.a = conv_layer.preprocess(data.a)
    data.a = sp_matrix_to_sp_tensor(data.a)

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


def predict_level_zero_CNN(nn, output_dir, hdf_path, mesh, conv_layer, batch_size=1000, log=False, stats=False,
                       corr_field_config=None, seed=1234, feature_names=[], config=None, mean_std_features=None, mean_std_target=None):
    # image_creator(output_dir, hdf_path, mesh, level=0, feature_names=feature_names)
    # exit()
    # Load data
    sample_time = 0
    if corr_field_config:
        sample_time = corr_field_sample_time(mesh, corr_field_config)
    else:
        raise Exception("No corr field config passed")

    #
    # config["n_train_samples"] = 2000
    # data = FlowDataset(output_dir=output_dir, log=log, config=config)

    data = ImageFlowDataset(data_dir=output_dir, config=config, mean_features=mean_std_features[0], std_features=mean_std_features[1],
                            mean_target=mean_std_target[0], std_target=mean_std_target[1])  # , mesh=mesh, corr_field_config=corr_field_config)
    # data = data  # [:10000]
    #data.shuffle(seed=seed)

    features, target = data[0]
    ot = (tf.float64, tf.float64)
    os = (tf.TensorShape([features.shape[0], features.shape[1], features.shape[2]]), tf.TensorShape([]))

    dataset = tf.data.Dataset.from_generator(data._generate_examples, output_types=ot,
                                              output_shapes=os)  # output_shapes=(512,512,1))


    # print("output_dir ", output_dir)
    # print("len(data) ", len(data))
    # print("data[0] ", data[0])

    predict_time_start = time.process_time()
    # data.a = conv_layer.preprocess(data.a)
    # data.a = sp_matrix_to_sp_tensor(data.a)

    #loader_te = BatchLoader(data, batch_size=batch_size)

    targets, predictions = nn.predict(dataset)
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
