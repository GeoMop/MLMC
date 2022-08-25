import tensorflow as tf
from mlmc.metamodel.graph_models import cnn_model
from tensorflow.keras.layers.experimental import preprocessing

# Following 3 lines prevent "Failed to get convolution algorithm. This is probably because cuDNN failed to initialize"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

import os
import numpy as np
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Run on CPU only
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError, SparseCategoricalCrossentropy, KLDivergence
from tensorflow.keras.metrics import mean_squared_error, kl_divergence
from tensorflow.keras.callbacks import History
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from mlmc.metamodel.postprocessing import analyze_results, plot_loss
from spektral.data import MixedLoader
from mlmc.metamodel.flow_dataset import FlowDataset
from spektral.layers import GCNConv, GlobalSumPool, ChebConv, GraphSageConv, ARMAConv, GATConv, APPNPConv, GINConv
from spektral.utils.sparse import sp_matrix_to_sp_tensor
from tensorflow.keras.layers.experimental import preprocessing
from mlmc.metamodel.custom_methods import abs_activation

from mlmc.metamodel.graph_models import Net1



##################################
# Convolutional neural network   #
##################################
#
# class CNN:
#     def __init__(self, **kwargs):
#         self._epochs = kwargs.get('epochs', 100)
#         self._val_split = kwargs.get('var_split', 0.2)
#         self._verbose = kwargs.get('verbose', False)
#
#         self._loss = kwargs.get('loss', 'mean_squared_error')
#         self._optimizer = kwargs.get('optimizer', tf.optimizers.Adam(learning_rate=0.001))
#         self._normalizer = kwargs.get('normalizer', preprocessing.Normalization())
#
#         self.history = None  # Set in fit method
#         self._model = cnn_model()
#         self._model.compile(loss=self._loss, optimizer=self._optimizer)
#
#     def fit(self, train_input, train_output):
#         self.history = self._model.fit(train_input, train_output, validation_split=self._val_split,
#                                        verbose=self._verbose, epochs=self._epochs)
#
#     def predict(self, test_input):
#         return self._model.predict(test_input)
#
#     def summary(self):
#         """
#         Should be called after fit method
#         """
#         return self._model.summary()


import os
import warnings
import logging
logging.getLogger('tensorflow').disabled = True
logging.getLogger('absl').disabled = True
warnings.simplefilter("ignore")
import numpy as np
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Run on CPU only
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError, SparseCategoricalCrossentropy, KLDivergence
from tensorflow.keras.metrics import mean_squared_error, kl_divergence
from tensorflow.keras.callbacks import History
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from mlmc.metamodel.postprocessing import analyze_results, plot_loss, estimate_density
from spektral.data import MixedLoader
from mlmc.metamodel.flow_dataset import FlowDataset
from spektral.layers import GCNConv, GlobalSumPool, ChebConv, GraphSageConv, ARMAConv, GATConv, APPNPConv, GINConv
from spektral.utils.sparse import sp_matrix_to_sp_tensor
from tensorflow.keras.layers.experimental import preprocessing
import copy
from mlmc.metamodel.custom_methods import abs_activation, var_loss_function
import keras.backend as K

from mlmc.metamodel.graph_models import Net1
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class CNN:
    def __init__(self, **kwargs):

        self._epochs = kwargs.get('epochs', 100)
        #self._val_split = kwargs.get('var_split', 0.2)
        #self._verbose = kwargs.get('verbose', False)

        self._hidden_activation = kwargs.get('hidden_activation', 'relu')
        self._hidden_regularizer = kwargs.get('hidden_reqularizer', None)
        self._output_activation = kwargs.get('output_activation', 'linear')
        self._conv_layer = kwargs.get('conv_layer', None)
        #self._n_hidden_layers = kwargs.get('n_hidden_layers', 1)
        #self._n_hidden_neurons = kwargs.get('n_hidden_neurons', [64])  # Number of hidden neurons for each hidden layer

        self._loss = kwargs.get('loss', MeanSquaredError)
        self._accuracy_func = kwargs.get('accuracy_func', mean_squared_error)
        self._optimizer = kwargs.get('optimizer', tf.optimizers.Adam(learning_rate=0.001))
        self._normalizer = kwargs.get('normalizer', preprocessing.Normalization())
        self._patience = kwargs.get('patience', 20)
        self._verbose = kwargs.get('verbose', True)

        self._train_loss = []
        self._val_loss = []
        self._test_loss = []
        self._train_acc = []
        self._learning_rates = []

        self.val_targets = []
        self._states = {}
        self._total_n_steps = 0

        if 'model_class' in kwargs:
            model_class = kwargs.get('model_class')
            net_model_config = kwargs.get('net_model_config')
            model = model_class(**net_model_config)
        else:
            model = kwargs.get('model')

        if model is None:
            self._model = Net1(conv_layer=self._conv_layer, hidden_activation=self._hidden_activation,
                               output_activation=self._output_activation,
                               kernel_regularization=self._hidden_regularizer,
                               normalizer=self._normalizer)
        else:
            self._model = model
            # self._model = model(conv_layer=self._conv_layer, hidden_activation=self._hidden_activation,
            #                    output_activation=self._output_activation,
            #                    kernel_regularization=self._hidden_regularizer,
            #                    normalizer=self._normalizer)
            #self._model = model(n_labels=1, output_activation="relu")

        self._model.optimizer = self._optimizer

    def fit(self, loader_tr, loader_va, loader_te, config):
        """
        Training procedure
        """
        # Setup training
        best_val_loss = np.inf
        current_patience = self._patience
        step = 0
        self._total_n_steps = 0

        train_targets = True
        train_targets_list = []

        epochs = config["epochs"]
        self._batch_size = config["batch_size"]
        for e in range(epochs):
            #print("e " ,e)
            # Training loop
            results_tr = []
            loader_tr = loader_tr.shuffle(np.min([config["n_train_samples"], 500]))
            train_loss = 0
            for batch in loader_tr.batch(self._batch_size):
                step += 1
                self._total_n_steps += 1

                # Training step
                inputs, target = batch
                if train_targets:
                    train_targets_list.extend(target)

                loss, acc = self.train_on_batch(inputs, target)

                self._train_loss.append(loss)
                self._train_acc.append(acc)
                results_tr.append((loss, acc, len(target)))

            results_va = self.evaluate(loader_va)

            self._val_loss.append(results_va[0])

            #if step == loader_tr.steps_per_epoch:  # step_per_epoch = int(np.ceil(len(self.dataset) / self.batch_size))
            train_targets = False
            if results_va[0] < best_val_loss:

                self._model.save(os.path.join(config["output_dir"], "saved_model"))
                #self._model = tf.keras.models.load_model(os.path.join(config["output_dir"], "saved_model"))

                best_val_loss = results_va[0]
                current_patience = self._patience
                self._states = {}
                #self._states[results_va[0]] = copy.deepcopy(self)
                results_te = self.evaluate(loader_te)
                self._test_loss.append(results_te[0])
            else:
                current_patience -= 1
                #results_tr_0 = np.array(results_tr)
                if current_patience == 0:
                    print("Early stopping")
                    break

            lr = K.eval(self._optimizer._decayed_lr(tf.float32))
            self._learning_rates.append(lr)
            # Print results
            results_tr = np.array(results_tr)
            results_tr = np.average(results_tr[:, :-1], 0, weights=results_tr[:, -1])
            if self._verbose:
                print(
                    "Train loss: {:.12f}, acc: {:.12f} | "
                    "Valid loss: {:.12f}, acc: {:.12f} | "
                    "Test loss: {:.12f}, acc: {:.12f} | LR: {:.12f}".format(
                        *results_tr, *results_va, *results_te, lr
                    )
                )

            # Reset epoch
            results_tr = []
            step = 0

        return train_targets_list

    # Training function
    @tf.function
    def train_on_batch(self, inputs, target):
        with tf.GradientTape() as tape:
            predictions = self._model(inputs, training=True)
            # print("targets ", target)
            # print("predictions ", predictions)
            loss = self._loss(target, predictions) + sum(self._model.losses) #+ 5 * var_loss_function(target, predictions)
            acc = tf.reduce_mean(self._accuracy_func(target, predictions))

        gradients = tape.gradient(loss, self._model.trainable_variables)
        self._optimizer.apply_gradients(zip(gradients, self._model.trainable_variables))

        return loss, acc

    def evaluate(self, loader, batch_size=None):
        step = 0
        results = []
        if batch_size is None:
            batch_size = self._batch_size

        for batch in loader.batch(batch_size):
            step += 1
            inputs, target = batch

            loss, acc = self.evaluate_batch(inputs, target)
            results.append((loss, acc, len(target)))  # Keep track of batch size

        results = np.array(results)
        return np.average(results[:, :-1], axis=0, weights=results[:, -1])

    @tf.function
    def evaluate_batch(self, inputs, target):
        predictions = self._model(inputs, training=False)

        loss = self._loss(target, predictions)
        acc = tf.reduce_mean(self._accuracy_func(target, predictions))

        return loss, acc

    def predict(self, loader, batch_size=20):
        targets = []
        predictions = []
        step = 0
        for batch in loader.batch(batch_size):
            step += 1
            inputs, target = batch
            targets.extend(target)
            predictions.extend(self._model(inputs, training=False))

        return targets, predictions






