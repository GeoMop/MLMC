import os
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
from spektral.layers.ops import sp_matrix_to_sp_tensor
from tensorflow.keras.layers.experimental import preprocessing
from mlmc.metamodel.custom_methods import abs_activation, var_loss_function

from mlmc.metamodel.graph_models import Net1


class GNN:
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

        self._loss = kwargs.get('loss', mean_squared_error)
        self._accuracy_func = kwargs.get('accuracy_func', mean_squared_error)
        self._optimizer = kwargs.get('optimizer', tf.optimizers.Adam(learning_rate=0.001))
        self._normalizer = kwargs.get('normalizer', preprocessing.Normalization())
        self._patience = kwargs.get('patience', 20)

        self._train_loss = []
        self._val_loss = []
        self._test_loss = []

        self.val_targets = []

        model = kwargs.get('model')
        if model is None:
            self._model = Net1(conv_layer=self._conv_layer, hidden_activation=self._hidden_activation,
                               output_activation=self._output_activation,
                               kernel_regularization=self._hidden_regularizer,
                               normalizer=self._normalizer)
        else:
            self._model = model(conv_layer=self._conv_layer, hidden_activation=self._hidden_activation,
                               output_activation=self._output_activation,
                               kernel_regularization=self._hidden_regularizer,
                               normalizer=self._normalizer)

    def fit(self, loader_tr, loader_va, loader_te):
        """
        Training procedure
        """
        # Setup training
        best_val_loss = np.inf
        current_patience = self._patience
        step = 0

        train_targets = True
        train_targets_list = []

        # Training loop
        results_tr = []
        for batch in loader_tr:
            step += 1

            # Training step
            inputs, target = batch

            if train_targets:
                train_targets_list.extend(target)
                print("len(train targets ", len(train_targets_list))

            loss, acc = self.train_on_batch(inputs, target)
            self._train_loss.append(loss)
            results_tr.append((loss, acc, len(target)))

            results_va = self.evaluate(loader_va)
            self._val_loss.append(results_va[0])

            if step == loader_tr.steps_per_epoch: # step_per_epoch = int(np.ceil(len(self.dataset) / self.batch_size))
                train_targets = False
                # results_va = self.evaluate(loader_va)
                # self._val_loss.append(results_va[0])
                if results_va[0] < best_val_loss:
                    best_val_loss = results_va[0]
                    current_patience = self._patience
                    results_te = self.evaluate(loader_te)
                    self._test_loss.append(results_te[0])
                else:
                    current_patience -= 1
                    if current_patience == 0:
                        print("Early stopping")
                        break

                # Print results
                results_tr = np.array(results_tr)
                results_tr = np.average(results_tr[:, :-1], 0, weights=results_tr[:, -1])
                print(
                    "Train loss: {:.4f}, acc: {:.4f} | "
                    "Valid loss: {:.4f}, acc: {:.4f} | "
                    "Test loss: {:.4f}, acc: {:.4f}".format(
                        *results_tr, *results_va, *results_te
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
            # @TODO: try to add KLDivergence to loss
            # print(KLDivergence(target, predictions))
            # print("self._loss(target, predictions) ", self._loss(target, predictions))
            # print("sum(self._model.losses) ", sum(self._model.losses))
            # print("var_loss_function(target, predictions) ", var_loss_function(target, predictions))
            loss = self._loss(target, predictions) + sum(self._model.losses) #+ 5 * var_loss_function(target, predictions)
            #loss = 100 * var_loss_function(target, predictions)
            acc = tf.reduce_mean(self._accuracy_func(target, predictions))

        gradients = tape.gradient(loss, self._model.trainable_variables)
        self._optimizer.apply_gradients(zip(gradients, self._model.trainable_variables))
        return loss, acc

    def evaluate(self, loader):
        step = 0
        results = []

        if len(self.val_targets) > 0:
            val_targets = False
        else:
            val_targets = True

        for batch in loader:
            step += 1
            inputs, target = batch

            if val_targets:
                self.val_targets.extend(target)

            predictions = self._model(inputs, training=False)

            loss = self._loss(target, predictions)
            acc = tf.reduce_mean(self._accuracy_func(target, predictions))
            results.append((loss, acc, len(target)))  # Keep track of batch size
            if step == loader.steps_per_epoch:
                results = np.array(results)
                return np.average(results[:, :-1], axis=0, weights=results[:, -1])

    def predict(self, loader):
        targets = []
        predictions = []
        step = 0
        for batch in loader:

            step += 1
            inputs, target = batch
            targets.extend(target)
            predictions.extend(self._model(inputs, training=False))

            if step == loader.steps_per_epoch:
                return targets, predictions

        return targets, predictions


# if __name__ == "__main__":
#     # Parameters
#     #conv_layer = GCNConv
#     conv_layer = ChebConv  # Seems better than GCNConv, good distribution of predictions
#     # # conv_layer = GraphSageConv  # Seems better than ChebConv, good loss but very narrow distribution of predictions
#     # # conv_layer = ARMAConv  # Seems worse than GraphSageConv
#     # # conv_layer = GATConv  # Slow and not better than GraphSageConv
#     # # conv_layer = APPNPConv  # Not bad but worse than GraphSageConv
#     # # conv_layer = GINConv  # it is comparable to APPNPConv
#     # act_func = "relu"  # "tanh"#"elu"  # ReLU keep predictions above zero
#     loss = MeanSquaredError()
#     optimizer = tf.optimizers.Adam(learning_rate=0.001)
#     batch_size = 500
#     epochs = 100
#
#     # Load data
#     data = FlowDataset()
#     data = data#[:10000]
#     #data.a = conv_layer.preprocess(data.a)
#     data.a = sp_matrix_to_sp_tensor(data.a)
#
#     train_data_len = int(len(data) * 0.8)
#     train_data_len = 10000
#
#     # Train/valid/test split
#     data_tr, data_te = data[:train_data_len], data[train_data_len:],
#     np.random.shuffle(data_tr)
#
#     val_data_len = int(len(data_tr) * 0.2)
#     data_tr, data_va = data_tr[:-val_data_len], data_tr[-val_data_len:]
#
#     print("data_tr len ", len(data_tr))
#     print("data_va len ", len(data_va))
#     print("data_te len ", len(data_te))
#
#     # We use a MixedLoader since the dataset is in mixed mode
#     loader_tr = MixedLoader(data_tr, batch_size=batch_size, epochs=epochs)
#     loader_va = MixedLoader(data_va, batch_size=batch_size)
#     loader_te = MixedLoader(data_te, batch_size=batch_size)
#
#     gnn = GNN(loss=loss, optimizer=optimizer, conv_layer=conv_layer, output_activation=abs_activation,
#               hidden_activation='relu', patience=20)
#     gnn.fit(loader_tr, loader_va, loader_te)
#
#     targets, predictions = gnn.predict(loader_te)
#     predictions = np.squeeze(predictions)
#
#     plot_loss(gnn._train_loss, gnn._val_loss)
#     analyze_results(targets, predictions)
#
#     estimate_density(targets)
#     estimate_density(predictions)
