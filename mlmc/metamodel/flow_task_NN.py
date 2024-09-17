import tensorflow as tf
import numpy as np
from tensorflow.keras.metrics import mean_squared_error, kl_divergence
from tensorflow.keras.layers.experimental import preprocessing
from mlmc.metamodel.graph_models import dnn_model
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing


# epochs = 200
#
#
# # Parameters
# loss_fn = "mean_squared_error"
# optimizer_nn = tf.optimizers.Adam(learning_rate=0.1)
# optimizer_dnn = tf.optimizers.Adam(learning_rate=0.001)
#
#
# # Load data
# data = FlowDataset()
#
# dataset = data.dataset[:5000]
# print("len(dataset) ", len(dataset))
# dataset = dataset.dropna()
#
# train_dataset = dataset.sample(frac=0.8, random_state=0)
# test_dataset = dataset.drop(train_dataset.index)
#
# train_x = np.squeeze(np.stack(train_dataset.x.to_numpy(), axis=0))
# train_x = np.asarray(train_x).astype('float64')
# train_y = train_dataset.y.to_numpy()
# train_y = np.asarray(train_y).astype('float64')
#
# test_x = np.squeeze(np.stack(test_dataset.x.to_numpy(), axis=0))
# test_x = np.asarray(test_x).astype('float32')
# test_y = test_dataset.y.to_numpy()
# test_y = np.asarray(test_y).astype('float32')


# ####################
# ## Neural network  - very bad for our purposes ##
# ####################
#
# normalizer = preprocessing.Normalization()
# linear_model = tf.keras.Sequential([
#     normalizer,
#     layers.Dense(units=1)
# ])
#
# linear_model.compile(
#     optimizer=optimizer_nn,
#     loss=loss_fn)
# #
# # history = linear_model.fit(
# #     train_x, train_y,
# #     epochs=100,
# #     # suppress logging
# #     verbose=0,
# #     # Calculate validation results on 20% of the training data
# #     validation_split=0.2)
# #
# # linear_model.summary()
# #
# #
# # plot_loss(history)
# # predictions = np.squeeze(linear_model.predict(test_x))
# # analyze_results(target=test_y, predictions=predictions)



#########################
# Deep neural network   #
#########################


class DNN:
    def __init__(self, **kwargs):
        self._epochs = kwargs.get('epochs', 100)
        self._val_split = kwargs.get('var_split', 0.2)
        self._verbose = kwargs.get('verbose', False)

        self._hidden_activation = kwargs.get('hidden_activation', 'relu')
        self._hidden_regularizer = kwargs.get('hidden_reqularizer', None)
        self._output_activation = kwargs.get('output_activation', 'linear')
        self._n_hidden_layers = kwargs.get('n_hidden_layers', 1)
        self._n_hidden_neurons = kwargs.get('n_hidden_neurons', [64])  # Number of hidden neurons for each hidden layer

        self._loss = kwargs.get('loss', 'mean_squared_error')
        self._optimizer = kwargs.get('optimizer', tf.optimizers.Adam(learning_rate=0.001))
        self._normalizer = kwargs.get('normalizer', preprocessing.Normalization())

        self.history = None  # Set in fit method
        self._create_model()

    def _create_model(self):
        hidden_layers = []
        for i in range(self._n_hidden_layers):
            if self._hidden_regularizer is not None:
                hidden_layers.append(
                    layers.Dense(self._n_hidden_neurons[i],
                                 kernel_regularizer=self._hidden_regularizer,
                                 activation=self._hidden_activation))
            else:
                hidden_layers.append(
                    layers.Dense(self._n_hidden_neurons[i],activation=self._hidden_activation))

        self._model = keras.Sequential([
            self._normalizer,
            *hidden_layers,
            layers.Dense(1, activation=self._output_activation)
        ])

        self._model.compile(loss=self._loss, optimizer=self._optimizer)

    def fit(self, train_input, train_output):
        self.history = self._model.fit(train_input, train_output, validation_split=self._val_split,
                                       verbose=self._verbose, epochs=self._epochs)

    def predict(self, test_input):
        return self._model.predict(test_input)

    def summary(self):
        """
        Should be called after fit method
        """
        return self._model.summary()


class DNN_2:
    def __init__(self, **kwargs):
        print("########  Create GNN  #########")

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
        self._final_loss = kwargs.get('final_loss', mean_squared_error)
        self._accuracy_func = kwargs.get('accuracy_func', mean_squared_error)
        self._optimizer = kwargs.get('optimizer', tf.optimizers.Adam(learning_rate=0.001))
        self._normalizer = kwargs.get('normalizer', preprocessing.Normalization())
        self._patience = kwargs.get('patience', 20)
        self._verbose = kwargs.get('verbose', True)
        #self._loss_changed = False

        self._train_loss = []
        self._val_loss = []
        self._test_loss = []

        self._loss_params = {}
        self._n_moments = 3

        self.val_targets = []
        self._states = {}
        self._total_n_steps = 0

        if 'model_class' in kwargs:
            model_class = kwargs.get('model_class')
            net_model_config = kwargs.get('net_model_config')
            model = model_class(**net_model_config)
            print("model class model ", model)
        else:
            model = kwargs.get('model')

        if model is None:
            self._model = DNNNet(conv_layer=self._conv_layer, hidden_activation=self._hidden_activation,
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

    def fit(self, train, validation, test):
        """
        Training procedure
        """
        print("fit init loss ", self._loss)
        # Setup training
        self._best_val_loss = np.inf
        self._current_patience = self._patience
        step = 0
        self._total_n_steps = 0

        train_targets = True
        train_targets_list = []

        # Training loop
        results_tr = []

        step += 1
        self._total_n_steps += 1

        # Training step
        inputs, target = train.x, train.y

        if train_targets:
            train_targets_list.extend(target)

        loss, acc = self.train_on_batch(inputs, target)
        results_tr.append((loss, acc, len(target)))

        results_va = self.evaluate(validation)
        self._val_loss.append(results_va[0])


        train_targets = False
        # results_va = self.evaluate(loader_va)
        # self._val_loss.append(results_va[0])
        #print("results_va[0] ", results_va[0])

        #print("self best val loss ", self._best_val_loss)

        if (results_va[0] + results_tr[-1][0]) < self._best_val_loss:# or (self._val_loss[-1] < self._val_loss[-2] and self._val_loss[-2] < self._val_loss[-3]):  # Continue to learn if validation loss is decreasing
            self._best_val_loss = (results_va[0] + results_tr[-1][0])#results_va[0]
            self._current_patience = self._patience
            self._states = {}
            results_te = self.evaluate(test)
            self._test_loss.append(results_te[0])
        else:
            self._current_patience -= 1
            #results_tr_0 = np.array(results_tr)
            loss_tr = results_va[0]
            self._states[loss_tr] = self

            if self._current_patience == 0:
                #if self._update_loss(patience=True):
                print("Early stopping")
                return

        # Print results
        results_tr = np.array(results_tr)
        results_tr = np.average(results_tr[:, :-1], 0, weights=results_tr[:, -1])

        self._train_loss.append(results_tr[0])
        if self._verbose:
            print(
                "Train loss: {:.4f}, acc: {:.4f} | "
                "Valid loss: {:.4f}, acc: {:.4f} | "
                "Test loss: {:.4f}, acc: {:.4f}".format(
                    *results_tr, *results_va, *results_te
                )
            )
        #self._update_loss()
        # Reset epoch
        results_tr = []
        step = 0

        return train_targets_list

    def _update_loss(self, patience=False):
        condition_max_loss = self._loss_params["loss_max"] #/ self._n_moments
        #condition_max_loss = self._loss_params["loss_max"]
        # print("self.train_loss ", self._train_loss)
        m_increment = 1

        if patience and self._n_moments <= self._loss_params["max_moments"]:
            self._n_moments += m_increment
            moments_fn = self._loss_params['moments_class'](self._n_moments, self._loss_params["domain"])
            self._loss = self._final_loss(moments_fn=moments_fn)
            # self._loss = MSE_moments_2(moments_fn=moments_fn)

            self._best_val_loss = np.inf
            print("self._loss ", self._loss)
        elif patience:
            return True

        # if self._train_loss[-1] > 1e10:
        #     moments_fn = self._loss_params['moments_class'](self._n_moments, self._loss_params["domain"])
        #     #self._loss = self._final_loss(moments_fn=moments_fn)
        #     self._loss = MSE_moments_2(moments_fn=moments_fn)
        # else:
        #     moments_fn = self._loss_params['moments_class'](self._n_moments, self._loss_params["domain"])
        #     self._loss = self._final_loss(moments_fn=moments_fn)

        if self._n_moments <= self._loss_params["max_moments"] and len(self._train_loss) > 0\
                and self._train_loss[-1] < condition_max_loss and self._val_loss[-1] < condition_max_loss:
            # print("self._train_loss ", self._train_loss)
            # print("change loss, n_moments  {}, last train loss: {}".format(self._n_moments, self._train_loss[-1]))
            #self._n_moments = self._loss_params["max_moments"]

            print("self._n_moments ", self._n_moments)
            self._n_moments += m_increment
            moments_fn = self._loss_params['moments_class'](self._n_moments, self._loss_params["domain"])
            self._loss = self._final_loss(moments_fn=moments_fn)
            #self._loss = MSE_moments_2(moments_fn=moments_fn)

            self._best_val_loss = np.inf
            print("self._loss ", self._loss)

    # Training function
    #@tf.function
    def train_on_batch(self, inputs, target):
        with tf.GradientTape() as tape:
            predictions = self._model(inputs, training=True)
            loss = self._loss(target, predictions) + sum(self._model.losses) #+ 5 * var_loss_function(target, predictions)
            #loss = 100 * var_loss_function(target, predictions)
            acc = tf.reduce_mean(self._accuracy_func(target, predictions))

        gradients = tape.gradient(loss, self._model.trainable_variables)
        self._optimizer.apply_gradients(zip(gradients, self._model.trainable_variables))

        return loss, acc

    def evaluate(self, data):
        step = 0
        results = []

        if len(self.val_targets) > 0:
            val_targets = False
        else:
            val_targets = True

        step += 1
        inputs, target = data.x, data.y

        if val_targets:
            self.val_targets.extend(target)

        predictions = self._model(inputs, training=False)

        #print("evaluate loss function ", self._loss)

        loss = self._loss(target, predictions)
        #print("target ", target)
        #print("loss ", np.mean((target - predictions)**2))
        acc = tf.reduce_mean(self._accuracy_func(target, predictions))
        results.append((loss, acc, len(target)))  # Keep track of batch size
        results = np.array(results)
        return np.average(results[:, :-1], axis=0, weights=results[:, -1])

    def predict(self, data):
        targets = []
        predictions = []

        inputs, target = data.x, data.y
        targets.extend(target)
        predictions.extend(self._model(inputs, training=False))

        return targets, predictions


# def build_and_compile_model(normalizer):
#   model = keras.Sequential([
#       normalizer,
#       layers.Dense(450, activation='relu'),  #64
#       #layers.Dense(64, activation='relu'),
#       layers.Dense(1, activation=abs_activation)
#   ])
#
#   model.compile(loss=loss_fn,
#                 optimizer=optimizer_dnn)
#   return model
#
#
# normalizer = preprocessing.Normalization()
# dnn_model = build_and_compile_model(normalizer)
# dnn_history = dnn_model.fit(
#     train_x, train_y,
#     validation_split=0.2,
#     verbose=0, epochs=epochs)
# dnn_model.summary()
#
# plot_loss(dnn_history)
#
# predictions = np.squeeze(dnn_model.predict(test_x))
#
# print("target ", test_y)
# print("predictions ", predictions)
#
#
# for index, (t, p) in enumerate(zip(test_y, predictions)):
#     if index > 100:
#         break
#     print("t: {}, p: {}".format(t, p))
#
# print("target mean ", np.mean(test_y))
# print("predictions mean ", np.mean(predictions))
#
# print("target var ", np.var(test_y))
# print("predictions var ", np.var(predictions))
#
# analyze_results(target=test_y, predictions=predictions)








