import numpy as np
from mlmc.metamodel.flow_dataset import FlowDataset
import tensorflow as tf
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
            hidden_layers.append(layers.Dense(self._n_hidden_neurons[i], activation=self._hidden_activation))

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








