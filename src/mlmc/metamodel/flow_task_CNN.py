import numpy as np
from mlmc.metamodel.flow_dataset import FlowDataset
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
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
from spektral.layers.ops import sp_matrix_to_sp_tensor
from tensorflow.keras.layers.experimental import preprocessing
from mlmc.metamodel.custom_methods import abs_activation

from mlmc.metamodel.graph_models import Net1


##################################
# Convolutional neural network   #
##################################

class CNN:
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
            #@TODO: Try normalization
            #self._normalizer,  # Seems worse results with normalization
            layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(958, 1)),
            #layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
            #layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
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





