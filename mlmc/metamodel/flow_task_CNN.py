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

class CNN:
    def __init__(self, **kwargs):
        self._epochs = kwargs.get('epochs', 100)
        self._val_split = kwargs.get('var_split', 0.2)
        self._verbose = kwargs.get('verbose', False)

        self._loss = kwargs.get('loss', 'mean_squared_error')
        self._optimizer = kwargs.get('optimizer', tf.optimizers.Adam(learning_rate=0.001))
        self._normalizer = kwargs.get('normalizer', preprocessing.Normalization())

        self.history = None  # Set in fit method
        self._model = cnn_model()
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





