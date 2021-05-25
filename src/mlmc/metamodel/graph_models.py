from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from spektral.layers import GlobalSumPool, GlobalMaxPool, GlobalAvgPool
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from mlmc.metamodel.custom_methods import abs_activation
from tensorflow.keras.layers.experimental import preprocessing


# Build model
class Net1(Model):
    def __init__(self, conv_layer, hidden_activation, output_activation, kernel_regularization, normalizer, **kwargs):
        super().__init__(**kwargs)
        #self.normalizer = normalizer
        #self.norm_layer = tf.keras.layers.LayerNormalization(axis=1)
        self.conv1 = conv_layer(256,  activation=hidden_activation, kernel_regularizer=kernel_regularization)
        # self.conv2 = conv_layer(128, K=2, activation=hidden_activation, kernel_regularizer=kernel_regularization)
        # self.conv3 = conv_layer(16, K=2, activation=hidden_activation, kernel_regularizer=kernel_regularization)
        # self.conv4 = conv_layer(8,  K=2,activation=hidden_activation, kernel_regularizer=kernel_regularization)
        # self.conv2 = conv_layer(32, K=2, activation=hidden_activation, kernel_regularizer=kernel_regularization)
        # self.conv3 = conv_layer(16, K=2, activation=hidden_activation, kernel_regularizer=kernel_regularization)
        # self.conv3 = conv_layer(8, activation=hidden_activation, kernel_regularizer=kernel_regularization)
        # self.conv4 = conv_layer(4, activation=hidden_activation, kernel_regularizer=kernel_regularization)
        #self.conv3 = conv_layer(64, activation=hidden_activation, kernel_regularizer=kernel_regularization)
        self.flatten = GlobalSumPool()
        #self.fc1 = Dense(32, activation=hidden_activation)
        self.fc2 = Dense(1)#, activation=output_activation)  # linear activation for output neuron

    def call(self, inputs):
        x, a = inputs
        #print("x ", x)
        #x = self.normalizer(x)
        #x = self.norm_layer(x)
        #print("normalized x ", x)

        #print("x[0,0,:] ", x[0, 0, :])
        x = self.conv1([x, a])

        #print("x[0,0,:] ", x[0,0,:])
        # print("x[0, 0, :] ", tf.make_ndarray(x[0,0,:].op.get_attr('net1/strided_slice_1:0')))
        #print("x.shape ", x.shape)
        # x = self.conv2([x, a])
        # # print("conv2 x shape", x.shape)
        # x = self.conv3([x, a])
        # x = self.conv4([x, a])
        output1 = self.flatten(x)
        #output2 = self.fc1(output1)
        output = self.fc2(output1)

        # print("x1 " ,x1)
        # print("output1 ", output1)
        # print("output2 ", output2)
        # print("output ", output)
        #print("output ", output.shape)

        return output


# Build model
class NetGCN(Model):
    # Setup from https://arxiv.org/pdf/1901.06181.pdf
    def __init__(self, conv_layer, hidden_activation, output_activation, kernel_regularization, normalizer, **kwargs):
        super().__init__(**kwargs)

        #self.normalizer = normalizer
        #self.norm_layer = tf.keras.layers.LayerNormalization(axis=1)
        self.conv1 = conv_layer(256, activation=hidden_activation, kernel_regularizer=kernel_regularization)
        #.conv2 = conv_layer(32, activation=hidden_activation, kernel_regularizer=kernel_regularization)
        # self.conv3 = conv_layer(16, activation=hidden_activation, kernel_regularizer=kernel_regularization)
        # self.conv4 = conv_layer(16, activation=hidden_activation, kernel_regularizer=kernel_regularization)
        # self.conv5 = conv_layer(32, activation=hidden_activation, kernel_regularizer=kernel_regularization)
        self.flatten = GlobalSumPool()
        #self.fc1 = Dense(16, activation=hidden_activation)
        self.fc2 = Dense(1)#, activation=output_activation)  # linear activation for output neuron

    def call(self, inputs):
        x, a = inputs
        #print("x ", x)
        #x = self.normalizer(x)
        #x = self.norm_layer(x)
        #print("normalized x ", x)

        #print("x[0,0,:] ", x[0, 0, :])
        x = self.conv1([x, a])
        #x = self.conv2([x, a])
        # x = self.conv3([x, a])
        # x = self.conv4([x, a])
        # x = self.conv5([x, a])
        #
        # #print("x[0,0,:] ", x[0,0,:])
        # print("x[0, 0, :] ", tf.make_ndarray(x[0,0,:].op.get_attr('net1/strided_slice_1:0')))
        # print("x.shape ", x.shape)
        # x = self.conv2([x, a])
        # x = self.conv3([x, a])
        output = self.flatten(x)
        #output = self.fc1(output)
        output = self.fc2(output)

        #print("output ", output.shape)

        return output


def cnn_model():
    return keras.Sequential([
            #@TODO: Try normalization
            #self._normalizer,  # Seems worse results with normalization
            layers.Conv1D(filters=256, kernel_size=3, activation='relu', input_shape=(958, 1)),#input_shape=(958, 1)),
            #layers.BatchNormalization(),
            layers.AveragePooling1D(pool_size=2),
            layers.Conv1D(filters=128, kernel_size=3, activation='relu'),
            #layers.BatchNormalization(),
            layers.AveragePooling1D(pool_size=2),

            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation=abs_activation)
        ])


def dnn_model():
    return keras.Sequential([
        #preprocessing.Normalization(),
        layers.Dense(128, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(1)
    ])