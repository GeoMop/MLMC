from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from spektral.layers import GlobalSumPool


# Build model
class Net1(Model):
    def __init__(self, conv_layer, hidden_activation, output_activation, kernel_regularization, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = conv_layer(32, activation=hidden_activation, kernel_regularizer=kernel_regularization)
        self.conv2 = conv_layer(32, activation=hidden_activation, kernel_regularizer=kernel_regularization)
        self.flatten = GlobalSumPool()
        self.fc1 = Dense(512, activation=hidden_activation)
        self.fc2 = Dense(1, activation=output_activation)  # linear activation for output neuron

    def call(self, inputs):
        x, a = inputs
        x = self.conv1([x, a])
        x = self.conv2([x, a])
        output = self.flatten(x)
        output = self.fc1(output)
        output = self.fc2(output)
        return output