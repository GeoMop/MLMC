from mlmc.metamodel.flow_task_GNN_2 import GNN
from spektral.layers import GCNConv, GlobalSumPool, ChebConv, GraphSageConv, ARMAConv, GATConv, APPNPConv, GINConv, GeneralConv
from tensorflow.keras.losses import MeanSquaredError, KLDivergence, MeanAbsoluteError
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from spektral.layers import GlobalSumPool, GlobalMaxPool, GlobalAvgPool
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing


def get_gnn():
    # Parameters
    # conv_layer = GCNConv
    conv_layer = ChebConv  # Seems better than GCNConv, good distribution of predictions
    # conv_layer = OwnChebConv
    # conv_layer = GraphSageConv  # Seems better than ChebConv, good loss but very narrow distribution of predictions
    # # conv_layer = ARMAConv  # Seems worse than GraphSageConv
    # conv_layer = GATConv  # Slow and not better than GraphSageConv
    # # conv_layer = APPNPConv  # Not bad but worse than GraphSageConv
    # # conv_layer = GINConv  # it is comparable to APPNPConv
    # act_func = "relu"  # "tanh"#"elu"

    loss = MeanSquaredError()  # var_loss_function#
    # loss = MeanAbsoluteError()
    # loss = MeanSquaredLogarithmicError()
    # loss = KLDivergence()
    # loss = total_loss_function
    optimizer = tf.optimizers.Adam(learning_rate=0.001)
    patience = 150
    hidden_regularization = None  # l2(2e-10)

    model_config = {
                 "conv_layer": conv_layer,
                 "hidden_activation": 'relu',
                 "output_activation": 'linear',
                 "kernel_regularization": hidden_regularization,
                 "normalizer": preprocessing.Normalization()
                 }

    model = Net(**model_config)

    model_config = {"loss": loss,
                    "optimizer": optimizer,
                    "patience": patience,
                    "model": model,
                    "verbose": False}

    print("get gnn")

    return GNN(**model_config), conv_layer


class Net(Model):
    def __init__(self, conv_layer, hidden_activation, output_activation, kernel_regularization, normalizer,
                 **kwargs):
        super().__init__(**kwargs)
        # self.normalizer = normalizer
        # self.norm_layer = tf.keras.layers.LayerNormalization(axis=1)
        self.conv1 = conv_layer(32, activation=hidden_activation, kernel_regularizer=kernel_regularization)
        # self.conv2 = conv_layer(128, K=2, activation=hidden_activation, kernel_regularizer=kernel_regularization)
        # self.conv3 = conv_layer(16, K=2, activation=hidden_activation, kernel_regularizer=kernel_regularization)
        # self.conv4 = conv_layer(8,  K=2,activation=hidden_activation, kernel_regularizer=kernel_regularization)
        # self.conv2 = conv_layer(32, K=2, activation=hidden_activation, kernel_regularizer=kernel_regularization)
        # self.conv3 = conv_layer(16, K=2, activation=hidden_activation, kernel_regularizer=kernel_regularization)
        # self.conv3 = conv_layer(8, activation=hidden_activation, kernel_regularizer=kernel_regularization)
        # self.conv4 = conv_layer(4, activation=hidden_activation, kernel_regularizer=kernel_regularization)
        # self.conv3 = conv_layer(64, activation=hidden_activation, kernel_regularizer=kernel_regularization)
        self.flatten = GlobalSumPool()
        # self.fc1 = Dense(32, activation=hidden_activation)
        self.fc2 = Dense(1)  # , activation=output_activation)  # linear activation for output neuron

    def call(self, inputs):
        x, a = inputs
        # print("x ", x)
        # x = self.normalizer(x)
        # x = self.norm_layer(x)
        # print("normalized x ", x)
        # print("x[0,0,:] ", x[0, 0, :])
        x = self.conv1([x, a])
        # print("x.shape ", x.shape)
        # x = self.conv2([x, a])
        # # print("conv2 x shape", x.shape)
        # x = self.conv3([x, a])
        # x = self.conv4([x, a])
        output1 = self.flatten(x)
        # output2 = self.fc1(output1)
        output = self.fc2(output1)
        return output


if __name__ == "__main__":
    get_gnn()
