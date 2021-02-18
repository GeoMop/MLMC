import os
import numpy as np
import matplotlib.pyplot as plt
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Run on CPU only
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanSquaredError, SparseCategoricalCrossentropy, KLDivergence
from tensorflow.keras.metrics import mean_squared_error, kl_divergence
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from spektral.data import MixedLoader
from mlmc.metamodel.flow_dataset import FlowDataset
from spektral.layers import GCNConv, GlobalSumPool, ChebConv, GraphSageConv, ARMAConv, GATConv, APPNPConv, GINConv
from spektral.layers.ops import sp_matrix_to_sp_tensor



#conv_layer = GCNConv
conv_layer = ChebConv  # Seems better than GCNConv, good distribution of predictions
#conv_layer = GraphSageConv  # Seems better than ChebConv, good loss but very narrow distribution of predictions
#conv_layer = ARMAConv  # Seems worse than GraphSageConv
#conv_layer = GATConv  # Slow and not better than GraphSageConv
# conv_layer = APPNPConv  # Not bad but worse than GraphSageConv
# conv_layer = GINConv  # it is comparable to APPNPConv
act_func = "relu"#"tanh"#"elu"  # ReLU keep predictions above zero

optimizer = Adam()
loss_fn = MeanSquaredError()
#loss_fn = KLDivergence()

acc_fn = mean_squared_error
acc_fn = kl_divergence

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Parameters
batch_size = 1000  # Batch size
epochs = 1000  # Number of training epochs
patience = 30 # Patience for early stopping
l2_reg = 0#5e-4  # Regularization rate for l2


# Load data
data = FlowDataset()

#print("data.a ", data.a)

data.a = conv_layer.preprocess(data.a)
data.a = sp_matrix_to_sp_tensor(data.a)

# Train/valid/test split
data_tr, data_te = data[:10000], data[10000:],
np.random.shuffle(data_tr)
data_tr, data_va = data_tr[:8000], data_tr[8000:]

# We use a MixedLoader since the dataset is in mixed mode
loader_tr = MixedLoader(data_tr, batch_size=batch_size, epochs=epochs)
loader_va = MixedLoader(data_va, batch_size=batch_size)
loader_te = MixedLoader(data_te, batch_size=batch_size)

# Build model
class Net(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = conv_layer(32, activation=act_func, kernel_regularizer=l2(l2_reg))
        self.conv2 = conv_layer(32, activation=act_func, kernel_regularizer=l2(l2_reg))
        self.flatten = GlobalSumPool()
        self.fc1 = Dense(512, activation="relu")
        self.fc2 = Dense(1, activation="linear")  # linear activation for output neuron

    def call(self, inputs):
        x, a = inputs
        x = self.conv1([x, a])
        x = self.conv2([x, a])
        output = self.flatten(x)
        output = self.fc1(output)
        output = self.fc2(output)

        return output


# Create model
model = Net()



# Training function
@tf.function
def train_on_batch(inputs, target):
    with tf.GradientTape() as tape:
        print("inputs data shape ", inputs[0].shape)  # (number of train samples, number of vertices, number of properties for each vertex)
        predictions = model(inputs, training=True)
        #@TODO: zkusit pridat k loss function KLDivergence
        # print(KLDivergence(target, predictions))
        # exit()
        loss = loss_fn(target, predictions) + sum(model.losses)# + KLDivergence(target, predictions)#+ sum(model.losses)
        acc = tf.reduce_mean(acc_fn(target, predictions))

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, acc


# Evaluation function
def evaluate(loader):
    step = 0
    results = []
    for batch in loader:
        step += 1
        inputs, target = batch
        print("loader ", loader)
        print("inputs data shape ", inputs[0].shape) # (number of validation or test samples, number of vertices, number of properties for each vertex)
        predictions = model(inputs, training=False)

        loss = loss_fn(target, predictions)
        acc = tf.reduce_mean(acc_fn(target, predictions))
        results.append((loss, acc, len(target)))  # Keep track of batch size
        if step == loader.steps_per_epoch:
            results = np.array(results)
            return np.average(results[:, :-1], axis=0, weights=results[:, -1]), target, predictions



def analyze_results(target, predictions):
    from scipy.stats import ks_2samp
    statistics, pvalue = ks_2samp(target, predictions)
    print("KS statistics: {}, pvalue: {}".format(statistics, pvalue))
    # The closer KS statistic is to 0 the more likely it is that the two samples were drawn from the same distribution

    print("len(target) ", len(target))
    print("len(predictions) ", len(predictions))

    plt.hist(target, bins=10, alpha=0.5, label='target', density=True)
    plt.hist(predictions, bins=10, alpha=0.5, label='predictions', density=True)
    plt.legend(loc='upper right')
    plt.show()



# Setup training
best_val_loss = np.inf
current_patience = patience
step = 0

# Training loop
results_tr = []
for batch in loader_tr:
    step += 1

    # Training step
    inputs, target = batch
    loss, acc = train_on_batch(inputs, target)
    results_tr.append((loss, acc, len(target)))

    all_targets = []
    all_predictions = []

    if step == loader_tr.steps_per_epoch:
        results_va, target, predictions = evaluate(loader_va)
        if results_va[0] < best_val_loss:
            best_val_loss = results_va[0]
            current_patience = patience
            results_te, target, predictions = evaluate(loader_te)

            print("target ", target)
            print("predictions ", np.squeeze(predictions.numpy()))
            print("len(target) ", len(target))
            print("len(predictions) ", len(np.squeeze(predictions.numpy())))

            analyze_results(target, np.squeeze(predictions.numpy()))


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

analyze_results(target, np.squeeze(predictions.numpy()))



