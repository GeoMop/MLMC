import os
import numpy as np
import random
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Run on CPU only
from mlmc.metamodel.flow_dataset import FlowDataset
# Make numpy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)
import tensorflow as tf
from scipy.stats import ks_2samp

from mlmc.tool import plot
from mlmc.moments import Legendre
import mlmc.tool.simple_distribution
import mlmc.estimator
from mlmc.sample_storage import Memory
from mlmc.quantity_spec import QuantitySpec, ChunkSpec
from mlmc.quantity import make_root_quantity
from mlmc.quantity_estimate import estimate_mean
import sklearn.model_selection
from mlmc.metamodel.custom_methods import abs_activation
from mlmc.metamodel.postprocessing import analyze_results, plot_loss
from mlmc.metamodel.flow_task_NN import DNN
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

epochs = 100


def prepare_dataset(dataset):
    # Load data
    dataset = dataset.dropna()
    train_x, test_x, train_y, test_y = sklearn.model_selection.train_test_split(dataset.x, dataset.y,
                                                                                test_size=0.2, random_state=123)
    train_x = np.squeeze(np.stack(train_x.to_numpy(), axis=0))
    train_x = np.asarray(train_x).astype('float64')
    train_y = train_y.to_numpy()
    train_y = np.asarray(train_y).astype('float64')

    test_x = np.squeeze(np.stack(test_x.to_numpy(), axis=0))
    test_x = np.asarray(test_x).astype('float64')
    test_y = test_y.to_numpy()
    test_y = np.asarray(test_y).astype('float64')

    return train_x, train_y, test_x, test_y


def estimate_density(values, title="Density"):
    sample_storage = Memory()
    n_levels = 1
    n_moments = 15
    distr_accuracy = 1e-7

    distr_plot = plot.Distribution(title=title,
                                   log_density=True)

    result_format = [QuantitySpec(name="flow", unit="m", shape=(1,), times=[0], locations=['0'])]

    sample_storage.save_global_data(result_format=result_format, level_parameters=np.ones(n_levels))

    successful_samples = {}
    failed_samples = {}
    n_ops = {}
    n_successful = len(values)
    for l_id in range(n_levels):
        sizes = []
        for quantity_spec in result_format:
            sizes.append(np.prod(quantity_spec.shape) * len(quantity_spec.times) * len(quantity_spec.locations))

        # Dict[level_id, List[Tuple[sample_id:str, Tuple[fine_result: ndarray, coarse_result: ndarray]]]]
        successful_samples[l_id] = []
        for sample_id in range(len(values)):
            successful_samples[l_id].append((str(sample_id), (values[sample_id], 0)))

        n_ops[l_id] = [random.random(), n_successful]

        sample_storage.save_scheduled_samples(l_id, samples=["S{:07d}".format(i) for i in range(n_successful)])

    print("len successful samples ", len(successful_samples[0]))
    sample_storage.save_samples(successful_samples, failed_samples)
    sample_storage.save_n_ops(list(n_ops.items()))

    quantity = make_root_quantity(storage=sample_storage, q_specs=result_format)
    length = quantity['flow']
    time = length[0]
    location = time['0']
    value_quantity = location[0]

    quantile = 0.001
    true_domain = mlmc.estimator.Estimate.estimate_domain(value_quantity, sample_storage, quantile=quantile)
    moments_fn = Legendre(n_moments, true_domain)

    estimator = mlmc.estimator.Estimate(quantity=value_quantity, sample_storage=sample_storage, moments_fn=moments_fn)

    reg_param = 0
    target_var = 1e-4
    distr_obj, info, result, moments_fn = estimator.construct_density(
        tol=distr_accuracy,
        reg_param=reg_param,
        orth_moments_tol=target_var)

    samples = value_quantity.samples(ChunkSpec(level_id=0, n_samples=sample_storage.get_n_collected()[0]))[..., 0]

    print("samples ", np.array(samples).shape)
    print("np.squeeze(samples) ", np.squeeze(samples).shape)

    distr_plot.add_raw_samples(np.squeeze(samples))

    distr_plot.add_distribution(distr_obj, label="")

    # kl = mlmc.tool.simple_distribution.KL_divergence(self.cut_distr.pdf, distr_obj.density,
    #                                                  self.cut_distr.domain[0], self.cut_distr.domain[1])
    #kl_divergences.append(kl)

    distr_plot.show(file=None)


def run():
    # Parameters
    loss = "mean_squared_error"
    optimizer = tf.optimizers.Adam(learning_rate=0.001)

    data = FlowDataset()
    dataset = data.dataset[:50000]
    train_input, train_output, test_input, test_output = prepare_dataset(dataset)

    print("len test(output) ", len(test_output))

    dnn = DNN(loss=loss, optimizer=optimizer, output_activation=abs_activation, hidden_activation='relu')
    dnn.fit(train_input, train_output)

    predictions = dnn.predict(test_input)
    predictions = np.squeeze(predictions)

    print("len(predictions) ", len(predictions))

    plot_loss(dnn.history.history['loss'], dnn.history.history['val_loss'])
    analyze_results(test_output, predictions)

    estimate_density(test_output)
    estimate_density(predictions)


def bootstrap():
    loss = "mean_absolute_error"
    optimizer = tf.optimizers.Adam(learning_rate=0.001)
    n_subsamples = 10
    size = 10000

    train_losses = []
    val_losses = []
    all_test_outputs = []
    all_predictions = []
    ks_statistics = []
    ks_p_values = []

    data = FlowDataset()
    dataset = data.dataset.dropna()

    for i in range(n_subsamples):
        dset = dataset.sample(size, replace=True)
        train_input, train_output, test_input, test_output = prepare_dataset(dset)

        print("Size TRAIN in: {}, out: {}, TEST in: {}, out: {}".format(len(train_input), len(train_output),
                                                                        len(test_input), len(test_output)))

        dnn = DNN(loss=loss, optimizer=optimizer, output_activation=abs_activation, hidden_activation='relu')
        dnn.fit(train_input, train_output)

        predictions = dnn.predict(test_input)
        predictions = np.squeeze(predictions)

        train_losses.append(dnn.history.history['loss'])
        val_losses.append(dnn.history.history['val_loss'])

        all_test_outputs.append(test_output)
        all_predictions.append(predictions)

        statistics, pvalue = ks_2samp(test_output, predictions)
        ks_statistics.append(statistics)
        ks_p_values.append(pvalue)

    analyze_results(np.mean(all_test_outputs, axis=0), np.mean(all_predictions, axis=0))
    analyze_results(np.var(all_test_outputs, axis=0), np.var(all_predictions, axis=0))
    #
    # estimate_density(np.mean(all_test_outputs, axis=0), title="Test outputs")
    # estimate_density(np.mean(all_predictions, axis=0), title="Predictions")


if __name__ == "__main__":
    #run()

    bootstrap()