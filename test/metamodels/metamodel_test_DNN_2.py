import os
import shutil
import numpy as np
import warnings
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Run on CPU only
import sys
import subprocess
from mlmc.metamodel.analyze_nn import run_GNN, run_SVR, statistics, analyze_statistics, process_results, run_DNN
from mlmc.moments import Legendre_tf, Monomial

from mlmc.metamodel.flow_task_GNN_2 import GNN
from mlmc.metamodel.flow_task_NN import DNN, DNN_2
from spektral.layers import GCNConv, GlobalSumPool, ChebConv, GraphSageConv, ARMAConv, GATConv, APPNPConv, GINConv, GeneralConv
from tensorflow.keras.losses import MeanSquaredError, KLDivergence, MeanAbsoluteError
from mlmc.metamodel.custom_methods import abs_activation, MSE_moments
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from spektral.layers import GlobalSumPool, GlobalMaxPool, GlobalAvgPool
import tensorflow as tf
from mlmc.tool import plot
from tensorflow.keras.layers.experimental import preprocessing
warnings.filterwarnings("ignore", category=DeprecationWarning)


def get_gnn():
    # Parameters
    conv_layer = GCNConv
    conv_layer = ChebConv  # Seems better than GCNConv, good distribution of predictions
    # conv_layer = OwnChebConv
    # conv_layer = GraphSageConv  # Seems better than ChebConv, good loss but very narrow distribution of predictions
    # # conv_layer = ARMAConv  # Seems worse than GraphSageConv
    # conv_layer = GATConv  # Slow and not better than GraphSageConv
    # # conv_layer = APPNPConv  # Not bad but worse than GraphSageConv
    # # conv_layer = GINConv  # it is comparable to APPNPConv
    # act_func = "relu"  # "tanh"#"elu"

    loss = MeanSquaredError()  # var_loss_function#
    #loss = MSE_moments
    final_loss = MSE_moments
    # loss = MeanAbsoluteError()
    # loss = MeanSquaredLogarithmicError()
    # loss = KLDivergence()
    # loss = total_loss_function
    optimizer = tf.optimizers.Adam(learning_rate=0.001)
    patience = 200
    hidden_regularization = None  # l2(2e-10)

    net_model_config = {
        "conv_layer": conv_layer,
        "hidden_activation": 'relu',
        "output_activation": abs_activation,
        # "output_activation": 'linear',
        "kernel_regularization": hidden_regularization,
        "normalizer": preprocessing.Normalization()
    }

    #model = Net(**net_model_config)

    model_config = {"loss": loss,
                    "optimizer": optimizer,
                    "patience": patience,
                    "model_class": DNNNet,
                    "net_model_config": net_model_config,
                    "verbose": True}

    corr_field_config = {'corr_length': 0.1, 'sigma': 1, 'log': True}

    return DNN_2, conv_layer, corr_field_config, model_config


# Build model
class DNNNet(Model):
    def __init__(self, conv_layer, hidden_activation, output_activation, kernel_regularization, normalizer, **kwargs):
        super().__init__(**kwargs)
        #self.normalizer = normalizer
        #self.norm_layer = tf.keras.layers.LayerNormalization(axis=1)
        self.conv1 = Dense(128, activation="relu")
        self.conv2 = Dense(64, activation="relu"),
        self.conv3 = Dense(32, activation="relu"),
        # self.conv2 = conv_layer(128, K=2, activation=hidden_activation, kernel_regularizer=kernel_regularization)
        # self.conv3 = conv_layer(16, K=2, activation=hidden_activation, kernel_regularizer=kernel_regularization)
        # self.conv4 = conv_layer(8,  K=2,activation=hidden_activation, kernel_regularizer=kernel_regularization)
        # self.conv2 = conv_layer(32, K=2, activation=hidden_activation, kernel_regularizer=kernel_regularization)
        # self.conv3 = conv_layer(16, K=2, activation=hidden_activation, kernel_regularizer=kernel_regularization)
        # self.conv3 = conv_layer(8, activation=hidden_activation, kernel_regularizer=kernel_regularization)
        # self.conv4 = conv_layer(4, activation=hidden_activation, kernel_regularizer=kernel_regularization)
        #self.conv3 = conv_layer(64, activation=hidden_activation, kernel_regularizer=kernel_regularization)
        #self.flatten = GlobalSumPool()
        #self.fc1 = Dense(32, activation=hidden_activation)
        self.fc2 = Dense(1)#, activation=output_activation)  # linear activation for output neuron

    def call(self, inputs):
        x = inputs.x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        #x = self.conv4(x)
        #output1 = self.flatten(x)
        #output2 = self.fc1(output1)
        output = self.fc2(x)
        return output


def get_config(data_dir, case=0):
    if case == 7: # mesh size comparison
        data_dir = "/home/martin/Documents/metamodels/data/mesh_size/"
        cl = "cl_0_1_s_1"
        level = 3
        nn_level = 0
        replace_level = False
        #mesh = os.path.join(data_dir, "l_step_1.0_common_files/mesh.msh".format(cl)) #L1, 7s
        #mesh = os.path.join(data_dir, "l_step_0.27232698153315_common_files/mesh.msh".format(cl)) #L2 10.5 s
        mesh = os.path.join(data_dir, "l_step_0.07416198487095663_common_files/mesh.msh".format(cl)) #L3 12s
        #mesh = os.path.join(data_dir, "l_step_0.020196309484414757_common_files/mesh.msh".format(cl)) #L4  22s
        #mesh = os.path.join(data_dir, "l_step_0.0055_common_files/mesh.msh".format(cl)) #L5
        output_dir = os.path.join(data_dir, "{}/L1_{}/test/01_cond_field/output/".format(cl, level))
        hdf_path = os.path.join(data_dir, "{}/L1_{}/mlmc_1.hdf5".format(cl, level))
        mlmc_hdf_path = os.path.join(data_dir, "{}/mlmc_hdf/L1_{}/mlmc_1.hdf5".format(cl, level))
        save_path = os.path.join(data_dir, "{}".format(cl))
        l_0_output_dir = os.path.join(data_dir, "{}/L0_MC/L1_{}/test/01_cond_field/output/".format(cl, level))
        l_0_hdf_path = os.path.join(data_dir, "{}/L0_MC/L1_{}/mlmc_1.hdf5".format(cl, level))
        sampling_info_path = os.path.join(data_dir, "{}/sampling_info".format(cl))
        ref_mlmc_file = os.path.join(data_dir, "{}/L1_benchmark/mlmc_1.hdf5".format(cl))

    elif case == 8 or case == 9:
        data_dir = "/home/martin/Documents/metamodels/data/5_ele/"
        cl = "cl_0_1_s_1"
        if case == 9:
            cl = "cl_0_3_s_4"
        nn_level = 0
        replace_level = False
        # mesh = os.path.join(data_dir, "{}/L5/l_step_0.020196309484414757_common_files/mesh.msh".format(cl))
        mesh = os.path.join(data_dir, "l_step_0.07416198487095663_common_files/mesh.msh".format(cl))
        output_dir = os.path.join(data_dir, "{}/L3_1/test/01_cond_field/output/".format(cl))
        hdf_path = os.path.join(data_dir, "{}/L3_1/mlmc_3.hdf5".format(cl))
        mlmc_hdf_path = os.path.join(data_dir, "{}/L3_2/mlmc_3.hdf5".format(cl))
        save_path = os.path.join(data_dir, "{}".format(cl))
        l_0_output_dir = os.path.join(data_dir, "{}/L1_2/test/01_cond_field/output/".format(cl,nn_level))
        l_0_hdf_path = os.path.join(data_dir, "{}/L1_2/mlmc_1.hdf5".format(cl, nn_level))
        sampling_info_path = os.path.join(data_dir, "{}/sampling_info".format(cl))
        ref_mlmc_file = os.path.join(data_dir, "{}/L1_benchmark/mlmc_1.hdf5".format(cl))

    elif case == 10 or case == 11:
        data_dir = "/home/martin/Documents/metamodels/data/5_ele/"
        cl = "cl_0_1_s_1"
        if case == 11:
            cl = "cl_0_3_s_4"
        nn_level = 0
        replace_level = False
        # mesh = os.path.join(data_dir, "{}/L5/l_step_0.020196309484414757_common_files/mesh.msh".format(cl))
        mesh = os.path.join(data_dir, "l_step_0.07416198487095663_common_files/mesh.msh".format(cl))
        output_dir = os.path.join(data_dir, "{}/L2_1/test/01_cond_field/output/".format(cl))
        hdf_path = os.path.join(data_dir, "{}/L2_1/mlmc_2.hdf5".format(cl))
        mlmc_hdf_path = os.path.join(data_dir, "{}/L2_2/mlmc_2.hdf5".format(cl))
        save_path = os.path.join(data_dir, "{}".format(cl))
        l_0_output_dir = os.path.join(data_dir, "{}/L1_2/test/01_cond_field/output/".format(cl,nn_level))
        l_0_hdf_path = os.path.join(data_dir, "{}/L1_2/mlmc_1.hdf5".format(cl, nn_level))
        sampling_info_path = os.path.join(data_dir, "{}/sampling_info".format(cl))
        ref_mlmc_file = os.path.join(data_dir, "{}/L1_benchmark/mlmc_1.hdf5".format(cl))

    # elif case == 6:
    #     nn_level = 0
    #     replace_level = True
    #     mesh = "/home/martin/Documents/metamodels/data/L1/test/01_cond_field/l_step_0.055_common_files/mesh.msh"
    #     output_dir = "/home/martin/Documents/metamodels/data/L1/test/01_cond_field/output"
    #     hdf_path = "/home/martin/Documents/metamodels/data/L1/test/01_cond_field/mlmc_1.hdf5"
    #
    #     save_path = "/home/martin/Documents/metamodels/data/L1/"
    #     l_0_output_dir = output_dir
    #     l_0_hdf_path = hdf_path
    #     sampling_info_path = "/home/martin/Documents/metamodels/data/L1/test/01_cond_field/"
    #     ref_mlmc_file = "/home/martin/Documents/metamodels/data/1000_ele/cl_0_1_s_1/L1_benchmark/mlmc_1.hdf5"

    return output_dir, hdf_path, l_0_output_dir, l_0_hdf_path, save_path, mesh, sampling_info_path, ref_mlmc_file, replace_level, nn_level, mlmc_hdf_path



def get_arguments(arguments):
    """
    Getting arguments from console
    :param arguments: list of arguments
    :return: namespace
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('work_dir', help='work directory')
    parser.add_argument('data_dir', help='data directory')
    args = parser.parse_args(arguments)
    return args


if __name__ == "__main__":
    args = get_arguments(sys.argv[1:])
    data_dir = args.data_dir
    work_dir = args.work_dir
    case = 7
    #data_dir = "/home/martin/Documents/metamodels/data/1000_ele/"
    output_dir, hdf_path, l_0_output_dir, l_0_hdf_path, save_path, mesh, sampling_info_path,\
    ref_mlmc_file, replace_level, nn_level, mlmc_hdf_path = get_config(data_dir, case)


    gnn, conv_layer, corr_field_config, model_config = get_gnn()

    # print("gnn ", gnn)
    #print("conv layer ", conv_layer)

    machine_learning_model = ("DNN_2_test", run_DNN, False)

    save_path = os.path.join(save_path, machine_learning_model[0])

    if os.path.exists(save_path):
        shutil.rmtree(save_path)

    print("save path ", save_path)
    graph_creation_time = 28#22#159#0#159#66

    config = {'machine_learning_model': machine_learning_model,
              'save_path': save_path,
              'sampling_info_path': sampling_info_path,
              'output_dir': output_dir,
              'nn_hdf_path': hdf_path,
              'mlmc_hdf_path': mlmc_hdf_path,
              'mesh': mesh,
              'l_0_output_dir': l_0_output_dir,
              'l_0_hdf_path': l_0_hdf_path,
              'ref_mlmc_file': ref_mlmc_file,
              'level': nn_level,
              'conv_layer': conv_layer,
              'gnn': gnn,
              'model_config': model_config,
              'replace_level': replace_level,
              'corr_field_config': corr_field_config,
              'n_train_samples': 2000,
              'val_samples_ratio': 0.3,
              'batch_size': 200,
              'epochs': 2000,
              'learning_rate': 0.01,
              'graph_creation_time': graph_creation_time,
              'save_model': False,
              'loss_params': {'moments_class': Legendre_tf, "max_moments": 20, 'loss_max': 0.5, 'quantile': 1e-3}
              }

    statistics(config)

    analyze_statistics(config)

    # save_path = os.path.join(save_path, "SVR")
    # statistics(run_SVR, output_dir, hdf_path, l_0_output_dir, l_0_hdf_path, save_path, mesh, level=nn_level, log=True)

