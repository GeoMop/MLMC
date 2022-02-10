import os
import numpy as np
import warnings
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Run on CPU only
import sys
import shutil
import subprocess
from mlmc.metamodel.analyze_nn import run_GNN, run_SVR, statistics, analyze_statistics, process_results
from mlmc.moments import Legendre_tf, Monomial
from keras.layers import Input
from mlmc.metamodel.flow_task_GNN_2 import GNN
from spektral.layers import GCNConv, GlobalSumPool, ChebConv, GraphSageConv, ARMAConv, GATConv, APPNPConv, GINConv, GeneralConv
from mlmc.metamodel.own_cheb_conv import OwnChebConv
from tensorflow.keras.losses import MeanSquaredError, KLDivergence, MeanAbsoluteError
from mlmc.metamodel.custom_methods import abs_activation, MSE_moments
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import regularizers
from spektral.layers import GlobalSumPool, GlobalMaxPool, GlobalAvgPool
import tensorflow as tf
from mlmc.plot import plots
from tensorflow.keras.layers.experimental import preprocessing
from mlmc.metamodel.analyze_nn import load_statistics, process_data
from scipy import stats
warnings.filterwarnings("ignore", category=DeprecationWarning)


def get_gnn():
    # Parameters
    # conv_layer = GCNConv
    conv_layer = ChebConv  # Seems better than GCNConv, good distribution of predictions
    conv_layer = OwnChebConv
    # conv_layer = GraphSageConv  # Seems better than ChebConv, good loss but very narrow distribution of predictions
    # # conv_layer = ARMAConv  # Seems worse than GraphSageConv
    # conv_layer = GATConv  # Slow and not better than GraphSageConv
    # # conv_layer = APPNPConv  # Not bad but worse than GraphSageConv
    # # conv_layer = GINConv  # it is comparable to APPNPConv
    # act_func = "relu"  # "tanh"#"elu"

    loss = MeanSquaredError()  # var_loss_function#
    #loss = MSE_moments
    # loss = MeanAbsoluteError()
    # loss = MeanSquaredLogarithmicError()
    # loss = KLDivergence()
    # loss = total_loss_function
    optimizer = tf.optimizers.Adam(learning_rate=0.001)
    patience = 150
    hidden_regularization = None  # l2(2e-10)

    net_model_config = {
                 "conv_layer": conv_layer,
                 "hidden_activation": 'relu',
                 "output_activation": abs_activation,
                 #"output_activation": 'linear',
                 "kernel_regularization": hidden_regularization,
                 "normalizer": preprocessing.Normalization()
                 }

    #model = Net(**net_model_config)

    model_config = {"loss": loss,
                    "optimizer": optimizer,
                    "patience": patience,
                    "model_class": Net,
                    "net_model_config": net_model_config,
                    "verbose": True}

    corr_field_config = {'02_conc': True, 'corr_length': 0.1, 'sigma': 1, 'log': True}

    return GNN, conv_layer, corr_field_config, model_config


class Net(Model):
    def __init__(self, conv_layer, hidden_activation, output_activation, kernel_regularization, normalizer,
                 **kwargs):
        super().__init__(**kwargs)

        # T19
        self._conv_layers = [
            conv_layer(8, K=4, activation=hidden_activation, kernel_regularizer=kernel_regularization)]
        self.flatten = GlobalSumPool()

        self._dense_layers = [Dense(32, activation=hidden_activation), Dense(16, activation=hidden_activation),
                              Dense(1)]

        # T34
        # self._conv_layers = [
        #     conv_layer(8, K=4, activation=hidden_activation, kernel_regularizer=kernel_regularization)]  # ,n)
        # self.flatten = GlobalSumPool()
        #
        # self._dense_layers = [Dense(256, activation=hidden_activation), Dense(128, activation=hidden_activation),
        #                       Dense(1)]


    def call(self, inputs):
        x, a = inputs

        for c_layer in self._conv_layers:
            x = c_layer([x, a])

        output = self.flatten(x)

        # print("output shape ", output.shape)
        # exit()1

        for d_layer in self._dense_layers:
            output = d_layer(output)

        return output


def get_config(data_dir, case=0):

    if case == 12:  # mesh size comparison
        data_dir = "/home/martin/Documents/metamodels/data/mesh_size/02_conc_cond/"
        # cl = "cl_0_1_s_1"
        level = 3
        nn_level = 0
        replace_level = False
        mesh = os.path.join(data_dir, "l_step_1.0_common_files/repo.msh")  # L1, 7s
        #mesh = os.path.join(data_dir, "l_step_0.3760603093086394_common_files/repo.msh") #L2 10.5 s
        mesh = os.path.join(data_dir, "l_step_0.1414213562373095_common_files/repo.msh") #L3 12s
        #mesh = os.path.join(data_dir, "l_step_0.053182958969449884_common_files/repo.msh")  # L4
        output_dir = os.path.join(data_dir, "L1_{}/test/02_conc/output/".format(level))
        hdf_path = os.path.join(data_dir, "L1_{}/mlmc_1.hdf5".format(level))
        mlmc_hdf_path = os.path.join(data_dir, "mlmc_hdf/L1_{}/mlmc_1.hdf5".format(level))
        save_path = data_dir
        l_0_output_dir = os.path.join(data_dir, "L0_MC/L1_{}/test/02_conc/output/".format(level))
        l_0_hdf_path = os.path.join(data_dir, "L0_MC/L1_{}/mlmc_1.hdf5".format(level))
        sampling_info_path = os.path.join(data_dir, "sampling_info")

        #ref_mlmc_file = "/home/martin/Documents/metamodels/data/mesh_size/02_conc_cond/L1_3/mlmc_1.hdf5"
        #ref_mlmc_file = "/home/martin/Documents/metamodels/data/mesh_size/02_conc_cond/L1_1/mlmc_1.hdf5"
        ref_mlmc_file = os.path.join(data_dir, "L1_benchmark/mlmc_1.hdf5")

        feature_names = [['conductivity_top', 'conductivity_bot', 'conductivity_repo']]

    if case == "L2":  # mesh size comparison
        data_dir = "/home/martin/Documents/metamodels/data/mesh_size/02_conc_cond/"
        level = 2
        nn_level = 0
        replace_level = False
        mesh = os.path.join(data_dir, "l_step_0.3760603093086394_common_files/repo.msh")
        graph_creation_time = 35

        output_dir = os.path.join(data_dir, "L1_{}/test/02_conc/output/".format(level))
        hdf_path = os.path.join(data_dir, "L1_{}/mlmc_1.hdf5".format(level))
        mlmc_hdf_path = os.path.join(data_dir, "mlmc_hdf/L1_{}/mlmc_1.hdf5".format(level))
        save_path = data_dir
        l_0_output_dir = os.path.join(data_dir, "L0_MC/L1_{}/test/02_conc/output/".format(level))
        l_0_hdf_path = os.path.join(data_dir, "L0_MC/L1_{}/mlmc_1.hdf5".format(level))
        sampling_info_path = os.path.join(data_dir, "sampling_info")

        #ref_mlmc_file = "/home/martin/Documents/metamodels/data/mesh_size/02_conc_cond/L1_3/mlmc_1.hdf5"
        #ref_mlmc_file = "/home/martin/Documents/metamodels/data/mesh_size/02_conc_cond/L1_1/mlmc_1.hdf5"
        ref_mlmc_file = os.path.join(data_dir, "L1_benchmark/mlmc_1.hdf5")

        feature_names = [['conductivity_top', 'conductivity_bot', 'conductivity_repo']]

    if case == 'L1':  # mesh size comparison
        data_dir = "/home/martin/Documents/metamodels/data/mesh_size/02_conc_cond/"
        level = 1
        nn_level = 0
        replace_level = False
        mesh = os.path.join(data_dir, "l_step_1.0_common_files/repo.msh")
        graph_creation_time = 68
        output_dir = os.path.join(data_dir, "L1_{}/test/02_conc/output/".format(level))
        hdf_path = os.path.join(data_dir, "L1_{}/mlmc_1.hdf5".format(level))
        mlmc_hdf_path = os.path.join(data_dir, "mlmc_hdf/L1_{}/mlmc_1.hdf5".format(level))
        save_path = data_dir
        l_0_output_dir = os.path.join(data_dir, "L0_MC/L1_{}/test/02_conc/output/".format(level))
        l_0_hdf_path = os.path.join(data_dir, "L0_MC/L1_{}/mlmc_1.hdf5".format(level))
        sampling_info_path = os.path.join(data_dir, "sampling_info")

        #ref_mlmc_file = "/home/martin/Documents/metamodels/data/mesh_size/02_conc_cond/L1_3/mlmc_1.hdf5"
        #ref_mlmc_file = "/home/martin/Documents/metamodels/data/mesh_size/02_conc_cond/L1_1/mlmc_1.hdf5"
        ref_mlmc_file = os.path.join(data_dir, "L1_benchmark/mlmc_1.hdf5")

        feature_names = [['conductivity_top', 'conductivity_bot', 'conductivity_repo']]

    if case == 'L3':  # mesh size comparison
        data_dir = "/home/martin/Documents/metamodels/data/mesh_size/02_conc_cond/"
        level = 3
        nn_level = 0
        replace_level = False
        mesh = os.path.join(data_dir, "l_step_0.1414213562373095_common_files/repo.msh")  # L3 12s
        graph_creation_time = 250
        output_dir = os.path.join(data_dir, "L1_{}/test/02_conc/output/".format(level))
        hdf_path = os.path.join(data_dir, "L1_{}/mlmc_1.hdf5".format(level))
        mlmc_hdf_path = os.path.join(data_dir, "mlmc_hdf/L1_{}/mlmc_1.hdf5".format(level))
        save_path = data_dir
        l_0_output_dir = os.path.join(data_dir, "L0_MC/L1_{}/test/02_conc/output/".format(level))
        l_0_hdf_path = os.path.join(data_dir, "L0_MC/L1_{}/mlmc_1.hdf5".format(level))
        sampling_info_path = os.path.join(data_dir, "sampling_info")

        # output_dir = l_0_output_dir
        # hdf_path = l_0_hdf_path

        #ref_mlmc_file = "/home/martin/Documents/metamodels/data/mesh_size/02_conc_cond/L1_3/mlmc_1.hdf5"
        #ref_mlmc_file = "/home/martin/Documents/metamodels/data/mesh_size/02_conc_cond/L1_1/mlmc_1.hdf5"
        ref_mlmc_file = os.path.join(data_dir, "L1_benchmark/mlmc_1.hdf5")

        feature_names = [['conductivity_top', 'conductivity_bot', 'conductivity_repo']]

    elif case == 429:  # mesh size comparison
        data_dir = "/home/martin/Documents/metamodels/data/mesh_size/02_conc_cond/"
        # cl = "cl_0_1_s_1"
        level = "3_429"
        nn_level = 0
        replace_level = False
        #mesh = os.path.join(data_dir, "l_step_1.0_common_files/repo.msh")  # L1, 7s
        #mesh = os.path.join(data_dir, "l_step_0.3760603093086394_common_files/repo.msh") #L2 10.5 s
        mesh = os.path.join(data_dir, "l_step_0.15_common_files/repo.msh") #L3 12s
        #mesh = os.path.join(data_dir, "l_step_0.053182958969449884_common_files/repo.msh")  # L4
        output_dir = os.path.join(data_dir, "L1_{}/test/02_conc/output/".format(level))
        hdf_path = os.path.join(data_dir, "L1_{}/mlmc_1.hdf5".format(level))
        mlmc_hdf_path = None#os.path.join(data_dir, "mlmc_hdf/L1_{}/mlmc_1.hdf5".format(level))
        save_path = data_dir
        l_0_output_dir = os.path.join(data_dir, "L0_MC/L1_{}/test/02_conc/output/".format(level))
        l_0_hdf_path = os.path.join(data_dir, "L0_MC/L1_{}/mlmc_1.hdf5".format(level))
        sampling_info_path = os.path.join(data_dir, "sampling_info")

        graph_creation_time = 241

        #ref_mlmc_file = "/home/martin/Documents/metamodels/data/mesh_size/02_conc_cond/L1_3/mlmc_1.hdf5"
        #ref_mlmc_file = "/home/martin/Documents/metamodels/data/mesh_size/02_conc_cond/L1_1/mlmc_1.hdf5"
        ref_mlmc_file = os.path.join(data_dir, "L1_benchmark/mlmc_1.hdf5")

        feature_names = [['conductivity_top', 'conductivity_bot', 'conductivity_repo']]

    elif case == 521:  # mesh size comparison
        data_dir = "/home/martin/Documents/metamodels/data/mesh_size/02_conc_cond/"
        # cl = "cl_0_1_s_1"
        level = "3_521"
        nn_level = 0
        replace_level = False
        #mesh = os.path.join(data_dir, "l_step_1.0_common_files/repo.msh")  # L1, 7s
        #mesh = os.path.join(data_dir, "l_step_0.3760603093086394_common_files/repo.msh") #L2 10.5 s
        mesh = os.path.join(data_dir, "l_step_0.13_common_files/repo.msh") #L3 12s
        #mesh = os.path.join(data_dir, "l_step_0.053182958969449884_common_files/repo.msh")  # L4
        output_dir = os.path.join(data_dir, "L1_{}/test/02_conc/output/".format(level))
        hdf_path = os.path.join(data_dir, "L1_{}/mlmc_1.hdf5".format(level))
        mlmc_hdf_path = None#os.path.join(data_dir, "mlmc_hdf/L1_{}/mlmc_1.hdf5".format(level))
        save_path = data_dir
        l_0_output_dir = os.path.join(data_dir, "L0_MC/L1_{}/test/02_conc/output/".format(level))
        l_0_hdf_path = os.path.join(data_dir, "L0_MC/L1_{}/mlmc_1.hdf5".format(level))
        sampling_info_path = os.path.join(data_dir, "sampling_info")

        #ref_mlmc_file = "/home/martin/Documents/metamodels/data/mesh_size/02_conc_cond/L1_3/mlmc_1.hdf5"
        #ref_mlmc_file = "/home/martin/Documents/metamodels/data/mesh_size/02_conc_cond/L1_1/mlmc_1.hdf5"
        ref_mlmc_file = os.path.join(data_dir, "L1_benchmark/mlmc_1.hdf5")

        feature_names = [['conductivity_top', 'conductivity_bot', 'conductivity_repo']]

        graph_creation_time = 285

    elif case == "case_2":  # mesh size comparison
        data_dir = "/home/martin/Documents/metamodels/data/mesh_size/02_conc_cond/case_2"
        # cl = "cl_0_1_s_1"
        level = "3"
        nn_level = 0
        replace_level = False
        # mesh = os.path.join(data_dir, "l_step_1.0_common_files/repo.msh")  # L1, 7s
        # mesh = os.path.join(data_dir, "l_step_0.3760603093086394_common_files/repo.msh") #L2 10.5 s
        mesh = os.path.join(data_dir, "l_step_0.1414213562373095_common_files/repo.msh") #L3 12s
        # mesh = os.path.join(data_dir, "l_step_0.053182958969449884_common_files/repo.msh")  # L4
        output_dir = os.path.join(data_dir, "L1_{}/test/02_conc/output/".format(level))
        hdf_path = os.path.join(data_dir, "L1_{}/mlmc_1.hdf5".format(level))
        mlmc_hdf_path = None  # os.path.join(data_dir, "mlmc_hdf/L1_{}/mlmc_1.hdf5".format(level))
        save_path = data_dir
        l_0_output_dir = os.path.join(data_dir, "L0_MC/L1_{}/test/02_conc/output/".format(level))
        l_0_hdf_path = os.path.join(data_dir, "L0_MC/L1_{}/mlmc_1.hdf5".format(level))
        sampling_info_path = os.path.join(data_dir, "sampling_info")

        # ref_mlmc_file = "/home/martin/Documents/metamodels/data/mesh_size/02_conc_cond/L1_3/mlmc_1.hdf5"
        # ref_mlmc_file = "/home/martin/Documents/metamodels/data/mesh_size/02_conc_cond/L1_1/mlmc_1.hdf5"
        ref_mlmc_file = "/home/martin/Documents/metamodels/data/mesh_size/02_conc_cond/L1_benchmark/mlmc_1.hdf5"
        #ref_mlmc_file = os.path.join(data_dir, "L1_benchmark/mlmc_1.hdf5")

        feature_names = [['conductivity_top', 'conductivity_bot', 'conductivity_repo']]

        graph_creation_time = 670  # case_2

    elif case == "case_3":  # mesh size comparison
        data_dir = "/home/martin/Documents/metamodels/data/mesh_size/02_conc_cond/case_3"
        # cl = "cl_0_1_s_1"
        level = "3"
        nn_level = 0
        replace_level = False
        # mesh = os.path.join(data_dir, "l_step_1.0_common_files/repo.msh")  # L1, 7s
        # mesh = os.path.join(data_dir, "l_step_0.3760603093086394_common_files/repo.msh") #L2 10.5 s
        mesh = os.path.join(data_dir, "l_step_0.1414213562373095_common_files/repo.msh") #L3 12s
        # mesh = os.path.join(data_dir, "l_step_0.053182958969449884_common_files/repo.msh")  # L4
        output_dir = os.path.join(data_dir, "L1_{}/test/02_conc/output/".format(level))
        hdf_path = os.path.join(data_dir, "L1_{}/mlmc_1.hdf5".format(level))
        mlmc_hdf_path = None  # os.path.join(data_dir, "mlmc_hdf/L1_{}/mlmc_1.hdf5".format(level))
        save_path = data_dir
        l_0_output_dir = os.path.join(data_dir, "L0_MC/L1_{}/test/02_conc/output/".format(level))
        l_0_hdf_path = os.path.join(data_dir, "L0_MC/L1_{}/mlmc_1.hdf5".format(level))
        sampling_info_path = os.path.join(data_dir, "sampling_info")

        # ref_mlmc_file = "/home/martin/Documents/metamodels/data/mesh_size/02_conc_cond/L1_3/mlmc_1.hdf5"
        # ref_mlmc_file = "/home/martin/Documents/metamodels/data/mesh_size/02_conc_cond/L1_1/mlmc_1.hdf5"
        #ref_mlmc_file = os.path.join(data_dir, "L1_benchmark/mlmc_1.hdf5")
        ref_mlmc_file = "/home/martin/Documents/metamodels/data/mesh_size/02_conc_cond/L1_benchmark/mlmc_1.hdf5"

        feature_names = [['conductivity_top', 'conductivity_bot', 'conductivity_repo']]

        graph_creation_time = 601  # case_3

    elif case == "case_4":  # mesh size comparison
        data_dir = "/home/martin/Documents/metamodels/data/mesh_size/02_conc_cond/case_4"
        # cl = "cl_0_1_s_1"
        level = "3"
        nn_level = 0
        replace_level = False
        # mesh = os.path.join(data_dir, "l_step_1.0_common_files/repo.msh")  # L1, 7s
        # mesh = os.path.join(data_dir, "l_step_0.3760603093086394_common_files/repo.msh") #L2 10.5 s
        mesh = os.path.join(data_dir, "l_step_0.1414213562373095_common_files/repo.msh") #L3 12s
        # mesh = os.path.join(data_dir, "l_step_0.053182958969449884_common_files/repo.msh")  # L4
        output_dir = os.path.join(data_dir, "L1_{}/test/02_conc/output/".format(level))
        hdf_path = os.path.join(data_dir, "L1_{}/mlmc_1.hdf5".format(level))
        mlmc_hdf_path = os.path.join(data_dir, "mlmc_hdf/L1_{}/mlmc_1.hdf5".format(level))
        save_path = data_dir
        l_0_output_dir = os.path.join(data_dir, "L0_MC/L1_{}/test/02_conc/output/".format(level))
        l_0_hdf_path = os.path.join(data_dir, "L0_MC/L1_{}/mlmc_1.hdf5".format(level))
        sampling_info_path = os.path.join(data_dir, "sampling_info")

        # ref_mlmc_file = "/home/martin/Documents/metamodels/data/mesh_size/02_conc_cond/L1_3/mlmc_1.hdf5"
        # ref_mlmc_file = "/home/martin/Documents/metamodels/data/mesh_size/02_conc_cond/L1_1/mlmc_1.hdf5"
        #ref_mlmc_file = os.path.join(data_dir, "L1_benchmark/mlmc_1.hdf5")
        ref_mlmc_file = mlmc_hdf_path #"/home/martin/Documents/metamodels/data/mesh_size/02_conc_cond/L1_benchmark/mlmc_1.hdf5"

        feature_names = [['conductivity_top', 'conductivity_bot', 'conductivity_repo']]

        graph_creation_time = 608  # case_4

    elif case == "case_5":  # mesh size comparison
        data_dir = "/home/martin/Documents/metamodels/data/mesh_size/02_conc_cond/case_5"
        # cl = "cl_0_1_s_1"
        level = "3"
        nn_level = 0
        replace_level = False
        # mesh = os.path.join(data_dir, "l_step_1.0_common_files/repo.msh")  # L1, 7s
        # mesh = os.path.join(data_dir, "l_step_0.3760603093086394_common_files/repo.msh") #L2 10.5 s
        mesh = os.path.join(data_dir, "l_step_0.1414213562373095_common_files/repo.msh") #L3 12s
        # mesh = os.path.join(data_dir, "l_step_0.053182958969449884_common_files/repo.msh")  # L4
        output_dir = os.path.join(data_dir, "L1_{}/test/02_conc/output/".format(level))
        hdf_path = os.path.join(data_dir, "L1_{}/mlmc_1.hdf5".format(level))
        mlmc_hdf_path = None  # os.path.join(data_dir, "mlmc_hdf/L1_{}/mlmc_1.hdf5".format(level))
        save_path = data_dir
        l_0_output_dir = os.path.join(data_dir, "L0_MC/L1_{}/test/02_conc/output/".format(level))
        l_0_hdf_path = os.path.join(data_dir, "L0_MC/L1_{}/mlmc_1.hdf5".format(level))
        sampling_info_path = os.path.join(data_dir, "sampling_info")

        # ref_mlmc_file = "/home/martin/Documents/metamodels/data/mesh_size/02_conc_cond/L1_3/mlmc_1.hdf5"
        # ref_mlmc_file = "/home/martin/Documents/metamodels/data/mesh_size/02_conc_cond/L1_1/mlmc_1.hdf5"
        #ref_mlmc_file = os.path.join(data_dir, "L1_benchmark/mlmc_1.hdf5")
        ref_mlmc_file = "/home/martin/Documents/metamodels/data/mesh_size/02_conc_cond/L1_benchmark/mlmc_1.hdf5"

        feature_names = [['conductivity_top', 'conductivity_bot', 'conductivity_repo']]
        graph_creation_time = 587  # case_5

    return output_dir, hdf_path, l_0_output_dir, l_0_hdf_path, save_path, mesh, sampling_info_path, ref_mlmc_file,\
           replace_level, nn_level, mlmc_hdf_path, feature_names, graph_creation_time


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


# def set_weights(new_model, old_model):
#     for new_conv_layer, old_conv_layer in zip(new_model._conv_layers, old_model._conv_layers):
#         new_conv_layer.kernel = old_conv_layer.kernel
#
#     print(old_conv_layer.kernel.numpy().shape)
#     input_imgs = Input(shape=(None, 108, 1))
#     print("old_model.flatten.weights", old_model.flatten().weights)
#
#     for new_dense_layer, old_dense_layer in zip(new_model._dense_layers, old_model._dense_layers):
#
#         print("old_dense_layer.get_weights() shape ", old_dense_layer.get_weights()[0].shape)
#         print("old_dense_layer.get_weights() shape ", old_dense_layer.get_weights()[1].shape)
#         input_imgs = Input(shape=(None, 108, 1))
#         new_dense_layer(input_imgs)
#         # model = Model(inputs=input_imgs, outputs=encoded)
#         # dense_layer.set_weights(weights)
#
#         print("new dense layer weights ", new_dense_layer.weights)
#         new_dense_layer.set_weights(old_dense_layer.get_weights())


if __name__ == "__main__":
    #######################
    # Load trained model  #
    #######################
    #machine_learning_model = ("L1_3_02_conc_cond_log_output_mult_T19_case_1", run_GNN, False)
    machine_learning_model = ("L1_3_T19_case_1_out_log_scale", run_GNN, False)
    save_path = os.path.join("/home/martin/Documents/metamodels/data/mesh_size/02_conc_cond/", machine_learning_model[0])
    data_dict = load_statistics(save_path)
    #data_dict = process_data(data_dict)
    #for i in range(len(data_dict["test_targets"])):

    list_train_MSE, list_test_MSE, list_all_train_RSE, list_all_test_RSE, list_nn_total_time, list_mlmc_total_time,\
    list_kl_mlmc_all, list_kl_nn_all, list_learning_times = [], [], [], [], [], [], [], [], []

    for index, model in enumerate(data_dict["model"][:2]):

        # newInput = Input(batch_shape=(None, 108, 1))
        # newOutputs = model(newInput)
        # newModel = Model(newInput, newOutputs)
        #
        # if type(model._conv_layers[0]).__name__ == 'OwnChebConv':
        #     conv_layer = OwnChebConv
        #
        # print("conv layer ", conv_layer)

        #####################
        ## New case config ##
        #####################
        args = get_arguments(sys.argv[1:])
        data_dir = args.data_dir
        work_dir = args.work_dir
        # case = "L2"
        # case = 521
        case = "case_3"
        #data_dir = "/home/martin/Documents/metamodels/data/1000_ele/"
        output_dir, hdf_path, l_0_output_dir, l_0_hdf_path, save_path, mesh, sampling_info_path, ref_mlmc_file,\
        replace_level, nn_level, mlmc_hdf_path, feature_names, graph_creation_time = get_config(data_dir, case)

        machine_learning_model = ("L1_2_02_conc_cond_log_output_mult_T19_case_1_trained_{}_{}".format(case, index), run_GNN, False)
        save_path = os.path.join(save_path, machine_learning_model[0])

        if os.path.exists(save_path):
            shutil.rmtree(save_path)

        #corr_field_config = {'02_conc': True, 'corr_length': 0.1, 'sigma': 1, 'log': True}

        gnn, conv_layer, corr_field_config, model_config = get_gnn()

        #gnn_au = gnn(**model_config)
        #
        # print("gnn au model ", gnn_au._model)
        # print("model ", model)

        #set_weights(gnn_au._model, model)

        # dataset_config = {"features_normalization": False,
        #                   "calc_output_mult_factor": True,
        #                   "output_mult_factor": 1,
        #                   "features_mult_factor": 1,
        #                   "features_log": False,
        #                   "output_log": True
        #                   }

        dataset_config = {"first_log_features": False,
                          "first_log_output": True,
                          "features_normalization": False,
                          "output_normalization": False,
                          "calc_output_mult_factor": False,
                          "output_mult_factor": 1,
                          "features_mult_factor": 1,
                          "features_scale": False,
                          "output_scale": True,
                          }

        config = {'machine_learning_model': machine_learning_model,
                  'save_path': save_path,
                  'output_dir': output_dir,
                  'hdf_path': hdf_path,
                  'mlmc_hdf_path': mlmc_hdf_path,
                  'mesh': mesh,
                  'l_0_output_dir': l_0_output_dir,
                  'l_0_hdf_path': l_0_hdf_path,
                  'sampling_info_path': sampling_info_path,
                  'ref_mlmc_file': ref_mlmc_file,
                  'level': nn_level,
                   'conv_layer': conv_layer,
                   'gnn': gnn,
                   'model_config': model_config,
                  'replace_level': replace_level,
                  'corr_field_config': corr_field_config,
                  'n_train_samples': 2000,
                  'val_samples_ratio': 0.2,
                  'batch_size': 200,
                  'epochs': 1,
                  'learning_rate': 0.001,
                  'graph_creation_time': graph_creation_time,
                  'save_model': True,
                  'feature_names': feature_names,
                  "train_model": False,
                  "set_model": model,
                  "dataset_config": dataset_config
                  }

        model_title, mch_l_model, log = config['machine_learning_model']

        train_MSE, test_MSE, all_train_RSE, all_test_RSE, nn_total_time, mlmc_total_time, kl_mlmc_all, kl_nn_all, \
        learning_times = statistics(config)

        list_train_MSE.append(train_MSE)
        list_test_MSE.append(test_MSE)
        list_all_train_RSE.append(all_train_RSE)
        list_all_test_RSE.append(all_test_RSE)
        list_nn_total_time.append(nn_total_time)
        list_mlmc_total_time.append(mlmc_total_time)
        list_kl_mlmc_all.append(kl_mlmc_all)
        list_kl_nn_all.append(kl_nn_all)
        list_learning_times.append(learning_times)

    print("learning time ", list_learning_times)
    print("mean learning time ", np.mean(list_learning_times))
    print("max learning time ", np.max(list_learning_times))


    print("############# OUTPUT ################")
    print("len(train MSE) ", len(list_train_MSE))
    print("train MSE ", np.mean(list_all_train_RSE))
    # print("train MSE sqrt var", np.sqrt(np.var(train_MSE)))
    # print("train MSE std", np.std(train_MSE))

    # output_mult_factor = 1437603411
    # print("orig train MSE ", train_MSE)
    # train_MSE = np.array(train_MSE) * output_mult_factor
    # print("train MSE ", train_MSE)
    # test_MSE = np.array(test_MSE) * output_mult_factor

    print("train MSE ", list_train_MSE)
    print("stats.sem(train_MSE) ", stats.sem(list_train_MSE))
    print("test MSE ", np.mean(list_test_MSE))
    print("test MSE ", list_test_MSE)
    print("stats.sem(test_MSE) ", stats.sem(list_test_MSE))
    # print("test MSE std", np.sqrt(np.var(test_MSE)))
    print("train RSE ", np.mean(list_all_train_RSE))
    print("test RSE ", np.mean(list_all_test_RSE))

    print("nn total time ", list_nn_total_time)
    print("mlmc total time ", list_mlmc_total_time)

    print("KL mlmc ", np.mean(list_kl_mlmc_all))
    print("KL nn ", np.mean(list_kl_nn_all))

    print("mean learning time ", np.mean(list_learning_times))
    print("max learning time ", np.max(list_learning_times))

    print("######################################")

    #analyze_statistics(config)

    # save_path = os.path.join(save_path, "SVR")
    # statistics(run_SVR, output_dir, hdf_path, l_0_output_dir, l_0_hdf_path, save_path, mesh, level=nn_level, log=True)

