import os
import numpy as np
import warnings
import logging
logging.getLogger('tensorflow').disabled = True
logging.getLogger('absl').disabled = True

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings('ignore')
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Run on CPU only
import sys
import shutil
import subprocess

warnings.simplefilter("ignore")
import tensorflow as tf
from mlmc.metamodel.analyze_nn import run_GNN, run_CNN, run_SVR, statistics, analyze_statistics, process_results, run_DeepMap
from mlmc.moments import Legendre_tf, Monomial
from mlmc.metamodel.flow_task_CNN import CNN
from spektral.layers import GCNConv, GlobalSumPool, ChebConv, GraphSageConv, ARMAConv, GATConv, APPNPConv, GINConv, GeneralConv
from mlmc.metamodel.own_cheb_conv import OwnChebConv
from tensorflow.keras.losses import MeanSquaredError, KLDivergence, MeanAbsoluteError
from mlmc.metamodel.custom_methods import abs_activation, MSE_moments
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras import regularizers
from spektral.layers import GlobalSumPool, GlobalMaxPool, GlobalAvgPool, DiffPool, TopKPool, SAGPool, DMoNPool, MinCutPool, SparseDropout
#from mlmc.metamodel.own_diffpool import DiffPool
from spektral.utils.sparse import sp_matrix_to_sp_tensor
from spektral.layers.pooling.global_pool import GlobalPool
from mlmc.plot import plots
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.optimizers.schedules import ExponentialDecay


def get_cnn():
    # Parameters
    # conv_layer = GCNConv
    conv_layer = ChebConv  # Seems better than GCNConv, good distribution of predictions
    #conv_layer = Conv2D
    # conv_layer = GraphSageConv  # Seems better than ChebConv, good loss but very narrow distribution of predictions
    # # conv_layer = ARMAConv  # Seems worse than GraphSageConv
    # conv_layer = GATConv  # Slow and not better than GraphSageConv
    # # conv_layer = APPNPConv  # Not bad but worse than GraphSageConv
    # # conv_layer = GINConv  # it is comparable to APPNPConv
    # act_func = "relu"  # "tanh"#"elu"
    loss = MeanSquaredError()  # var_loss_function#
    # loss = MSE_moments
    # loss = MeanAbsoluteError()
    # loss = MeanSquaredLogarithmicError()
    # loss = KLDivergence()
    # loss = total_loss_function
    optimizer = tf.optimizers.Adam(learning_rate=0.001)
    patience = 1000000
    hidden_regularization = None  # l2(2e-10)

    net_model_config = {
        "conv_layer": conv_layer,
        "hidden_activation": 'relu',
        "output_activation": abs_activation,
        # "output_activation": 'linear',
        "kernel_regularization": hidden_regularization,
        "normalizer": preprocessing.Normalization()
    }

    #model = NetCNN(**net_model_config)

    model_config = {"loss": loss,
                    "optimizer": optimizer,
                    "patience": patience,
                    "model_class": NetCNN,
                    "net_model_config": net_model_config,
                    "verbose": True}

    corr_field_config = {'02_conc': False, 'corr_length': 0.1, 'sigma': 1, 'log': True}

    return CNN, conv_layer, corr_field_config, model_config


class NetCNN(Model):
    def __init__(self, conv_layer, hidden_activation, output_activation, kernel_regularization, normalizer,
                 **kwargs):
        super().__init__(**kwargs)

        dropout_rate = 0.5


        # self.normalizer = normalizer
        # self.norm_layer = tf.keras.layers.LayerNormalization(axis=1)
        # self.normalizer = normalizer
        # self.norm_layer = tf.keras.layers.LayerNormalization(axis=1)
        self._conv_layers = [Conv2D(16, (3, 3), activation=hidden_activation, input_shape=(512, 512, 1)),
                             MaxPooling2D((2, 2)),
                             Conv2D(32, (3, 3), activation=hidden_activation),
                             MaxPooling2D((2, 2)),
                             #Conv2D(64, (3, 3), activation=hidden_activation),
                             ]
        # conv_layer(64, K=1, activation=hidden_activation, kernel_regularizer=kernel_regularization)]
        # self.conv3 = conv_layer(32, K=1, activation=hidden_activation, kernel_regularizer=kernel_regularization)
        # self.conv4 = conv_layer(32, K=1, activation=hidden_activation, kernel_regularizer=kernel_regularization)
        # self.conv2 = conv_layer(32, K=2, activation=hidden_activation, kernel_regularizer=kernel_regularization)
        # self.conv3 = conv_layer(16, K=2, activation=hidden_activation, kernel_regularizer=kernel_regularization)
        # self.conv3 = conv_layer(8, activation=hidden_activation, kernel_regularizer=kernel_regularization)
        # self.conv4 = conv_layer(4, activation=hidden_activation, kernel_regularizer=kernel_regularization)
        # self.conv3 = conv_layer(64, activation=hidden_activation, kernel_regularizer=kernel_regularization)
        self.flatten = Flatten()

        #self.dropout = Dropout(dropout_rate, input_shape=(200, 50, 128))

        # self.flatten = GlobalAvgPool()
        # self._submodel = Sequential()
        # self._dense_layers = [Dense(64, activation=hidden_activation), Dense(32, activation=hidden_activation),
        #                       Dense(1)]

        self._dense_layers = [Dense(64, activation=hidden_activation), Dense(32, activation=hidden_activation),
                              Dense(1)]

        #self._dense_layers = [Dense(1)]

        # for d_layer in self._dense_layers:
        #     self._submodel.add(d_layer)
        # self.fc1 = Dense(32, activation=hidden_activation)
       #, activation=output_activation)  # linear activation for output neuron

    def call(self, inputs):
        x = inputs

        for c_layer in self._conv_layers:
            #print("type c layer ", type(c_layer))
            if isinstance(c_layer, MaxPooling2D):
                x = c_layer(x)
                #print("x " ,x)
                #print("a " ,a)
                #print("diff pool x shape ", x.shape)

                #print("np.array(x).shape ", np.array(x).shape)
            else:
                #print("cov layer type ", type(c_layer))
                #print("x.shape ", x.shape)
                x = c_layer(x)
                #print("x before dropout ", x.shape)
                # print("num zeros ", np.count_nonzero(x == 0))
                #x = self.dropout(x, training=True)
                # print("after dropout x", np.array(x))
                # print("num zeros ", np.count_nonzero(x == 0))

        #print("x.shape ", x.shape)
        output = self.flatten(x)
        # print("output flatten shape ", output.shape)
        # exit()

        for d_layer in self._dense_layers:
            output = d_layer(output)
            #output = self.dropout(output, training=True)
        return output


def get_config(data_dir, case=0):
    feature_names = [['conductivity']]
    graph_creation_time = -1

    predict_dir, predict_hdf = None, None

    if case == 0:
        cl = "cl_0_3_s_4"
        nn_level = 0
        replace_level = False
        mesh = os.path.join(data_dir, "l_step_0.055_common_files/mesh.msh")
        output_dir = os.path.join(data_dir, "{}/L5/test/01_cond_field/output/".format(cl))
        hdf_path = os.path.join(data_dir, "{}/L5/mlmc_5.hdf5".format(cl))
        save_path = os.path.join(data_dir, "{}".format(cl))
        l_0_output_dir = os.path.join(data_dir, "{}/L{}/test/01_cond_field/output/".format(cl, nn_level + 1))
        l_0_hdf_path = os.path.join(data_dir, "{}/L{}/mlmc_{}.hdf5".format(cl, nn_level + 1, nn_level + 1))
        sampling_info_path = os.path.join(data_dir, "{}/sampling_info".format(cl))
        ref_mlmc_file = os.path.join(data_dir, "{}/L1_benchmark/mlmc_1.hdf5".format(cl))

    elif case == 1:
        cl = "cl_0_1_s_1"
        nn_level = 1
        replace_level = False
        mesh = os.path.join(data_dir, "l_step_0.027624156655057155_common_files/mesh.msh")
        output_dir = os.path.join(data_dir, "{}/L5/test/01_cond_field/output/".format(cl))
        hdf_path = os.path.join(data_dir, "{}/L5/mlmc_5.hdf5".format(cl))
        save_path = os.path.join(data_dir, "{}".format(cl))
        l_0_output_dir = os.path.join(data_dir, "{}/L{}/test/01_cond_field/output/".format(cl, nn_level + 1))
        l_0_hdf_path = os.path.join(data_dir, "{}/L{}/mlmc_{}.hdf5".format(cl, nn_level + 1, nn_level))
        sampling_info_path = os.path.join(data_dir, "{}/sampling_info".format(cl))
        ref_mlmc_file = os.path.join(data_dir, "{}/L1_benchmark/mlmc_1.hdf5".format(cl))

    elif case == 2:
        data_dir = "/home/martin/Documents/metamodels/data/5_ele/"
        cl = "cl_0_1_s_1"
        nn_level = 3
        replace_level = False
        mesh = os.path.join(data_dir, "cl_0_1_s_1/L5/l_step_0.020196309484414757_common_files/mesh.msh")
        output_dir = os.path.join(data_dir, "{}/L5/test/01_cond_field/output/".format(cl))
        hdf_path = os.path.join(data_dir, "{}/L5/mlmc_5.hdf5".format(cl))
        save_path = os.path.join(data_dir, "{}".format(cl))
        l_0_output_dir = os.path.join(data_dir, "{}/L1_{}/test/01_cond_field/output/".format(cl, nn_level))
        l_0_hdf_path = os.path.join(data_dir, "{}/L1_{}/mlmc_1.hdf5".format(cl, nn_level))
        sampling_info_path = os.path.join(data_dir, "{}/sampling_info".format(cl))
        ref_mlmc_file = os.path.join(data_dir, "{}/L1_3/mlmc_1.hdf5".format(cl))

    elif case == 3 or case == 4:
        #data_dir = "/home/martin/Documents/metamodels/data/5_ele/"
        cl = "cl_0_1_s_1"
        if case == 4:
            cl = "cl_0_3_s_4"
        nn_level = 0
        replace_level = False
        # mesh = os.path.join(data_dir, "{}/L5/l_step_0.020196309484414757_common_files/mesh.msh".format(cl))
        mesh = os.path.join(data_dir, "l_step_0.07416198487095663_common_files/mesh.msh".format(cl))
        output_dir = os.path.join(data_dir, "{}/L5/test/01_cond_field/output/".format(cl))
        hdf_path = os.path.join(data_dir, "{}/L5/mlmc_5.hdf5".format(cl))
        save_path = os.path.join(data_dir, "{}".format(cl))
        l_0_output_dir = os.path.join(data_dir, "{}/L1_{}/test/01_cond_field/output/".format(cl,nn_level))
        l_0_hdf_path = os.path.join(data_dir, "{}/L1_{}/mlmc_1.hdf5".format(cl, nn_level))
        sampling_info_path = os.path.join(data_dir, "{}/sampling_info".format(cl))
        ref_mlmc_file = os.path.join(data_dir, "{}/L1_benchmark/mlmc_1.hdf5".format(cl))

    elif case == 5:
        #data_dir = "/home/martin/Documents/metamodels/data/5_ele/"
        cl = "cl_0_3_s_4"
        nn_level = 0
        replace_level = False
        mesh = os.path.join(data_dir, "cl_0_1_s_1/L5/l_step_0.020196309484414757_common_files/mesh.msh")
        output_dir = os.path.join(data_dir,"{}/L1_3/test/01_cond_field/output/".format(cl))
        hdf_path = os.path.join(data_dir,"{}/L1_3/mlmc_1.hdf5".format(cl))
        save_path = os.path.join(data_dir,"{}".format(cl))
        l_0_output_dir = os.path.join(data_dir,"{}/L1_{}/test/01_cond_field/output/".format(cl,nn_level))
        l_0_hdf_path = os.path.join(data_dir,"{}/L1_{}/mlmc_1.hdf5".format(cl, nn_level))
        sampling_info_path = os.path.join(data_dir, "{}/sampling_info".format(cl))
        ref_mlmc_file = os.path.join(data_dir,"{}/L1_3/mlmc_1.hdf5".format(cl))

    # elif case == 6: # mesh size comparison
    #     cl = "cl_0_1_s_1"
    #     nn_level = 0
    #     replace_level = False
    #     mesh = os.path.join(data_dir, "l_step_0.07416198487095663_common_files/mesh.msh".format(cl))
    #     output_dir = os.path.join(data_dir, "{}/L1/test/01_cond_field/output/".format(cl))
    #     hdf_path = os.path.join(data_dir, "{}/L1/mlmc_1.hdf5".format(cl))
    #     save_path = os.path.join(data_dir, "{}".format(cl))
    #     l_0_output_dir = os.path.join(data_dir, "{}/L1_{}/test/01_cond_field/output/".format(cl, nn_level))
    #     l_0_hdf_path = os.path.join(data_dir, "{}/L1_{}/mlmc_1.hdf5".format(cl, nn_level))
    #     sampling_info_path = os.path.join(data_dir, "{}/sampling_info".format(cl))
    #     ref_mlmc_file = os.path.join(data_dir, "{}/L1_benchmark/mlmc_1.hdf5".format(cl))

    elif case == 6: # mesh size comparison
        data_dir = "/home/martin/Documents/metamodels/data/mesh_size/"
        cl = "cl_0_3_s_4"
        level = 3
        nn_level = 0
        replace_level = False
        #mesh = os.path.join(data_dir, "l_step_1.0_common_files/mesh.msh".format(cl)) #L1, 7s
        mesh = os.path.join(data_dir, "l_step_0.27232698153315_common_files/mesh.msh".format(cl)) #L2 10.5 s
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

    # elif case == 7: # mesh size comparison
    #     data_dir = "/home/martin/Documents/metamodels/data/mesh_size/"
    #     cl = "cl_0_1_s_1"
    #     level = 5
    #     nn_level = 0
    #     replace_level = False
    #     #mesh = os.path.join(data_dir, "l_step_1.0_common_files/mesh.msh".format(cl)) #L1, 7s
    #     #mesh = os.path.join(data_dir, "l_step_0.27232698153315_common_files/mesh.msh".format(cl)) #L2 10.5 s
    #     #mesh = os.path.join(data_dir, "l_step_0.07416198487095663_common_files/mesh.msh".format(cl)) #L3 12s
    #     #mesh = os.path.join(data_dir, "l_step_0.020196309484414757_common_files/mesh.msh".format(cl)) #L4 mlmc  22s
    #     #mesh = os.path.join(data_dir, "l_step_0.031_common_files/mesh.msh".format(cl)) #L4 mlmc
    #     mesh = os.path.join(data_dir, "l_step_0.016_common_files/mesh.msh".format(cl)) #L5 graph creation time 2938.779510805 s
    #     output_dir = os.path.join(data_dir, "{}/L1_{}/test/01_cond_field/output/".format(cl, level))
    #     hdf_path = os.path.join(data_dir, "{}/L1_{}/mlmc_1.hdf5".format(cl, level))
    #     mlmc_hdf_path = os.path.join(data_dir, "{}/mlmc_hdf/L1_{}/mlmc_1.hdf5".format(cl, level))
    #     save_path = os.path.join(data_dir, "{}".format(cl))
    #     l_0_output_dir = os.path.join(data_dir, "{}/L0_MC/L1_{}/test/01_cond_field/output/".format(cl, level))
    #     l_0_hdf_path = os.path.join(data_dir, "{}/L0_MC/L1_{}/mlmc_1.hdf5".format(cl, level))
    #     sampling_info_path = os.path.join(data_dir, "{}/sampling_info".format(cl))
    #     ref_mlmc_file = os.path.join(data_dir, "{}/L4_benchmark/mlmc_1.hdf5".format(cl))
    #

    elif case == 7: # mesh size comparison
        data_dir = "/home/martin/Documents/metamodels/data/mesh_size/"
        cl = "cl_0_1_s_1"
        level = 1
        nn_level = 0
        replace_level = False
        mesh = os.path.join(data_dir, "l_step_1.0_common_files/mesh.msh".format(cl)) #L1, 7s
        #mesh = os.path.join(data_dir, "l_step_0.27232698153315_common_files/mesh.msh".format(cl)) #L2 10.5 s
        #mesh = os.path.join(data_dir, "l_step_0.07416198487095663_common_files/mesh.msh".format(cl)) #L3 12s
        #mesh = os.path.join(data_dir, "l_step_0.020196309484414757_common_files/mesh.msh".format(cl)) #L4 mlmc  22s
        #mesh = os.path.join(data_dir, "l_step_0.031_common_files/mesh.msh".format(cl)) #L4 mlmc
        #mesh = os.path.join(data_dir, "l_step_0.016_common_files/mesh.msh".format(cl)) #L5 graph creation time 2938.779510805 s
        output_dir = os.path.join(data_dir, "{}/L1_{}/test/01_cond_field/output/".format(cl, level))
        hdf_path = os.path.join(data_dir, "{}/L1_{}/mlmc_1.hdf5".format(cl, level))
        mlmc_hdf_path = os.path.join(data_dir, "{}/mlmc_hdf/L1_{}/mlmc_1.hdf5".format(cl, level))
        save_path = os.path.join(data_dir, "{}".format(cl))
        l_0_output_dir = os.path.join(data_dir, "{}/L0_MC/L1_{}/test/01_cond_field/output/".format(cl, level))
        l_0_hdf_path = os.path.join(data_dir, "{}/L0_MC/L1_{}/mlmc_1.hdf5".format(cl, level))
        sampling_info_path = os.path.join(data_dir, "{}/sampling_info".format(cl))
        ref_mlmc_file = os.path.join(data_dir, "{}/L4_benchmark/mlmc_1.hdf5".format(cl))

    elif case == "cnn":  # mesh size comparison
        data_dir = "/home/martin/Documents/metamodels/data/mesh_size/"
        cl = "cl_0_1_s_1"
        level = 2
        nn_level = 0
        replace_level = False
        # mesh = os.path.join(data_dir, "l_step_1.0_common_files/mesh.msh".format(cl)) #L1, 7s
        mesh = os.path.join(data_dir, "l_step_0.27232698153315_common_files/mesh.msh".format(cl))  # L2 10.5 s
        # mesh = os.path.join(data_dir, "l_step_0.07416198487095663_common_files/mesh.msh".format(cl)) #L3 12s
        # mesh = os.path.join(data_dir, "l_step_0.020196309484414757_common_files/mesh.msh".format(cl)) #L4 mlmc  22s
        # mesh = os.path.join(data_dir, "l_step_0.031_common_files/mesh.msh".format(cl)) #L4 mlmc
        # mesh = os.path.join(data_dir, "l_step_0.016_common_files/mesh.msh".format(cl)) #L5 graph creation time 2938.779510805 s
        output_dir = os.path.join(data_dir, "{}/L1_{}_cnn/test/01_cond_field/output/".format(cl, level))
        hdf_path = os.path.join(data_dir, "{}/L1_{}_cnn/mlmc_1.hdf5".format(cl, level))
        mlmc_hdf_path = os.path.join(data_dir, "{}/mlmc_hdf/L1_{}/mlmc_1.hdf5".format(cl, level))
        save_path = os.path.join(data_dir, "{}".format(cl))
        l_0_output_dir = os.path.join(data_dir, "{}/L0_MC/L1_{}/test/01_cond_field/output/".format(cl, level))
        l_0_hdf_path = os.path.join(data_dir, "{}/L0_MC/L1_{}/mlmc_1.hdf5".format(cl, level))
        sampling_info_path = os.path.join(data_dir, "{}/sampling_info".format(cl))
        ref_mlmc_file = os.path.join(data_dir, "{}/L4_benchmark/mlmc_1.hdf5".format(cl))

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

    elif case == 12:  # mesh size comparison
        data_dir = "/home/martin/Documents/metamodels/data/mesh_size/02_conc_cond/"
        # cl = "cl_0_1_s_1"
        level = "1"
        nn_level = 0
        replace_level = False
        mesh = os.path.join(data_dir, "l_step_1.0_common_files/repo.msh")  # L1, 7s
        #mesh = os.path.join(data_dir, "l_step_0.3760603093086394_common_files/repo.msh") #L2 10.5 s
        #mesh = os.path.join(data_dir, "l_step_0.1414213562373095_common_files/repo.msh") #L3 12s
        #mesh = os.path.join(data_dir, "l_step_0.053182958969449884_common_files/repo.msh")  # L4 # 4388 - for 50k
        #mesh = os.path.join(data_dir, "l_step_0.027_common_files/repo.msh")  # L5 - graph creation time: 2564.6843196170003
        output_dir = os.path.join(data_dir, "L1_{}_50k/test/02_conc/output/".format(level))
        predict_dir = os.path.join(data_dir, "L1_{}/test/02_conc/output/".format(level))
        predict_hdf = os.path.join(data_dir, "L1_{}/mlmc_1.hdf5".format(level))
        hdf_path = os.path.join(data_dir, "L1_{}_50k/mlmc_1.hdf5".format(level))
        mlmc_hdf_path = os.path.join(data_dir, "mlmc_hdf/L1_{}/mlmc_1.hdf5".format(level))
        save_path = data_dir
        l_0_output_dir = os.path.join(data_dir, "L0_MC/L1_{}_50k/test/02_conc/output/".format(level))
        l_0_hdf_path = os.path.join(data_dir, "L0_MC/L1_{}_50k/mlmc_1.hdf5".format(level))
        sampling_info_path = os.path.join(data_dir, "sampling_info")

        #ref_mlmc_file = "/home/martin/Documents/metamodels/data/mesh_size/02_conc_cond/L1_3/mlmc_1.hdf5"
        #ref_mlmc_file = "/home/martin/Documents/metamodels/data/mesh_size/02_conc_cond/L1_1/mlmc_1.hdf5"
        ref_mlmc_file = os.path.join(data_dir, "L1_benchmark/mlmc_1.hdf5")

        feature_names = [['conductivity_top', 'conductivity_bot', 'conductivity_repo']]

    # elif case == 12:  # mesh size comparison
    #     data_dir = "/home/martin/Documents/metamodels/data/mesh_size/02_conc_cond/"
    #     # cl = "cl_0_1_s_1"
    #     level = "6"
    #     nn_level = 0
    #     replace_level = False
    #     mesh = os.path.join(data_dir, "l_step_1.0_common_files/repo.msh")  # L1, 7s
    #     #mesh = os.path.join(data_dir, "l_step_0.3760603093086394_common_files/repo.msh") #L2 10.5 s
    #     #mesh = os.path.join(data_dir, "l_step_0.1414213562373095_common_files/repo.msh") #L3 12s
    #     #mesh = os.path.join(data_dir, "l_step_0.053182958969449884_common_files/repo.msh")  # L4 # 4388 - for 50k
    #     #mesh = os.path.join(data_dir, "l_step_0.027_common_files/repo.msh")  # L5 - graph creation time: 2564.6843196170003
    #     mesh = os.path.join(data_dir, "l_step_0.02_common_files/repo.msh")  # L6 - graph creation time: 11633
    #     output_dir = os.path.join(data_dir, "L1_{}/test/02_conc/output/".format(level))
    #     predict_dir = os.path.join(data_dir, "L1_{}/test/02_conc/output/".format(level))
    #     predict_hdf = os.path.join(data_dir, "L1_{}/mlmc_1.hdf5".format(level))
    #     hdf_path = os.path.join(data_dir, "L1_{}/mlmc_1.hdf5".format(level))
    #     mlmc_hdf_path = os.path.join(data_dir, "mlmc_hdf/L1_{}/mlmc_1.hdf5".format(level))
    #     save_path = data_dir
    #     l_0_output_dir = os.path.join(data_dir, "L0_MC/L1_{}/test/02_conc/output/".format(level))
    #     l_0_hdf_path = os.path.join(data_dir, "L0_MC/L1_{}/mlmc_1.hdf5".format(level))
    #     sampling_info_path = os.path.join(data_dir, "sampling_info")
    #
    #     #ref_mlmc_file = "/home/martin/Documents/metamodels/data/mesh_size/02_conc_cond/L1_3/mlmc_1.hdf5"
    #     #ref_mlmc_file = "/home/martin/Documents/metamodels/data/mesh_size/02_conc_cond/L1_1/mlmc_1.hdf5"
    #     ref_mlmc_file = os.path.join(data_dir, "mlmc_hdf/L1_5/mlmc_1.hdf5")
    #
    #     feature_names = [['conductivity_top', 'conductivity_bot', 'conductivity_repo']]

    elif case == 13:  # mesh size comparison
        data_dir = "/home/martin/Documents/metamodels/data/mesh_size/02_conc_por/"
        # cl = "cl_0_1_s_1"
        level = 3
        nn_level = 0
        replace_level = False
        mesh = os.path.join(data_dir, "l_step_1.0_common_files/repo.msh")  # L1, 7s
        #mesh = os.path.join(data_dir, "l_step_0.3760603093086394_common_files/repo.msh")  # L2 10.5 s
        mesh = os.path.join(data_dir, "l_step_0.1414213562373095_common_files/repo.msh") #L3 12s
        output_dir = os.path.join(data_dir, "L1_{}/test/02_conc/output/".format(level))
        hdf_path = os.path.join(data_dir, "L1_{}/mlmc_1.hdf5".format(level))
        mlmc_hdf_path = os.path.join(data_dir, "mlmc_hdf/L1_{}/mlmc_1.hdf5".format(level))
        save_path = data_dir
        l_0_output_dir = os.path.join(data_dir, "L0_MC/L1_{}/test/02_conc/output/".format(level))
        l_0_hdf_path = os.path.join(data_dir, "L0_MC/L1_{}/mlmc_1.hdf5".format(level))
        sampling_info_path = os.path.join(data_dir, "sampling_info")
        ref_mlmc_file = os.path.join(data_dir, "L1_benchmark/mlmc_1.hdf5")
        feature_names = [['porosity_top', 'porosity_bot', 'porosity_repo']]

    elif case == 14:  # mesh size comparison
        data_dir = "/home/martin/Documents/metamodels/data/mesh_size/02_conc_2_features/"
        # cl = "cl_0_1_s_1"
        level = 3
        nn_level = 0
        replace_level = False
        mesh = os.path.join(data_dir, "l_step_1.0_common_files/repo.msh")  # L1, 7s
        #mesh = os.path.join(data_dir, "l_step_0.3760603093086394_common_files/repo.msh")  # L2 10.5 s
        #mesh = os.path.join(data_dir, "l_step_0.1414213562373095_common_files/repo.msh") #L3 12s
        mesh = os.path.join(data_dir, "l_step_0.053182958969449884_common_files/repo.msh")  # L3 12s
        output_dir = os.path.join(data_dir, "L1_{}/test/02_conc/output/".format(level))
        hdf_path = os.path.join(data_dir, "L1_{}/mlmc_1.hdf5".format(level))
        mlmc_hdf_path = os.path.join(data_dir, "mlmc_hdf/L1_{}/mlmc_1.hdf5".format(level))
        save_path = data_dir
        l_0_output_dir = os.path.join(data_dir, "L0_MC/L1_{}/test/02_conc/output/".format(level))
        l_0_hdf_path = os.path.join(data_dir, "L0_MC/L1_{}/mlmc_1.hdf5".format(level))
        sampling_info_path = os.path.join(data_dir, "sampling_info")
        ref_mlmc_file = os.path.join(data_dir, "L1_benchmark/mlmc_1.hdf5")

        feature_names = [['conductivity_top', 'conductivity_bot', 'conductivity_repo'],
                         ['porosity_top', 'porosity_bot', 'porosity_repo']]

    # elif case == 15:
    #     data_dir_3_level = "/home/martin/Documents/metamodels/data/3_level/02_conc_cond/"
    #     data_dir = "/home/martin/Documents/metamodels/data/mesh_size/02_conc_cond/"
    #     nn_level = 0
    #     replace_level = False
    #
    #     mesh = os.path.join(data_dir, "l_step_0.1414213562373095_common_files/repo.msh")  # L3 12s
    #     output_dir = os.path.join(data_dir_3_level, "L3_1/test/02_conc/output/")
    #     hdf_path = os.path.join(data_dir_3_level, "L3_1/mlmc_3.hdf5")
    #     mlmc_hdf_path = os.path.join(data_dir_3_level, "L3_2/mlmc_3.hdf5")
    #     save_path = data_dir_3_level
    #     l_0_output_dir = os.path.join(data_dir, "L0_MC/L1_3_50k/test/02_conc/output/")
    #     l_0_hdf_path = os.path.join(data_dir, "L0_MC/L1_3_50k/mlmc_1.hdf5")
    #     ref_mlmc_file = os.path.join(data_dir, "L1_benchmark/mlmc_1.hdf5")
    #     sampling_info_path = os.path.join(data_dir, "sampling_info")
    #
    #     graph_creation_time = 1460
    #
    #     feature_names = [['conductivity_top', 'conductivity_bot', 'conductivity_repo']]

    elif case == 15:
        data_dir_3_level = "/home/martin/Documents/metamodels/data/3_level/02_conc_cond/"
        data_dir = "/home/martin/Documents/metamodels/data/mesh_size/02_conc_cond/"
        nn_level = 0
        replace_level = False

        mesh = os.path.join(data_dir, "l_step_0.1414213562373095_common_files/repo.msh")  # L3 12s
        output_dir = os.path.join(data_dir, "L1_3_50k/test/02_conc/output/")
        predict_dir = os.path.join(data_dir, "L1_3/test/02_conc/output/")
        predict_hdf = os.path.join(data_dir, "L1_3/mlmc_1.hdf5")
        hdf_path = os.path.join(data_dir_3_level, "L3_1/mlmc_3.hdf5")
        mlmc_hdf_path = os.path.join(data_dir_3_level, "L3_2/mlmc_3.hdf5")
        save_path = data_dir_3_level
        save_path = data_dir
        l_0_output_dir = os.path.join(data_dir, "L0_MC/L1_3_50k/test/02_conc/output/")
        l_0_hdf_path = os.path.join(data_dir, "L0_MC/L1_3_50k/mlmc_1.hdf5")
        #ref_mlmc_file = os.path.join(data_dir, "L1_benchmark/mlmc_1.hdf5")
        ref_mlmc_file = os.path.join(data_dir, "mlmc_hdf/L1_5/mlmc_1.hdf5")
        ref_mlmc_file = os.path.join(data_dir, "L0_MC/L1_3_150k/mlmc_1.hdf5")
        sampling_info_path = os.path.join(data_dir, "sampling_info")

        graph_creation_time = 1460

        feature_names = [['conductivity_top', 'conductivity_bot', 'conductivity_repo']]

    # elif case == 15:
    #     data_dir_3_level = "/home/martin/Documents/metamodels/data/3_level/02_conc_cond/"
    #     data_dir = "/home/martin/Documents/metamodels/data/mesh_size/02_conc_cond/"
    #     nn_level = 0
    #     replace_level = False
    #
    #     mesh = os.path.join(data_dir, "l_step_0.1414213562373095_common_files/repo.msh")  # L3 12s
    #     output_dir = os.path.join(data_dir, "L1_4_50k/test/02_conc/output/")
    #     predict_dir = os.path.join(data_dir, "L1_4/test/02_conc/output/")
    #     predict_hdf = os.path.join(data_dir, "L1_4/mlmc_1.hdf5")
    #     hdf_path = os.path.join(data_dir_3_level, "L3_1/mlmc_3.hdf5")
    #     mlmc_hdf_path = os.path.join(data_dir_3_level, "L3_2/mlmc_3.hdf5")
    #     save_path = data_dir_3_level
    #     save_path = data_dir
    #     l_0_output_dir = os.path.join(data_dir, "L0_MC/L1_4_50k/test/02_conc/output/")
    #     l_0_hdf_path = os.path.join(data_dir, "L0_MC/L1_4_50k/mlmc_1.hdf5")
    #     #ref_mlmc_file = os.path.join(data_dir, "L1_benchmark/mlmc_1.hdf5")
    #     ref_mlmc_file = os.path.join(data_dir, "mlmc_hdf/L1_5/mlmc_1.hdf5")
    #     sampling_info_path = os.path.join(data_dir, "sampling_info")
    #
    #     graph_creation_time = 1460
    #     feature_names = [['conductivity_top', 'conductivity_bot', 'conductivity_repo']]

    # elif case == 15:
    #     data_dir_3_level = "/home/martin/Documents/metamodels/data/2_level/02_conc_cond/"
    #     data_dir = "/home/martin/Documents/metamodels/data/mesh_size/02_conc_cond/"
    #     nn_level = 0
    #     replace_level = False
    #
    #     mesh = os.path.join(data_dir, "l_step_0.1414213562373095_common_files/repo.msh")  # L3 12s
    #     output_dir = os.path.join(data_dir, "L1_3_50k/test/02_conc/output/")
    #     predict_dir = os.path.join(data_dir, "L1_3/test/02_conc/output/")
    #     predict_hdf = os.path.join(data_dir, "L1_3/mlmc_1.hdf5")
    #     hdf_path = os.path.join(data_dir_3_level, "L2_1/mlmc_2.hdf5")
    #     mlmc_hdf_path = os.path.join(data_dir_3_level, "L2_2/mlmc_2.hdf5")
    #     save_path = data_dir_3_level
    #     save_path = data_dir
    #     l_0_output_dir = os.path.join(data_dir, "L0_MC/L1_3_50k/test/02_conc/output/")
    #     l_0_hdf_path = os.path.join(data_dir, "L0_MC/L1_3_50k/mlmc_1.hdf5")
    #     #ref_mlmc_file = os.path.join(data_dir, "L1_benchmark/mlmc_1.hdf5")
    #     ref_mlmc_file = os.path.join(data_dir, "mlmc_hdf/L1_5/mlmc_1.hdf5")
    #     sampling_info_path = os.path.join(data_dir, "sampling_info")
    #
    #     graph_creation_time = 1460
    #
    #     feature_names = [['conductivity_top', 'conductivity_bot', 'conductivity_repo']]

    # elif case == 15:
    #     data_dir_3_level = "/home/martin/Documents/metamodels/data/5_level/02_conc_cond/"
    #     data_dir = "/home/martin/Documents/metamodels/data/mesh_size/02_conc_cond/"
    #     nn_level = 0
    #     replace_level = False
    #
    #     mesh = os.path.join(data_dir, "l_step_0.1414213562373095_common_files/repo.msh")  # L3 12s
    #     output_dir = os.path.join(data_dir, "L1_3_50k/test/02_conc/output/")
    #     predict_dir = os.path.join(data_dir, "L1_3/test/02_conc/output/")
    #     predict_hdf = os.path.join(data_dir, "L1_3/mlmc_1.hdf5")
    #     hdf_path = os.path.join(data_dir_3_level, "L5_1/mlmc_5.hdf5")
    #     mlmc_hdf_path = os.path.join(data_dir_3_level, "L5_1/mlmc_5.hdf5")
    #     save_path = data_dir_3_level
    #     save_path = data_dir
    #     l_0_output_dir = os.path.join(data_dir, "L0_MC/L1_3_50k/test/02_conc/output/")
    #     l_0_hdf_path = os.path.join(data_dir, "L0_MC/L1_3_50k/mlmc_1.hdf5")
    #     #ref_mlmc_file = os.path.join(data_dir, "L1_benchmark/mlmc_1.hdf5")
    #     ref_mlmc_file = os.path.join(data_dir, "mlmc_hdf/L1_5/mlmc_1.hdf5")
    #     sampling_info_path = os.path.join(data_dir, "sampling_info")
    #
    #     graph_creation_time = 1460
    #
    #     feature_names = [['conductivity_top', 'conductivity_bot', 'conductivity_repo']]

    # elif case == 15:
    #     data_dir_3_level = "/home/martin/Documents/metamodels/data/3_level/02_conc_cond/"
    #     data_dir = "/home/martin/Documents/metamodels/data/mesh_size/02_conc_cond/"
    #     nn_level = 0
    #     replace_level = False
    #
    #     mesh = os.path.join(data_dir, "l_step_0.1414213562373095_common_files/repo.msh")  # L3 12s
    #     output_dir = os.path.join(data_dir, "L1_6/test/02_conc/output/")
    #     predict_dir = os.path.join(data_dir, "L1_6/test/02_conc/output/")
    #     predict_hdf = os.path.join(data_dir, "L1_6/mlmc_1.hdf5")
    #     hdf_path = os.path.join(data_dir_3_level, "L3_1/mlmc_3.hdf5")
    #     mlmc_hdf_path = os.path.join(data_dir_3_level, "L3_2/mlmc_3.hdf5")
    #     save_path = data_dir_3_level
    #     save_path = data_dir
    #     l_0_output_dir = os.path.join(data_dir, "L0_MC/L1_6/test/02_conc/output/")
    #     l_0_hdf_path = os.path.join(data_dir, "L0_MC/L1_6/mlmc_1.hdf5")
    #     #ref_mlmc_file = os.path.join(data_dir, "L1_benchmark/mlmc_1.hdf5")
    #     ref_mlmc_file = os.path.join(data_dir, "mlmc_hdf/L1_5/mlmc_1.hdf5")
    #     sampling_info_path = os.path.join(data_dir, "sampling_info")
    #
    #     graph_creation_time = 1460
    #
    #     feature_names = [['conductivity_top', 'conductivity_bot', 'conductivity_repo']]

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

        graph_creation_time = 240

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

        graph_creation_time = 290#285

    # elif case == "cases":  # mesh size comparison
    #     data_dir = "/home/martin/Documents/metamodels/data/mesh_size/02_conc_cond/case_5"
    #     # cl = "cl_0_1_s_1"
    #     level = "3"
    #     nn_level = 0
    #     replace_level = False
    #     # mesh = os.path.join(data_dir, "l_step_1.0_common_files/repo.msh")  # L1, 7s
    #     # mesh = os.path.join(data_dir, "l_step_0.3760603093086394_common_files/repo.msh") #L2 10.5 s
    #     mesh = os.path.join(data_dir, "l_step_0.1414213562373095_common_files/repo.msh") #L3 12s
    #     # mesh = os.path.join(data_dir, "l_step_0.053182958969449884_common_files/repo.msh")  # L4
    #     output_dir = os.path.join(data_dir, "L1_{}/test/02_conc/output/".format(level))
    #     hdf_path = os.path.join(data_dir, "L1_{}/mlmc_1.hdf5".format(level))
    #     mlmc_hdf_path = None  # os.path.join(data_dir, "mlmc_hdf/L1_{}/mlmc_1.hdf5".format(level))
    #     save_path = data_dir
    #     l_0_output_dir = os.path.join(data_dir, "L0_MC/L1_{}/test/02_conc/output/".format(level))
    #     l_0_hdf_path = os.path.join(data_dir, "L0_MC/L1_{}/mlmc_1.hdf5".format(level))
    #     sampling_info_path = os.path.join(data_dir, "sampling_info")
    #
    #     # ref_mlmc_file = "/home/martin/Documents/metamodels/data/mesh_size/02_conc_cond/L1_3/mlmc_1.hdf5"
    #     # ref_mlmc_file = "/home/martin/Documents/metamodels/data/mesh_size/02_conc_cond/L1_1/mlmc_1.hdf5"
    #     ref_mlmc_file = os.path.join(data_dir, "L1_benchmark/mlmc_1.hdf5")
    #
    #     feature_names = [['conductivity_top', 'conductivity_bot', 'conductivity_repo']]
    #
    #     graph_creation_time = 670  # case_2
    #     graph_creation_time = 601  # case_3
    #     graph_creation_time = 608  # case_4
    #     graph_creation_time = 587  # case_5

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
           replace_level, nn_level, mlmc_hdf_path, feature_names, graph_creation_time, predict_dir, predict_hdf


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
    case = 7 #"cnn"
    #data_dir = "/home/martin/Documents/metamodels/data/1000_ele/"
    output_dir, hdf_path, l_0_output_dir, l_0_hdf_path, save_path, mesh, sampling_info_path, ref_mlmc_file,\
    replace_level, nn_level, mlmc_hdf_path, feature_names, graph_creation_time,\
    predict_dir, predict_hdf = get_config(data_dir, case)


    # plot_results_corr_length()
    # exit()

    # if os.path.exists(os.path.join(work_dir, "mlmc_{}.hdf5".format(nn_level + 1))):
    #     l_0_hdf_path = os.path.join(work_dir, "mlmc_{}.hdf5".format(nn_level + 1))
    #     hdf_path = os.path.join(work_dir, "mlmc_5.hdf5")
    #     ref_mlmc_file = os.path.join(work_dir, "benchmark_mlmc_1.hdf5")


    # import cProfile
    # import pstats
    # pr = cProfile.Profile()
    # pr.enable()
    #gnn, conv_layer, corr_field_config = get_gnn()
    #
    # my_result = run_GNN(output_dir, hdf_path, l_0_output_dir, l_0_hdf_path, save_path, mesh, level=nn_level, log=True, conv_layer=conv_layer)
    #
    # pr.disable()
    # ps = pstats.Stats(pr).sort_stats('cumtime')
    # ps.print_stats()

    #run_SVR(output_dir, hdf_path, l_0_output_dir, l_0_hdf_path, save_path, mesh, sampling_info_path, ref_mlmc_file, level=nn_level, log=True, conv_layer=conv_layer)  # , gnn=gnn)

    #run_GNN(output_dir, hdf_path, l_0_output_dir, l_0_hdf_path, save_path, mesh, level=nn_level, log=True)  # , gnn=gnn)
    #run_CNN(output_dir, hdf_path, l_0_output_dir, l_0_hdf_path, save_path, mesh, level=nn_level, log=True)  # , gnn=gnn)
    #process_results(hdf_path, sampling_info_path, ref_mlmc_file, save_path, nn_level, replace_level)

    # Graph creation time for cl_0_1_s_1 case 1 = 100 s

    # gnn, conv_layer, corr_field_config = get_gnn()
    # # # # #run_GNN(output_dir, hdf_path, l_0_output_dir, l_0_hdf_path, save_path, mesh, model=GCN, level=nn_level, log=True) # CGN model leads to constant value
    # # # # #run_GNN(output_dir, hdf_path, l_0_output_dir, l_0_hdf_path, save_path, mesh, level=nn_level, log=True, gnn=gnn, conv_layer=conv_layer)
    # run_GNN(output_dir, hdf_path, l_0_output_dir, l_0_hdf_path, save_path, mesh, sampling_info_path, ref_mlmc_file,
    #            level=nn_level, log=True, conv_layer=conv_layer, gnn=gnn, corr_field_config=corr_field_config, graph_creation_time=100)
    # process_results(hdf_path, sampling_info_path, ref_mlmc_file, save_path, nn_level, replace_level)
    #

    #gnn, conv_layer, corr_field_config, model_config = get_gnn()
    gnn, conv_layer, corr_field_config, model_config = get_cnn()

    # print("gnn ", gnn)
    #print("conv layer ", conv_layer)

    #machine_learning_model = ("L2_test", run_GNN, False)

    machine_learning_model = ("ChC8L3_log", run_GNN, False)
    #machine_learning_model = ("ChC8L2_log", run_GNN, False)
    #machine_learning_model = ("SVR_L3_log", run_GNN, False)
    # machine_learning_model = ("ChC32L3T25000", run_GNN, False)
    #
    # machine_learning_model = ("ChC32Loss2_adding_moments_2", run_GNN, False)
    #machine_learning_model = ("ChC32Loss2_add_mom", run_GNN, True)
    #
    #machine_learning_model = ("ChC32L3M10_test", run_GNN, False)

    #machine_learning_model = ("mesh_L3", run_GNN, False)

    #machine_learning_model = ("SVR_mesh_L3_log", run_GNN, False)


    machine_learning_model = ("GCN_mesh_L3_log", run_GNN, False)
    #
    #machine_learning_model = ("mesh_moments_test_2", run_GNN, True)
    #machine_learning_model = ("mesh_L3_test_m", run_GNN, False)

    # #models = {"ChebConv": (run_GNN, False), "SVR": (run_SVR, False)}
    machine_learning_model = ("5eleChebConvL3_2", run_GNN, False)
    #machine_learning_model = ("5eleChebConvK2", run_GNN, False)
    # # machine_learning_model = ("5eleChebConvK3", run_GNN, False)
    #machine_learning_model = ("5eleChebConv32abs", run_GNN, False)
    #machine_learning_model = ("5eleChebConv32msemom", run_GNN, False)

    ####################
    ### Compare number of training samples ###
    # machine_learning_model = ("mesh_L3", run_GNN, False)
    # machine_learning_model = ("mesh_L3_t_5k", run_GNN, False)
    # machine_learning_model = ("mesh_L3_t_10k", run_GNN, False)
    # #machine_learning_model = ("mesh_L3_t_15k", run_GNN, False)
    # ################
    # ################
    #
    # ####################
    # ### Compare number of training samples LOG ###
    # machine_learning_model = ("mesh_L3_t_100_log", run_GNN, False)
    # machine_learning_model = ("mesh_L3_t_500_log", run_GNN, False)
    # machine_learning_model = ("mesh_L3_t_1000_log", run_GNN, False)
    # machine_learning_model = ("mesh_L3_t_1500_log", run_GNN, False)
    # machine_learning_model = ("mesh_L3_t_2000_log", run_GNN, False)
    # machine_learning_model = ("mesh_L3_t_4000_log", run_GNN, False)
    # machine_learning_model = ("mesh_L3_t_8000_log", run_GNN, False)
    # # # machine_learning_model = ("mesh_L3_t_15k", run_GNN, False)
    # ################
    # ################
    #
    # ########
    # # Test different correlation lengths
    # machine_learning_model = ("mesh_L3_log", run_GNN, False)
    # #machine_learning_model = ("SVR_mesh_L3_log", run_GNN, False)
    # #######
    #
    #machine_learning_model = ("DNN_mesh_L3_log_deep", run_DNN, True)
    #
    #machine_learning_model = ("DNN_mesh_L3_6", run_DNN, True)
    machine_learning_model = ("GCN_mesh_L3_log_16", run_GNN, True)
    machine_learning_model = ("mesh_L3_log_test_saved_model", run_GNN, True)

    if case == 6:
        machine_learning_model = ("L1_3_T19_case_1_out_log_scale", run_GNN, False)
        machine_learning_model = ("L1_3_T19_case_1_f_log_out_log_scale", run_GNN, False)
        #machine_learning_model = ("L1_3_T19_case_1_f_log_norm_out_log_scale", run_GNN, False)
        #machine_learning_model = ("L1_3_T19_case_1_f_log_scale_out_log_scale", run_GNN, False)
        #machine_learning_model = ("L1_3_T19_case_1_f_norm_out_log_scale", run_GNN, False)

    if case == 7:
        machine_learning_model = ("mesh_L3_log_50k", run_GNN, True)  # best results
        #machine_learning_model = ("mesh_L3_log_50k", run_GNN, True)
        # machine_learning_model = ("L1_3_cl_0_1_s_1_all_log_output_mult", run_GNN, False)
        #
        machine_learning_model = ("L1_3_cl_0_1_s_1_log_output_mult_T19_f_log_case_1", run_GNN, False)
        machine_learning_model = ("L1_3_cl_0_1_s_1_log_output_mult_T19_f_2_log_case_1", run_GNN, False)
        #machine_learning_model = ("L1_3_cl_0_1_s_1_log_output_mult_T19_case_1", run_GNN, False)
        #machine_learning_model = ("L1_3_cl_0_1_s_1_log_output_mult_T19_K1_case_1", run_GNN, False)

        machine_learning_model = ("L1_3_T19_case_1_out_log_scale", run_GNN, False)
        #machine_learning_model = ("L1_3_T19_case_1_f_log_norm_out_log_scale", run_GNN, False)
        machine_learning_model = ("L1_3_T19_case_1_f_norm_out_log_scale", run_GNN, False)

        machine_learning_model = ("L1_2_T19_case_1_out_log_scale", run_GNN, False)
        machine_learning_model = ("L1_4_T34_case_1_out_log_scale", run_GNN, False)

        machine_learning_model = ("L1_4_T27_case_1_f_log_out_log_scale", run_GNN, False)

        machine_learning_model = ("L1_3_T19_case_1_f_log_out_log_scale", run_GNN, False)
        machine_learning_model = ("L1_4_T19_case_1_f_log_out_log_scale", run_GNN, False)

        machine_learning_model = ("L1_4_T27_case_1_f_log_out_log_scale_CH2", run_GNN, False)

        machine_learning_model = ("L1_3_T1_case_1_f_log_out_log_scale_is", run_GNN, False)

        #machine_learning_model = ("L1_5_T27_GPU_learning_time_test", run_GNN, False)

        #machine_learning_model = ("L1_1_T27_test_independent_samples", run_GNN, False)

        #machine_learning_model = ("L1_2_T27_case_1_f_log_out_log_scale_BS2", run_GNN, False)

        ## Different number of channels
        #machine_learning_model = ("L1_3_T19_case_1_out_log_scale_CH2_lr_schedule_15", run_GNN, False)

        machine_learning_model = ("L1_1_test_diffpool", run_GNN, False)

        machine_learning_model = ("L1_1_test_CNN", run_CNN, False)

    if case == "cnn":
        machine_learning_model = ("L1_2_test_CNN", run_CNN, False)


    # machine_learning_model = ("test_02_conc", run_GNN, False)
    #
    # machine_learning_model = ("L1_1_02_conc_cond_5", run_GNN, False)
    # # #
    # machine_learning_model = ("L1_1_02_conc_cond", run_GNN, False)
    # machine_learning_model = ("L1_1_02_conc_cond_norm", run_GNN, False)
    # machine_learning_model = ("L1_1_02_conc_cond_norm_output_mult", run_GNN, False)
    if case == 12:
        machine_learning_model = ("L1_1_02_conc_cond_norm_output_mult", run_GNN, False)
        machine_learning_model = ("L1_1_02_conc_cond_output_mult", run_GNN, False)
        machine_learning_model = ("L1_1_02_conc_cond_norm", run_GNN, False)

        #machine_learning_model = ("L1_1_02_conc_cond", run_GNN, False)
        #
        # #machine_learning_model = ("L1_1_02_conc_cond_all_log_output_mult", run_GNN, False)
        # #machine_learning_model = ("L1_1_02_conc_cond_features_log_output_mult", run_GNN, False)
        # machine_learning_model = ("L1_1_02_conc_cond_output_log_output_mult", run_GNN, False)
        #
        # #machine_learning_model = ("L1_3_02_conc_cond_test", run_GNN, False)
        # machine_learning_model = ("L1_3_02_conc_cond_all_log_output_mult", run_GNN, False)
        machine_learning_model = ("L1_3_02_conc_cond_all_log_output_mult_case_1", run_GNN, False)

        #### CASE 1 ####
        machine_learning_model = ("L1_3_02_conc_cond_features_norm_case_1", run_GNN, False)
        machine_learning_model = ("L1_3_02_conc_cond_features_norm_mult_output_case_1", run_GNN, False)

        machine_learning_model = ("L1_3_02_conc_cond_output_mult_case_1", run_GNN, False)
        ### log
        machine_learning_model = ("L1_3_02_conc_cond_log_features_norm_case_1", run_GNN, False)
        #machine_learning_model = ("L1_3_02_conc_cond_true_features_norm_mult_output_case_1", run_GNN, False)
        #machine_learning_model = ("L1_3_02_conc_cond_log_output_mult_case_1", run_GNN, False)

        #machine_learning_model = ("L1_3_02_conc_cond_all_log_output_mult_T1_case_1", run_GNN, False)

        machine_learning_model = ("L1_3_02_conc_cond_log_output_mult_T19_case_1", run_GNN, False)
        #machine_learning_model = ("L1_3_02_conc_cond_log_output_mult_T19_f_log_case_1", run_GNN, False)
        #machine_learning_model = ("L1_3_02_conc_cond_log_output_mult_T20_case_1", run_GNN, False)

        machine_learning_model = ("L1_3_02_conc_cond_test", run_GNN, False)


        ######################################
        ## Test input output preprocessing  ##
        ######################################

        #machine_learning_model = ("L1_3_T19_case_1_f_scale_out_log_scale", run_GNN, False)
        #machine_learning_model = ("L1_3_T19_case_1_f_log_scale_out_log_scale", run_GNN, False)
        #machine_learning_model = ("L1_3_T19_case_1_out_log_scale", run_GNN, False)

        machine_learning_model = ("L1_3_T19_case_1_f_norm_out_log_scale", run_GNN, False)
        machine_learning_model = ("L1_3_T19_case_1_f_log_norm_out_log_scale", run_GNN, False)
        #machine_learning_model = ("L1_3_T19_case_1_f_log_out_log_scale", run_GNN, False)
        #machine_learning_model = ("L1_3_T19_case_1_out_log_scale", run_GNN, False)

        #machine_learning_model = ("L1_3_02_conc_cond_log_output_mult_T19_K1_case_1", run_GNN, False)
        #machine_learning_model = ("L1_3_02_conc_cond_log_output_mult_T24_case_1", run_GNN, False)

        #######################################
        ## Test different number of channels ##
        #######################################
        machine_learning_model = ("L1_3_T19_case_1_out_log_scale_CH1", run_GNN, False)
        machine_learning_model = ("L1_3_T19_case_1_out_log_scale_CH2", run_GNN, False)
        machine_learning_model = ("L1_3_T19_case_1_out_log_scale_CH3", run_GNN, False)
        machine_learning_model = ("L1_3_T19_case_1_out_log_scale_CH4", run_GNN, False)
        machine_learning_model = ("L1_3_T19_case_1_out_log_scale_CH5", run_GNN, False)

        machine_learning_model = ("L1_3_T19_case_1_out_log_scale_CH4_patience", run_GNN, False)

        ####
        ## Learning rate schedule
        ###
        machine_learning_model = ("L1_3_T19_case_1_out_log_scale_CH5_lr_schedule_14", run_GNN, False)

        machine_learning_model = ("test_data_augmentation", run_GNN, False)

        machine_learning_model = ("L1_3_02_conc_cond_log_output_mult_T24_case_1", run_GNN, False)

        machine_learning_model = ("L1_3_T19_02_conc_cond_test", run_GNN, False)

        machine_learning_model = ("L1_4_T34_case_1_out_log_scale_1250", run_GNN, False)
        machine_learning_model = ("L1_4_T19_case_1_out_log_scale_K1", run_GNN, False)

        machine_learning_model = ("L1_5_T27_case_1_out_log_scale_is", run_GNN, False)
        machine_learning_model = ("L1_6_T27_case_1_out_log_scale", run_GNN, False)
        #machine_learning_model = ("L1_6_T27_case_1_out_log_scale", run_GNN, False)

        #machine_learning_model = ("L1_2_02_conc_cond_log_output_mult_T19_case_1_trained_12_0", run_GNN, False)
        #machine_learning_model = ("L1_3_02_conc_cond_log_output_mult_T27_case_1_trained_L3_0", run_GNN, False)


        #machine_learning_model = ("L1_1_preprocess_data", run_GNN, False)
        # machine_learning_model = ("L1_4_test_epochs_cpu", run_GNN, False)
        machine_learning_model = ("L1_1_test_diffpool", run_GNN, False)
        #machine_learning_model = ("L1_1_test_deep_map", run_DeepMap, False)

    if case == 13:
        machine_learning_model = ("L1_1_02_conc_por_all_log_output_mult", run_GNN, False)
        machine_learning_model = ("L1_1_02_conc_por_features_log_output_mult", run_GNN, False)
        machine_learning_model = ("L1_1_02_conc_por_output_log_output_mult", run_GNN, False)

        machine_learning_model = ("L1_1_02_conc_por_test", run_GNN, False)

        machine_learning_model = ("L1_3_02_conc_por_all_log_output_mult", run_GNN, False)

    if case == 14:
        machine_learning_model = ("L1_1_02_conc_2_features", run_GNN, False)
        #machine_learning_model = ("L1_1_02_conc_2_features_log", run_GNN, True)

        #machine_learning_model = ("L1_1_02_conc_2_features_log_output_mult", run_GNN, False)
        machine_learning_model = ("L1_1_02_conc_2_features_all_log_mult_output", run_GNN, False)
        #machine_learning_model = ("L1_1_02_conc_2_features_output_log_mult_output", run_GNN, False)
        #machine_learning_model = ("L1_1_02_conc_2_features_features_log_mult_output", run_GNN, False)

        machine_learning_model = ("L1_1_02_conc_2_features_test", run_GNN, False)

        machine_learning_model = ("L1_3_02_conc_2_features_all_log_output_mult", run_GNN, False)

        machine_learning_model = ("L1_3_02_conc_2_features_log_output_mult_T19_case_1", run_GNN, False)

    if case == 15:
        machine_learning_model = ("L3_02_conc_cond_log_output_mult_T19_case_1", run_GNN, False)
        machine_learning_model = ("L3_T27_case_1_out_log_scale", run_GNN, False)

        machine_learning_model = ("L1_3_T27_case_1_out_log_scale_is", run_GNN, False)
        #machine_learning_model = ("L1_4_T27_case_1_out_log_scale_is", run_GNN, False)
        #machine_learning_model = ("L1_6_T27_case_1_out_log_scale", run_GNN, False)

    if case == 429:
        machine_learning_model = ("L1_3_429_test", run_GNN, False)

    if case == 521:
        machine_learning_model = ("L1_3_521_test", run_GNN, False)

    if case == "cases":
        machine_learning_model = ("cases data preprocessing", run_GNN, False)

    if case == "case_3":
        machine_learning_model = ("L1_3_T19_case_3_out_log_scale", run_GNN, False)
        machine_learning_model = ("L1_3_T19_case_3_f_log_out_log_scale", run_GNN, False)
        #machine_learning_model = ("L1_3_T19_case_4_f_log_norm_out_log_scale", run_GNN, False)
        #machine_learning_model = ("L3_1_test", run_GNN, False)

    #machine_learning_model = ("L1_1_02_conc_2_features_test", run_GNN, False)

    #machine_learning_model = ("mesh_L3_log_50k_weights_5", run_GNN, False)

    #machine_learning_model = ("mesh_L3_log_sigmoid", run_GNN, False) # ReLU is much better

    save_path = os.path.join(save_path, machine_learning_model[0])
    #
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    # #
    # print("save path ", save_path)

    # 02 proc times
    # graph creation time: 2 features: 53 sec
    #                       conductivity: 35 sec
    #                       porosity: 35 sec

    # L2
    # graph creation time: 2 features: 104 sec
    #                       conductivity: 68 sec
    #                       porosity: 66 sec

    # L3
    # graph creation time: 2 features: 396 sec
    #                       conductivity: 251 sec
    #                       porosity: 250 sec

    # L4
    # graph creation time: 2 features:
    #                       conductivity: 1670
    #                       porosity:

    # if graph_creation_time == -1:
    #     graph_creation_time = 250#25#11#22#159#0#159#66

    graph_creation_time = 2500  # 72#4400
    graph_creation_time = 0  # 6045.724339538  #6944  # cl_0_1_s_1, L1_2
    graph_creation_time = 868  # 1850.33 sec -  cl_0_1_s_1, L1 (50k samples)

    # cl_0_1_s_1 config
    dataset_config = {"first_log_features": True,
                      "first_log_output": True,
                      "features_normalization": False,
                      "output_normalization": False,
                      "calc_output_mult_factor": False,
                      "output_mult_factor": 1,
                      "features_mult_factor": 1,
                      "features_scale": False,
                      "output_scale": True,
                      }

    # # 02_conc config
    dataset_config = {"features_normalization": False,
                      "calc_output_mult_factor": False,
                      "output_mult_factor": 1,
                      "features_mult_factor": 1,
                      "first_log_features": False,
                      "first_log_output": True,
                      "output_scale": True,
                      }

    # CNN cl_0_1_s_1 config
    dataset_config = {"first_log_features": False,
                      "first_log_output": False,
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
              'n_test_samples': 1000,
              'val_samples_ratio': 0.2,
              'batch_size': 20,
              'epochs': 2,
              'learning_rate': 0.001,
              'file_name': "bypixel_512.npz",
              'graph_creation_time': graph_creation_time,
              'save_model': True,
              'feature_names': feature_names,
              'dataset_config': dataset_config,
              'independent_samples': True,
              'predict_dir': predict_dir,
              'predict_hdf ': predict_hdf
              }

    import cProfile
    import pstats

    # pr = cProfile.Profile()
    # pr.enable()
    #

    statistics(config)
    #
    # my_result = statistics(config)
    #
    # pr.disable()
    # ps = pstats.Stats(pr).sort_stats('cumtime')
    # ps.print_stats()

    analyze_statistics(config)

    # save_path = os.path.join(save_path, "SVR")
    # statistics(run_SVR, output_dir, hdf_path, l_0_output_dir, l_0_hdf_path, save_path, mesh, level=nn_level, log=True)

