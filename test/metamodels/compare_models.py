import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Run on CPU only
import sys
import subprocess
from mlmc.metamodel.analyze_nn import run_GNN, run_SVR, statistics, analyze_statistics, process_results
import tensorflow as tf

from mlmc.metamodel.flow_task_GNN_2 import GNN
from spektral.layers import GCNConv, GlobalSumPool, ChebConv, GraphSageConv, ARMAConv, GATConv, APPNPConv, GINConv, GeneralConv
from tensorflow.keras.losses import MeanSquaredError, KLDivergence, MeanAbsoluteError
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from spektral.layers import GlobalSumPool, GlobalMaxPool, GlobalAvgPool
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing


def compare_channels():
    data_path = "/home/martin/Documents/metamodels/data/comparison/ChebConv_channels"
    channels = [8, 16, 32, 64, 128, 256]
    channels = [2, 4, 8, 16, 32, 128]

    for channel in channels:
        channel_path = os.path.join(data_path, "{0}/ChebConv{0}".format(channel))
        print("channel path ", channel_path)
        analyze_statistics(channel_path)




if __name__ == "__main__":
    compare_channels()
