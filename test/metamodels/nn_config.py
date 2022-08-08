from mlmc.metamodel.flow_task_GNN_2 import GNN
from spektral.layers import GCNConv, GlobalSumPool, ChebConv, GraphSageConv, ARMAConv, GATConv, APPNPConv, GINConv, GeneralConv
from tensorflow.keras.losses import MeanSquaredError, KLDivergence, MeanAbsoluteError
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from spektral.layers import GlobalSumPool, GlobalMaxPool, GlobalAvgPool
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing

