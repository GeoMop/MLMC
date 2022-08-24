import os
import sys
import os.path as osp
from math import ceil

import torch
import torch.nn.functional as F
from spektral.data import MixedLoader

import torch_geometric.transforms as T
#from torch_geometric.datasets import TUDataset
from mlmc.metamodel.flow_dataset_pytorch import FlowDataset
from mlmc.metamodel.flow_dataset import FlowDataset
from torch_geometric.loader import DenseDataLoader
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool, ChebConv


def get_config(data_dir, case=0):
    feature_names = [['conductivity']]
    graph_creation_time = -1

    predict_dir, predict_hdf = None, None

    if case == 12:  # mesh size comparison
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


# args = get_arguments(sys.argv[1:])
# data_dir = args.data_dir
# work_dir = args.work_dir
case = 12
data_dir = "/home/martin/Documents/metamodels/data/mesh_size/02_conc_cond/"
output_dir, hdf_path, l_0_output_dir, l_0_hdf_path, save_path, mesh, sampling_info_path, ref_mlmc_file,\
replace_level, nn_level, mlmc_hdf_path, feature_names, graph_creation_time,\
predict_dir, predict_hdf = get_config(data_dir, case)



# # 02_conc config
dataset_config = {"features_normalization": False,
                  "calc_output_mult_factor": False,
                  "output_mult_factor": 1,
                  "features_mult_factor": 1,
                  "first_log_features": False,
                  "first_log_output": True,
                  "output_scale": True,
                  }

config = {#'machine_learning_model': machine_learning_model,
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
          #'conv_layer': conv_layer,
          #'gnn': gnn,
          #'model_config': model_config,
          'replace_level': replace_level,
          #'corr_field_config': corr_field_config,
          'n_train_samples': 2000,
          'val_samples_ratio': 0.2,
          'batch_size': 20,
          'epochs': 2000,
          'learning_rate': 0.001,
          'graph_creation_time': graph_creation_time,
          'save_model': True,
          'feature_names': feature_names,
          'dataset_config': dataset_config,
          'independent_samples': False,
          'predict_dir': predict_dir,
          'predict_hdf ': predict_hdf
          }



# path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data',
#                 'PROTEINS_dense')

dataset = FlowDataset(output_dir=config['output_dir'], level=config['level'], log=False, config=config,
                           index=0, n_test_samples=100000)

# data.shuffle()
# dataset = data.dataset

num_features = 1
n = (len(dataset) + 9) // 10
print("n ", n)
train_dataset = dataset[:config["n_train_samples"]]
val_dataset = train_dataset[:int(config["n_train_samples"]*config["val_samples_ratio"])]
test_dataset = dataset[config["n_train_samples"]:]

print("train dataset ", len(train_dataset))

test_loader = DenseDataLoader(test_dataset, batch_size=config["batch_size"])
val_loader = DenseDataLoader(val_dataset, batch_size=config["batch_size"])
train_loader = DenseDataLoader(train_dataset, batch_size=config["batch_size"])


########################
### Orig loader     ####
########################
batch_size = config["batch_size"]
epochs = config["epochs"]
log = False
index = 0
data = FlowDataset(output_dir=config['output_dir'], level=config['level'], log=log, config=config,
                           index=index, n_test_samples=100000)
len_all_samples = len(data)

last_train_sample = index * config['n_train_samples'] + config['n_train_samples']
last_test_sample = len_all_samples - (index * config['n_train_samples'] + config['n_train_samples'])

print("last train sample ", last_train_sample)
print("last test sample ", last_test_sample)

data_tr = FlowDataset(output_dir=config['output_dir'], level=config['level'], log=log, config=config,
                      index=index, train_samples=True, independent_sample=True)

print("len data tr ", len(data_tr))

data_te = FlowDataset(output_dir=config['output_dir'], level=config['level'], log=log, config=config,
                      index=index, predict=True, test_samples=True, independent_samples=True)

print("len data te ", len(data_te))

val_data_len = int(len(data_tr) * config['val_samples_ratio'])
data_tr, data_va = data_tr[:-val_data_len], data_tr[-val_data_len:]
loader_tr = MixedLoader(data_tr, batch_size=batch_size, epochs=epochs)
loader_va = MixedLoader(data_va, batch_size=batch_size)
loader_te = MixedLoader(data_te, batch_size=batch_size)


max_nodes = 50#len(train_dataset.loc[0].x)

class MyFilter(object):
    def __call__(self, data):
        return data.num_nodes <= max_nodes


class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 normalize=False, lin=True):
        super().__init__()

        self.conv1 = ChebConv(-1, hidden_channels, K=1)#DenseSAGEConv(in_channels, hidden_channels, normalize)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = ChebConv(hidden_channels, hidden_channels, K=1)#DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv3 = ChebConv(hidden_channels, out_channels, K=1)#DenseSAGEConv(hidden_channels, out_channels, normalize)
        self.bn3 = torch.nn.BatchNorm1d(out_channels)

        if lin is True:
            self.lin = torch.nn.Linear(2 * hidden_channels + out_channels,
                                       out_channels)
        else:
            self.lin = None

    def bn(self, i, x):
        batch_size, num_nodes, num_channels = x.size()

        x = x.view(-1, num_channels)
        x = getattr(self, f'bn{i}')(x)
        x = x.view(batch_size, num_nodes, num_channels)
        return x

    def forward(self, x, adj, mask=None):
        batch_size, num_nodes, in_channels = x.shape

        x0 = x
        #print("x0.shape ", x0.shape)
        x1 = self.bn(1, self.conv1(x0, adj, mask).relu())
        x2 = self.bn(2, self.conv2(x1, adj, mask).relu())
        x3 = self.bn(3, self.conv3(x2, adj, mask).relu())

        x = torch.cat([x1, x2, x3], dim=-1)

        if self.lin is not None:
            x = self.lin(x).relu()

        return x

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()

        num_nodes = ceil(0.25 * max_nodes)
        self.gnn1_pool = GNN(num_features, 64, num_nodes)
        self.gnn1_embed = GNN(num_features, 64, 64, lin=False)

        num_nodes = ceil(0.25 * num_nodes)
        self.gnn2_pool = GNN(3 * 64, 64, num_nodes)
        self.gnn2_embed = GNN(3 * 64, 64, 64, lin=False)

        self.gnn3_embed = GNN(3 * 64, 64, 64, lin=False)

        self.lin1 = torch.nn.Linear(3 * 64, 64)
        self.lin2 = torch.nn.Linear(64, 1)  # single output neuron

    def forward(self, x, adj, mask=None):
        s = self.gnn1_pool(x, adj, mask)
        x = self.gnn1_embed(x, adj, mask)

        x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask)

        s = self.gnn2_pool(x, adj)
        x = self.gnn2_embed(x, adj)

        x, adj, l2, e2 = dense_diff_pool(x, adj, s)

        x = self.gnn3_embed(x, adj)

        x = x.mean(dim=1)
        x = self.lin1(x).relu()
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1), l1 + l2, e1 + e2


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train(epoch):
    model.double()
    model.train()
    loss_all = 0

    for data in loader_tr:

        #data = data.to(device)
        optimizer.zero_grad()

        inputs, target = data
        x, adj = inputs

        x = torch.from_numpy(x).to(device)
        adj = torch.from_numpy(adj).to(device)
        target = torch.from_numpy(target).to(device)

        x = x.double()
        adj = adj.double()
        target = target.double()

        # print("data.x.shape ", x.size())
        # print("data.adj.shape ", adj.size())

        output, _, _ = model(x, adj)
        output = torch.squeeze(output)

        loss = F.mse_loss(output, target.view(-1))
        print("loss ", loss)
        loss.backward()
        loss_all += target.size(0) * float(loss)
        optimizer.step()
    return loss_all / len(train_dataset)


@torch.no_grad()
def test(loader):
    model.double()
    model.eval()
    correct = 0

    for data in loader:
        inputs, target = data
        x, adj = inputs


        x = torch.from_numpy(x).to(device)
        adj = torch.from_numpy(adj).to(device)
        target = torch.from_numpy(target).to(device)

        x = x.double()
        adj = adj.double()
        target = target.double()


        #data = data.to(device)
        pred = model(x, adj)[0].max(dim=1)[1]
        correct += int(pred.eq(target.view(-1)).sum())
    return correct / len(loader.dataset)


best_val_acc = test_acc = 0
for epoch in range(1, 151):
    train_loss = train(epoch)
    print("train loss ", train_loss)
    val_acc = test(loader_va)
    print("val_acc ", val_acc)
    if val_acc > best_val_acc:
        test_acc = test(loader_te)
        best_val_acc = val_acc
    print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, '
          f'Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')
