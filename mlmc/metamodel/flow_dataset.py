import os
import re
import copy
import random
import numpy as np
import pandas as pd
from mlmc.tool import gmsh_io
from mlmc.tool.flow_mc import FlowSim, create_corr_field
from spektral.data import Dataset, Graph
import pickle

#MESH = "/home/martin/Documents/metamodels/data/L1/test/01_cond_field/l_step_0.055_common_files/mesh.msh"
FIELDS_SAMPLE = "fine_fields_sample.msh"
#OUTPUT_DIR = "/home/martin/Documents/metamodels/data/1000_ele/test/01_cond_field/output/"
#OUTPUT_DIR = "/home/martin/Documents/metamodels/data/cl_0_3_s_4/L5/test/01_cond_field/output/"
#OUTPUT_DIR = "/home/martin/Documents/metamodels/data/cl_0_1_s_1/L5/test/01_cond_field/output/"
#OUTPUT_DIR = "/home/martin/Documents/metamodels/data/1000_ele/cl_0_1_s_1/L5/test/01_cond_field/output/"


class FlowDataset(Dataset):
    GRAPHS_FILE = "graphs"
    DATA_FILE = "data"

    def __init__(self, output_dir=None, level=0, log=False, mesh=None, corr_field_config=None, config={}, index=None,
                 predict=False, **kwargs):
        self._output_dir = output_dir
        # if self._output_dir is None:
        #     self._output_dir = OUTPUT_DIR
        self._log = log
        self.level = level
        self._mesh = mesh
        self._corr_field_config = corr_field_config
        self.adjacency_matrix = np.load(os.path.join(self._output_dir, "adjacency_matrix.npy"), allow_pickle=True)  # adjacency matrix
        self.data = []
        self._config = config
        self._index = index
        self._predict = predict
        self._dataset_config = config.get('dataset_config', {})
        self._augment_data = config.get('augment_data', False)

        if predict:
            self._min_features = self._dataset_config.get('min_features', None)
            self._max_features = self._dataset_config.get('max_features', None)
            self._mean_features = self._dataset_config.get('mean_features', None)
            self._var_features = self._dataset_config.get('var_features', None)
            self._min_output = self._dataset_config.get('min_output', None)
            self._max_output = self._dataset_config.get('max_output', None)
            self._mean_output = self._dataset_config.get('mean_output', None)
            self._var_output = self._dataset_config.get('var_output', None)
            self._output_mult_factor = self._dataset_config.get('output_mult_factor', 1)
        else:
            self._min_features = None
            self._max_features = None
            self._mean_features = None
            self._var_features = None
            self._min_output = None
            self._max_output = None
            self._mean_output = None
            self._var_output = None
            self._output_mult_factor = 1

        self._aug_data = []
        self._columns = None

        super().__init__(**kwargs)

        #self.a = self.adjacency_matrix
        self.dataset = pd.DataFrame(self.data)

        self._df_for_augmentation = pd.DataFrame(self._aug_data, columns=self._columns)

        #self._data_augmentation()

    def get_train_data(self, index, length):
        new_dataset = self.dataset[index * length: index * length + length]
        new_graphs = self.graphs[index * length: index * length + length]  # self.graphs is read() method output

        if self._augment_data:
            new_dataset, new_graphs = self._data_augmentation(self._df_for_augmentation[index * length: index * length + length], new_graphs)

        new_obj = copy.deepcopy(self)
        new_obj.dataset = new_dataset
        new_obj.graphs = new_graphs

        return new_obj

    def split_val_train(self, len_val_data):
        tr_dataset = self.dataset[:-len_val_data]
        va_dataset = self.dataset[-len_val_data:]

        tr_graphs = self.graphs[:-len_val_data]
        va_graphs = self.graphs[-len_val_data:]
        #new_graphs = self.graphs[index * length: index * length + length]  # self.graphs is read() method output

        tr_obj = copy.deepcopy(self)
        va_obj = copy.deepcopy(self)

        tr_obj.dataset = tr_dataset
        va_obj.dataset = va_dataset

        tr_obj.graphs = tr_graphs
        va_obj.graphs = va_graphs

        return tr_obj, va_obj

    def get_test_data(self, index, length):
        new_dataset = self.dataset[0:index * length] + self.dataset[index * length + length:]
        new_graphs = self.graphs[0:index * length] + self.graphs[index * length + length:]
        new_obj = copy.deepcopy(self)
        new_obj.dataset = new_dataset
        new_obj.graphs = new_graphs
        return new_obj

    def shuffle(self, seed=None):
        if seed is not None:
            random.seed(seed)

        random.shuffle(self.data)
        self.dataset = pd.DataFrame(self.data)

    def _data_augmentation(self, df_slice, new_graphs):
        import smogn
        import matplotlib.pyplot as plt
        import seaborn

        # df_slice = self._df_for_augmentation[self._index * self._config['n_train_samples']:
        #                             self._index * self._config['n_train_samples'] + self._config['n_train_samples']]

        df_slice = df_slice.reset_index(drop=True)

        if 'augmentation_config' in self._config:
            dataset_modified = smogn.smoter(data=df_slice, y="y", **self._config["augmentation_config"])
        else:
            dataset_modified = smogn.smoter(data=df_slice, y="y", k=9, samp_method="extreme")

        # print("dataset modified shape",  dataset_modified.shape)
        # print("dataset modified ", dataset_modified)
        #
        # print("df stats ", smogn.box_plot_stats(df_slice['y'])['stats'])
        # print("modified stats ", smogn.box_plot_stats(dataset_modified['y'])['stats'])
        #
        # fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        # ax.hist(df_slice['y'], bins=50, alpha=0.5, label='target', density=True)
        # plt.title("original")
        # plt.show()
        #
        # fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        # ax.hist(dataset_modified['y'], bins=50, alpha=0.5, label='target', density=True)
        # plt.title("modified")
        # plt.show()

        appended_dataset = df_slice.append(dataset_modified)

        numpy_frame = dataset_modified.to_numpy()

        for i in range(numpy_frame.shape[0]):
            features = numpy_frame[i][:-1]
            features = features.reshape((len(features), 1))
            new_graphs.append(Graph(x=features, y=numpy_frame[i][-1]))

        return appended_dataset, copy.deepcopy(new_graphs)

        # seaborn.kdeplot(df_slice['y'], label="Original")
        # seaborn.kdeplot(dataset_modified['y'], label="Modified")


    # def generate_data(self):
    #     n_samples = 10**5
    #     graphs = []
    #     mesh_data = FlowSim.extract_mesh(self._mesh)
    #     fields = create_corr_field(model="exp", dim=2,
    #                                sigma=self._corr_field_config['sigma'],
    #                                corr_length=self._corr_field_config['corr_length'],
    #                                log=self._corr_field_config['log'])
    #
    #     # # Create fields both fine and coarse
    #     fields = FlowSim.make_fields(fields, mesh_data, None)
    #
    #     for i in range(n_samples):
    #         fine_input_sample, coarse_input_sample = FlowSim.generate_random_sample(fields, coarse_step=0,
    #                                                                                 n_fine_elements=len(
    #                                                                                     mesh_data['points']))
    #         # print("len fine input sample ", len(fine_input_sample["conductivity"]))
    #         # print("fine input sample ", fine_input_sample["conductivity"])
    #
    #         features = fine_input_sample["conductivity"]
    #         output = 1
    #
    #         # gmsh_io.GmshIO().write_fields('fields_sample.msh', mesh_data['ele_ids'], fine_input_sample)
    #         #
    #         # mesh = gmsh_io.GmshIO('fields_sample.msh')
    #         # element_data = mesh.current_elem_data
    #         # features = list(element_data.values())
    #
    #         if self._log:
    #             features = np.log(features)
    #             output = np.log(output)
    #             # features = (features - minimum) / (maximum - minimum)
    #         graphs.append(Graph(x=features, y=output))  # , a=self.adjacency_matrix))
    #         # Save data for pandas dataframe creation, not used with Graph neural network
    #         self.data.append({'x': features, 'y': output})
    #
    #     self.a = self.adjacency_matrix
    #     return graphs

    def read(self):
        all_outputs = []
        all_features = []

        for s_dir in os.listdir(self._output_dir):
            try:
                l = re.findall(r'L(\d+)_S', s_dir)[0]
                if int(l) != self.level:
                    continue
            except IndexError:
                    continue
            if os.path.isdir(os.path.join(self._output_dir, s_dir)):
                sample_dir = os.path.join(self._output_dir, s_dir)
                if os.path.exists(os.path.join(sample_dir, "nodes_features.npy")):
                    features = np.load(os.path.join(sample_dir, "nodes_features.npy"))
                    all_features.append(features)

                    output = np.load(os.path.join(sample_dir, "output.npy"))
                    all_outputs.append(output)

        if self._dataset_config.get("first_log_features", False):
            all_features = np.log(all_features)

        if self._dataset_config.get("first_log_output", False):
            all_outputs = np.log(all_outputs)

        #print("all outputs ", np.array(all_outputs).shape)
        # min_output = np.min(all_outputs)
        # max_output = np.max(all_outputs)

        if not self._predict:
            if self._index is not None:
                train_outputs = all_outputs[self._index * self._config['n_train_samples']:
                                            self._index * self._config['n_train_samples'] + self._config['n_train_samples']]
                train_features = all_features[self._index * self._config['n_train_samples']:
                                              self._index * self._config['n_train_samples'] + self._config['n_train_samples']]
            else:
                train_outputs = all_outputs
                train_features = all_features

            if self._dataset_config.get("calc_output_mult_factor", False) is True:
                self._output_mult_factor = 1/np.mean(train_outputs)
                self._dataset_config["output_mult_factor"] = self._output_mult_factor
                self._save_data_config()

            if self._dataset_config.get("features_normalization", False):
                self._min_features = np.min(train_features, axis=0)
                self._max_features = np.max(train_features, axis=0)
                self._dataset_config["min_features"] = self._min_features
                self._dataset_config["max_features"] = self._max_features

            if self._dataset_config.get("features_scale", False):
                self._mean_features = np.mean(train_features, axis=0)
                self._var_features = np.var(train_features, axis=0)
                self._dataset_config["mean_features"] = self._mean_features
                self._dataset_config["var_features"] = self._var_features

            if self._dataset_config.get("output_normalization", False):
                self._min_output = np.min(train_outputs, axis=0)
                self._max_output = np.max(train_outputs, axis=0)
                self._dataset_config["min_output"] = self._min_output
                self._dataset_config["max_output"] = self._max_output

            if self._dataset_config.get("output_scale", False):
                self._mean_output = np.mean(train_outputs, axis=0)
                self._var_output = np.var(train_outputs, axis=0)
                self._dataset_config["mean_output"] = self._mean_output
                self._dataset_config["var_output"] = self._var_output

            self._save_data_config()

        graphs = []
        for features, output in zip(all_features, all_outputs):
            if self._dataset_config.get("features_normalization", False):
                features = (features - self._min_features) / (self._max_features - self._min_features)
                features = np.nan_to_num(features)

            if self._dataset_config.get("output_normalization", False):
                output = (output - self._min_output) / (self._max_output - self._min_output)
                output = np.nan_to_num(output)

            if self._dataset_config.get("features_scale", False):
                features -= self._mean_features
                features /= self._var_features

            if self._dataset_config.get("output_scale", False):
                output -= self._mean_output
                output /= self._var_output

                # output = (output - min_output) / (max_output - min_output)
                # print("max ", maximum)
                # print("max ", minimum)
                #
                # print("new featuers max ", np.max(new_features))
                # print("new featuers min ", np.min(new_features))
                # exit()

            output_mult_factor = self._output_mult_factor
            features_mult_factor = self._dataset_config.get("features_mult_factor", 1)

            features *= features_mult_factor
            output *= output_mult_factor

            if self._log and self._dataset_config.get("features_log", False) is False and self._dataset_config.get("output_log", False) is False:
                features = np.log(features)
                output = np.log(output)

            if self._dataset_config.get("last_log_features", False):
                features = np.log(features)

            if self._dataset_config.get("last_log_output", False):
                output = np.log(output)

            graphs.append(Graph(x=features, y=output))#, a=self.adjacency_matrix))

            # Save data for pandas dataframe creation, not used with Graph neural network
            self.data.append({'x': features, 'y': output})

            if self._augment_data:
                if self._columns is None:
                    d = [pd.DataFrame(features.reshape((features.shape[1], features.shape[0])).tolist()).add_prefix("x_")]
                    new_df = pd.concat(d, axis=1)
                    #new_df.insert(loc=0, column="y", value=output)
                    self._columns = []
                    for col in new_df.columns:
                        self._columns.append(col)
                    self._columns.append("y")

                squeezed_features = list(np.squeeze(features))
                squeezed_features.append(output)
                self._aug_data.append(squeezed_features)


            #new_df["y"] = output
            # print("new df y ", new_df["y"])
            # print("new df .shape ", new_df.shape)
            #print("new df ", new_df)

            # if self._df_for_augmentation is not None:
            #     self._df_for_augmentation = self._df_for_augmentation.append(new_df)
            # else:
            #     self._df_for_augmentation = new_df


        self.a = self.adjacency_matrix
        return graphs

    def _save_data_config(self):
        # Save config to Pickle
        import pickle
        import shutil

        if "iter_dir" in self._config:
            if os.path.exists(os.path.join(self._config['iter_dir'], "dataset_config.pkl")):
                os.remove(os.path.join(self._config['iter_dir'], "dataset_config.pkl"))

            # create a binary pickle file
            with open(os.path.join(self._config['iter_dir'], "dataset_config.pkl"), "wb") as writer:
                pickle.dump(self._dataset_config, writer)

    @staticmethod
    def pickle_data(data, output_dir, file_path):
        with open(os.path.join(output_dir, file_path), 'wb') as writer:
            pickle.dump(data, writer)


def extract_mesh_gmsh_io(mesh_file):
    """
    Extract mesh from file
    :param mesh_file: Mesh file path
    :return: Dict
    """
    mesh = gmsh_io.GmshIO(mesh_file)
    is_bc_region = {}
    region_map = {}
    for name, (id, _) in mesh.physical.items():
        unquoted_name = name.strip("\"'")
        is_bc_region[id] = (unquoted_name[0] == '.')
        region_map[unquoted_name] = id

    bulk_elements = []

    for id, el in mesh.elements.items():
        _, tags, i_nodes = el
        region_id = tags[0]
        if not is_bc_region[region_id]:
            bulk_elements.append(id)

    n_bulk = len(bulk_elements)
    centers = np.empty((n_bulk, 3))
    ele_ids = np.zeros(n_bulk, dtype=int)
    ele_nodes = {}
    point_region_ids = np.zeros(n_bulk, dtype=int)

    for i, id_bulk in enumerate(bulk_elements):
        _, tags, i_nodes = mesh.elements[id_bulk]
        region_id = tags[0]
        centers[i] = np.average(np.array([mesh.nodes[i_node] for i_node in i_nodes]), axis=0)
        point_region_ids[i] = region_id
        ele_ids[i] = id_bulk
        ele_nodes[id_bulk] = i_nodes

    return ele_nodes
