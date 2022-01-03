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

    def __init__(self, output_dir=None, level=0, log=False, mesh=None, corr_field_config=None, config={}, **kwargs):
        self._output_dir = output_dir
        # if self._output_dir is None:
        #     self._output_dir = OUTPUT_DIR
        self._log = log
        self.level = level
        self._mesh = mesh
        self._corr_field_config = corr_field_config
        self.adjacency_matrix = np.load(os.path.join(self._output_dir, "adjacency_matrix.npy"), allow_pickle=True)  # adjacency matrix
        self.data = []
        self._save_path = config['save_path']
        self._dataset_config = config.get('dataset_config', {})
        self._min_feature = None
        self._max_feature = None
        super().__init__(**kwargs)

        #self.a = self.adjacency_matrix
        self.dataset = pd.DataFrame(self.data)

    def get_test_data(self, index, length):
        new_dataset = self.dataset[0:index * length] + self.dataset[index * length + length:]
        new_obj = copy.deepcopy(self)
        new_obj.dataset = new_dataset

        return new_obj

    def shuffle(self, seed=None):
        if seed is not None:
            random.seed(seed)

        random.shuffle(self.data)
        self.dataset = pd.DataFrame(self.data)

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
        # if self._mesh is not None:
        #     return self.generate_data()

        # with open(os.path.join(OUTPUT_DIR, FlowDataset.GRAPHS_FILE), 'rb') as reader:
        #     graphs = pickle.loads(reader)
        #
        # if os.path.exists(os.path.join(OUTPUT_DIR, FlowDataset.DATA_FILE)):
        #     with open(os.path.join(OUTPUT_DIR, FlowDataset.DATA_FILE), 'rb') as reader:
        #         self.data = pickle.loads(reader)
        #
        # return graphs

        if self._dataset_config.get("features_normalization", False) or self._dataset_config.get("calc_output_mult_factor", False):
            # i = 0
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
                        all_features.extend(features)

                        if self._dataset_config.get("calc_output_mult_factor", False) is True:
                            output = np.load(os.path.join(sample_dir, "output.npy"))
                            all_outputs.append(output)

            #print("all outputs ", np.array(all_outputs).shape)
            # min_output = np.min(all_outputs)
            # max_output = np.max(all_outputs)

            if self._dataset_config.get("calc_output_mult_factor", False) is True:
                self._dataset_config["output_mult_factor"] = 1/np.mean(all_outputs)
                self._save_output_mult_factor()
                print("output mult factor ", self._dataset_config["output_mult_factor"])

            if self._dataset_config.get("features_normalization", False):
                self._min_feature = np.min(all_features)
                self._max_feature = np.max(all_features)

            # if self._log and self._dataset_config.get("features_log", False) is False and self._dataset_config.get("output_log",
            #                                                                                        False) is False:
            #     all_features = np.log(all_features)
            #     all_outputs = np.log(all_outputs)
            #
            # if self._dataset_config.get("features_log", False):
            #     all_features = np.log(all_features)
            #
            # if self._dataset_config.get("output_log", False):
            #     all_outputs = np.log(all_outputs)


            # self.min_output = min_output
            # self.max_output = max_output

        graphs = []
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
                    output = np.load(os.path.join(sample_dir, "output.npy"))

                    if self._min_feature is not None and self._max_feature is not None:
                        features = (features - self._min_feature) / (self._max_feature - self._min_feature)

                        # output = (output - min_output) / (max_output - min_output)
                        # print("max ", maximum)
                        # print("max ", minimum)
                        #
                        # print("new featuers max ", np.max(new_features))
                        # print("new featuers min ", np.min(new_features))
                        # exit()

                    output_mult_factor = self._dataset_config.get("output_mult_factor", 1)
                    features_mult_factor = self._dataset_config.get("features_mult_factor", 1)

                    features *= features_mult_factor
                    output *= output_mult_factor

                    if self._log and self._dataset_config.get("features_log", False) is False and self._dataset_config.get("output_log", False) is False:
                        features = np.log(features)
                        output = np.log(output)

                    if self._dataset_config.get("features_log", False):
                        features = np.log(features)

                    if self._dataset_config.get("output_log", False):
                        output = np.log(output)

                    graphs.append(Graph(x=features, y=output))#, a=self.adjacency_matrix))

                    # Save data for pandas dataframe creation, not used with Graph neural network
                    self.data.append({'x': features, 'y': output})

        self.a = self.adjacency_matrix
        return graphs

    def _save_output_mult_factor(self):
        # Save config to Pickle
        import pickle
        import shutil

        if os.path.exists(os.path.join(self._save_path, "dataset_config.pkl")):
            os.remove(os.path.join(self._save_path, "dataset_config.pkl"))

        # create a binary pickle file
        with open(os.path.join(self._save_path, "dataset_config.pkl"), "wb") as writer:
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
