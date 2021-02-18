import os
import numpy as np
import pandas as pd
from mlmc.tool import gmsh_io
from spektral.data import Dataset, Graph

MESH = "/home/martin/Documents/metamodels/data/L1/test/01_cond_field/l_step_0.055_common_files/mesh.msh"
FIELDS_SAMPLE = "fine_fields_sample.msh"
OUTPUT_DIR = "/home/martin/Documents/metamodels/data/L1/test/01_cond_field/output/"


class FlowDataset(Dataset):
    """
    """
    def __init__(self, **kwargs):
        self.adjacency_matrix = np.load(os.path.join(OUTPUT_DIR, "adjacency_matrix.npy"), allow_pickle=True)  # adjacency matrix
        self.data = []
        super().__init__(**kwargs)
        self.a = self.adjacency_matrix

        self.dataset = pd.DataFrame(self.data)

    def read(self):
        graphs = []
        for s_dir in os.listdir(OUTPUT_DIR):
            if os.path.isdir(os.path.join(OUTPUT_DIR, s_dir)):
                sample_dir = os.path.join(OUTPUT_DIR, s_dir)
                if os.path.exists(os.path.join(sample_dir, "nodes_features.npy")):
                    features = np.load(os.path.join(sample_dir, "nodes_features.npy"))
                    output = np.load(os.path.join(sample_dir, "output.npy"))
                    graphs.append(Graph(x=features, y=output))#, a=self.adjacency_matrix))

                    # Save data for pandas dataframe creation, not used with Graph neural network
                    self.data.append({'x': features, 'y': output})
        return graphs




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
