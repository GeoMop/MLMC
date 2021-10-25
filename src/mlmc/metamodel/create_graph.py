import os
import os.path
import numpy as np
import networkx as nx
from mlmc.tool import gmsh_io
from mlmc.tool.hdf5 import HDF5
from spektral.data import Graph
from mlmc.metamodel.flow_dataset import FlowDataset


MESH = "/home/martin/Documents/metamodels/data/L1/test/01_cond_field/l_step_0.055_common_files/mesh.msh"
#FIELDS_SAMPLE_MESH = "/home/martin/Documents/metamodels/data/L1/test/01_cond_field/output/L00_S0000000/fine_fields_sample.msh"
FIELDS_SAMPLE = "fine_fields_sample.msh"
# OUTPUT_DIR = "/home/martin/Documents/metamodels/data/1000_ele/test/01_cond_field/output/"
# HDF_PATH = "/home/martin/Documents/metamodels/data/1000_ele/test/01_cond_field/mlmc_1.hdf5"

# OUTPUT_DIR = "/home/martin/Documents/metamodels/data/cl_0_3_s_4/L5/test/01_cond_field/output/"
# HDF_PATH = "/home/martin/Documents/metamodels/data/cl_0_3_s_4/L5/mlmc_5.hdf5"

# OUTPUT_DIR = "/home/martin/Documents/metamodels/data/cl_0_1_s_1/L5/test/01_cond_field/output/"
# HDF_PATH = "/home/martin/Documents/metamodels/data/cl_0_1_s_1/L5/mlmc_5.hdf5"

# OUTPUT_DIR = "/home/martin/Documents/metamodels/data/1000_ele/cl_0_1_s_1/L5/test/01_cond_field/output/"
# HDF_PATH = "/home/martin/Documents/metamodels/data/1000_ele/cl_0_1_s_1/L5/mlmc_5.hdf5"


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


def get_node_features(fields_mesh):
    mesh = gmsh_io.GmshIO(fields_mesh)
    element_data = mesh.current_elem_data
    features = list(element_data.values())
    return features


def create_adjacency_matrix(ele_nodes, joint_nodes=2):
    """
    Create adjacency matrix
    :param ele_nodes: Mesh file path
    :param joint_nodes: number of joint nodes that determines the adjacent vertices
    """
    adjacency_matrix = np.zeros((len(ele_nodes), len(ele_nodes)))
    #adjacency_matrix = sparse.csr_matrix((len(ele_nodes), len(ele_nodes)))  #

    nodes = list(ele_nodes.values())
    for i in range(adjacency_matrix.shape[0]):
        ele_nodes = nodes[i]

        for j in range(i+1, len(nodes)):
            if i == j:
                continue
            ele_n = nodes[j]

            if len(list(set(ele_nodes).intersection(ele_n))) >= joint_nodes:
                adjacency_matrix[j][i] = adjacency_matrix[i][j] = 1

    print(np.count_nonzero(adjacency_matrix))
    assert np.allclose(adjacency_matrix, adjacency_matrix.T)  # symmetry
    return adjacency_matrix


def plot_graph(adjacency_matrix):
    import matplotlib.pyplot as plt
    #G = nx.from_scipy_sparse_matrix(adjacency_matrix)
    G = nx.from_numpy_matrix(adjacency_matrix)
    nx.draw_kamada_kawai(G, with_labels=True, node_size=1, font_size=6)
    plt.axis('equal')
    plt.show()


def reject_outliers(data, m=2):
    #print("abs(data - np.mean(data)) < m * np.std(data) ", abs(data - np.mean(data)) < m * np.std(data))
    #return data[abs(data - np.mean(data)) < m * np.std(data)]
    return abs(data - np.mean(data)) < m * np.std(data)


def graph_creator(output_dir, hdf_path, mesh, level=0, joint_nodes=2):
    adjacency_matrix = create_adjacency_matrix(extract_mesh_gmsh_io(mesh), joint_nodes=joint_nodes)
    np.save(os.path.join(output_dir, "adjacency_matrix_{}".format(joint_nodes)), adjacency_matrix, allow_pickle=True)
    loaded_adjacency_matrix = np.load(os.path.join(output_dir, "adjacency_matrix_{}.npy".format(joint_nodes)), allow_pickle=True)

    plot_graph(loaded_adjacency_matrix)

    hdf = HDF5(file_path=hdf_path, load_from_file=True)
    level_group = hdf.add_level_group(level_id=str(level))

    # print("collected ", level_group.collected()[:, 0, :])
    # indices = reject_outliers(np.squeeze(level_group.collected()[:, 0, :]), m=6)
    # print("removed outliers ", np.count_nonzero(~indices))
    indices = np.ones(len(level_group.collected()))

    collected = zip(level_group.get_collected_ids(), level_group.collected())

    graphs = []
    data = []
    i = 0
    for keep, (sample_id, col_values) in zip(indices, collected):
        if not keep:
            continue
        output_value = col_values[0, 0]

        sample_dir = os.path.join(output_dir, sample_id)
        field_mesh = os.path.join(sample_dir, FIELDS_SAMPLE)
        if os.path.exists(field_mesh):
            # i += 1
            # if i > 150:
            #     break

            features = get_node_features(field_mesh)
            np.save(os.path.join(sample_dir, "nodes_features"), features)
            np.save(os.path.join(sample_dir, "output"), output_value)

            #graphs.append(Graph(x=features, y=output_value))  # , a=self.adjacency_matrix))
            # Save data for pandas dataframe creation, not used with Graph neural network
            #data.append({'x': features, 'y': output_value})

            #loaded_features = np.load(os.path.join(sample_dir, "nodes_features.npy"))
            #print("loaded features ", loaded_features)

    #FlowDataset.pickle_data(graphs, FlowDataset.GRAPHS_FILE)
    #FlowDataset.pickle_data(data, FlowDataset.DATA_FILE)


if __name__ == "__main__":

    # mesh = "/home/martin/Documents/metamodels/data/5_ele/cl_0_1_s_1/L5/l_step_0.020196309484414757_common_files/mesh.msh"
    # output_dir = "/home/martin/Documents/metamodels/data/5_ele/cl_0_3_s_4/L1_3/test/01_cond_field/output/"
    # hdf_path = "/home/martin/Documents/metamodels/data/5_ele/cl_0_1_s_1/L1_3/mlmc_1.hdf5"

    import cProfile
    import pstats
    pr = cProfile.Profile()
    pr.enable()

    my_result = graph_creator(output_dir, hdf_path, mesh)

    pr.disable()
    ps = pstats.Stats(pr).sort_stats('cumtime')
    ps.print_stats()
