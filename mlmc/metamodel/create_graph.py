import os
import os.path
import numpy as np
import networkx as nx
from mlmc.tool import gmsh_io
from mlmc.tool.hdf5 import HDF5
from mlmc.sample_storage_hdf import SampleStorageHDF
from mlmc.quantity.quantity import make_root_quantity
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


def extract_mesh_gmsh_io(mesh_file, get_points=False):
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

    if get_points:

        min_pt = np.min(centers, axis=0)
        max_pt = np.max(centers, axis=0)
        diff = max_pt - min_pt
        min_axis = np.argmin(diff)
        non_zero_axes = [0, 1, 2]
        # TODO: be able to use this mesh_dimension in fields
        if diff[min_axis] < 1e-10:
            non_zero_axes.pop(min_axis)
        points = centers[:, non_zero_axes]

        return {'points': points, 'point_region_ids': point_region_ids, 'ele_ids': ele_ids, 'region_map': region_map}

    return ele_nodes


def get_node_features(fields_mesh, feature_names):
    """
    Extract mesh from file
    :param fields_mesh: Mesh file
    :param feature_names: [[], []] - fields in each sublist are joint to one feature, each sublist corresponds to one vertex feature
    :return: list
    """
    mesh = gmsh_io.GmshIO(fields_mesh)

    features = []
    for f_names in feature_names:
        joint_features = join_fields(mesh._fields, f_names)

        features.append(list(joint_features.values()))

    return np.array(features).T


def join_fields(fields, f_names):
    if len(f_names) > 0:
        x_name = len(set([*fields[f_names[0]]]))
    assert all(x_name == len(set([*fields[f_n]])) for f_n in f_names)

    # # Using defaultdict
    # c = [collections.Counter(fields[f_n]) for f_n in f_names]
    # Cdict = collections.defaultdict(int)

    joint_dict = {}
    for f_n in f_names:
        for key, item in fields[f_n].items():
            #print("key: {}, item: {}".format(key, np.squeeze(item)))
            joint_dict.setdefault(key, 0)

            if joint_dict[key] != 0 and np.squeeze(item) != 0:
                raise ValueError("Just one field value should be non zero for each element")
            joint_dict[key] += np.squeeze(item)

    return joint_dict


def create_adjacency_matrix(ele_nodes):

    adjacency_matrix = np.zeros((len(ele_nodes), len(ele_nodes)))
    #adjacency_matrix = sparse.csr_matrix((len(ele_nodes), len(ele_nodes)))  #

    nodes = list(ele_nodes.values())
    for i in range(adjacency_matrix.shape[0]):
        ele_nodes = nodes[i]

        for j in range(i+1, len(nodes)):
            if i == j:
                continue
            ele_n = nodes[j]

            if len(list(set(ele_nodes).intersection(ele_n))) == 2:
                adjacency_matrix[j][i] = adjacency_matrix[i][j] = 1

    #print(np.count_nonzero(adjacency_matrix))
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


def graph_creator(output_dir, hdf_path, mesh, level=0, feature_names=[['conductivity']], quantity_name="conductivity"):
    adjacency_matrix = create_adjacency_matrix(extract_mesh_gmsh_io(mesh))
    np.save(os.path.join(output_dir, "adjacency_matrix"), adjacency_matrix, allow_pickle=True)
    loaded_adjacency_matrix = np.load(os.path.join(output_dir, "adjacency_matrix.npy"), allow_pickle=True)

    #plot_graph(loaded_adjacency_matrix)

    sample_storage = SampleStorageHDF(file_path=hdf_path)
    sample_storage.chunk_size = 1e8
    result_format = sample_storage.load_result_format()
    root_quantity = make_root_quantity(sample_storage, result_format)

    #@TODO:
    conductivity = root_quantity[quantity_name]
    time = conductivity[1]  # times: [1]
    location = time['0']  # locations: ['0']
    q_value = location[0, 0]

    hdf = HDF5(file_path=hdf_path, load_from_file=True)
    level_group = hdf.add_level_group(level_id=str(level))

    chunk_spec = next(sample_storage.chunks(level_id=level, n_samples=sample_storage.get_n_collected()[int(level)]))
    collected_values = q_value.samples(chunk_spec=chunk_spec)[0]

    collected_ids = sample_storage.collected_ids(level_id=level)

    indices = np.ones(len(collected_values))
    collected = zip(collected_ids, collected_values)

    graphs = []
    data = []
    i = 0
    for keep, (sample_id, col_values) in zip(indices, collected):
        if not keep:
            continue

        output_value = col_values[0]

        sample_dir = os.path.join(output_dir, sample_id)
        field_mesh = os.path.join(sample_dir, FIELDS_SAMPLE)
        if os.path.exists(field_mesh):
            # i += 1
            # if i > 150:
            #     break

            features = get_node_features(field_mesh, feature_names)
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
