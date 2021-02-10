import os
import os.path
import numpy as np
from mlmc.tool import gmsh_io

MESH = "/home/martin/Documents/metamodels/data/L1/test/01_cond_field/l_step_0.055_common_files/mesh.msh"
#FIELDS_SAMPLE_MESH = "/home/martin/Documents/metamodels/data/L1/test/01_cond_field/output/L00_S0000000/fine_fields_sample.msh"
FIELDS_SAMPLE = "fine_fields_sample.msh"
OUTPUT_DIR = "/home/martin/Documents/metamodels/data/L1/test/01_cond_field/output/"


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


def create_adjacency_matrix(ele_nodes):
    adjacency_matrix = np.zeros((len(ele_nodes), len(ele_nodes)))
    #adjacency_matrix = sparse.csr_matrix((len(ele_nodes), len(ele_nodes)))  #

    nodes = list(ele_nodes.values())
    for i in range(adjacency_matrix.shape[0]):
        ele_nodes = nodes[i]
        for j in range(i+1, len(nodes)):
            #print("i: {}, j: {}".format(i, j))
            if i == j:
                continue
            ele_n = nodes[j]
            if len(list(set(ele_nodes).intersection(ele_n))) == 2:
                adjacency_matrix[j][i] = adjacency_matrix[i][j] = 1

    print(np.count_nonzero(adjacency_matrix))
    assert np.allclose(adjacency_matrix, adjacency_matrix.T)  # symmetry
    return adjacency_matrix


def extract_mesh():
    adjacency_matrix = create_adjacency_matrix(extract_mesh_gmsh_io(MESH))
    np.save(os.path.join(OUTPUT_DIR, "adjacency_matrix"), adjacency_matrix, allow_pickle=True)
    loaded_adjacency_matrix = np.load(os.path.join(OUTPUT_DIR, "adjacency_matrix.npy"), allow_pickle=True)

    print("loaded adjacency matrix ", loaded_adjacency_matrix)
    ele_nodes = extract_mesh_gmsh_io(MESH)

    for s_dir in os.listdir(OUTPUT_DIR):
        if os.path.isdir(os.path.join(OUTPUT_DIR, s_dir)):
            sample_dir = os.path.join(OUTPUT_DIR, s_dir)
            field_mesh = os.path.join(sample_dir, FIELDS_SAMPLE)
            features = get_node_features(field_mesh)

            np.save(os.path.join(sample_dir, "nodes_features"), features)
            #loaded_features = np.load(os.path.join(sample_dir, "nodes_features.npy"))
            #print("loaded features ", loaded_features)


if __name__ == "__main__":
    import cProfile
    import pstats
    pr = cProfile.Profile()
    pr.enable()

    my_result = extract_mesh()

    pr.disable()
    ps = pstats.Stats(pr).sort_stats('cumtime')
    ps.print_stats()
