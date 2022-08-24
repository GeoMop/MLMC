import numpy as np
import networkx as nx
# import breadth_first_search as bfs
# import feature_maps as fm
from scipy.sparse import save_npz
from scipy.sparse import csc_matrix
from scipy.sparse import lil_matrix


def compute_centrality(adj):
    n = len(adj)
    adj = adj + np.eye(n)
    cen = np.zeros(n)
    G = nx.from_numpy_matrix(adj)
    nodes = nx.eigenvector_centrality(G, max_iter=1000, tol=1.0e-4)
    for i in range(len(nodes)):
        cen[i] = nodes[i]

    return cen


def canonicalization(adj, graph_data, filter_size, feature_type, graphlet_size, max_h):
    depth = 10
    graphs = {}
    labels = {}
    attributes = {}
    num_graphs = len(graph_data[0])
    centrality_vector = {}

    num_sample = 0

    print("graph_data ", graph_data)

    for gidx in range(num_graphs):
        # adj = graph_data[0][gidx]['am'].toarray()
        #adj = graph_data[0][gidx]['am']
        n = len(adj)
        if n >= num_sample:
            num_sample = n

        graphs[gidx] = adj
        v = compute_centrality(adj)
        # print("v ", v.shape)
        # exit()
        centrality_vector[gidx] = v

        degree = np.sum(adj, axis=1)
        if hasnl == 0:
            labels[gidx] = degree
        else:
            label = graph_data[0][gidx]['nl'].T
            labels[gidx] = label[0]

    if feature_type == 1:
        features = graphlet_feature_map(num_graphs, graphs, graphlet_size, 20)

    elif feature_type == 2:
        features = shortest_path_feature_map(num_graphs, graphs, labels)

    elif feature_type == 3:
        features = wl_subtree_feature_map(num_graphs, graphs, labels, max_h)

    else:
        raise Exception("Unknown feature type!")

    for gidx in range(num_graphs):
        path_feature = features[gidx]
        attributes[gidx] = path_feature

    # building tree-structured filters
    all_samples = {}

    for gidx in range(num_graphs):
        adj = graphs[gidx]
        nx_G = nx.from_numpy_matrix(adj)
        label = labels[gidx]
        nodetrees = []
        n = len(adj)
        cen = centrality_vector[gidx]

        sorting_vertex = -1 * np.ones(num_sample)
        cen_v = np.zeros(n)
        vertex = np.zeros(n)
        for i in range(n):
            vertex[i] = i
            cen_v[i] = cen[i]
        sub = np.argsort(cen_v)
        vertex = vertex[sub]

        if num_sample <= n:
            for i in range(num_sample):
                sorting_vertex[i] = vertex[i]

        else:
            for i in range(n):
                sorting_vertex[i] = vertex[i]

        sample = []
        for node in sorting_vertex:

            if node != -1:
                edges = list(bfs.bfs_edges(nx_G, cen, source=int(node), depth_limit=depth))
                truncated_edges = edges[:filter_size - 1]
                if not truncated_edges or len(truncated_edges) != filter_size - 1:
                    continue
                else:
                    tmp = []
                    tmp_cen = []
                    tmp.append(int(node))
                    tmp_cen.append(cen[int(node)])
                    for u, v in truncated_edges:
                        tmp.append(int(v))
                        tmp_cen.append(cen[int(v)])
                    tmp_cen = np.array(tmp_cen)
                    tmp_cen = -1 * tmp_cen
                    sub = np.argsort(tmp_cen)
                    tmp = np.array(tmp)
                    tmp = tmp[sub]
                    for v in tmp:
                        sample.append(v)
            else:
                for i in range(filter_size):
                    sample.append(-1)

        all_samples[gidx] = sample

    att = attributes[0]
    feature_size = att.shape[1]

    graph_tensor = []
    for gidx in range(num_graphs):
        sample = all_samples[gidx]
        att = attributes[gidx]
        feature_matrix = csc_matrix((num_sample * filter_size, feature_size), dtype=np.float32)
        pointer = 0
        for node in sample:
            if node != -1:
                feature_matrix[pointer, :] = att[node, :]

            pointer += 1

        print("feature_matrix.shape ", feature_matrix.shape)
        graph_tensor.append(feature_matrix)

    return graph_tensor, feature_size, num_sample



import networkx as nx
from gensim import corpora
import gensim
import breadth_first_search as bfs
from collections import defaultdict
import numpy as np
import copy, pickle
import pynauty
from sklearn.preprocessing import normalize
from scipy.sparse import csc_matrix


def get_graphlet(window, nsize):
    """
    This function takes the upper triangle of a nxn matrix and computes its canonical map
    """
    adj_mat = {idx: [i for i in list(np.where(edge)[0]) if i!=idx] for idx, edge in enumerate(window)}

    g = pynauty.Graph(number_of_vertices=nsize, directed=False, adjacency_dict = adj_mat)
    cert = pynauty.certificate(g)
    return cert


def get_maps(n):
    # canonical_map -> {canonical string id: {"graph", "idx", "n"}}
    file_counter = open("canonical_maps/canonical_map_n%s.p"%n, "rb")
    canonical_map = pickle.load(file_counter, encoding='bytes')
    file_counter.close()
    # weight map -> {parent id: {child1: weight1, ...}}
    file_counter = open("graphlet_counter_maps/graphlet_counter_nodebased_n%s.p"%n, "rb")
    weight_map = pickle.load(file_counter, encoding='bytes')
    file_counter.close()
    weight_map = {parent: {child: weight/float(sum(children.values())) for child, weight in children.items()} for parent, children in weight_map.items()}
    child_map = {}
    for parent, children in weight_map.items():
        for k,v in children.items():
            if k not in child_map:
                child_map[k] = {}
            child_map[k][parent] = v
    weight_map = child_map
    return canonical_map, weight_map


def adj_wrapper(g):
    am_ = g["al"]
    size = max(np.shape(am_))
    am = np.zeros((size, size))
    for idx, i in enumerate(am_):
        for j in i:
            am[idx][j-1] = 1
    return am


def graphlet_feature_map(num_graphs, graphs, num_graphlets, samplesize):
    # if no graphlet is found in a graph, we will fall back to 0th graphlet of size k
    fallback_map = {1: 1, 2: 2, 3: 4, 4: 8, 5: 19, 6: 53, 7: 209, 8: 1253, 9: 13599}
    canonical_map, weight_map = get_maps(num_graphlets)
    canonical_map1, weight_map1 = get_maps(2)
    # randomly sample graphlets
    graph_map = {}
    graphlet_graph = []
    for gidx in range(num_graphs):
        #print(gidx)
        am = graphs[gidx]
        m = len(am)
        for node in range(m):
            graphlet_node = []
            for j in range(samplesize):
                rand = np.random.permutation(range(m))
                r = []
                r.append(node)
                for ele in rand:
                    if ele != node:
                        r.append(ele)

                for n in [num_graphlets]:
                #for n in range(3,6):
                    if m >= num_graphlets:
                        window = am[np.ix_(r[0:n], r[0:n])]
                        g_type = canonical_map[get_graphlet(window, n)]
                        #for key, value in g_type.items():
                        #    print(key.decode("utf-8"))
                        #    print(value)
                        graphlet_idx = str(g_type["idx".encode()])
                    else:
                        window = am[np.ix_(r[0:2], r[0:2])]
                        g_type = canonical_map1[get_graphlet(window, 2)]
                        graphlet_idx = str(g_type["idx".encode()])

                    graphlet_node.append(graphlet_idx)

            graphlet_graph.append(graphlet_node)

    dictionary = corpora.Dictionary(graphlet_graph)
    corpus = [dictionary.doc2bow(graphlet_node) for graphlet_node in graphlet_graph]
    M = gensim.matutils.corpus2csc(corpus, dtype=np.float32)
    M = normalize(M, norm='l1', axis=0)
    M = M.T

    allFeatures = {}
    index = 0
    for gidx in range(num_graphs):
        adj = graphs[gidx]
        n = len(adj)
        graphlet_feature = M[index:index + n, :]
        index += n
        allFeatures[gidx] = graphlet_feature

    return allFeatures


def wl_subtree_feature_map(num_graphs, graphs, labels, max_h):
    alllabels = {}
    label_lookup = {}
    label_counter = 0
    wl_graph_map = {it: {gidx: defaultdict(lambda: 0) for gidx in range(num_graphs)} for it in range(-1, max_h)}

    alllabels[0] = labels
    new_labels = {}
    # initial labeling
    for gidx in range(num_graphs):
        adj = graphs[gidx]
        n = len(adj)
        new_labels[gidx] = np.zeros(n, dtype=np.int32)
        label = labels[gidx]

        for node in range(n):
            la = label[node]
            if la not in label_lookup:
                label_lookup[la] = label_counter
                new_labels[gidx][node] = label_counter
                label_counter += 1
            else:
                new_labels[gidx][node] = label_lookup[la]
            wl_graph_map[-1][gidx][label_lookup[la]] = wl_graph_map[-1][gidx].get(label_lookup[la], 0) + 1
    compressed_labels = copy.deepcopy(new_labels)
    # WL iterations started
    for it in range(max_h - 1):
        label_lookup = {}
        label_counter = 0
        for gidx in range(num_graphs):
            adj = graphs[gidx]
            n = len(adj)
            nx_G = nx.from_numpy_matrix(adj)
            for node in range(n):
                node_label = tuple([new_labels[gidx][node]])
                neighbors = []
                edges = list(bfs.bfs_edges(nx_G, np.zeros(n), source=node, depth_limit=1))
                for u, v in edges:
                    neighbors.append(v)

                if len(neighbors) > 0:
                    neighbors_label = tuple([new_labels[gidx][i] for i in neighbors])
                    node_label = tuple(tuple(node_label) + tuple(sorted(neighbors_label)))
                if node_label not in label_lookup:
                    label_lookup[node_label] = str(label_counter)
                    compressed_labels[gidx][node] = str(label_counter)
                    label_counter += 1
                else:
                    compressed_labels[gidx][node] = label_lookup[node_label]
                wl_graph_map[it][gidx][label_lookup[node_label]] = wl_graph_map[it][gidx].get(label_lookup[node_label],
                                                                                              0) + 1
        # print("Number of compressed labels at iteration %s: %s"%(it, len(label_lookup)))
        new_labels = copy.deepcopy(compressed_labels)
        # print("labels")
        # print(labels)
        alllabels[it + 1] = new_labels

    subtrees_graph = []
    for gidx in range(num_graphs):
        adj = graphs[gidx]
        n = len(adj)
        for node in range(n):
            subtrees_node = []
            for it in range(max_h):
                graph_label = alllabels[it]
                label = graph_label[gidx]
                subtrees_node.append(str(label[node]))

            subtrees_graph.append(subtrees_node)

    dictionary = corpora.Dictionary(subtrees_graph)
    corpus = [dictionary.doc2bow(subtrees_node) for subtrees_node in subtrees_graph]
    M = gensim.matutils.corpus2csc(corpus, dtype=np.float32)
    M = M.T

    allFeatures = {}
    index = 0
    for gidx in range(num_graphs):
        adj = graphs[gidx]
        n = len(adj)
        subtree_feature = M[index:index + n, :]
        index += n
        allFeatures[gidx] = subtree_feature

    return allFeatures


def shortest_path_feature_map(num_graphs, graphs, labels):
    sp_graph = []
    for gidx in range(num_graphs):
        adj = graphs[gidx]
        n = len(adj)
        label = labels[gidx]
        nx_G = nx.from_numpy_matrix(adj)

        for i in range(n):
            sp_node = []
            for j in range(n):
                if i != j:
                    try:
                        path = list(nx.shortest_path(nx_G, i, j))
                    except nx.exception.NetworkXNoPath:
                        continue

                    if not path:
                        continue
                    if label[i] <=label[j]:
                        sp_label = str(int(label[i])) + ',' + str(int(label[j])) + ',' + str(len(path))
                    else:
                        sp_label = str(int(label[j])) + ',' + str(int(label[i])) + ',' + str(len(path))
                    sp_node.append(sp_label)
            sp_graph.append(sp_node)

    dictionary = corpora.Dictionary(sp_graph)
    corpus = [dictionary.doc2bow(sp_node) for sp_node in sp_graph]
    M = gensim.matutils.corpus2csc(corpus, dtype=np.float32)
    M = M.T

    allFeatures = {}
    index = 0
    for gidx in range(num_graphs):
        adj = graphs[gidx]
        n = len(adj)
        sp_feature = M[index:index + n, :]
        index += n
        allFeatures[gidx] = sp_feature

    return allFeatures


# breadth_first_search.py - breadth-first traversal of a graph
#
# Copyright (C) 2004-2018 NetworkX Developers
#   Aric Hagberg <hagberg@lanl.gov>
#   Dan Schult <dschult@colgate.edu>
#   Pieter Swart <swart@lanl.gov>
#
# This file is part of NetworkX.
#
# NetworkX is distributed under a BSD license; see LICENSE.txt for more
# information.
#
# Authors:
#     Aric Hagberg <aric.hagberg@gmail.com>
#
"""Basic algorithms for breadth-first searching the nodes of a graph."""
import networkx as nx
import copy
from collections import deque

__all__ = ['bfs_edges', 'bfs_tree', 'bfs_predecessors', 'bfs_successors']


def generic_bfs_edges(G, label, source, neighbors=None, depth_limit=None):
    """Iterate over edges in a breadth-first search.

    The breadth-first search begins at `source` and enqueues the
    neighbors of newly visited nodes specified by the `neighbors`
    function.

    Parameters
    ----------
    G : NetworkX graph

    source : node
        Starting node for the breadth-first search; this function
        iterates over only those edges in the component reachable from
        this node.

    neighbors : function
        A function that takes a newly visited node of the graph as input
        and returns an *iterator* (not just a list) of nodes that are
        neighbors of that node. If not specified, this is just the
        ``G.neighbors`` method, but in general it can be any function
        that returns an iterator over some or all of the neighbors of a
        given node, in any order.

    depth_limit : int, optional(default=len(G))
        Specify the maximum search depth

    Yields
    ------
    edge
        Edges in the breadth-first search starting from `source`.

    Examples
    --------
    >>> G = nx.path_graph(3)
    >>> print(list(nx.bfs_edges(G,0)))
    [(0, 1), (1, 2)]
    >>> print(list(nx.bfs_edges(G, source=0, depth_limit=1)))
    [(0, 1)]

    Notes
    -----
    This implementation is from `PADS`_, which was in the public domain
    when it was first accessed in July, 2004.  The modifications
    to allow depth limits based on the Wikipedia article
    "`Depth-limited-search`_".

    .. _PADS: http://www.ics.uci.edu/~eppstein/PADS/BFS.py
    .. _Depth-limited-search: https://en.wikipedia.org/wiki/Depth-limited_search
    """
    visited = {source}
    if depth_limit is None:
        depth_limit = len(G)

    neigh = list(neighbors(source))
    # print(neigh)
    neighlabel = []
    for nei in neigh:
        neighlabel.append(label[int(nei)])
    # print(neighlabel)
    neighindex = sorted(range(len(neighlabel)), key=lambda k: neighlabel[k], reverse=True)
    sortedneighbor = []
    for ele in neighindex:
        sortedneighbor.append(neigh[ele])

    queue = deque([(source, depth_limit, iter(sortedneighbor))])
    # print(type(neighbors(source)))
    # print(list(iter(sortedneighbor)))

    while queue:
        parent, depth_now, children = queue[0]
        try:
            child = next(children)
            if child not in visited:
                yield parent, child
                visited.add(child)
                if depth_now > 1:
                    chil = list(neighbors(child))
                    chillabel = []
                    for ch in chil:
                        chillabel.append(label[int(ch)])

                    chilindex = sorted(range(len(chillabel)), key=lambda k: chillabel[k], reverse=True)
                    sortedneighbor = []
                    for ele in chilindex:
                        sortedneighbor.append(chil[ele])

                    queue.append((child, depth_now - 1, iter(sortedneighbor)))
        except StopIteration:
            queue.popleft()


def bfs_edges(G, label, source, reverse=False, depth_limit=None):
    """Iterate over edges in a breadth-first-search starting at source.

    Parameters
    ----------
    G : NetworkX graph

    source : node
       Specify starting node for breadth-first search and return edges in
       the component reachable from source.

    reverse : bool, optional
       If True traverse a directed graph in the reverse direction

    depth_limit : int, optional(default=len(G))
        Specify the maximum search depth

    Returns
    -------
    edges: generator
       A generator of edges in the breadth-first-search.

    Examples
    --------
    To get the edges in a breadth-first search::

        >>> G = nx.path_graph(3)
        >>> list(nx.bfs_edges(G, 0))
        [(0, 1), (1, 2)]
        >>> list(nx.bfs_edges(G, source=0, depth_limit=1))
        [(0, 1)]

    To get the nodes in a breadth-first search order::

        >>> G = nx.path_graph(3)
        >>> root = 2
        >>> edges = nx.bfs_edges(G, root)
        >>> nodes = [root] + [v for u, v in edges]
        >>> nodes
        [2, 1, 0]

    Notes
    -----
    Based on http://www.ics.uci.edu/~eppstein/PADS/BFS.py.
    by D. Eppstein, July 2004. The modifications
    to allow depth limits based on the Wikipedia article
    "`Depth-limited-search`_".

    .. _Depth-limited-search: https://en.wikipedia.org/wiki/Depth-limited_search
    """
    if reverse and G.is_directed():
        successors = G.predecessors
    else:
        successors = G.neighbors
    # TODO In Python 3.3+, this should be `yield from ...`
    for e in generic_bfs_edges(G, label, source, successors, depth_limit):
        yield e


def bfs_tree(G, source, reverse=False, depth_limit=None):
    """Return an oriented tree constructed from of a breadth-first-search
    starting at source.

    Parameters
    ----------
    G : NetworkX graph

    source : node
       Specify starting node for breadth-first search and return edges in
       the component reachable from source.

    reverse : bool, optional
       If True traverse a directed graph in the reverse direction

    depth_limit : int, optional(default=len(G))
        Specify the maximum search depth

    Returns
    -------
    T: NetworkX DiGraph
       An oriented tree

    Examples
    --------
    >>> G = nx.path_graph(3)
    >>> print(list(nx.bfs_tree(G,1).edges()))
    [(1, 0), (1, 2)]
    >>> H = nx.Graph()
    >>> nx.add_path(H, [0, 1, 2, 3, 4, 5, 6])
    >>> nx.add_path(H, [2, 7, 8, 9, 10])
    >>> print(sorted(list(nx.bfs_tree(H, source=3, depth_limit=3).edges())))
    [(1, 0), (2, 1), (2, 7), (3, 2), (3, 4), (4, 5), (5, 6), (7, 8)]


    Notes
    -----
    Based on http://www.ics.uci.edu/~eppstein/PADS/BFS.py
    by D. Eppstein, July 2004. The modifications
    to allow depth limits based on the Wikipedia article
    "`Depth-limited-search`_".

    .. _Depth-limited-search: https://en.wikipedia.org/wiki/Depth-limited_search
    """
    T = nx.DiGraph()
    T.add_node(source)
    edges_gen = bfs_edges(G, source, reverse=reverse, depth_limit=depth_limit)
    T.add_edges_from(edges_gen)
    return T


def bfs_predecessors(G, source, depth_limit=None):
    """Returns an iterator of predecessors in breadth-first-search from source.

    Parameters
    ----------
    G : NetworkX graph

    source : node
       Specify starting node for breadth-first search and return edges in
       the component reachable from source.

    depth_limit : int, optional(default=len(G))
        Specify the maximum search depth

    Returns
    -------
    pred: iterator
        (node, predecessors) iterator where predecessors is the list of
        predecessors of the node.

    Examples
    --------
    >>> G = nx.path_graph(3)
    >>> print(dict(nx.bfs_predecessors(G, 0)))
    {1: 0, 2: 1}
    >>> H = nx.Graph()
    >>> H.add_edges_from([(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)])
    >>> print(dict(nx.bfs_predecessors(H, 0)))
    {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2}
    >>> M = nx.Graph()
    >>> nx.add_path(M, [0, 1, 2, 3, 4, 5, 6])
    >>> nx.add_path(M, [2, 7, 8, 9, 10])
    >>> print(sorted(nx.bfs_predecessors(M, source=1, depth_limit=3)))
    [(0, 1), (2, 1), (3, 2), (4, 3), (7, 2), (8, 7)]


    Notes
    -----
    Based on http://www.ics.uci.edu/~eppstein/PADS/BFS.py
    by D. Eppstein, July 2004. The modifications
    to allow depth limits based on the Wikipedia article
    "`Depth-limited-search`_".

    .. _Depth-limited-search: https://en.wikipedia.org/wiki/Depth-limited_search
    """
    for s, t in bfs_edges(G, source, depth_limit=depth_limit):
        yield (t, s)


def bfs_successors(G, source, depth_limit=None):
    """Returns an iterator of successors in breadth-first-search from source.

    Parameters
    ----------
    G : NetworkX graph

    source : node
       Specify starting node for breadth-first search and return edges in
       the component reachable from source.

    depth_limit : int, optional(default=len(G))
        Specify the maximum search depth

    Returns
    -------
    succ: iterator
       (node, successors) iterator where successors is the list of
       successors of the node.

    Examples
    --------
    >>> G = nx.path_graph(3)
    >>> print(dict(nx.bfs_successors(G,0)))
    {0: [1], 1: [2]}
    >>> H = nx.Graph()
    >>> H.add_edges_from([(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)])
    >>> print(dict(nx.bfs_successors(H, 0)))
    {0: [1, 2], 1: [3, 4], 2: [5, 6]}
    >>> G = nx.Graph()
    >>> nx.add_path(G, [0, 1, 2, 3, 4, 5, 6])
    >>> nx.add_path(G, [2, 7, 8, 9, 10])
    >>> print(dict(nx.bfs_successors(G, source=1, depth_limit=3)))
    {1: [0, 2], 2: [3, 7], 3: [4], 7: [8]}


    Notes
    -----
    Based on http://www.ics.uci.edu/~eppstein/PADS/BFS.py
    by D. Eppstein, July 2004.The modifications
    to allow depth limits based on the Wikipedia article
    "`Depth-limited-search`_".

    .. _Depth-limited-search: https://en.wikipedia.org/wiki/Depth-limited_search
    """
    parent = source
    children = []
    for p, c in bfs_edges(G, source, depth_limit=depth_limit):
        if p == parent:
            children.append(c)
            continue
        yield (parent, children)
        children = [c]
        parent = p
    yield (parent, children)
