import warnings
from copy import deepcopy

import networkx as nx
import numpy as np
import xxhash

budget_eps = 1e-5


class GNNGraph(object):
    def __init__(self, g):
        self.picked_nodes = None
        self.total_capacity = None
        self.capacity_og = None
        self.num_nodes = g.number_of_nodes()
        self.node_labels = np.arange(self.num_nodes)
        self.all_nodes_set = set(self.node_labels)
        self.nx_graph = g
        self.banned_actions = None
        self.load = None
        self.capacity = None

        x, y = zip(*g.edges())
        self.num_edges = len(x)
        self.edge_pairs = np.ndarray(shape=(self.num_edges, 2), dtype=np.int32)
        self.edge_pairs[:, 0] = x
        self.edge_pairs[:, 1] = y
        self.edge_pairs = np.ravel(self.edge_pairs)

        self.node_degrees = np.array([deg for (node, deg) in sorted(g.degree(), key=lambda deg_pair: deg_pair[0])])
        self.first_node = None

        self.reset()

    def reset(self, capacity_budget=None, validate=False):
        self.capacity_og = nx.betweenness_centrality(
            self.nx_graph,
            k=int(1.0 * len(self.nx_graph)),  # TODO: add c_approx
            normalized=True,
            endpoints=True,
        )
        self.load = self.capacity_og
        self.capacity = self.capacity_og.copy()

        self.total_capacity = sum(self.capacity.values())
        self.picked_nodes = np.zeros(self.num_nodes, dtype=int)

        # All have some extra capacity to begin with
        self.capacity.update(
            (x, y * (1.0 + 0.25)) for x, y in self.capacity.items()  # TODO: add self.prm["r"]
        )
        if validate:
            # Evenly distribute capacity budget
            self.capacity.update(
                (x, y + (self.total_capacity * capacity_budget)/self.num_nodes) for x, y in self.capacity.items()  # TODO: add self.prm["r"]
            )

    def add_edge(self, first_node, second_node):
        nx_graph = self.to_networkx()
        nx_graph.add_edge(first_node, second_node)
        s2v_graph = GNNGraph(nx_graph)
        return s2v_graph, 1

    def to_networkx(self):
        edges = self.convert_edges()
        g = nx.Graph()
        g.add_edges_from(edges)
        return g

    def convert_edges(self):
        return np.reshape(self.edge_pairs, (self.num_edges, 2))

    def display(self, ax=None):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            nx_graph = self.to_networkx()
            nx.draw_shell(nx_graph, with_labels=True, ax=ax)

    def display_with_positions(self, node_positions, ax=None):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            nx_graph = self.to_networkx()
            nx.draw(nx_graph, pos=node_positions, with_labels=True, ax=ax)

    def draw_to_file(self, filename):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig_size_length = self.num_nodes / 5
        figsize = (fig_size_length, fig_size_length)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        self.display(ax=ax)
        fig.savefig(filename)
        plt.close()

    def get_adjacency_matrix(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            nx_graph = self.to_networkx()
            adj_matrix = np.asarray(
                nx.convert_matrix.to_numpy_matrix(nx_graph, nodelist=self.node_labels)
            )

        return adj_matrix

    def copy(self):
        return deepcopy(self)

    def __repr__(self):
        gh = get_graph_hash(self, size=32, include_first=True)
        return f"Graph State with hash {gh}"

    def add_capacity_to_node(self, node, add_increment):
        self.capacity[node] += add_increment


def get_graph_hash(g, size=32, include_first=False):
    if size == 32:
        hash_instance = xxhash.xxh32()
    elif size == 64:
        hash_instance = xxhash.xxh64()
    else:
        raise ValueError("only 32 or 64-bit hashes supported.")

    if include_first:
        if g.first_node is not None:
            hash_instance.update(np.array([g.first_node]))
        else:
            hash_instance.update(np.zeros(g.num_nodes))

    hash_instance.update(g.edge_pairs)
    graph_hash = hash_instance.intdigest()
    return graph_hash
