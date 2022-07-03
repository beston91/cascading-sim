import json
import math
from abc import ABC, abstractmethod
from pathlib import Path

import networkx as nx

from relnet.evaluation.file_paths import FilePaths
from relnet.state.gnn_graph import GNNGraph
from relnet.utils.config_utils import get_logger_instance


class NetworkGenerator(ABC):
    enforce_connected = True

    def __init__(self, store_graphs=False, graph_storage_root=None, logs_file=None):
        super().__init__()
        self.store_graphs = store_graphs
        if self.store_graphs:
            self.graph_storage_root = graph_storage_root
            self.graph_storage_dir = graph_storage_root / self.name
            self.graph_storage_dir.mkdir(parents=True, exist_ok=True)

        if logs_file is not None:
            self.logger_instance = get_logger_instance(logs_file)
        else:
            self.logger_instance = None

    def generate(self, gen_params, random_seed):
        if self.store_graphs:
            filename = self.get_data_filename(gen_params, random_seed)
            filepath = self.graph_storage_dir / filename

            should_create = True
            if filepath.exists():
                try:
                    instance = self.read_graphml_with_ordered_int_labels(filepath)
                    state = self.post_generate_instance(instance)
                    should_create = False
                except Exception:
                    should_create = True

            if should_create:
                instance = self.generate_instance(gen_params, random_seed)
                state = self.post_generate_instance(instance)
                nx.readwrite.write_graphml(instance, filepath.resolve())

                drawing_filename = self.get_drawing_filename(gen_params, random_seed)
                drawing_path = self.graph_storage_dir / drawing_filename
                state.draw_to_file(drawing_path)
        else:
            instance = self.generate_instance(gen_params, random_seed)
            state = self.post_generate_instance(instance)

        return state

    def read_graphml_with_ordered_int_labels(self, filepath):
        instance = nx.readwrite.read_graphml(filepath.resolve())
        num_nodes = len(instance.nodes)
        relabel_map = {str(i): i for i in range(num_nodes)}
        nx.relabel_nodes(instance, relabel_map, copy=False)

        G = nx.Graph()
        G.add_nodes_from(sorted(instance.nodes(data=True)))
        G.add_edges_from(instance.edges(data=True))

        return G

    def generate_many(self, gen_params, random_seeds):
        return [self.generate(gen_params, random_seed) for random_seed in random_seeds]

    @abstractmethod
    def generate_instance(self, gen_params, random_seed):
        pass

    @abstractmethod
    def post_generate_instance(self, instance):
        pass

    def get_data_filename(self, gen_params, random_seed):
        n = gen_params["n"]
        filename = f"{n}-{random_seed}.graphml"
        return filename

    def get_drawing_filename(self, gen_params, random_seed):
        n = gen_params["n"]
        filename = f"{n}-{random_seed}.png"
        return filename

    @staticmethod
    def compute_number_edges(n, edge_percentage):
        total_possible_edges = (n * (n - 1)) / 2
        return int(math.ceil((total_possible_edges * edge_percentage / 100)))

    @staticmethod
    def construct_network_seeds(
        num_train_graphs, num_validation_graphs, num_test_graphs
    ):
        train_seeds = list(range(0, num_train_graphs))
        validation_seeds = list(
            range(num_train_graphs, num_train_graphs + num_validation_graphs)
        )
        offset = num_train_graphs + num_validation_graphs
        test_seeds = list(range(offset, offset + num_test_graphs))
        return train_seeds, validation_seeds, test_seeds


class OrdinaryGraphGenerator(NetworkGenerator, ABC):
    def post_generate_instance(self, instance):
        state = GNNGraph(instance)
        return state


class GNMNetworkGenerator(OrdinaryGraphGenerator):
    name = "random_network"
    num_tries = 10000

    def generate_instance(self, gen_params, random_seed):
        number_vertices = gen_params["n"]
        number_edges = gen_params["m"]

        if not self.enforce_connected:
            random_graph = nx.generators.random_graphs.gnm_random_graph(
                number_vertices, number_edges, seed=random_seed
            )
            return random_graph
        else:
            for try_num in range(0, self.num_tries):
                random_graph = nx.generators.random_graphs.gnm_random_graph(
                    number_vertices, number_edges, seed=(random_seed + (try_num * 1000))
                )
                if nx.is_connected(random_graph):
                    return random_graph
                else:
                    continue
            raise ValueError("Maximum number of tries exceeded, giving up...")


class BANetworkGenerator(OrdinaryGraphGenerator):
    name = "barabasi_albert"

    def generate_instance(self, gen_params, random_seed):
        n, m = gen_params["n"], gen_params["m_ba"]
        ba_graph = nx.generators.random_graphs.barabasi_albert_graph(
            n, m, seed=random_seed
        )
        return ba_graph
