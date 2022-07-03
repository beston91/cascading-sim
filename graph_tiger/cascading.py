import itertools as it
from collections import defaultdict
from locale import normalize

import numpy as np
from tqdm import tqdm

from graph_tiger.attacks import get_attack_category, run_attack_method
from graph_tiger.defenses import get_defense_category, run_defense_method
from graph_tiger.graphs import *
from graph_tiger.measures import run_measure
from graph_tiger.simulations import Simulation


class Cascading(Simulation):
    """
    This class simulates cascading failures on a network :cite:`crucitti2004model`.

    :param gnn_graph: an undirected NetworkX graph
    :param runs: an integer number of times to run the simulation
    :param steps: an integer number of steps to run a single simulation
    :param l: a float representing the maximum initial load for each node
    :param r: a float representing the amount of redundancy in the network
    :param **kwargs: see parent class Simulation for additional options
    """

    def __init__(self, gnn_graph, runs=10, steps=100, l=0.8, r=0.2, **kwargs):
        super().__init__(gnn_graph, runs, steps, **kwargs)

        self.capacity = None
        self.prm.update(
            {
                "l": l,
                "r": r,
                "c_approx": len(gnn_graph.nx_graph),
                "robust_measure": "largest_connected_component",
                "k_a": 10,
                "attack": "id_node",
                "attack_approx": None,
                "k_d": None,
                "defense": None,
            }
        )

        self.prm.update(kwargs)

        if self.prm["plot_transition"] or self.prm["gif_animation"]:
            self.node_pos, self.edge_pos = self.get_graph_coordinates()

        self.save_dir = os.path.join(os.getcwd(), "plots", self.get_plot_title(steps))
        os.makedirs(self.save_dir, exist_ok=True)

        self.capacity_og = self.gnn_graph.capacity_og.copy()
        self.capacity = self.gnn_graph.capacity.copy()
        self.load = self.gnn_graph.load.copy()

        self.max_val = max(self.capacity_og.values()) + self.gnn_graph.total_capacity * 0.1

        self.protected = set()
        self.failed = set()
        self.load = defaultdict()
        self.sim_info = defaultdict()

        self.reset_simulation()

    def reset_simulation(self):
        """
         Resets the simulation between each run
         """
        assert self.prm["k_a"] < self.gnn_graph.num_nodes
        self.protected = set()
        self.failed = set()
        self.load = defaultdict()
        self.sim_info = defaultdict()

        self.set_load()
        self.track_simulation(step=0)

        # attacked nodes or edges
        if self.prm["attack"] is not None and self.prm["k_a"] > 0:
            self.failed = set(
                run_attack_method(
                    self.gnn_graph.nx_graph,
                    self.prm["attack"],
                    self.prm["k_a"],
                    approx=self.prm["attack_approx"],
                    seed=self.prm["seed"],
                )
            )

            if get_attack_category(self.prm["attack"]) == "node":
                for n in self.failed:
                    # increase load by 2x when attacked
                    self.load[n] = 2 * self.load[n]

            elif get_attack_category(self.prm["attack"]) == "edge":
                self.gnn_graph.nx_graph.remove_edges_from(self.failed)
                self.failed = set()

        # defended nodes or edges
        if self.prm["defense"] is not None and self.prm["k_d"] > 0:

            if get_defense_category(self.prm["defense"]) == "node":
                self.protected = run_defense_method(
                    self.gnn_graph.nx_graph,
                    self.prm["defense"],
                    self.prm["k_d"],
                    seed=self.prm["seed"],
                )
                for n in self.protected:
                    # double the capacity when defended
                    self.capacity[n] = 2 * self.capacity[n]

            elif get_defense_category(self.prm["defense"]) == "edge":
                edge_info = run_defense_method(
                    self.gnn_graph.nx_graph,
                    self.prm["defense"],
                    self.prm["k_d"],
                    seed=self.prm["seed"],
                )

                self.gnn_graph.nx_graph.add_edges_from(edge_info["added"])

                if "removed" in edge_info:
                    self.gnn_graph.nx_graph.remove_edges_from(edge_info["removed"])

        elif self.prm["defense"] is not None:
            print(self.prm["defense"], "not available or k <= 0")

        self.track_simulation(step=1)

    def set_load(self):
        self.capacity_og = self.gnn_graph.capacity_og.copy()
        self.capacity = self.gnn_graph.capacity.copy()
        self.load = self.gnn_graph.load.copy()
        # self.load = self.capacity_og
        # # Uniform load percentage in the beginning
        # self.capacity.update(
        #     (x, y * (1.0 + self.prm["r"])) for x, y in self.capacity.items()
        # )

    def compute_failed_new(self):
        failed_new = set()

        view = nx.subgraph_view(
            self.gnn_graph.nx_graph, filter_node=lambda node: node not in self.failed
        )

        new_load = nx.betweenness_centrality(
            view,
            k=int(self.prm["c_approx"] * len(view)),
            normalized=True,
            endpoints=True,
        )

        self.load.update(new_load)

        for n in view.nodes:
            if self.load[n] > self.capacity[n]:
                failed_new.add(n)

        return failed_new

    def track_simulation(self, step):
        """
        Keeps track of important simulation information at each step of the simulation

        :param step: current simulation iteration
        """

        nodes_functioning = set(self.gnn_graph.nx_graph.nodes).difference(self.failed)

        measure = 0
        if len(nodes_functioning) > 0:
            measure = run_measure(
                self.gnn_graph.nx_graph.subgraph(nodes_functioning),
                self.prm["robust_measure"],
            )

        self.sim_info[step] = {
            "status": [self.load[n] for n in self.gnn_graph.nx_graph.nodes],
            "failed": len(self.failed),
            "measure": measure,
            "protected": self.protected,
            "state": 0,
        }

    def run_single_sim(self):
        """
        Run the attack simulation
        """
        for step in tqdm(range(self.prm["steps"]), leave=False, colour="blue"):
            cur_timestep = step + 2
            failed_new = self.compute_failed_new()
            self.failed = self.failed.union(failed_new)
            self.track_simulation(cur_timestep)

        robustness = [
            v["measure"] if v["measure"] is not None else 0
            for k, v in self.sim_info.items()
        ]
        return robustness

    def update_graph(self, n):
        for nb in list(self.gnn_graph.nx_graph.neighbors(n)):
            self.gnn_graph.nx_graph.remove_edge(n, nb)


def main():
    graph = electrical()

    params = {
        "runs": 1,
        "steps": 100,
        "seed": 1,
        "l": 0.8,
        "r": 0.2,
        "c_approx": int(0.1 * len(graph)),
        "k_a": 5,
        "attack": "id_node",
        "attack_approx": None,  # int(0.1 * len(graph)),
        "k_d": 0,
        "defense": None,
        "robust_measure": "largest_connected_component",
        "plot_transition": False,
        "gif_animation": True,
        "gif_snaps": True,
        "edge_style": None,
        "node_style": "spectral",
        "fa_iter": 2000,
    }

    cf = Cascading(graph, **params)
    results = cf.run_simulation()
    cf.plot_results(results)


if __name__ == "__main__":
    main()
