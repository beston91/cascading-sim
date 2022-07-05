from copy import deepcopy

import numpy as np

from relnet.state.network_generators import NetworkGenerator
from graph_tiger.cascading import Cascading


class GraphEdgeEnv(object):
    def __init__(self, capacity_budget_percent):
        self.final_objective_values = None
        self.increments = None
        self.logger_instance = None
        self.rewards = None
        self.objective_function_values = None
        self.training = None
        self.exhausted_budgets = None
        self.used_capacity_budgets = None
        self.n_steps = None
        self.g_list = None
        self.capacity_budgets = None
        # Capacity budget is a percentage
        self.capacity_budget_percent = capacity_budget_percent
        # Add increment is a fraction of the capacity_budget

        self.reward_eps = 1e-4
        self.reward_scale_multiplier = 100

    def setup(self, g_list, initial_objective_function_values, training=False):
        self.g_list = g_list
        self.n_steps = 0

        g_len = len(g_list)
        self.used_capacity_budgets = np.zeros(g_len, dtype=np.float)
        self.exhausted_budgets = np.zeros(g_len, dtype=np.bool)
        self.capacity_budgets = np.zeros(g_len, dtype=np.float)
        self.increments = np.zeros(g_len, dtype=np.float)

        for i in range(len(self.g_list)):
            g = g_list[i]
            g.reset()
            g_budget = g.total_capacity_og * self.capacity_budget_percent
            self.capacity_budgets[i] = g_budget
            self.increments[i] = g_budget * (1 / (g.num_nodes * 4))

        self.training = training
        self.final_objective_values = np.zeros(len(self.g_list), dtype=np.float)
        self.rewards = np.zeros(len(g_list), dtype=np.float)

        # if self.training:
        #     self.objective_function_values = np.multiply(
        #         self.objective_function_values, self.reward_scale_multiplier
        #     )

    def pass_logger_instance(self, logger):
        self.logger_instance = logger

    def get_final_values(self):
        return self.final_objective_values

    def get_remaining_budget(self, i):
        return self.capacity_budgets[i] - self.used_capacity_budgets[i]

    @staticmethod
    def get_valid_actions(g, banned_actions):
        all_nodes_set = g.all_nodes_set
        valid_first_nodes = all_nodes_set - banned_actions
        return valid_first_nodes

    def apply_action(self, g, action, node, copy_state=False):
        # Every time a node is picked 1/10 of the original budget is assigned
        g.add_capacity_to_node(action, self.increments[node])
        g.picked_nodes[action] += 1
        budget_used = self.increments[node]
        return g, budget_used

    def step(self, actions, show_graph=False):
        for i in range(len(self.g_list)):
            if not self.exhausted_budgets[i]:
                if actions[i] == -1:
                    if self.logger_instance is not None:
                        self.logger_instance.warn(
                            "budget not exhausted but trying to apply dummy action!"
                        )
                        self.logger_instance.error(
                            f"the remaining budget: {self.get_remaining_budget(i)}"
                        )
                        g = self.g_list[i]

                self.g_list[i], budget_used = self.apply_action(
                    self.g_list[i], actions[i], i
                )

                self.used_capacity_budgets[i] += budget_used

                if self.capacity_budgets[i] - self.used_capacity_budgets[i] <= 0:
                    # TODO: simulate cascade failure
                    if show_graph:
                        print("showing graph")
                    self.rewards[i] = self.simulate_to_failure(self.g_list[i], show_graph)
                    self.final_objective_values[i] = self.rewards[i]
                    self.exhausted_budgets[i] = True

        self.n_steps += 1

    def exploratory_actions(self, agent_exploration_policy):
        act_list = []
        for i in range(len(self.g_list)):
            selected_node = agent_exploration_policy(i)
            act_list.append(selected_node)

        return act_list

    def get_max_graph_size(self):
        max_graph_size = np.max([g.num_nodes for g in self.g_list])
        return max_graph_size

    def is_terminal(self):
        return np.all(self.exhausted_budgets)

    def get_state_ref(self):
        return self.g_list

    def clone_state(self, indices=None):
        if indices is None:
            cp_first = [g.first_node for g in self.g_list][:]
            b_list = [g.banned_actions for g in self.g_list][:]
            return list(zip(deepcopy(self.g_list), cp_first, b_list))
        else:
            cp_g_list = []

            for i in indices:
                cp_g_list.append(deepcopy(self.g_list[i]))

            return list(cp_g_list)

    @staticmethod
    def simulate_to_failure(gnn_graph, show_graph=False):
        params = {
            'runs': 1,
            'steps': 20,
            'seed': 1,

            'l': 0.8,
            'r': 0.2,
            'c_approx': 1.0,

            'k_a': 1,
            'attack': 'rb_node',
            'attack_approx': int(0.1 * gnn_graph.num_nodes),

            'k_d': 0,
            # 'defense': None,

            'robust_measure': 'largest_connected_component',

            'plot_transition': show_graph,  # False turns off key simulation image "snapshots"
            'gif_animation': False,  # True creates a video of the simulation (MP4 file)
            'gif_snaps': False,  # True saves each frame of the simulation as an image

            'edge_style': 'bundled',
            'node_style': 'force_atlas',
            'fa_iter': 2000,
        }

        cascading = Cascading(gnn_graph, **params)
        results = cascading.run_simulation()
        # reward is the negative difference between starting robustness and ending robustness
        reward = results[-1] / results[0]
        return reward
