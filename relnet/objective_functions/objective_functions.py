import random

import networkx as nx


def critical_fraction_random(state, **kwargs):
    num_plays = 50
    random_seed = 42
    if "num_plays" in kwargs:
        num_plays = kwargs["num_plays"]
    if "random_seed" in kwargs:
        random_seed = kwargs["random_seed"]

    critical_fractions_sum = 0.0
    for play_num in range(num_plays):
        rand_instance = random.Random(random_seed * play_num)
        nx_graph = state.to_networkx()
        if nx.number_connected_components(nx_graph) > 1:
            return 0.0
        nodes_to_remove = list(nx_graph.nodes)
        N = state.num_nodes
        critical_q = float(N)
        for q in range(0, N - 1):
            node_to_remove = rand_instance.choice(nodes_to_remove)
            nx_graph.remove_node(node_to_remove)
            nodes_to_remove.remove(node_to_remove)
            num_connected_components = nx.number_connected_components(nx_graph)
            if num_connected_components > 1 or q == N - 2:
                critical_q = float(q)
                break
        critical_fractions_sum += (critical_q + 1) / float(N)
    return critical_fractions_sum / num_plays


def critical_fraction_targeted(state, **kwargs):
    num_plays = 50
    random_seed = 42
    if "num_plays" in kwargs:
        num_plays = kwargs["num_plays"]
    if "random_seed" in kwargs:
        random_seed = kwargs["random_seed"]
    starting_graph = nx.from_numpy_matrix(state.adjacency_matrix)
    degrees_dict = nx.degree_centrality(starting_graph)
    # TODO: A possible optimization to both random/targeted is to hash the permutation
    # And cache the previous result
    # May be particularly helpful for targeted.
    critical_fractions_sum = 0.0
    for play_num in range(num_plays):
        rand_instance = random.Random(random_seed * play_num * state.get_identifier())
        nx_graph = nx.from_numpy_matrix(state.adjacency_matrix)
        if nx.number_connected_components(nx_graph) > 1:
            return 0.0
        degrees_descending_order = sorted(
            degrees_dict.items(),
            key=lambda x: (x[1], rand_instance.random()),
            reverse=True,
        )
        # print(f"Sorted by degrees order: as {[x[0] for x in degrees_descending_order]}")
        # print(f"Actual degrees order: as {['{:.3f}'.format(x[1]) for x in degrees_descending_order]}")
        N = state.number_vertices
        critical_q = float(N)
        for q in range(0, N - 1):
            nx_graph.remove_node(degrees_descending_order[q][0])
            num_connected_components = nx.number_connected_components(nx_graph)
            if num_connected_components > 1 or q == N - 2:
                critical_q = float(q)
                break

        critical_fractions_sum += (critical_q + 1) / float(N)
    return critical_fractions_sum / num_plays
