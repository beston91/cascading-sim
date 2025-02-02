import sys
from pathlib import Path

import wandb
from relnet.agent.rnet_dqn.rnet_dqn_agent import RNetDQNAgent
from relnet.environment.graph_edge_env import GraphEdgeEnv
from relnet.evaluation.file_paths import FilePaths
from relnet.objective_functions.objective_functions import critical_fraction_random
from relnet.state.network_generators import BANetworkGenerator, NetworkGenerator

sys.path.append("/relnet")

wandb.init(project="PCF", group="playground", name="play" + wandb.util.generate_id())


def get_gen_params():
    gp = {"n": 20, "m_ba": 2, "m_percentage_er": 20}
    gp["m"] = NetworkGenerator.compute_number_edges(gp["n"], gp["m_percentage_er"])
    return gp


def get_options(file_paths):
    options = {
        "log_progress": True,
        "log_filename": str(file_paths.construct_log_filepath()),
        "log_tf_summaries": True,
        "random_seed": 42,
        "models_path": file_paths.models_dir,
        "restore_model": False,
    }
    return options


def get_file_paths():
    parent_dir = "/Users/beston/Documents/GitHub/cascading-sim/experiment_data"
    experiment_id = "development"
    file_paths = FilePaths(parent_dir, experiment_id)
    return file_paths


if __name__ == "__main__":
    NUM_TRAINING_STEPS = 1000
    NUM_TRAIN_GRAPHS = 100
    NUM_VALIDATION_GRAPHS = 20
    NUM_TEST_GRAPHS = 20

    gen_params = get_gen_params()
    file_paths = get_file_paths()

    options = get_options(file_paths)
    storage_root = Path(
        "/Users/beston/Documents/GitHub/cascading-sim/experiment_data/stored_graphs"
    )
    original_dataset_dir = Path(
        "/Users/beston/Documents/GitHub/cascading-sim/experiment_data/real_world_graphs/processed_data"
    )
    kwargs = {"store_graphs": True, "graph_storage_root": storage_root}
    gen = BANetworkGenerator(**kwargs)

    (
        train_graph_seeds,
        validation_graph_seeds,
        test_graph_seeds,
    ) = NetworkGenerator.construct_network_seeds(
        NUM_TRAIN_GRAPHS, NUM_VALIDATION_GRAPHS, NUM_TEST_GRAPHS
    )

    train_graphs = gen.generate_many(gen_params, train_graph_seeds)
    validation_graphs = gen.generate_many(gen_params, validation_graph_seeds)
    test_graphs = gen.generate_many(gen_params, test_graph_seeds)

    EDGE_PERCENTAGE = 2.5
    obj_fun_kwargs = {"random_seed": 42, "num_mc_sims": gen_params["n"] * 2}
    targ_env = GraphEdgeEnv(critical_fraction_random, obj_fun_kwargs, EDGE_PERCENTAGE)

    agent = RNetDQNAgent(targ_env)
    agent.setup(options, agent.get_default_hyperparameters())
    agent.train(train_graphs, validation_graphs, NUM_TRAINING_STEPS)
    avg_perf = agent.eval(test_graphs)
