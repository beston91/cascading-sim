
from graph_tiger.graphs import graph_loader
from graph_tiger.cascading import Cascading
import sys
sys.path.append('/graph_tiger')


graph = graph_loader('BA', n=500, seed=1)
# graph = graph_loader('electrical')

params = {
    'runs': 1,
    'steps': 100,
    'seed': 1,

    'l': 0.8,
    'r': 0.4,
    'c_approx': 1.0,

    'load_method': 'shortest',  # shortest, random
    'propagation_method': 'shortest',  # shortest, random
    'centrality_method': 'betweeness',  # betweeness, load

    'k_a': 2,
    'attack': 'rb_node',
    'attack_approx': int(0.1 * len(graph)),

    'k_d': 0,
    'defense': None,

    'robust_measure': 'largest_connected_component',

    'plot_transition': True,  # False turns off key simulation image "snapshots"
    # True creates a video of the simulation (MP4 file)
    'gif_animation': False,
    'gif_snaps': False,  # True saves each frame of the simulation as an image

    'edge_style': 'bundled',
    'node_style': 'force_atlas',
    'fa_iter': 2000,
}

cascading = Cascading(graph, **params)
results = cascading.run_simulation()

cascading.plot_results(results)
