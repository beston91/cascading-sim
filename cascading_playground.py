
from graph_tiger.graphs import graph_loader
from graph_tiger.cascading import Cascading
from omegaconf import DictConfig, OmegaConf
import hydra
import sys
sys.path.append('/graph_tiger')

graph = graph_loader('BA', n=500, seed=1)


@hydra.main(config_path="./config", config_name="config")
def run(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    params = cfg.playground
    if 'attack_approx' in params:
        params.update({
            'attack_approx': int(params['attack_approx'] * len(graph))})
    cascading = Cascading(graph, **params)
    results = cascading.run_simulation()
    cascading.plot_results(results)


if __name__ == "__main__":
    run()
