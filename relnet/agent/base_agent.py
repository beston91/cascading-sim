import random
from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np
import torch

from relnet.evaluation.eval_utils import eval_on_dataset, get_values_for_g_list
from relnet.utils.config_utils import get_logger_instance


class Agent(ABC):
    def __init__(self, environment):
        self.environment = environment
        self.options = None
        self.log_filename = None
        self.random_seed = None
        self.local_random = None
        self.log_progress = None
        self.hyperparams = None
        self.logger = None

    def train(self, train_g_list, validation_g_list, max_steps, **kwargs):
        pass

    def eval(self, g_list, initial_obj_values=None, validation=False, make_action_kwargs=None):

        eval_nets = [deepcopy(g) for g in g_list]
        initial_obj_values, final_obj_values = get_values_for_g_list(
            self, eval_nets, initial_obj_values, validation, make_action_kwargs
        )
        return np.mean(np.absolute(initial_obj_values)), eval_on_dataset(initial_obj_values, final_obj_values)

    @abstractmethod
    def make_actions(self, t, **kwargs):
        pass

    def setup(self, options, hyperparams):
        self.options = options
        if "log_filename" in options:
            self.log_filename = options["log_filename"]
        if "log_progress" in options:
            self.log_progress = options["log_progress"]
        else:
            self.log_progress = False
        if self.log_progress:
            self.logger = get_logger_instance(self.log_filename)
            self.environment.pass_logger_instance(self.logger)
        else:
            self.logger = None

        if "random_seed" in options:
            self.set_random_seeds(options["random_seed"])
        else:
            self.set_random_seeds(42)
        self.hyperparams = hyperparams

    @abstractmethod
    def finalize(self):
        pass

    @abstractmethod
    def pick_random_actions(self, i):
        pass

    def set_random_seeds(self, random_seed):
        self.random_seed = random_seed
        self.local_random = random.Random()
        self.local_random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed(self.random_seed)
        torch.cuda.manual_seed_all(self.random_seed)
