import datetime
import traceback
from abc import ABC
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import wandb

from relnet.agent.base_agent import Agent
from relnet.evaluation.file_paths import FilePaths
from relnet.environment.graph_edge_env import GraphEdgeEnv


class PyTorchAgent(Agent, ABC):
    DEFAULT_BATCH_SIZE = 20

    def __init__(self, environment):
        super().__init__(environment)
        self.summaries_path = None
        self.net = None
        self.algorithm_name = None
        self.train_g_list = None
        self.validation_g_list = None
        self.train_initial_obj_values = None
        self.validation_initial_obj_values = None
        self.sample_indexes = None
        self.batch_size = None
        self.validation_check_interval = None
        self.max_validation_consecutive_steps = None
        self.model_identifier_prefix = None
        self.restore_model = None
        self.models_path = None
        self.checkpoints_path = None
        self.log_tf_summaries = None
        self.eval_histories_path = None
        self.enable_assertions = True
        self.hist_out = None

        self.validation_change_threshold = 1e-5
        self.best_validation_changed_step = -1
        self.best_validation_loss = float("inf")

        self.pos = 0
        self.step = 0

    def setup_graphs(self, train_g_list, validation_g_list):
        self.train_g_list = train_g_list
        self.validation_g_list = validation_g_list
        # self.train_initial_obj_values = self.environment.get_objective_function_values(
        #     self.train_g_list
        # )
        # self.validation_initial_obj_values = self.environment.get_objective_function_values(
        #     self.validation_g_list
        # )

    def setup_sample_indexes(self, dataset_size):
        self.sample_indexes = list(range(dataset_size))

    def advance_pos_and_sample_indices(self):
        if (self.pos + 1) * self.batch_size > len(self.sample_indexes):
            self.pos = 0
            np.random.shuffle(self.sample_indexes)

        selected_idx = self.sample_indexes[
                       self.pos * self.batch_size: (self.pos + 1) * self.batch_size
                       ]
        self.pos += 1
        return selected_idx

    def save_model_checkpoints(self):
        model_dir = self.checkpoints_path / self.model_identifier_prefix
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"{self.algorithm_name}_agent.model"
        torch.save(self.net.state_dict(), model_path)

    def restore_model_from_checkpoint(self):
        model_path = (
                self.checkpoints_path
                / self.model_identifier_prefix
                / f"{self.algorithm_name}_agent.model"
        )
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        self.net.load_state_dict(checkpoint)

    def check_validation_loss(
            self,
            step_number,
            max_steps,
            make_action_kwargs=None,
            model_tag=None,
            save_model_if_better=True,
    ):
        if (
                step_number % self.validation_check_interval == 0
                or step_number == max_steps
        ):
            validation_loss = self.log_validation_loss(
                step_number, make_action_kwargs=make_action_kwargs
            )
            if self.log_progress:
                self.logger.info(
                    f"{model_tag if model_tag is not None else 'model'} validation loss: {validation_loss: .4f} at step {step_number}. "
                )
            if (
                    self.best_validation_loss - validation_loss
            ) > self.validation_change_threshold:
                if self.log_progress:
                    self.logger.info(
                        f"rejoice! found a better validation loss at step {step_number}."
                    )
                self.best_validation_changed_step = step_number
                self.best_validation_loss = validation_loss
                if save_model_if_better:
                    if self.log_progress:
                        self.logger.info("saving model.")
                    self.save_model_checkpoints()

    def log_validation_loss(self, step, make_action_kwargs=None):

        max_improvement, performance = self.eval(
            self.validation_g_list,
            None,
            validation=True,
            make_action_kwargs=make_action_kwargs,
        )

        # max_improvement = self.environment.objective_function.upper_limit
        validation_loss = max_improvement - performance
        wandb.log({"validation_loss": validation_loss})
        wandb.log({"performance": performance})

        return validation_loss

    def print_model_parameters(self):
        param_list = self.net.parameters()
        for params in param_list:
            print(params.data)

    def check_stopping_condition(self, step_number, max_steps):
        if step_number >= max_steps or (
                step_number - self.best_validation_changed_step
                > self.max_validation_consecutive_steps
        ):
            if self.log_progress:
                self.logger.info(
                    "number steps exceeded or validation plateaued for too long, stopping training."
                )
            if self.log_progress:
                self.logger.info("restoring best model to use for predictions.")
            self.restore_model_from_checkpoint()

            if self.log_tf_summaries:
                self.file_writer.close()
            return True
        return False

    def setup(self, options, hyperparams):
        super().setup(options, hyperparams)
        if "batch_size" in options:
            self.batch_size = options["batch_size"]
        else:
            self.batch_size = self.DEFAULT_BATCH_SIZE

        if "validation_check_interval" in options:
            self.validation_check_interval = options["validation_check_interval"]
        else:
            self.validation_check_interval = 100

        if "max_validation_consecutive_steps" in options:
            self.max_validation_consecutive_steps = options[
                "max_validation_consecutive_steps"
            ]
        else:
            self.max_validation_consecutive_steps = 200000

        if "pytorch_full_print" in options:
            if options["pytorch_full_print"]:
                torch.set_printoptions(profile="full")

        if "enable_assertions" in options:
            self.enable_assertions = options["enable_assertions"]

        if "model_identifier_prefix" in options:
            self.model_identifier_prefix = options["model_identifier_prefix"]
        else:
            self.model_identifier_prefix = FilePaths.DEFAULT_MODEL_PREFIX

        if "restore_model" in options:
            self.restore_model = options["restore_model"]
        else:
            self.restore_model = False

        if "models_path" in options:
            self.models_path = Path(options["models_path"])
        else:
            self.models_path = Path.cwd() / FilePaths.MODELS_DIR_NAME

        self.checkpoints_path = self.models_path / FilePaths.CHECKPOINTS_DIR_NAME

        self.log_tf_summaries = False

    def get_summaries_run_path(self):
        now = datetime.datetime.now().strftime(FilePaths.DATE_FORMAT)
        return self.summaries_path / f"{self.model_identifier_prefix}-run-{now}"

    def setup_histories_file(self):
        self.eval_histories_path = self.models_path / FilePaths.EVAL_HISTORIES_DIR_NAME
        model_history_filename = (
                self.eval_histories_path
                / FilePaths.construct_history_file_name(self.model_identifier_prefix)
        )
        model_history_file = Path(model_history_filename)
        if model_history_file.exists():
            model_history_file.unlink()
        self.hist_out = open(model_history_filename, "a")

    def finalize(self):
        if self.hist_out is not None and not self.hist_out.closed:
            self.hist_out.close()
