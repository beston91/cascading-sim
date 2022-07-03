import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb
from torch.autograd import Variable
from tqdm import tqdm

from relnet.agent.pytorch_agent import PyTorchAgent
from relnet.agent.rnet_dqn.nstep_replay_mem import NstepReplayMem
from relnet.agent.rnet_dqn.q_net import NStepQNet, QNet, greedy_actions
from relnet.utils.config_utils import get_device_placement


class RNetDQNAgent(PyTorchAgent):
    algorithm_name = "rnet_dqn"
    is_deterministic = False
    is_trainable = True

    def __init__(self, environment):
        super().__init__(environment)
        self.net = None
        self.old_net = None
        self.mem_pool = None
        self.learning_rate = None
        self.eps_start = None
        self.eps_step = None
        self.eps_end = None
        self.burn_in = None
        self.net_copy_interval = None
        self.eps = None
        self.next_exploration_actions = None

    def setup(self, options, hyperparams):
        super().setup(options, hyperparams)
        self.setup_nets()
        self.take_snapshot()

    def train(self, train_g_list, validation_g_list, max_steps, **kwargs):
        self.setup_graphs(train_g_list, validation_g_list)
        self.setup_sample_indexes(len(train_g_list))

        self.setup_mem_pool(max_steps, self.hyperparams["mem_pool_to_steps_ratio"])
        self.setup_histories_file()
        self.setup_training_parameters(max_steps)

        pbar = tqdm(range(self.burn_in), unit="batch", disable=None)
        for p in pbar:
            with torch.no_grad():
                self.run_simulation()
        pbar = tqdm(range(max_steps + 1), unit="steps", disable=None)
        for name, param in self.net.named_parameters():
            if param.requires_grad:
                print(name, param.data)
        optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)
        for self.step in pbar:
            with torch.no_grad():
                self.run_simulation()
            if self.step % self.net_copy_interval == 0:
                self.take_snapshot()
            self.check_validation_loss(self.step, max_steps)

            (
                cur_time,
                list_st,
                list_at,
                list_rt,
                list_s_primes,
                list_term,
            ) = self.mem_pool.sample(batch_size=self.batch_size)
            list_target = torch.Tensor(list_rt)
            if get_device_placement() == "GPU":
                list_target = list_target.cuda()

            cleaned_sp = []
            nonterms = []
            for i in range(len(list_st)):
                if not list_term[i]:
                    cleaned_sp.append(list_s_primes[i])
                    nonterms.append(i)

            if len(cleaned_sp):
                _, _, banned = zip(*cleaned_sp)
                _, q_t_plus_1, prefix_sum_prime = self.old_net(
                    cleaned_sp, None
                )
                _, q_rhs = greedy_actions(q_t_plus_1, prefix_sum_prime, None)
                list_target[nonterms] = q_rhs

            list_target = Variable(list_target.view(-1, 1))
            _, q_sa, _ = self.net(list_st, list_at)

            loss = F.mse_loss(q_sa, list_target)
            wandb.log({"training_loss": loss})
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_description("exp: %.5f, loss: %0.5f" % (self.eps, loss))

            should_stop = self.check_stopping_condition(self.step, max_steps)
            if should_stop:
                break

    def setup_nets(self):
        self.net = QNet(self.hyperparams, None)
        self.old_net = QNet(self.hyperparams, None)
        if get_device_placement() == "GPU":
            self.net = self.net.cuda()
            self.old_net = self.old_net.cuda()
        if self.restore_model:
            self.restore_model_from_checkpoint()

    def setup_mem_pool(self, num_steps, mem_pool_to_steps_ratio):
        exp_replay_size = int(num_steps * mem_pool_to_steps_ratio)
        self.mem_pool = NstepReplayMem(memory_size=exp_replay_size, n_steps=2)

    def setup_training_parameters(self, max_steps):
        self.learning_rate = self.hyperparams["learning_rate"]
        self.eps_start = self.hyperparams["epsilon_start"]

        eps_step_denominator = (
            self.hyperparams["eps_step_denominator"]
            if "eps_step_denominator" in self.hyperparams
            else 2
        )
        self.eps_step = max_steps / eps_step_denominator
        self.eps_end = 0.1
        self.burn_in = 1
        self.net_copy_interval = 1

    def finalize(self):
        pass

    def take_snapshot(self):
        self.old_net.load_state_dict(self.net.state_dict())

    def make_actions(self, t, **kwargs):
        greedy = kwargs["greedy"] if "greedy" in kwargs else True
        if greedy:
            return self.do_greedy_actions(t)
        else:
            self.eps = self.eps_end + max(0.0, (self.eps_start - self.eps_end) * (
                        self.eps_step - max(0.0, self.step)) / self.eps_step, )
            if self.local_random.random() < self.eps:
                exploration_actions = self.environment.exploratory_actions(
                    self.agent_exploration_policy)
                return exploration_actions
            else:
                return self.do_greedy_actions(t)

    def do_greedy_actions(self, time_t):
        cur_state = self.environment.get_state_ref()
        actions, _, _ = self.net(cur_state, None, greedy_acts=True)
        actions = list(actions.cpu().numpy())
        return actions

    def agent_exploration_policy(self, i):
        return self.pick_random_actions(i)

    def pick_random_actions(self, i):
        g = self.environment.g_list[i]
        valid_actions = g.all_nodes_set
        return self.local_random.choice(tuple(valid_actions))

    def run_simulation(self):
        selected_idx = self.advance_pos_and_sample_indices()
        self.environment.setup(
            [self.train_g_list[idx] for idx in selected_idx],
            None,
            training=True,
        )
        self.post_env_setup()

        final_st = [None] * len(selected_idx)
        final_acts = np.empty(len(selected_idx), dtype=np.int)
        final_acts.fill(-1)

        t = 0
        while not self.environment.is_terminal():
            list_at = self.make_actions(t, greedy=False)

            (non_exhausted_before,) = np.where(~self.environment.exhausted_budgets)
            list_st = self.environment.clone_state(non_exhausted_before)
            self.environment.step(list_at)

            (non_exhausted_after,) = np.where(~self.environment.exhausted_budgets)
            (exhausted_after,) = np.where(self.environment.exhausted_budgets)

            nonterm_indices = np.flatnonzero(
                np.isin(non_exhausted_before, non_exhausted_after)
            )
            nonterm_st = [list_st[i] for i in nonterm_indices]
            nonterm_at = [list_at[i] for i in non_exhausted_after]
            rewards = np.zeros(len(nonterm_at), dtype=np.float)
            nonterm_s_prime = self.environment.clone_state(non_exhausted_after)

            now_term_indices = np.flatnonzero(
                np.isin(non_exhausted_before, exhausted_after)
            )
            term_st = [list_st[i] for i in now_term_indices]
            for i in range(len(term_st)):
                g_list_index = non_exhausted_before[now_term_indices[i]]

                final_st[g_list_index] = term_st[i]
                final_acts[g_list_index] = list_at[g_list_index]

            if len(nonterm_at) > 0:
                self.mem_pool.add_list(
                    nonterm_st,
                    nonterm_at,
                    rewards,
                    nonterm_s_prime,
                    [False] * len(nonterm_at),
                    t % 2,
                )

            t += 1

        final_at = list(final_acts)
        rewards = self.environment.rewards
        final_s_prime = None
        self.mem_pool.add_list(
            final_st,
            final_at,
            rewards,
            final_s_prime,
            [True] * len(final_at),
            (t - 1) % 2,
        )

    def post_env_setup(self):
        pass

    def get_default_hyperparameters(self):
        hyperparams = {
            "learning_rate": 0.0001,
            "epsilon_start": 1,
            "mem_pool_to_steps_ratio": 1,
            "latent_dim": 64,
            "hidden": 32,
            "max_lv": 3,
            "eps_step_denominator": 10,
        }
        return hyperparams
