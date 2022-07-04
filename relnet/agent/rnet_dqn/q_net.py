import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric.data import Batch
from torch_geometric.utils.convert import from_networkx

from relnet.state.graph_neural_network import GCN
from relnet.utils.config_utils import get_device_placement

sys.path.append("/usr/lib/pytorch_structure2vec/s2v_lib")


def jmax(arr, prefix_sum):
    actions = []
    values = []
    for i in range(len(prefix_sum[0:,])):
        if i == 0:
            start_index = 0
            end_index = prefix_sum[i]
        else:
            start_index = prefix_sum[i - 1]
            end_index = prefix_sum[i]

        arr_vals = arr[start_index:end_index]
        act = np.argmax(arr_vals)
        val = arr_vals[act]

        actions.append(act)
        values.append(val)

    actions_tensor = torch.LongTensor(actions)
    values_tensor = torch.Tensor(values)

    return actions_tensor, values_tensor


def greedy_actions(q_values, v_p, banned_list):
    actions = []
    offset = 0
    banned_acts = []
    prefix_sum = v_p.data.cpu().numpy()
    for i in range(len(prefix_sum)):
        n_nodes = prefix_sum[i] - offset

        if banned_list is not None and banned_list[i] is not None:
            for j in banned_list[i]:
                banned_acts.append(offset + j)
        offset = prefix_sum[i]

    q_values = q_values.data.clone()
    q_values.resize_(len(q_values))

    banned = torch.LongTensor(banned_acts)
    device_placement = get_device_placement()
    if device_placement == "GPU":
        banned = banned.cuda()

    if len(banned_acts):
        min_tensor = torch.tensor(float(np.finfo(np.float32).min))
        if device_placement == "GPU":
            min_tensor = min_tensor.cuda()
        q_values.index_fill_(0, banned, min_tensor)

    q_vals_cpu = q_values.data.cpu().numpy()
    return jmax(q_vals_cpu, prefix_sum)


class QNet(nn.Module):
    def __init__(self, hyperparams, s2v_module):
        super().__init__()

        embed_dim = hyperparams["latent_dim"]

        self.linear_1 = nn.Linear(embed_dim * 2, hyperparams["hidden"])
        self.linear_out = nn.Linear(hyperparams["hidden"], 1)
        # weights_init(self)

        self.num_node_feats = 2
        self.num_edge_feats = 0

        if s2v_module is None:
            self.gnn = GCN(in_channels=2, hidden_channels=32, out_channels=64)
        else:
            self.gnn = s2v_module

    def add_offset(self, actions, v_p):
        prefix_sum = v_p.data.cpu().numpy()

        shifted = []
        for i in range(len(prefix_sum)):
            if i > 0:
                offset = prefix_sum[i - 1]
            else:
                offset = 0
            shifted.append(actions[i] + offset)

        return shifted

    def rep_global_embed(self, graph_embed, v_p):
        prefix_sum = v_p.data.cpu().numpy()

        rep_idx = []
        for i in range(len(prefix_sum)):
            if i == 0:
                n_nodes = prefix_sum[i]
            else:
                n_nodes = prefix_sum[i] - prefix_sum[i - 1]
            rep_idx += [i] * n_nodes

        rep_idx = Variable(torch.LongTensor(rep_idx))
        if get_device_placement() == "GPU":
            rep_idx = rep_idx.cuda()
        graph_embed = torch.index_select(graph_embed, 0, rep_idx)
        return graph_embed

    def prepare_node_features(self, batch_graph, picked_nodes):
        n_nodes = 0
        prefix_sum = []
        picked_ones = []
        for i in range(len(batch_graph)):
            if picked_nodes is not None and picked_nodes[i] is not None:
                assert 0 <= picked_nodes[i] < batch_graph[i].num_nodes

                picked_ones.append(n_nodes + picked_nodes[i])
            n_nodes += batch_graph[i].num_nodes
            prefix_sum.append(n_nodes)

        node_feat = torch.zeros(n_nodes, self.num_node_feats)
        node_feat[:, 0] = 1.0

        if len(picked_ones):
            node_feat.numpy()[picked_ones, 1] = 1.0
            node_feat.numpy()[picked_ones, 0] = 0.0

        return node_feat, torch.LongTensor(prefix_sum)

    def forward(self, states, actions, greedy_acts=False):
        batch_graph = states

        node_feat, prefix_sum = self.prepare_node_features(batch_graph, None)
        # if get_device_placement() == "GPU":
        #     node_feat = node_feat.cuda()
        #     prefix_sum = prefix_sum.cuda()
        # edge_feat = None
        # embed, graph_embed = self.s2v(
        #     batch_graph, node_feat, edge_feat, pool_global=True)
        pyg_graphs = []
        for graph in batch_graph:
            nx_graph = graph.nx_graph
            picked_nodes = graph.picked_nodes
            node_degrees = graph.node_degrees
            for n in np.arange(len(nx_graph)):
                nx_graph.nodes[n]["x"] = [np.float32(graph.capacity_og[n]), np.float32(graph.capacity[n])]
            pyg_graph = from_networkx(nx_graph)
            pyg_graphs.append(pyg_graph)
        batch = Batch.from_data_list(pyg_graphs)
        embed, graph_embed = self.gnn(
            batch.x, batch.edge_index, batch.edge_weight, batch.batch
        )
        # embed, graph_embed, prefix_sum = self.run_s2v_embedding(batch_graph, prefix_sum)

        prefix_sum = Variable(prefix_sum)
        if actions is None:
            graph_embed = self.rep_global_embed(graph_embed, prefix_sum)
        else:
            shifted = self.add_offset(actions, prefix_sum)
            embed = embed[shifted, :]

        embed_s_a = torch.cat((embed, graph_embed), dim=1)
        embed_s_a = F.relu(self.linear_1(embed_s_a))
        raw_pred = self.linear_out(embed_s_a)

        if greedy_acts:
            actions, _ = greedy_actions(raw_pred, prefix_sum, None)

        return actions, raw_pred, prefix_sum


class NStepQNet(nn.Module):
    def __init__(self, hyperparams, num_steps):
        super(NStepQNet, self).__init__()

        list_mod = [QNet(hyperparams, None)]

        for i in range(1, num_steps):
            list_mod.append(QNet(hyperparams, list_mod[0].gnn))

        self.list_mod = nn.ModuleList(list_mod)
        self.num_steps = num_steps

    def forward(self, time_t, states, actions, greedy_acts=False):
        assert 0 <= time_t < self.num_steps

        return self.list_mod[time_t](states, actions, greedy_acts)
