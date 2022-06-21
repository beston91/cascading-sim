import argparse
import os.path as osp
import random

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
import wandb
from torch_geometric.datasets import Planetoid
from torch_geometric.utils.convert import from_networkx
from torch_geometric.nn import GCNConv
import networkx as nx
import numpy as np
import sys


from graph_tiger.graphs import graph_loader
sys.path.append('/graph_tiger')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Cora')
parser.add_argument('--hidden_channels', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--use_gdc', action='store_true', help='Use GDC')
parser.add_argument('--wandb', action='store_true', help='Track experiment')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
wandb.init(project="PCF", group="playground",
           name="play"+wandb.util.generate_id())

graph = graph_loader('BA', n=500, seed=1)


def setup_nx_graph(graph, train_ratio=0.8):
    for n in np.arange(len(graph)):
        if n < 250:
            graph.nodes[n]['x'] = np.float32(
                np.random.uniform(low=0.0, high=0.5, size=(5,)))
            graph.nodes[n]['y'] = np.float32(1.0)
        else:
            graph.nodes[n]['x'] = np.float32(
                np.random.uniform(low=0.5, high=1.0, size=(5,)))
            graph.nodes[n]['y'] = np.float32(0.0)
    pyg_graph = from_networkx(graph)

    # Split the data
    num_nodes = pyg_graph.x.shape[0]
    num_train = int(num_nodes * train_ratio)
    num_test = int((num_nodes - num_train)/2)
    idx = [i for i in range(num_nodes)]

    np.random.shuffle(idx)
    train_mask = torch.full_like(pyg_graph.y, False, dtype=bool)
    train_mask[idx[:num_train]] = True
    test_mask = torch.full_like(pyg_graph.y, False, dtype=bool)
    test_mask[idx[num_train: num_train+num_test]] = True
    val_mask = torch.full_like(pyg_graph.y, False, dtype=bool)
    val_mask[idx[num_train+num_test:]] = True

    pyg_graph.train_mask = train_mask
    pyg_graph.test_mask = test_mask
    pyg_graph.val_mask = val_mask
    if args.use_gdc:
        transform = T.GDC(
            self_loop_weight=1,
            normalization_in='sym',
            normalization_out='col',
            diffusion_kwargs=dict(method='ppr', alpha=0.05),
            sparsification_kwargs=dict(method='topk', k=128, dim=0),
            exact=True,
        )
        pyg_graph = transform(pyg_graph)
    return pyg_graph


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=True,
                             normalize=not args.use_gdc)
        self.conv2 = GCNConv(hidden_channels, hidden_channels, cached=True,
                             normalize=not args.use_gdc)
        self.conv3 = GCNConv(hidden_channels, out_channels, cached=True,
                             normalize=not args.use_gdc)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        x = self.conv3(x, edge_index, edge_weight)
        return torch.sigmoid(x)


model = GCN(-1, args.hidden_channels, 1)
# model, data = model.to(device), data.to(device)
optimizer = torch.optim.Adam([
    dict(params=model.conv1.parameters(), weight_decay=5e-4),
    dict(params=model.conv2.parameters(), weight_decay=0)
], lr=args.lr)  # Only perform weight-decay on first convolution.


def train(data):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, data.edge_weight)
    loss = F.binary_cross_entropy(
        out[data.train_mask], data.y[data.train_mask][:, None])
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test(data):
    model.eval()
    pred = model(data.x, data.edge_index, data.edge_weight).argmax(dim=-1)

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        accs.append(
            int((pred[:, None][mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs


data = setup_nx_graph(graph)
best_val_acc = final_test_acc = 0
for epoch in range(1, args.epochs + 1):
    loss = train(data)
    train_acc, val_acc, tmp_test_acc = test(data)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    wandb.log({'accuracy': train_acc, 'test_acc': test_acc, 'loss': loss})
