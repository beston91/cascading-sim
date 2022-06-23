from __future__ import print_function

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(
            in_channels, hidden_channels, cached=True, normalize=not True
        )
        self.conv2 = GCNConv(
            hidden_channels, out_channels, cached=True, normalize=not True
        )
        self.conv3 = GCNConv(
            hidden_channels, out_channels, cached=True, normalize=not True
        )

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        embed = self.conv2(x, edge_index, edge_weight)
        graph_embed = self.conv3(x, edge_index, edge_weight)
        graph_embed = global_add_pool(graph_embed, batch=batch)
        return embed, graph_embed
