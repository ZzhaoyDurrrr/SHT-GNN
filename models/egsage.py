import torch
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops

from torch.nn.init import xavier_uniform_, zeros_

import torch.nn as nn
import torch.nn.functional as F
from utils.utils import get_activation

class EGraphSage(MessagePassing):
    """Non-minibatch version of GraphSage."""
    def __init__(self, in_channels, out_channels,
                 edge_channels, activation, edge_mode,
                 normalize_emb,
                 aggr):
        super(EGraphSage, self).__init__(aggr=aggr)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_channels = edge_channels
        self.edge_mode = edge_mode

        self.message_lin = nn.Linear(in_channels+edge_channels, out_channels)
        self.agg_lin = nn.Linear(in_channels+out_channels, out_channels)

        self.message_activation = get_activation(activation)
        self.update_activation = get_activation(activation)
        self.normalize_emb = normalize_emb

    def forward(self, x, edge_attr, edge_index):
        num_nodes = x.size(0)
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, size=(num_nodes, num_nodes))

    def message(self, x_i, x_j, edge_attr, edge_index, size):
        m_j = torch.cat((x_j, edge_attr),dim=-1)
        m_j = self.message_activation(self.message_lin(m_j))
        return m_j

    def update(self, aggr_out, x):
        aggr_out = self.update_activation(self.agg_lin(torch.cat((aggr_out, x),dim=-1)))
        if self.normalize_emb:
            aggr_out = F.normalize(aggr_out, p=2, dim=-1)
        return aggr_out
