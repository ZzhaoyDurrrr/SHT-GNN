from builtins import print
from numpy.core.numeric import NaN
import torch
from torch._C import get_num_interop_threads
import torch.nn as nn
import torch.nn.functional as F
import pyro
import scipy.sparse as sp
import numpy as np
import math
from sklearn import preprocessing
import time

# WeightGraph
# Longitudinal 
class WeightGraph(nn.Module):
    def __init__(self, dim_feats, dim_h, dim_z, activation):
        super(WeightGraph, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GCNLayer(dim_feats, dim_h, 1, None, 0, bias=False))
        self.layers.append(GCNLayer(dim_h, dim_z, 1, activation, 0, bias=False))

    def forward(self, adj, features, target_edge_index):
        Z = features
        edge_logits = (Z[target_edge_index[0,:]] * Z[target_edge_index[1,:]]).sum(axis=1)
        
        min_val = edge_logits.min()
        max_val = edge_logits.max()
        normalized_edge_logits = (edge_logits - min_val) / (max_val - min_val) 
        return normalized_edge_logits

class WeightGraph(nn.Module):
    def __init__(self, dim_feats, dim_h, dim_z, activation):
        super(WeightGraph, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GCNLayer(dim_feats, dim_h, 1, None, 0, bias=False))
        self.layers.append(GCNLayer(dim_h, dim_z, 1, activation, 0, bias=False))

    def forward(self, features, target_edge_index, mode):
        if mode == 1:
            Z = features
            edge_logits = (Z[target_edge_index[0,:]] * Z[target_edge_index[1,:]]).sum(axis=1)
        return edge_logits

    
class GCNLayer(nn.Module):
    """ one layer of GCN """
    def __init__(self, input_dim, output_dim, n_heads, activation, dropout, bias=True):
        super(GCNLayer, self).__init__()
        self.W = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        self.activation = activation
        if bias:
            self.b = nn.Parameter(torch.FloatTensor(output_dim))
        else:
            self.b = None
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0
        self.init_params()

    def init_params(self):
        """ Initialize weights with xavier uniform and biases with all zeros """
        for param in self.parameters():
            if len(param.size()) == 2:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.constant_(param, 0.0)

    def forward(self, adj, h):
        if self.dropout:
            h = self.dropout(h)
        x = h @ self.W
        x = adj @ x
        if self.b is not None:
            x = x + self.b
        if self.activation:
            x = self.activation(x)
        return x
