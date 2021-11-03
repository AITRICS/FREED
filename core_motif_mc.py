from copy import deepcopy
import math
import time
from scipy import sparse
import scipy.signal
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.distributions as td
from torch.distributions.normal import Normal

import gym
import dgl
import dgl.function as fn
from dgl.nn.pytorch.glob import SumPooling

from rdkit import Chem

from gym_molecule.envs.env_utils_graph import ATOM_VOCAB, FRAG_VOCAB, FRAG_VOCAB_MOL

from descriptors import ecfp, rdkit_descriptors


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)
def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])
    
# DGL operations
msg = fn.copy_src(src='x', out='m')
def reduce_mean(nodes):
    accum = torch.mean(nodes.mailbox['m'], 1)
    return {'x': accum}

def reduce_sum(nodes):
    accum = torch.sum(nodes.mailbox['m'], 1)
    return {'x': accum}  

def MC_dropout(act_vec, p=0.3, mask=True):
    return F.dropout(act_vec, p=p, training=mask, inplace=True)

class GCN_MC(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=.3, agg="sum", is_normalize=False, residual=True):
        super().__init__()
        self.residual = residual
        assert agg in ["sum", "mean"], "Wrong agg type"
        self.agg = agg
        self.is_normalize = is_normalize
        self.linear1 = nn.Linear(in_channels, out_channels, bias=False)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, g):
        h_in = g.ndata['x']
        if self.agg == "sum":
            g.update_all(msg, reduce_sum)
        elif self.agg == "mean":
            g.update_all(msg, reduce_mean)
        h = self.linear1(g.ndata['x'])
        # apply MC dropout
        h = self.dropout(h)
        h = self.activation(h)
        if self.is_normalize:
            h = F.normalize(h, p=2, dim=1)
        
        if self.residual:
            h += h_in
        return h

from scipy.special import kl_div 

LOG_STD_MAX = 2
LOG_STD_MIN = -20

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)

class GCNEmbed_MC(nn.Module):
    def __init__(self, args):

        ### GCN
        super().__init__()
        self.device = args.device
        self.possible_atoms = ATOM_VOCAB
        self.bond_type_num = 4
        self.d_n = len(self.possible_atoms)+18
        
        self.emb_size = args.emb_size * 2
        self.gcn_aggregate = args.gcn_aggregate

        in_channels = 8
        self.emb_linear = nn.Linear(self.d_n, in_channels, bias=False)

        self.dropout = nn.Dropout(args.dropout)

        self.gcn_type = args.gcn_type
        assert args.gcn_type in ['GCN', 'GINE'], "Wrong gcn type"
        assert args.gcn_aggregate in ['sum', 'gmt'], "Wrong gcn agg type"

        # if args.gcn_type == 'GCN':
        self.gcn_layers = nn.ModuleList([GCN_MC(in_channels, self.emb_size, 
                            dropout=args.dropout, agg="sum", residual=False)])
        for _ in range(args.layer_num_g-1):
            self.gcn_layers.append(GCN_MC(self.emb_size, self.emb_size, 
                            dropout=args.dropout, agg="sum"))
    
        if self.gcn_aggregate == 'sum':
            self.pool = SumPooling()
        else:
            pass
        
    def forward(self, ob):
        ## Graph
        
        ob_g = [o['g'] for o in ob]
        ob_att = [o['att'] for o in ob]

        # create attachment point mask as one-hot
        for i, x_g in enumerate(ob_g):
            att_onehot = F.one_hot(torch.LongTensor(ob_att[i]), 
                        num_classes=x_g.number_of_nodes()).sum(0)
            ob_g[i].ndata['att_mask'] = att_onehot.bool()

        g = deepcopy(dgl.batch(ob_g)).to(self.device)
        
        g.ndata['x'] = self.emb_linear(g.ndata['x'])
        g.ndata['x'] = self.dropout(g.ndata['x'])

        for i, conv in enumerate(self.gcn_layers):
            h = conv(g)
            g.ndata['x'] = h
        
        emb_node = g.ndata['x']

        ## Get graph embedding
        emb_graph = self.pool(g, g.ndata['x'])
        
        return g, emb_node, emb_graph


