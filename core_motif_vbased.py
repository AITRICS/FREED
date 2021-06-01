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
from torch.distributions.categorical import Categorical

import gym
import dgl
import dgl.function as fn
from dgl.nn.pytorch.glob import SumPooling

from rdkit import Chem

from layers.gin_e_layer import *
from gym_molecule.envs.env_utils_graph import ATOM_VOCAB, SFS_VOCAB, SFS_VOCAB_MOL

from descriptors import ecfp, rdkit_descriptors
from core_motif_mc import GCNEmbed_MC


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

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input: 
        vector x, 
        [x0, 
         x1, 
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class GCN(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0., agg="sum", is_normalize=False, residual=True):
        super().__init__()
        self.residual = residual
        assert agg in ["sum", "mean"], "Wrong agg type"
        self.agg = agg
        self.is_normalize = is_normalize
        self.linear1 = nn.Linear(in_channels, out_channels, bias=False)
        self.activation = nn.ReLU()
        # self.dropout = nn.Dropout(dropout)

    def forward(self, g):
        h_in = g.ndata['x']
        if self.agg == "sum":
            g.update_all(msg, reduce_sum)
        elif self.agg == "mean":
            g.update_all(msg, reduce_mean)
        h = self.linear1(g.ndata['x'])
        h = self.activation(h)
        if self.is_normalize:
            h = F.normalize(h, p=2, dim=1)
        # h = self.dropout(h)
        if self.residual:
            h += h_in
        return h

class GCNActorCritic(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        # build policy and value functions
        self.embed = GCNEmbed(args)
        ob_space = env.observation_space
        ac_space = env.action_space
        self.env = env
        self.pi = SFSPolicy(ob_space, ac_space, env, args)
        self.v = GCNVFunction(ac_space, args)
        self.cand = self.create_candidate_motifs()

    def create_candidate_motifs(self):
        motif_gs = [self.env.get_observation_mol(mol) for mol in SFS_VOCAB_MOL]
        return motif_gs

    def step(self, o_g_emb, o_n_emb, o_g, cands):
        with torch.no_grad():
            
            ac, ac_prob, log_ac_prob = self.pi(o_g_emb, o_n_emb, o_g, cands)

            dists = self.pi._distribution(ac_prob.cpu())
            logp_a = self.pi._log_prob_from_distribution(dists, ac.cpu())

            v = self.v(o_g_emb)
            
        return ac.cpu().numpy(), v.cpu().numpy(), logp_a.cpu().numpy()

    def act(self, o_g_emb, o_n_emb, o_g, cands):
        return self.step(o_g_emb, o_n_emb, o_g, cands)[0]

from scipy.special import kl_div 

class GCNVFunction(nn.Module):
    def __init__(self, ac_space, args, override_seed=False):
        super().__init__()
        if override_seed:
            seed = args.seed + 1
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.batch_size = args.batch_size
        self.device = args.device
        self.emb_size = args.emb_size
        self.max_action2 = len(ATOM_VOCAB)
        self.max_action_stop = 2

        self.d = 2 * args.emb_size 
        self.out_dim = 1

        self.embed = GCNEmbed(args)
        
        self.vpred_layer = nn.Sequential(
                            nn.Linear(self.d, int(self.d//2), bias=False),
                            nn.ReLU(inplace=False),
                            nn.Linear(int(self.d//2), self.out_dim, bias=True))
    
    def forward(self, o_g_emb):
        
        qpred = self.vpred_layer(o_g_emb)
        return qpred

LOG_STD_MAX = 2
LOG_STD_MIN = -20

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)

class SFSPolicy(nn.Module):
    def __init__(self, ob_space, ac_space, env, args):
        super().__init__()
        self.device = args.device
        self.batch_size = args.batch_size
        self.ac_dim = len(SFS_VOCAB)-1
        self.emb_size = args.emb_size
        self.tau = args.tau
        
        # init candidate atoms
        self.bond_type_num = 4

        self.env = env # env utilized to init cand motif mols
        self.cand = self.create_candidate_motifs()
        self.cand_g = dgl.batch([x['g'] for x in self.cand])
        self.cand_ob_len = self.cand_g.batch_num_nodes().tolist()
        # Create candidate descriptors

        if args.desc == 'ecfp':
            desc = ecfp
            self.desc_dim = 1024
        elif args.desc == 'desc':
            desc = rdkit_descriptors
            self.desc_dim = 199
        self.cand_desc = torch.Tensor([desc(Chem.MolFromSmiles(x['smi'])) 
                                for x in self.cand]).to(self.device)
        self.motif_type_num = len(self.cand)

        self.action1_layers = nn.ModuleList([nn.Bilinear(2*args.emb_size, 2*args.emb_size, args.emb_size).to(self.device),
                                nn.Linear(2*args.emb_size, args.emb_size, bias=False).to(self.device),
                                nn.Linear(2*args.emb_size, args.emb_size, bias=False).to(self.device), 
                                nn.Sequential(
                                nn.Linear(args.emb_size, args.emb_size//2, bias=False),
                                nn.ReLU(inplace=False),
                                nn.Linear(args.emb_size//2, 1, bias=True)).to(self.device)])
                       
        self.action2_layers = nn.ModuleList([nn.Bilinear(self.desc_dim,args.emb_size, args.emb_size).to(self.device),
                                nn.Linear(self.desc_dim, args.emb_size, bias=False).to(self.device),
                                nn.Linear(args.emb_size, args.emb_size, bias=False).to(self.device), 
                                nn.Sequential(
                                nn.Linear(args.emb_size, args.emb_size, bias=False),
                                nn.ReLU(inplace=False),
                                nn.Linear(args.emb_size, args.emb_size, bias=True),
                                nn.ReLU(inplace=False),
                                nn.Linear(args.emb_size, 1, bias=True),
                                )])

        self.action3_layers = nn.ModuleList([nn.Bilinear(2*args.emb_size, 2*args.emb_size, args.emb_size).to(self.device),
                               nn.Linear(2*args.emb_size, args.emb_size, bias=False).to(self.device),
                               nn.Linear(2*args.emb_size, args.emb_size, bias=False).to(self.device),
                               nn.Sequential(
                                nn.Linear(args.emb_size, args.emb_size//2, bias=False),
                                nn.ReLU(inplace=False),
                                nn.Linear(args.emb_size//2, 1, bias=True)).to(self.device)])

        # Zero padding with max number of actions
        self.max_action = 40 # max atoms
        
        print('number of candidate motifs : ', len(self.cand))
        self.ac3_att_len = torch.LongTensor([len(x['att']) 
                                for x in self.cand]).to(self.device)
        self.ac3_att_mask = torch.cat([torch.LongTensor([i]*len(x['att'])) 
                                for i, x in enumerate(self.cand)], dim=0).to(self.device)

    def create_candidate_motifs(self):
        motif_gs = [self.env.get_observation_mol(mol) for mol in SFS_VOCAB_MOL]
        return motif_gs


    def gumbel_softmax(self, logits: torch.Tensor, tau: float = 1, hard: bool = False, eps: float = 1e-10, dim: int = -1, \
                    g_ratio: float = 1e-3) -> torch.Tensor:
        gumbels = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
        )  # ~Gumbel(0,1)
        gumbels = (logits + gumbels * g_ratio) / tau  # ~Gumbel(logits,tau)
        y_soft = gumbels.softmax(dim)
        
        if hard:
            # Straight through.
            index = y_soft.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            # Reparametrization trick.
            ret = y_soft
        return ret

    def forward(self, graph_emb, node_emb, g, cands, deterministic=False):
        """
        graph_emb : bs x hidden_dim
        node_emb : (bs x num_nodes) x hidden_dim)
        g: batched graph
        att: indexs of attachment points, list of list
        """
        
        g.ndata['node_emb'] = node_emb
        cand_g, cand_node_emb, cand_graph_emb = cands

        # Only acquire node embeddings with attatchment points
        ob_len = g.batch_num_nodes().tolist()
        att_mask = g.ndata['att_mask'] # used to select att embs from node embs
        
        if g.batch_size != 1:
            att_mask_split = torch.split(att_mask, ob_len, dim=0)
            att_len = [torch.sum(x, dim=0) for x in att_mask_split]
        else:
            att_len = torch.sum(att_mask, dim=-1) # used to torch.split for att embs

        cand_att_mask = cand_g.ndata['att_mask']
        cand_att_mask_split = torch.split(cand_att_mask, self.cand_ob_len, dim=0)
        cand_att_len = [torch.sum(x, dim=0) for x in cand_att_mask_split]

        # =============================== 
        # step 1 : where to add
        # =============================== 
        # select only nodes with attachment points
        att_emb = torch.masked_select(node_emb , att_mask.unsqueeze(-1))
        att_emb = att_emb.view(-1, 2*self.emb_size)
        
        if g.batch_size != 1:
            graph_expand = torch.cat([graph_emb[i].unsqueeze(0).repeat(att_len[i],1) for i in range(g.batch_size)], dim=0).contiguous()
        else:
            graph_expand = graph_emb.repeat(att_len, 1)

        att_emb = self.action1_layers[0](att_emb, graph_expand) + self.action1_layers[1](att_emb) \
                    + self.action1_layers[2](graph_expand)
        logits_first = self.action1_layers[3](att_emb)

        if g.batch_size != 1:
            ac_first_prob = [torch.softmax(logit, dim=0)
                            for i, logit in enumerate(torch.split(logits_first, att_len, dim=0))]
            ac_first_prob = [p+1e-8 for p in ac_first_prob]
            log_ac_first_prob = [x.log() for x in ac_first_prob]

        else:
            ac_first_prob = torch.softmax(logits_first, dim=0) + 1e-8
            log_ac_first_prob = ac_first_prob.log()

        if g.batch_size != 1:  
            first_stack = []
            first_ac_stack = []
            for i, node_emb_i in enumerate(torch.split(att_emb, att_len, dim=0)):
                ac_first_hot_i = self.gumbel_softmax(ac_first_prob[i], tau=self.tau, hard=True, dim=0).transpose(0,1)
                ac_first_i = torch.argmax(ac_first_hot_i, dim=-1)
                first_stack.append(torch.matmul(ac_first_hot_i, node_emb_i))
                first_ac_stack.append(ac_first_i)

            emb_first = torch.stack(first_stack, dim=0).squeeze(1)
            ac_first = torch.stack(first_ac_stack, dim=0).squeeze(1)
            
            ac_first_prob = torch.cat([
                                torch.cat([ac_first_prob_i, ac_first_prob_i.new_zeros(
                                    max(self.max_action - ac_first_prob_i.size(0),0),1)]
                                        , 0).contiguous().view(1,self.max_action)
                                for i, ac_first_prob_i in enumerate(ac_first_prob)], dim=0).contiguous()

            log_ac_first_prob = torch.cat([
                                    torch.cat([log_ac_first_prob_i, log_ac_first_prob_i.new_zeros(
                                        max(self.max_action - log_ac_first_prob_i.size(0),0),1)]
                                            , 0).contiguous().view(1,self.max_action)
                                    for i, log_ac_first_prob_i in enumerate(log_ac_first_prob)], dim=0).contiguous()
            
        else:            
            ac_first_hot = self.gumbel_softmax(ac_first_prob, tau=self.tau, hard=True, dim=0).transpose(0,1)
            ac_first = torch.argmax(ac_first_hot, dim=-1)
            emb_first = torch.matmul(ac_first_hot, att_emb)
            ac_first_prob = torch.cat([ac_first_prob, ac_first_prob.new_zeros(
                            max(self.max_action - ac_first_prob.size(0),0),1)]
                                , 0).contiguous().view(1,self.max_action)
            log_ac_first_prob = torch.cat([log_ac_first_prob, log_ac_first_prob.new_zeros(
                            max(self.max_action - log_ac_first_prob.size(0),0),1)]
                                , 0).contiguous().view(1,self.max_action)

        # =============================== 
        # step 2 : which motif to add - Using Descriptors
        # ===============================   

        emb_first_expand = emb_first.view(-1, 1, self.emb_size).repeat(1, self.motif_type_num, 1)
        cand_expand = self.cand_desc.unsqueeze(0).repeat(g.batch_size, 1, 1)
        
        emb_cat = self.action2_layers[0](cand_expand, emb_first_expand) + \
                    self.action2_layers[1](cand_expand) + self.action2_layers[2](emb_first_expand)

        logit_second = self.action2_layers[3](emb_cat).squeeze(-1)
        ac_second_prob = F.softmax(logit_second, dim=-1) + 1e-8
        log_ac_second_prob = ac_second_prob.log()
        
        ac_second_hot = self.gumbel_softmax(ac_second_prob, tau=self.tau, hard=True, g_ratio=1e-3)                                    
        emb_second = torch.matmul(ac_second_hot, cand_graph_emb)
        ac_second = torch.argmax(ac_second_hot, dim=-1)

        # Print gumbel otuput
        ac_second_gumbel = self.gumbel_softmax(ac_second_prob, tau=self.tau, hard=False, g_ratio=1e-3)                                 
        print('ac second prob', ac_second_prob[0])
        print('ac second gumbel', ac_second_gumbel[0])
        
        # ===============================  
        # step 4 : where to add on motif
        # ===============================
        # Select att points from candidate
        cand_att_emb = torch.masked_select(cand_node_emb, cand_att_mask.unsqueeze(-1))
        cand_att_emb = cand_att_emb.view(-1, 2*self.emb_size)

        ac3_att_mask = self.ac3_att_mask.repeat(g.batch_size, 1) # bs x (num cands * num att size)

        # for cpu usage...
        ac3_att_mask = torch.where(ac3_att_mask==ac_second.view(-1, 1),
                            1, 0).view(g.batch_size, -1) # (num_cands * num_nodes)
        ac3_att_mask = ac3_att_mask.bool()

        ac3_cand_emb = torch.masked_select(cand_att_emb.view(1, -1, 2*self.emb_size), 
                                ac3_att_mask.view(g.batch_size, -1, 1)).view(-1, 2*self.emb_size)
        
        
        ac3_att_len = torch.index_select(self.ac3_att_len, 0, ac_second).tolist()
        emb_second_expand = torch.cat([emb_second[i].unsqueeze(0).repeat(ac3_att_len[i],1) for i in range(g.batch_size)]).contiguous()

        emb_cat_ac3 = self.action3_layers[0](emb_second_expand, ac3_cand_emb) + self.action3_layers[1](emb_second_expand) \
                  + self.action3_layers[2](ac3_cand_emb)
        
        logits_third = self.action3_layers[3](emb_cat_ac3)

        # predict logit
        if g.batch_size != 1:
            ac_third_prob = [torch.softmax(logit,dim=-1)
                            for i, logit in enumerate(torch.split(logits_third.squeeze(-1), ac3_att_len, dim=0))]
            ac_third_prob = [p+1e-8 for p in ac_third_prob]
            log_ac_third_prob = [x.log() for x in ac_third_prob]

        else:
            logits_third = logits_third.transpose(1,0)
            ac_third_prob = torch.softmax(logits_third, dim=-1) + 1e-8
            log_ac_third_prob = ac_third_prob.log()
        
        # gumbel softmax sampling and zero-padding
        if g.batch_size != 1:
            third_stack = []
            third_ac_stack = []
            for i, node_emb_i in enumerate(torch.split(emb_cat_ac3, ac3_att_len, dim=0)):
                ac_third_hot_i = self.gumbel_softmax(ac_third_prob[i], tau=self.tau, hard=True, dim=-1)
                ac_third_i = torch.argmax(ac_third_hot_i, dim=-1)
                third_stack.append(torch.matmul(ac_third_hot_i, node_emb_i))
                third_ac_stack.append(ac_third_i)

                del ac_third_hot_i
            emb_third = torch.stack(third_stack, dim=0).squeeze(1)
            ac_third = torch.stack(third_ac_stack, dim=0)
            ac_third_prob = torch.cat([
                                torch.cat([ac_third_prob_i, ac_third_prob_i.new_zeros(
                                    self.max_action - ac_third_prob_i.size(0))]
                                        , dim=0).contiguous().view(1,self.max_action)
                                for i, ac_third_prob_i in enumerate(ac_third_prob)], dim=0).contiguous()
            
            log_ac_third_prob = torch.cat([
                                    torch.cat([log_ac_third_prob_i, log_ac_third_prob_i.new_zeros(
                                        self.max_action - log_ac_third_prob_i.size(0))]
                                            , 0).contiguous().view(1,self.max_action)
                                    for i, log_ac_third_prob_i in enumerate(log_ac_third_prob)], dim=0).contiguous()

        else:
            ac_third_hot = self.gumbel_softmax(ac_third_prob, tau=self.tau, hard=True, dim=-1)
            ac_third = torch.argmax(ac_third_hot, dim=-1)
            emb_third = torch.matmul(ac_third_hot, emb_cat_ac3)
            
            ac_third_prob = torch.cat([ac_third_prob, ac_third_prob.new_zeros(
                                        1, self.max_action - ac_third_prob.size(1))] 
                                , -1).contiguous()
            log_ac_third_prob = torch.cat([log_ac_third_prob, log_ac_third_prob.new_zeros(
                                        1, self.max_action - log_ac_third_prob.size(1))]
                                , -1).contiguous()

        # ==== concat everything ====

        ac_prob = torch.cat([ac_first_prob, ac_second_prob, ac_third_prob], dim=1).contiguous()
        log_ac_prob = torch.cat([log_ac_first_prob, 
                            log_ac_second_prob, log_ac_third_prob], dim=1).contiguous()
        ac = torch.stack([ac_first, ac_second, ac_third], dim=1)

        return ac, ac_prob, log_ac_prob

    def _distribution(self, ac_prob):
        
        ac_prob_split = torch.split(ac_prob, [self.max_action, len(SFS_VOCAB), self.max_action], dim=1)
        dists = [Categorical(probs=pr) for pr in ac_prob_split]
        return dists

    def _log_prob_from_distribution(self, dists, act):
        log_probs = [p.log_prob(act[0][i]) for i, p in enumerate(dists)]
        
        return torch.cat(log_probs, dim=0)
    
    def sample(self, ac, graph_emb, node_emb, g, cands):
        g.ndata['node_emb'] = node_emb
        cand_g, cand_node_emb, cand_graph_emb = cands 

        # Only acquire node embeddings with attatchment points
        ob_len = g.batch_num_nodes().tolist()
        att_mask = g.ndata['att_mask'] # used to select att embs from node embs
        att_len = torch.sum(att_mask, dim=-1) # used to torch.split for att embs

        cand_att_mask = cand_g.ndata['att_mask']
        cand_att_mask_split = torch.split(cand_att_mask, self.cand_ob_len, dim=0)
        cand_att_len = [torch.sum(x, dim=0) for x in cand_att_mask_split]

        # =============================== 
        # step 1 : where to add
        # =============================== 
        # select only nodes with attachment points
        att_emb = torch.masked_select(node_emb, att_mask.unsqueeze(-1))
        att_emb = att_emb.view(-1, 2*self.emb_size)
        graph_expand = graph_emb.repeat(att_len, 1)
        
        att_emb = self.action1_layers[0](att_emb, graph_expand) + self.action1_layers[1](att_emb) \
                    + self.action1_layers[2](graph_expand)
        logits_first = self.action1_layers[3](att_emb).transpose(1,0)
            
        ac_first_prob = torch.softmax(logits_first, dim=-1) + 1e-8
        
        log_ac_first_prob = ac_first_prob.log()
        ac_first_prob = torch.cat([ac_first_prob, ac_first_prob.new_zeros(1,
                        max(self.max_action - ac_first_prob.size(1),0))]
                            , 1).contiguous()
        
        log_ac_first_prob = torch.cat([log_ac_first_prob, log_ac_first_prob.new_zeros(1,
                        max(self.max_action - log_ac_first_prob.size(1),0))]
                            , 1).contiguous()
        emb_first = att_emb[ac[0]].unsqueeze(0)
        
        # =============================== 
        # step 2 : which motif to add     
        # ===============================   
        emb_first_expand = emb_first.repeat(1, self.motif_type_num, 1)
        cand_expand = self.cand_desc.unsqueeze(0).repeat(g.batch_size, 1, 1)     
        
        emb_cat = self.action2_layers[0](cand_expand, emb_first_expand) + \
                    self.action2_layers[1](cand_expand) + self.action2_layers[2](emb_first_expand)

        logit_second = self.action2_layers[3](emb_cat).squeeze(-1)
        ac_second_prob = F.softmax(logit_second, dim=-1) + 1e-8
        log_ac_second_prob = ac_second_prob.log()
        
        ac_second_hot = self.gumbel_softmax(ac_second_prob, tau=self.tau, hard=True, g_ratio=1e-3)                                    
        emb_second = torch.matmul(ac_second_hot, cand_graph_emb)
        ac_second = torch.argmax(ac_second_hot, dim=-1)

        # ===============================  
        # step 3 : where to add on motif
        # ===============================
        # Select att points from candidates
        
        cand_att_emb = torch.masked_select(cand_node_emb, cand_att_mask.unsqueeze(-1))
        cand_att_emb = cand_att_emb.view(-1, 2*self.emb_size)

        ac3_att_mask = self.ac3_att_mask.repeat(g.batch_size, 1) # bs x (num cands * num att size)
        
        ac3_att_mask = torch.where(ac3_att_mask==ac[1],
                            1, 0).view(g.batch_size, -1) # (num_cands * num_nodes)
        ac3_att_mask = ac3_att_mask.bool()

        

        ac3_cand_emb = torch.masked_select(cand_att_emb.view(1, -1, 2*self.emb_size), 
                                ac3_att_mask.view(g.batch_size, -1, 1)).view(-1, 2*self.emb_size)
        
        ac3_att_len = self.ac3_att_len[ac[1]]
        emb_second_expand = emb_second.repeat(ac3_att_len,1)
        emb_cat_ac3 = self.action3_layers[0](emb_second_expand, ac3_cand_emb) + self.action3_layers[1](emb_second_expand) \
                  + self.action3_layers[2](ac3_cand_emb)

        logits_third = self.action3_layers[3](emb_cat_ac3)
        logits_third = logits_third.transpose(1,0)
        ac_third_prob = torch.softmax(logits_third, dim=-1) + 1e-8
        log_ac_third_prob = ac_third_prob.log()

        # gumbel softmax sampling and zero-padding
        emb_third = emb_cat_ac3[ac[2]].unsqueeze(0)
        ac_third_prob = torch.cat([ac_third_prob, ac_third_prob.new_zeros(
                                        1, self.max_action - ac_third_prob.size(1))] 
                                , -1).contiguous()
        log_ac_third_prob = torch.cat([log_ac_third_prob, log_ac_third_prob.new_zeros(
                                        1, self.max_action - log_ac_third_prob.size(1))]
                                , -1).contiguous()

        # ==== concat everything ====
        ac_prob = torch.cat([ac_first_prob, ac_second_prob, ac_third_prob], dim=1).contiguous()
        log_ac_prob = torch.cat([log_ac_first_prob, 
                            log_ac_second_prob, log_ac_third_prob], dim=1).contiguous()

        return ac_prob, log_ac_prob


class GCNEmbed(nn.Module):
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

        self.gcn_type = args.gcn_type

        self.gcn_layers = nn.ModuleList([GCN(in_channels, self.emb_size, agg="sum", residual=False)])
        for _ in range(args.layer_num_g-1):
            self.gcn_layers.append(GCN(self.emb_size, self.emb_size, agg="sum"))
        
        self.pool = SumPooling()
        
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

        for i, conv in enumerate(self.gcn_layers):
            h = conv(g)
            g.ndata['x'] = h
        
        emb_node = g.ndata['x']

        ## Get graph embedding
        emb_graph = self.pool(g, g.ndata['x'])
        
        return g, emb_node, emb_graph
