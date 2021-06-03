import time
from copy import deepcopy
import itertools

import numpy as np
from rdkit import Chem
import pickle

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam, lr_scheduler
from torch.autograd import Variable
from tensorboardX import SummaryWriter

import gym

import core_motif as core
from gym_molecule.envs.env_utils_graph import SFS_VOCAB, ATOM_VOCAB

def delete_multiple_element(list_object, indices):
    indices = sorted(indices, reverse=True)
    for idx in indices:
        if idx < len(list_object):
            list_object.pop(idx)

def get_att_points(mol):
    att_points = []
    for a in mol.GetAtoms(): 
        if a.GetSymbol() == '*':
            att_points.append(a.GetIdx())
    return att_points

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = [] # o
        self.obs2_buf = [] # o2
        self.act_buf = np.zeros((size, 3), dtype=np.int32) # ac
        self.rew_buf = np.zeros(size, dtype=np.float32) # r
        self.done_buf = np.zeros(size, dtype=np.float32) # d
        
        self.ac_prob_buf = []
        self.log_ac_prob_buf = []
        
        self.ac_first_buf = []
        self.ac_second_buf = []
        self.ac_third_buf = []

        self.o_embeds_buf = []
        
        self.ptr, self.size, self.max_size = 0, 0, size
        self.done_location = []

    def store(self, obs, act, rew, next_obs, done, ac_prob, log_ac_prob, ac_first_prob, ac_second_hot, ac_third_prob, o_embeds):
        if self.size == self.max_size:
            self.obs_buf.pop(0)
            self.obs2_buf.pop(0)
            
            self.ac_prob_buf.pop(0)
            self.log_ac_prob_buf.pop(0)
            
            self.ac_first_buf.pop(0)
            self.ac_second_buf.pop(0)
            self.ac_third_buf.pop(0)

            self.o_embeds_buf.pop(0)

        self.obs_buf.append(obs)
        self.obs2_buf.append(next_obs)
        
        self.ac_prob_buf.append(ac_prob)
        self.log_ac_prob_buf.append(log_ac_prob)
        
        self.ac_first_buf.append(ac_first_prob)
        self.ac_second_buf.append(ac_second_hot)
        self.ac_third_buf.append(ac_third_prob)
        self.o_embeds_buf.append(o_embeds)

        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done

        if done:
            self.done_location.append(self.ptr)
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def rew_store(self, rew, batch_size=32):
        rew_ls = list(rew)
        
        done_location_np = np.array(self.done_location)
        zeros = np.where(rew==0.0)[0]
        nonzeros = np.where(rew!=0.0)[0]
        zero_ptrs = done_location_np[zeros]        

        done_location_np = done_location_np[nonzeros]
        rew = rew[nonzeros]
        
        if len(self.done_location) > 0:
            self.rew_buf[done_location_np] += rew
            self.done_location = []

        self.act_buf = np.delete(self.act_buf, zero_ptrs, axis=0)
        self.rew_buf = np.delete(self.rew_buf, zero_ptrs)
        self.done_buf = np.delete(self.done_buf, zero_ptrs)
        delete_multiple_element(self.obs_buf, zero_ptrs.tolist())
        delete_multiple_element(self.obs2_buf, zero_ptrs.tolist())

        delete_multiple_element(self.ac_prob_buf, zero_ptrs.tolist())
        delete_multiple_element(self.log_ac_prob_buf, zero_ptrs.tolist())
        
        delete_multiple_element(self.ac_first_buf, zero_ptrs.tolist())
        delete_multiple_element(self.ac_second_buf, zero_ptrs.tolist())
        delete_multiple_element(self.ac_third_buf, zero_ptrs.tolist())

        delete_multiple_element(self.o_embeds_buf, zero_ptrs.tolist())

        self.size = min(self.size-len(zero_ptrs), self.max_size)
        self.ptr = (self.ptr-len(zero_ptrs)) % self.max_size
        
    def sample_batch(self, device, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        obs_batch = [self.obs_buf[idx] for idx in idxs]
        obs2_batch = [self.obs2_buf[idx] for idx in idxs]

        ac_prob_batch = [self.ac_prob_buf[idx] for idx in idxs]
        log_ac_prob_batch = [self.log_ac_prob_buf[idx] for idx in idxs]
        
        ac_first_batch = torch.stack([self.ac_first_buf[idx].to(device) for idx in idxs]).squeeze(1)
        ac_second_batch = torch.stack([self.ac_second_buf[idx].to(device) for idx in idxs]).squeeze(1)
        ac_third_batch = torch.stack([self.ac_third_buf[idx].to(device) for idx in idxs]).squeeze(1)
        o_g_emb_batch = torch.stack([self.o_embeds_buf[idx][2] for idx in idxs]).squeeze(1)

        act_batch = torch.as_tensor(self.act_buf[idxs], dtype=torch.float32).unsqueeze(-1).to(device)
        rew_batch = torch.as_tensor(self.rew_buf[idxs], dtype=torch.float32).to(device)
        done_batch = torch.as_tensor(self.done_buf[idxs], dtype=torch.float32).to(device)

        batch = dict(obs=obs_batch,
                     obs2=obs2_batch,
                     act=act_batch,
                     rew=rew_batch,
                     done=done_batch,
                     ac_prob=ac_prob_batch,
                     log_ac_prob=log_ac_prob_batch,
                     
                     ac_first=ac_first_batch,
                     ac_second=ac_second_batch,
                     ac_third=ac_third_batch,
                     o_g_emb=o_g_emb_batch)
        return batch

def xavier_uniform_init(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)

def xavier_normal_init(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal(m.weight)

class sac:
    """
    Soft Actor-Critic (SAC)
    """
    def __init__(self, writer, args, env_fn, actor_critic=core.GCNActorCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99, 
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=500, 
        update_after=100, update_every=50, update_freq=50, expert_every=5, num_test_episodes=10, max_ep_len=1000, 
        save_freq=1, train_alpha=True):
        super().__init__()
        self.device = args.device

        torch.manual_seed(seed)
        np.random.seed(seed)
        self.gamma = gamma
        self.polyak = polyak
        self.num_test_episodes = num_test_episodes
        self.writer = writer
        self.fname = 'molecule_gen/'+args.name_full+'.csv'
        self.test_fname = 'molecule_gen/'+args.name_full+'_test.csv'
        self.save_name = './ckpt/' + args.name_full + '_'
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.batch_size = batch_size
        self.replay_size = replay_size
        self.start_steps = start_steps
        self.max_ep_len = args.max_action
        self.update_after = update_after
        self.update_every = update_every
        self.update_freq = update_freq
        self.docking_every = int(update_every/2)
        self.save_freq = save_freq
        self.train_alpha = train_alpha
        
        self.pretrain_q = -1

        self.env, self.test_env = env_fn, deepcopy(env_fn)

        # self.obs_dim = 128
        self.obs_dim = args.emb_size * 2
        self.act_dim = len(SFS_VOCAB)-1

        # gan
        self.adv_rew = args.adv_rew
        self.adv_rew_ratio = args.adv_rew_ratio

        # intrinsic reward
        self.intr_rew = args.intr_rew
        self.intr_rew_ratio = args.intr_rew_ratio

        self.ac1_dims = 40 
        self.ac2_dims = len(SFS_VOCAB) # 76
        self.ac3_dims = 40 
        self.action_dims = [self.ac1_dims, self.ac2_dims, self.ac3_dims]

        self.target_entropy = args.target_entropy

        self.log_alpha = torch.tensor([np.log(alpha)], requires_grad=train_alpha) 
        alpha = self.log_alpha.exp().item()

        # Create actor-critic module and target networks
        self.ac = actor_critic(self.env, args).to(args.device)
        self.ac_targ = deepcopy(self.ac).to(args.device).eval()

        if args.load==1:
            fname = args.name_full_load
            self.ac.load_state_dict(torch.load(fname))
            self.ac_targ = deepcopy(self.ac).to(args.device)
            print(f"loaded model {fname} successfully")

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False

        for q in self.ac.parameters():
            q.requires_grad = True

        # Experience buffer
        self.replay_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=replay_size)

        # Count variables (protip: try to get a feel for how different size networks behave!)
        self.var_counts = tuple(core.count_vars(module) for module in [self.ac.pi, self.ac.q1, self.ac.q2])
        self.iter_so_far = 0
        self.ep_so_far = 0

        ## OPTION1: LEARNING RATE
        pi_lr = 1e-4
        q_lr = 1e-4

        alpha_lr = 5e-4
        d_lr = 1e-3
        p_lr = 1e-3
    
        ## OPTION2: OPTIMIZER SETTING        
        self.pi_params = list(self.ac.pi.parameters()) 
        self.q_params = list(self.ac.q1.parameters()) + list(self.ac.q2.parameters()) + list(self.ac.embed.parameters())
        self.alpha_params = [self.log_alpha]

        self.emb_params = list(self.ac.embed.parameters())

        self.pi_optimizer = Adam(self.pi_params, lr=pi_lr, weight_decay=1e-4)
        self.q_optimizer = Adam(self.q_params, lr=q_lr, weight_decay=1e-4)
        self.alpha_optimizer = Adam(self.alpha_params, lr=alpha_lr, eps=1e-4)

        if args.intr_rew is not None:
            self.p_params = list(self.ac.p.parameters())
            self.p_optimizer = Adam(self.p_params, lr=p_lr, weight_decay=1e-4)
            self.p_scheduler = lr_scheduler.ReduceLROnPlateau(self.p_optimizer, factor=0.1, patience=500)

        self.q_scheduler = lr_scheduler.ReduceLROnPlateau(self.q_optimizer, factor=0.1, patience=768) 
        self.pi_scheduler = lr_scheduler.ReduceLROnPlateau(self.pi_optimizer, factor=0.1, patience=768)        
  
        self.L2_loss = torch.nn.MSELoss()

        torch.set_printoptions(profile="full")
        self.possible_bonds = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE]

        self.alpha_start = self.start_steps
        self.alpha_end = self.start_steps + 30000
        self.t = 0

        self.ac.apply(xavier_uniform_init)
        # self.ac.apply(xavier_normal_init)

        tm = time.localtime(time.time())
        self.init_tm = time.strftime('_%Y-%m-%d_%I:%M:%S-%p', tm)
        
    def compute_loss_q(self, data):

        ac_first, ac_second, ac_third = data['ac_first'], data['ac_second'], data['ac_third']             
        self.ac.q1.train()
        self.ac.q2.train()
        o = data['obs']
        _, _, o_g_emb = self.ac.embed(o)
        q1 = self.ac.q1(o_g_emb, ac_first, ac_second, ac_third).squeeze()
        q2 = self.ac.q2(o_g_emb.detach(), ac_first, ac_second, ac_third).squeeze()

        # Target actions come from *current* policy
        o2 = data['obs2']
        r, d = data['rew'], data['done']
        alpha = min(self.log_alpha.exp().item(), 5)
        with torch.no_grad():
            o2_g, o2_n_emb, o2_g_emb = self.ac.embed(o2)
            cands = self.ac.embed(self.ac.pi.cand)
            a2, (a2_prob, log_a2_prob), (ac2_first, ac2_second, ac2_third) = self.ac.pi(o2_g_emb, o2_n_emb, o2_g, cands)
            # Target Q-values
            q1_pi_targ = self.ac_targ.q1(o2_g_emb, ac2_first, ac2_second, ac2_third)
            q2_pi_targ = self.ac_targ.q2(o2_g_emb, ac2_first, ac2_second, ac2_third)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ).squeeze() 
            backup = r + self.gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2
        print('Q loss', loss_q1, loss_q2)

        return loss_q

    # Set up function for computing SAC pi loss
    def compute_loss_pi(self, data):
        with torch.no_grad():
            o_embeds = self.ac.embed(data['obs'])   
            o_g, o_n_emb, o_g_emb = o_embeds
            cands = self.ac.embed(self.ac.pi.cand)

        _, (ac_prob, log_ac_prob), (ac_first, ac_second, ac_third) = \
            self.ac.pi(o_g_emb, o_n_emb, o_g, cands)

        q1_pi = self.ac.q1(o_g_emb, ac_first, ac_second, ac_third)
        q2_pi = self.ac.q2(o_g_emb, ac_first, ac_second, ac_third)
        q_pi = torch.min(q1_pi, q2_pi)

        ac_prob_sp = torch.split(ac_prob, self.action_dims, dim=1)
        log_ac_prob_sp = torch.split(log_ac_prob, self.action_dims, dim=1)
        
        loss_policy = torch.mean(-q_pi)        

        # Entropy-regularized policy loss
        alpha = min(self.log_alpha.exp().item(), 20.)
        alpha = max(self.log_alpha.exp().item(), .05)

        loss_entropy = 0
        loss_alpha = 0

        # New version
        ent_weight = [1, 1, 1]
        # get ac1 x ac2 x ac3 
        
        ac_prob_comb = torch.einsum('by, bz->byz', ac_prob_sp[1], ac_prob_sp[2]).reshape(self.batch_size, -1) # (bs , 73 x 40)
        ac_prob_comb = torch.einsum('bx, bz->bxz', ac_prob_sp[0], ac_prob_comb).reshape(self.batch_size, -1) # (bs , 40 x 73 x 40)
        # order by (a1, b1, c1) (a1, b1, c2)! Be advised!
        
        log_ac_prob_comb = log_ac_prob_sp[0].reshape(self.batch_size, self.action_dims[0], 1, 1).repeat(
                                    1, 1, self.action_dims[1], self.action_dims[2]).reshape(self.batch_size, -1)\
                            + log_ac_prob_sp[1].reshape(self.batch_size, 1, self.action_dims[1], 1).repeat(
                                    1, self.action_dims[0], 1, self.action_dims[2]).reshape(self.batch_size, -1)\
                            + log_ac_prob_sp[2].reshape(self.batch_size, 1, 1, self.action_dims[2]).repeat(
                                    1, self.action_dims[0], self.action_dims[1], 1).reshape(self.batch_size, -1)
        loss_entropy = (alpha * ac_prob_comb * log_ac_prob_comb).sum(dim=1).mean()
        loss_alpha = -(self.log_alpha.to(self.device) * \
                        ((ac_prob_comb*log_ac_prob_comb).sum(dim=1) + self.target_entropy).detach()).mean()

        
        print('loss policy', loss_policy)
        print('loss entropy', loss_entropy)
        print('loss_alpha', loss_alpha)

        # Record things
        if self.writer is not None:
            self.writer.add_scalar("Entropy", sum(-(x * ac_prob_sp[i]).mean() for i, x in enumerate(log_ac_prob_sp)), self.iter_so_far)
            self.writer.add_scalar("Alpha", alpha, self.iter_so_far)

        return loss_entropy, loss_policy, loss_alpha

    # Curiosity driven model with intr / mc

    def compute_intr_rew(self, o_embed, rew):
        pred = self.ac.p(o_embed).squeeze()
        rew = torch.tensor(rew).to(self.device).float()
        rew_intr = torch.abs(pred - rew).cpu().detach().numpy()
        loss_p = self.L2_loss(pred, rew).float()
        return loss_p, rew_intr

    def L2_dist(self, x, y):
        return (x-y)**2

    def MCDropoutLoss(self, x, y, logvar):
        return torch.mean(.5*(-logvar).exp().unsqueeze(1)*self.L2_dist(x, y)) + .5*torch.mean(logvar)

    def total_var(self, x_samples, logvar_samples):
        return torch.var(x_samples, dim=1) + torch.mean(logvar_samples.exp(), dim=1)

    def compute_active_loss(self, ob, rew):
        
        # Acquire single sample for loss calculation
        pred_mean, pred_logvar  = self.ac.p(ob)
        rew = torch.tensor(rew).to(self.device).float()
        
        loss_p = self.MCDropoutLoss(pred_mean, pred_logvar, rew).float()

        return loss_p

    def compute_active_rew(self, ob):

        # Acquire MC samples first
        
        mean_infer = []
        logvar_infer = []
        with torch.no_grad():
            mean_samples, logvar_samples = self.ac.p.forward_n_samples(ob)

        # # Use epistemic + aleatoric uncertainty
        rew_intr = self.total_var(mean_samples, logvar_samples).cpu().detach().numpy()
        
        return rew_intr.squeeze()
    
    def update(self, data):
        # First run one gradient descent step for Q1 and Q2
        ave_pi_grads, ave_q_grads = [], []
        
        loss_q = self.compute_loss_q(data)
        self.q_optimizer.zero_grad()
        loss_q.backward()
        clip_grad_norm_(self.q_params, 5)
        for q in list(self.q_params):
            ave_q_grads.append(q.grad.abs().mean().item())
        self.writer.add_scalar("grad_q", np.array(ave_q_grads).mean(), self.iter_so_far)
        self.q_optimizer.step()
        self.q_scheduler.step(loss_q)

        # Freeze Q-networks so you don't waste computational effort 
        # computing gradients for them during the policy learning step.
        for q in self.q_params:
            q.requires_grad = False

        loss_entropy, loss_policy, loss_alpha = self.compute_loss_pi(data)
        loss_pi = loss_entropy + loss_policy
        self.pi_optimizer.zero_grad()
        loss_pi.backward()
        clip_grad_norm_(self.pi_params, 5)
        for p in self.pi_params:
            ave_pi_grads.append(p.grad.abs().mean().item())
        self.writer.add_scalar("grad_pi", np.array(ave_pi_grads).mean(), self.iter_so_far)
        self.pi_optimizer.step()
        self.pi_scheduler.step(loss_policy)
        
        if self.train_alpha:
            if self.alpha_end > self.t >= self.alpha_start:
                self.alpha_optimizer.zero_grad()
                loss_alpha.backward()
                self.alpha_optimizer.step()
        
        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.q_params:
            p.requires_grad = True
        
        # Record things
        if self.writer is not None:
            self.writer.add_scalar("loss_Q", loss_q.item(), self.iter_so_far)
            self.writer.add_scalar("loss_Pi", loss_pi.item(), self.iter_so_far)
            self.writer.add_scalar("loss_Policy", loss_policy.item(), self.iter_so_far)
            self.writer.add_scalar("loss_Ent", loss_entropy.item(), self.iter_so_far)
            self.writer.add_scalar("loss_alpha", loss_alpha.item(), self.iter_so_far)
            if self.adv_rew:
                self.writer.add_scalar("loss_D", loss_d.item(), self.iter_so_far)

        # Finally, update target networks by polyak averaging.

        with torch.no_grad():
            self.ac_targ.load_state_dict(self.ac.state_dict())
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

    def get_action(self, o, deterministic=False):
        return self.ac.act(o, deterministic)

    def train(self):
        # Prepare for interaction with environment
        total_steps = self.steps_per_epoch * self.epochs
        start_time = time.time()

        o, ep_ret, ep_len = self.env.reset(), 0, 0
        ep_len_batch = 0
        o_embed_list = []
        ob_list = []

        # Main loop: collect experience in env and update/log each epoch
        for t in range(total_steps):
            self.t = t
            with torch.no_grad():
                cands = self.ac.embed(self.ac.pi.cand)
                o_embeds = self.ac.embed([o])
                o_g, o_n_emb, o_g_emb = o_embeds
                if t > self.start_steps:
                    ac, (ac_prob, log_ac_prob), (ac_first, ac_second, ac_third) = \
                    self.ac.pi(o_g_emb, o_n_emb, o_g, cands)
                    print(ac, ' pi')
                else:
                    ac = self.env.sample_motif()[np.newaxis]
                    (ac_prob, log_ac_prob), (ac_first, ac_second, ac_third) = \
                    self.ac.pi.sample(ac[0], o_g_emb, o_n_emb, o_g, cands)
                    print(ac, 'sample')

            # Step the env
            o2, r, d, info = self.env.step(ac)

            if d and self.intr_rew is not None:
                o_embed_list.append(o_g_emb)
                ob_list.append(o2)


            r_d = info['stop']
            # Store experience to replay buffer
            # Problems: attachment points may not exists in o2
            # Only store Obs where attachment point exits in o2
            if any(o2['att']):
                if type(ac) == np.ndarray:
                    self.replay_buffer.store(o, ac, r, o2, r_d, 
                                            ac_prob, log_ac_prob, ac_first, ac_second, ac_third,
                                            o_embeds)
                else:    
                    self.replay_buffer.store(o, ac.detach().cpu().numpy(), r, o2, r_d, 
                                            ac_prob, log_ac_prob, ac_first, ac_second, ac_third,
                                            o_embeds)

            # Super critical, easy to overlook step: make sure to update 
            # most recent observation!
            o = o2

            # End of trajectory handling
            if get_att_points(self.env.mol) == []: # Temporally force attachment calculation
                d = True
            if not any(o2['att']):
                d = True

            if d:
                o, ep_ret, ep_len = self.env.reset(), 0, 0
                self.ep_so_far += 1

            if t>1 and t % self.docking_every == 0 and self.env.smile_list != []:
                n_smi = len(self.env.smile_list)
                print('=================== num smiles : ', n_smi)
                print('=================== t : ', t)
                if n_smi > 0:
                    ext_rew = self.env.reward_batch()
                    
                    # if self.intr_rew:

                    if self.intr_rew is not None:
                        if self.intr_rew == 'pe':
                            loss_p, intr_rew = self.compute_intr_rew(ob_list, ext_rew)
                        elif self.intr_rew == 'bu':
                            loss_p = self.compute_active_loss(ob_list, ext_rew)
                            intr_rew = self.compute_active_rew(ob_list)
                        r_batch = self.intr_rew_ratio * intr_rew + ext_rew

                        for e in self.emb_params:
                            e.requires_grad = False

                        if not torch.isnan(loss_p).any():
                            self.p_optimizer.zero_grad()
                            loss_p.backward()
                            clip_grad_norm_(self.p_params, 5)
                            self.p_optimizer.step()
                            self.p_scheduler.step(loss_p)
                            self.writer.add_scalar("loss_P", loss_p.item(), self.iter_so_far)
                        
                        for e in self.emb_params:
                            e.requires_grad = True

                        o_embed_list = []
                        ob_list = []
                        if self.writer:
                            self.writer.add_scalar("EpIntRet", sum(intr_rew)/n_smi, self.iter_so_far)
                    else:
                        r_batch = ext_rew

                    self.replay_buffer.rew_store(r_batch, self.docking_every)

                    with open(self.fname[:-4]+self.init_tm+'.csv', 'a') as f:
                        for i in range(n_smi):
                            str = f'{self.env.smile_list[i]},{ext_rew[i]},{t}'+'\n'
                            f.write(str)
                    if self.writer:
                        n_nonzero_smi = np.count_nonzero(ext_rew)
                        self.writer.add_scalar("EpRet", sum(ext_rew)/n_nonzero_smi, self.iter_so_far)
                    self.env.reset_batch()
                    ep_len_batch = 0

            # Update handling
            if t >= self.update_after and t % self.update_every == 0:
                for j in range(self.update_freq):
                    batch = self.replay_buffer.sample_batch(self.device, self.batch_size)
                    t_update = time.time()
                    self.update(data=batch)
                    dt_update = time.time()
                    print('update time : ', j, dt_update-t_update)
            
            # End of epoch handling
            if (t+1) % self.steps_per_epoch == 0:
                epoch = (t+1) // self.steps_per_epoch

            # Save model
            if (t % self.save_freq == 0):
                fname = self.save_name + f'{self.iter_so_far}'
                torch.save(self.ac.state_dict(), fname+"_rl")
                print('model saved!',fname)
            self.iter_so_far += 1


