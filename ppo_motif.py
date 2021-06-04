import time
from copy import deepcopy
import itertools

import numpy as np
from rdkit import Chem
import pickle

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam, lr_scheduler
from torch.autograd import Variable
from tensorboardX import SummaryWriter

import gym

import core_motif_vbased as core
from gym_molecule.envs.env_utils_graph import SFS_VOCAB, ATOM_VOCAB

from mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs


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

def get_final_smi(smi):
    m = Chem.DeleteSubstructs(Chem.MolFromSmiles(smi), Chem.MolFromSmiles("*"))
    Chem.SanitizeMol(m, sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
    return Chem.MolToSmiles(m)

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = [] # o
        self.obs2_buf = [] # o2
        self.act_buf = np.zeros((size, 3), dtype=np.int32) # ac
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32) # r
        self.ret_buf = np.zeros(size, dtype=np.float32) # v
        self.val_buf = np.zeros(size, dtype=np.float32) # v
        self.logp_buf = np.zeros((size, 3), dtype=np.float32) # v
        self.done_buf = np.zeros(size, dtype=np.float32) # d
        
        self.ac_prob_buf = []
        self.log_ac_prob_buf = []
        
        self.ac_first_buf = []
        self.ac_second_buf = []
        self.ac_third_buf = []

        self.o_embeds_buf = []
        
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        self.tmp_size = self.max_size

    def store(self, obs, next_obs, act, rew, val, logp, done):
        
        assert self.ptr < self.max_size     # buffer has to have room so you can store

        self.obs_buf.append(obs)
        self.obs2_buf.append(next_obs)

        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.done_buf[self.ptr] = done

        self.ptr += 1

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

        self.size = min(self.size-len(zero_ptrs), self.max_size)
        self.ptr = (self.ptr-len(zero_ptrs)) % self.max_size
        
    def sample_batch(self, device, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        obs_batch = [self.obs_buf[idx] for idx in idxs]
        obs2_batch = [self.obs2_buf[idx] for idx in idxs]

        act_batch = torch.as_tensor(self.act_buf[idxs], dtype=torch.float32).unsqueeze(-1).to(device)
        rew_batch = torch.as_tensor(self.rew_buf[idxs], dtype=torch.float32).to(device)
        done_batch = torch.as_tensor(self.done_buf[idxs], dtype=torch.float32).to(device)

        batch = dict(obs=obs_batch,
                     obs2=obs2_batch,
                     act=act_batch,
                     rew=rew_batch,
                     done=done_batch)
                     
        return batch

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.
        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        
        data = dict(obs=self.obs_buf, 
                    act=torch.as_tensor(self.act_buf),
                    ret=torch.as_tensor(self.ret_buf, dtype=torch.float32),
                    adv=torch.as_tensor(self.adv_buf, dtype=torch.float32), 
                    logp=torch.as_tensor(self.logp_buf, dtype=torch.float32))
                    
        self.obs_buf = []
        self.obs2_buf = []
        return {k:v for k, v in data.items()}

class ppo:
    """
    
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

        self.update_after = update_after
        self.update_every = update_every
        self.update_freq = update_freq
        self.docking_every = int(update_every/2)
        self.save_freq = save_freq
        self.train_alpha = train_alpha
        
        self.pretrain_q = -1

        self.env, self.test_env = env_fn, deepcopy(env_fn)

        self.obs_dim = args.emb_size * 2
        self.act_dim = len(SFS_VOCAB)-1
        
        self.ac1_dims = 40 
        self.ac2_dims = len(SFS_VOCAB) # 76
        self.ac3_dims = 40 
        self.action_dims = [self.ac1_dims, self.ac2_dims, self.ac3_dims]

        # On-policy

        self.train_pi_iters = 10
        self.train_v_iters = 10
        self.target_kl = 0.01
        self.steps_per_epoch = steps_per_epoch
        self.local_steps_per_epoch = steps_per_epoch//num_procs()
        self.epochs = epochs
        self.clip_ratio = .2
        self.ent_coeff = .01

        self.n_cpus = args.n_cpus

        self.target_entropy = 1.0

        self.log_alpha = torch.tensor([np.log(alpha)], requires_grad=train_alpha)
        alpha = self.log_alpha.exp().item()

        # Create actor-critic module and target networks
        self.ac = actor_critic(self.env, args).to(args.device)
        self.ac_targ = deepcopy(self.ac).to(args.device).eval()

        # Sync params across processes
        sync_params(self.ac)

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


        self.replay_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=self.local_steps_per_epoch,
                                            gamma=1., lam=.95)

        # Count variables (protip: try to get a feel for how different size networks behave!)
        self.var_counts = tuple(core.count_vars(module) for module in [self.ac.pi, self.ac.v])
        self.iter_so_far = 0
        self.ep_so_far = 0

        ## OPTION1: LEARNING RATE
        pi_lr = 1e-3
        vf_lr = 1e-3
    
        ## OPTION2: OPTIMIZER SETTING        
        self.pi_params = list(self.ac.pi.parameters())
        self.vf_params = list(self.ac.v.parameters()) + list(self.ac.embed.parameters())
        self.emb_params = list(self.ac.embed.parameters())

        self.pi_optimizer = Adam(self.pi_params, lr=pi_lr, weight_decay=1e-8)
        self.vf_optimizer = Adam(self.vf_params, lr=vf_lr, weight_decay=1e-8)
        
        self.vf_scheduler = lr_scheduler.ReduceLROnPlateau(self.vf_optimizer, factor=0.1, patience=768) 
        self.pi_scheduler = lr_scheduler.ReduceLROnPlateau(self.pi_optimizer, factor=0.1, patience=768)
  
        self.L2_loss = torch.nn.MSELoss()

        torch.set_printoptions(profile="full")
        self.possible_bonds = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE]

        self.t = 0

        tm = time.localtime(time.time())
        self.init_tm = time.strftime('_%Y-%m-%d_%I:%M:%S-%p', tm)
        
    def compute_loss_v(self, data):
        cands = self.ac.embed(self.ac.pi.cand)
        obs, ret = data['obs'], data['ret'].to(self.device)
        o_g, o_n_emb, o_g_emb = self.ac.embed(obs)
        
        return ((self.ac.v(o_g_emb) - ret)**2).mean()

    def compute_loss_pi(self, data):
        obs, act, adv, logp_old = data['obs'], data['act'], \
                                data['adv'].to(self.device).unsqueeze(1), data['logp'].to(self.device)
        with torch.no_grad():
            o_embeds = self.ac.embed(data['obs'])   
            o_g, o_n_emb, o_g_emb = o_embeds
            cands = self.ac.embed(self.ac.pi.cand)

        # Policy loss
        ac, ac_prob, log_ac_prob = self.ac.pi(o_g_emb, o_n_emb, o_g, cands)
        dists = self.ac.pi._distribution(ac_prob)

        logp = self.ac.pi._log_prob_from_distribution(dists, ac).view(-1, len(self.action_dims))
        ratio = torch.exp(logp.sum(1) - logp_old.sum(1))
        
        clip_adv = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Entropy Regularization
        ac_prob_sp = torch.split(ac_prob, self.action_dims, dim=1)
        log_ac_prob_sp = torch.split(log_ac_prob, self.action_dims, dim=1)
        
        if self.n_cpus == 1:
            split_size = self.batch_size+1
        else:
            split_size = self.batch_size//self.n_cpus
        ac_prob_comb = torch.einsum('by, bz->byz', ac_prob_sp[1], ac_prob_sp[2]).reshape(split_size, -1) # (bs , 91 x 40)
        ac_prob_comb = torch.einsum('bx, bz->bxz', ac_prob_sp[0], ac_prob_comb).reshape(split_size, -1) # (bs , 40 x 91 x 40)
        # order by (a1, b1, c1) (a1, b1, c2)! Be advised!

        log_ac_prob_comb = log_ac_prob_sp[0].reshape(split_size, self.action_dims[0], 1, 1).repeat(
                                    1, 1, self.action_dims[1], self.action_dims[2]).reshape(split_size, -1)\
                            + log_ac_prob_sp[1].reshape(split_size, 1, self.action_dims[1], 1).repeat(
                                    1, self.action_dims[0], 1, self.action_dims[2]).reshape(split_size, -1)\
                            + log_ac_prob_sp[2].reshape(split_size, 1, 1, self.action_dims[2]).repeat(
                                    1, self.action_dims[0], self.action_dims[1], 1).reshape(split_size, -1)
        loss_entropy = -self.ent_coeff*(ac_prob_comb * log_ac_prob_comb).sum(dim=1).mean()
        loss_pi += loss_entropy

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()

        clipped = ratio.gt(1+self.clip_ratio) | ratio.lt(1-self.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, cf=clipfrac)
        
        return loss_pi, pi_info
    
    def update(self):
        # First run one gradient descent step for Q1 and Q2
        ave_pi_grads, ave_q_grads = [], []
        data = self.replay_buffer.get()

        pi_l_old, pi_info_old = self.compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = self.compute_loss_v(data).item()

        # Train policy with multiple steps of gradient descent
        for i in range(self.train_pi_iters):
            self.pi_optimizer.zero_grad()
            loss_pi, pi_info = self.compute_loss_pi(data)
            kl = mpi_avg(pi_info['kl'])
            
            loss_pi.backward()
            mpi_avg_grads(self.ac.pi)    # average grads across MPI processes
            self.pi_optimizer.step()
            self.pi_scheduler.step(loss_pi)

        for i in range(self.train_v_iters):
            self.vf_optimizer.zero_grad()
            loss_v = self.compute_loss_v(data)
            loss_v.backward()
            mpi_avg_grads(self.ac.v)    # average grads across MPI processes
            self.vf_optimizer.step()
            self.vf_scheduler.step(loss_v)

        # Log changes from update
        kl, cf = pi_info['kl'], pi_info['cf']

        # Record things
        if proc_id() == 0:
            if self.writer is not None:
                iter_so_far_mpi = self.iter_so_far*self.n_cpus
                self.writer.add_scalar("loss_V", loss_v.item(), iter_so_far_mpi)
                self.writer.add_scalar("loss_Pi", loss_pi.item(), iter_so_far_mpi)

    def get_action(self, o, deterministic=False):
        return self.ac.act(o, deterministic)

    def train(self):
        total_steps = self.steps_per_epoch * self.epochs
        start_time = time.time()
        self.iter_so_far = 0

        o, ep_ret, ep_len = self.env.reset(), 0, 0
        ep_len_batch = 0
        o_embed_list = []

        for epoch in range(self.epochs):
            for t in range(self.local_steps_per_epoch):
                self.t = t
                with torch.no_grad():
                    cands = self.ac.embed(self.ac.pi.cand)
                    o_embeds = self.ac.embed([o])
                    o_g, o_n_emb, o_g_emb = o_embeds
                    ac, v, logp = self.ac.step(o_g_emb, o_n_emb, o_g, cands)
                
                o2, r, d, info = self.env.step(ac)
                r_d = info['stop']

                # Store experience to replay buffer

                if type(ac) == np.ndarray:
                    self.replay_buffer.store(o, o2, ac, r, v, logp, r_d)
                else:    
                    self.replay_buffer.store(o, o2, ac.detach().cpu().numpy(), r, v, logp, r_d)
                        
                # Super critical, easy to overlook step: make sure to update 
                # most recent observation!
                o = o2

                # End of trajectory handling
                if get_att_points(self.env.mol) == []: # Temporally force attachment calculation
                    d = True
                if not any(o2['att']):
                    d = True
                if d:
                    final_smi = get_final_smi(o2['smi'])
                    ext_rew = self.env.reward_single(
                                    [final_smi])

                    if ext_rew[0] > 0:
                        iter_so_far_mpi = self.iter_so_far*self.n_cpus
                        if proc_id() == 0:
                            self.writer.add_scalar("EpRet", ext_rew[0], iter_so_far_mpi)
                        
                        with open(self.fname[:-3]+self.init_tm+'.csv', 'a') as f:
                            strng = f'{final_smi},{ext_rew[0]},{iter_so_far_mpi}'+'\n'
                            f.write(strng)
                        
                    self.replay_buffer.finish_path(ext_rew)
                    o, ep_ret, ep_len = self.env.reset(), 0, 0
                    self.env.smile_list = []
                    self.ep_so_far += 1
                self.iter_so_far += 1
                    
            t_update = time.time()
            self.update()
            dt_update = time.time()
            print('update time : ', t, dt_update-t_update)
            
