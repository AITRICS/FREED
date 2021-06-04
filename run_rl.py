#!/usr/bin/env python3

import os
import random

import numpy as np
import dgl
import torch 
from tensorboardX import SummaryWriter

import gym

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    dgl.random.seed(seed)
    if torch.cuda.is_available():
       torch.cuda.manual_seed(seed)
       torch.cuda.manual_seed_all(seed)

def gpu_setup(use_gpu, gpu_id):
    if torch.cuda.is_available() and use_gpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  
        print('cuda available with GPU:',torch.cuda.get_device_name(0))
        device = torch.device("cuda:"+str(gpu_id))
    else:
        print('cuda not available')
        device = torch.device("cpu")

    return device

def train(args,seed,writer=None):
    
    if args.rl_model == 'sac':
        if args.active_learning == "freed_bu":
            from sac_motif_freed_bu import sac
        elif args.active_learning == "freed_pe":
            from sac_motif_freed_pe import sac
        elif args.active_learning == "per":
            from sac_motif_per import sac
        elif args.active_learning is None:
            from sac_motif import sac

    elif args.rl_model == 'td3':
        from td3_motif import td3
    elif args.rl_model == 'ddpg':
        from ddpg_motif import ddpg
    elif args.rl_model == 'ppo':
        from ppo_motif import ppo
    elif args.rl_model == 'vpg':
        from vpg_motif import vpg

    if args.rl_model in ['ppo', 'vpg']:
        from core_motif_vbased import GCNActorCritic
    else:
        from core_motif import GCNActorCritic

    workerseed = args.seed
    set_seed(workerseed)
    
    # device
    gpu_use = False
    gpu_id = None
    if args.gpu_id is not None:
        gpu_id = int(args.gpu_id)
        gpu_use = True

    device = gpu_setup(gpu_use, gpu_id)

    env = gym.make('molecule-v0')
    env.init(docking_config=args.docking_config, data_type=args.dataset, ratios = args.ratios, reward_step_total=args.reward_step_total,is_normalize=args.normalize_adj,reward_type=args.reward_type,reward_target=args.reward_target,has_feature=bool(args.has_feature),is_conditional=bool(args.is_conditional),conditional=args.conditional,max_action=args.max_action,min_action=args.min_action)  
    env.seed(workerseed)

    if args.rl_model == 'sac':
        SAC = sac(writer, args, env, actor_critic=GCNActorCritic, ac_kwargs=dict(), seed=seed, 
            steps_per_epoch=500, epochs=100, replay_size=int(1e6), gamma=0.99, 
            # polyak=0.995, lr=args.init_lr, alpha=args.init_alpha, batch_size=args.batch_size, start_steps=128,    
            polyak=0.995, lr=args.init_lr, alpha=args.init_alpha, batch_size=args.batch_size, start_steps=args.start_steps,
            update_after=args.update_after, update_every=args.update_every, update_freq=args.update_freq, 
            expert_every=5, num_test_episodes=8, max_ep_len=args.max_action, 
            save_freq=2000, train_alpha=True)
        SAC.train()
    
    elif args.rl_model == 'ppo':
        from mpi_tools import mpi_fork
        mpi_fork(args.n_cpus)
        epochs = 200
        PPO = ppo(writer, args, env, actor_critic=GCNActorCritic, ac_kwargs=dict(), seed=seed, 
            steps_per_epoch=args.steps_per_epoch, epochs=200, replay_size=int(1e6), gamma=0.99, 
            polyak=0.995, lr=args.init_lr, alpha=args.init_alpha, batch_size=args.batch_size, start_steps=args.start_steps,
            update_after=args.update_after, update_every=args.update_every, update_freq=args.update_freq, 
            expert_every=5, num_test_episodes=8, max_ep_len=args.max_action, 
            save_freq=2000, train_alpha=True)
        PPO.train()

    env.close()

def arg_parser():
    """
    Create an empty argparse.ArgumentParser.
    """
    import argparse
    return argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def molecule_arg_parser():
    parser = arg_parser()

    # Choose RL model
    parser.add_argument('--rl_model', type=str, default='sac') # sac, td3, ddpg

    parser.add_argument('--gpu_id', type=int, default=None)
    parser.add_argument('--train', type=int, default=1, help='training or inference')
    # env
    parser.add_argument('--env', type=str, help='environment name: molecule; graph', default='molecule')
    parser.add_argument('--seed', help='RNG seed', type=int, default=666)
    parser.add_argument('--num_steps', type=int, default=int(5e7))
    
    parser.add_argument('--dataset', type=str, default='zinc',help='caveman; grid; ba; zinc; gdb')
    parser.add_argument('--dataset_load', type=str, default='zinc')

    parser.add_argument('--name',type=str,default='')
    parser.add_argument('--name_full',type=str,default='')
    parser.add_argument('--name_full_load',type=str,default='')
    
    # rewards
    parser.add_argument('--reward_type', type=str, default='crystal',help='logppen;logp_target;qed;qedsa;qed_target;mw_target;gan')
    parser.add_argument('--reward_target', type=float, default=0.5,help='target reward value')
    parser.add_argument('--reward_step_total', type=float, default=0.5)
    parser.add_argument('--target', type=str, default='jak2', help='jak2, tgfr1, braf, 2oh4A')
    
    # GAN
    parser.add_argument('--gan_type', type=str, default='normal')# normal, recommend, wgan
    parser.add_argument('--gan_step_ratio', type=float, default=1)
    parser.add_argument('--gan_final_ratio', type=float, default=1)
    parser.add_argument('--has_d_step', type=int, default=1)
    parser.add_argument('--has_d_final', type=int, default=1)

    parser.add_argument('--intr_rew', type=str, default=None) # intr, mc
    parser.add_argument('--intr_rew_ratio', type=float, default=5e-1)
    
    parser.add_argument('--tau', type=float, default=1)
    
    # Expert
    parser.add_argument('--expert_start', type=int, default=0)
    parser.add_argument('--expert_end', type=int, default=int(1e6))
    parser.add_argument('--curriculum', type=int, default=0)
    parser.add_argument('--curriculum_num', type=int, default=6)
    parser.add_argument('--curriculum_step', type=int, default=200)
    parser.add_argument('--supervise_time', type=int, default=4)

    # model update
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--init_lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--update_every', type=int, default=256)
    parser.add_argument('--update_freq', type=int, default=256)
    parser.add_argument('--update_after', type=int, default=1000)
    parser.add_argument('--start_steps', type=int, default=2000)
    
    # model save and load
    parser.add_argument('--save_every', type=int, default=200)
    parser.add_argument('--load', type=int, default=0)
    parser.add_argument('--load_step', type=int, default=250)
    
    # graph embedding
    parser.add_argument('--gcn_type', type=str, default='GCN') # GCN, GINE
    parser.add_argument('--gcn_aggregate', type=str, default='sum') # sum, mean, concat, gmt
    parser.add_argument('--graph_emb', type=int, default=0)
    parser.add_argument('--emb_size', type=int, default=64) # default 64
    parser.add_argument('--has_residual', type=int, default=0)
    parser.add_argument('--has_feature', type=int, default=0)

    parser.add_argument('--normalize_adj', type=int, default=0)
    parser.add_argument('--bn', type=int, default=0)

    parser.add_argument('--layer_num_g', type=int, default=3)
        
    parser.add_argument('--stop_shift', type=int, default=-3)
    parser.add_argument('--has_concat', type=int, default=0)
        
    parser.add_argument('--gate_sum_d', type=int, default=0)
    parser.add_argument('--mask_null', type=int, default=0)

    # action
    parser.add_argument('--is_conditional', type=int, default=0) 
    parser.add_argument('--conditional', type=str, default='low')
    parser.add_argument('--max_action', type=int, default=12) 
    parser.add_argument('--min_action', type=int, default=3) 

    # SAC
    parser.add_argument('--target_entropy', type=float, default=1.)
    parser.add_argument('--init_alpha', type=float, default=1.)
    parser.add_argument('--desc', type=str, default='ecfp') # ecfp / desc

    # MC dropout
    parser.add_argument('--active_learning', type=str, default=None) # "mc", "per", None
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--n_samples', type=int, default=5)

    # On-policy
    parser.add_argument('--n_cpus', type=int, default=1)
    parser.add_argument('--steps_per_epoch', type=int, default=257)
    
    return parser

def main():
    args = molecule_arg_parser().parse_args()
    print(args)
    args.name_full = args.env + '_' + args.dataset + '_' + args.name

    docking_config = dict()
    
    assert args.target in ['2oh4A', 'tgfr1', 'jak2', 'braf', 'fa7', 'drd3', 'parp1', '5ht1b'], "Wrong target type"
    if args.target == '2oh4A':
        box_center = (37.4,36.1,12.0)
        box_size = (15.9,26.7,15.0)
        docking_config['receptor_file'] = 'ReLeaSE_Vina/docking/2oh4A_receptor.pdbqt'
    elif args.target == 'tgfr1':
        box_center = (14.421,66.480,5.632)
        box_size = (18.238,16.117,19.066)
        docking_config['receptor_file'] = 'ReLeaSE_Vina/docking/tgfr1/receptor.pdbqt'
    elif args.target == 'jak2':
        box_center = (114.758,65.496,11.345)
        box_size= (19.033,17.929,20.283)
        docking_config['receptor_file'] = 'ReLeaSE_Vina/docking/jak2/receptor.pdbqt'
    elif args.target == 'braf':
        box_center = (84.194,6.949,-7.081)
        box_size = (22.032,19.211,14.106)
        docking_config['receptor_file'] = 'ReLeaSE_Vina/docking/braf/receptor.pdbqt'
    elif args.target == 'fa7':
        box_center = (10.131, 41.879, 32.097)
        box_size = (20.673, 20.198, 21.362)
        docking_config['receptor_file'] = 'ReLeaSE_Vina/docking/fa7/receptor.pdbqt'
    elif args.target == 'drd3':
        box_center = (9.140, 21.192, 24.264)
        box_size = (16.568, 20.410, 15.768)
        docking_config['receptor_file'] = 'ReLeaSE_Vina/docking/drd3/receptor.pdbqt'
    elif args.target == 'parp1':
        box_center = (26.413, 11.282, 27.238)
        box_size = (18.521, 17.479, 19.995)
        docking_config['receptor_file'] = 'ReLeaSE_Vina/docking/parp1/receptor.pdbqt'
    elif args.target == '5ht1b':
        box_center = (-26.602, 5.277, 17.898)
        box_size = (22.5, 22.5, 22.5)
        docking_config['receptor_file'] = 'ReLeaSE_Vina/docking/5ht1b/receptor.pdbqt'
        docking_config['temp_dir'] = '5ht1b_tmp'

    box_parameter = (box_center, box_size)
    docking_config['vina_program'] = 'qvina02'
    docking_config['box_parameter'] = box_parameter
    docking_config['temp_dir'] = 'tmp'
    if args.train:
        docking_config['exhaustiveness'] = 1
    else: 
        docking_config['exhaustiveness'] = 4
    docking_config['num_sub_proc'] = 10
    docking_config['num_cpu_dock'] = 5
    docking_config['num_modes'] = 10 
    docking_config['timeout_gen3d'] = 30
    docking_config['timeout_dock'] = 100 

    ratios = dict()
    ratios['logp'] = 0
    ratios['qed'] = 0
    ratios['sa'] = 0
    ratios['mw'] = 0
    ratios['filter'] = 0
    ratios['docking'] = 1

    args.docking_config = docking_config
    args.ratios = ratios
    
    # check and clean
    if not os.path.exists('molecule_gen'):
        os.makedirs('molecule_gen')
    if not os.path.exists('ckpt'):
        os.makedirs('ckpt')

    writer = SummaryWriter(comment='_'+args.dataset+'_'+args.name)

    # device
    gpu_use = False
    gpu_id = None
    if args.gpu_id is not None:
        gpu_id = int(args.gpu_id)
        gpu_use = True
    device = gpu_setup(gpu_use, gpu_id)
    args.device = device

    if args.gpu_id is None:
        torch.set_num_threads(256)
        print(torch.get_num_threads())

    train(args,seed=args.seed,writer=writer)

if __name__ == '__main__':
    main()
