export PATH="bin:$PATH" # for docking use

CUDA_LAUNCH_BLOCKING=1 python run_rl.py --name='docking_ac4_gcn_ecfp_fa7_rand92_vanilla_alr5e-4_141' \
                       --load=0 --train=1 --has_feature=1 --is_conditional=0 \
                       --name_full_load='' \
                       --min_action=1 --max_action=4 \
                       --graph_emb=1 --gcn_aggregate='sum' --gcn_type='GCN'\
                       --seed=141 --adv_rew=0 --target='fa7' \
                       --update_after=3000 --start_steps=4000 --update_every=256 --init_alpha=1.0 \
                       --target_entropy=3. \
                       --desc='ecfp' \
                       --rl_model='sac' \
                       --gpu_id=1 --emb_size=64 --tau=1e-1 --batch_size=256 
