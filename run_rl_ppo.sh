export PATH="bin:$PATH" # for docking use

CUDA_LAUNCH_BLOCKING=1 python run_rl.py --name='docking_ac4_gcn_ecfp_5ht1b_rand92_ppo_ent.01_141' \
                       --load=0 --train=1 --has_feature=1 --is_conditional=0 \
                       --name_full_load='' \
                       --min_action=1 --max_action=4 \
                       --graph_emb=1 --gcn_aggregate='sum' --gcn_type='GCN'\
                       --seed=141 --intr_rew=0 --intr_rew_ratio=5e-1 --target='5ht1b' \
                       --update_after=1 --update_every=256 --steps_per_epoch=257 \
                       --start_steps=128 --init_alpha=1.0 \
                       --desc='ecfp' \
                       --rl_model='ppo' \
                       --n_cpus=1 \
                       --gpu_id=4 \
                       --emb_size=64 --tau=1e-1 --batch_size=256
