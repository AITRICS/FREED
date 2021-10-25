export PATH="bin:$PATH" # for docking use

CUDA_LAUNCH_BLOCKING=1 python run_rl.py --name='motifs_1k_fa7_5krandom_141_ent0.05' \
                       --load=0 --train=1 --has_feature=1 \
                       --name_full_load='' \
                       --min_action=1 --max_action=4 \
                       --graph_emb=1 --gcn_aggregate='sum' --gcn_type='GCN'\
                       --seed=141 --intr_rew=0 --intr_rew_ratio=5e-1 --target='fa7' \
                       --update_after=2000 --start_steps=3000 --update_every=256 --init_alpha=1.0 \
                       --desc='ecfp' \
                       --rl_model='sac' \
                       --active_learning='freed_pe' \
                       --gpu_id=4 --emb_size=64 --tau=1e-1 --batch_size=128 --target_entropy=0.05