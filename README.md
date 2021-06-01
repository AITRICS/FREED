# Fragment-based generative RL with Explorative Experience replay for Drug design (FREED)

Source codes for "Hit and Lead Discovery with Explorative RL andFragment-based Molecule Generation"

![figure_concept_1](./figures/figure_concept_1.png)

![figure_concept_2](./figures/figure_concept_2.png)

## Setup Python environment
for GPU usage,
DGL requires CUDA **10.0** or higher.

```
# Install python environment
conda env create -f environment_gpu.yml

# Activate environment
conda activate freed_pt
```

## Usage

```
# Start training with FREED - predictive error(PE), target: fa7, fragment vocab: 91 random fragments

# Check which target you want to optimize for.
vim run_rl_XXXX.sh # -> set --target='fa7'
# Currently supported Targets are 'fa7', 'parp1', '5ht1b'.

# Check which fragment vocab you are using.
vim gym_molecule/envs/env_utils_graph.py
# SFS_VOCAB = open('gym_molecule/dataset/VOCAB_TO_USE.txt','r').readlines()
# Currently supported VOCABs are 'motifs_zinc_random_92.txt'(91 random fragments), 'motif_cleaned.txt'(66 filtered fragments)

# To run FREED - predictive error(PE):
bash run_rl_intr.sh

# To run FREED - Bayesian uncertainty(BU):
bash run_rl_mc.sh

# To run PER with TD error:
bash run_rl_per.sh

# To run Curiosity driven model with predictive error:
bash run_rl_curio_intr.sh

# To run Curiosity driven model with Bayesian uncertainty:
bash run_rl_curio_mc.sh

# To run Vanilla SAC model:
bash run_rl.sh

# To run baseline PPO:
bash run_rl_ppo.sh
```

Generated molecules are stored in ./molecule_gen

## Generated Molecules
