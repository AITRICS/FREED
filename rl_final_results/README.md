# Descriptions to generated molecules

Format of file names goes as follows.

## For SAC based models

[target]\_[vocab_used]\_[algorithm]\_[seed].csv

targets: {5ht1b, fa7, parp1}\
vocabs: {cleaned, rand91}\
algorithms: {freed_pe, freed_be, per, curio_pe, curio_bu, vanilla}\
e.g. 5ht1b_cleaned_freed_pe_23.csv

Each columns in csv file represents ['SMILES', 'DOCKING', 'ITER'].\
Generated molecules until ITER=4000 are generated from random policy.
## For non-SAC based models

[target]\_[algorithm]\_[seed].csv

algorithms: {rei, morld, rand91_ppo}\
where rei for REINCENT, rand91_ppo for PPO with rand91 vocab.

molecules from REINVENT and MORLD with quality scores(Glaxo, PAINS, SureChEMBL)\
are marked as [target]\_[algorithm]\_filter\_[seed].csv

For MORLD, Each column in csv file represents ['SMILES', 'DOCKING', 'SA', 'QED'].\
For REINVENT, Each column represents ['ITER', 'SMILES', 'DOCKING']

## Hit-to-lead generation

In case of hit-to-lead scenario, {scaff, scaff2} are added after target name.

scaff: start from scaffold with 

scaff2: start from scaffold where all atoms given as possible attachment points

