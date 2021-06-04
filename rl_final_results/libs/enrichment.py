import sys
from rdkit import Chem
import glob
import pandas as pd
import numpy as np

if __name__ == '__main__':

    COLUMNS = ['SMILES', 'DOCKING', 'ITER'] # SAC

    d_path = sys.argv[1]
    protein = d_path.split('_')[0]

    ef = []
    tot = []
    unique = []
    hit = []
    top_k_each = []
    df_append = None

    if 'morld_' in d_path:
        COLUMNS = ['SMILES', 'DOCKING', 'SA', 'QED'] # morld
    elif 'rei_' in d_path:
        COLUMNS = ['ITER', 'SMILES', 'DOCKING', 'd', 'r'] # REINVENT
    df = pd.read_csv(d_path, names = COLUMNS)

    if ('_rei' not in d_path) and ('_morld' not in d_path):
        df = df.loc[df['ITER']>4000].loc[df['ITER']<20000] # remark for morld

    df = df.head(3000)
    n_total_smi = len(df)

    print('Total molecules : ', n_total_smi)

    df = df.drop_duplicates(subset=['SMILES'])
    df['MOL'] = df['SMILES'].apply(Chem.MolFromSmiles)
    df = df.dropna(subset=['MOL'])

    n_unique_smi = len(df)
    print('Unique molecules : ', n_unique_smi)

    if 'fa7_' in d_path:
        df_hit = df.loc[df['DOCKING']>8.5] # fa7
    elif 'parp1_' in d_path:
        df_hit = df.loc[df['DOCKING']>10.] # parp1
    elif '5ht1b_' in d_path:
        df_hit = df.loc[df['DOCKING']>8.7845] # 5ht1b

    n_hit = len(df_hit)
    print('Hit molecules : ', n_hit)

    print('Hit ratio : ', float(n_hit/n_total_smi))

    idx_tmp = int(len(df)*.05)
    if len(df)<20:
        top_5_score = df.sort_values(by='DOCKING', ascending=False).loc[:,'DOCKING'].iloc[0]
        print('Top 5% score : ', top_5_score)
    else: 
        top_5_score = df.sort_values(by='DOCKING', ascending=False).loc[:,'DOCKING'].iloc[:idx_tmp].mean()
        print('Top 5% score : ', top_5_score)
