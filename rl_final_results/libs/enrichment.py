import sys
from rdkit import Chem
import glob
import pandas as pd
import numpy as np

if __name__ == '__main__':
    # COLUMNS_DUDE = ['MOL_ID', 'SMILES', 'Activity', 'Docking1', 'Docking', 'Docking_re', 'pic_int', 'pinfo']
    COLUMNS = ['SMILES', 'DOCKING', 'ITER'] # SAC
    
    # COLUMNS = ['ITER', 'SMILES', 'DOCKING', 'SA'] # ReLEASE
    # COLUMNS = ['ITER', 'SMILES', 'DOCKING', 'd', 'r'] # REINVENT

    # our_data_name = sys.argv[1]
    # protein = our_data_name.split('_')[0]

    # our_datas = glob.glob(f'../molecule_gen/{our_data_name}*')
    d_path = sys.argv[1]
    protein = d_path.split('_')[0]
    # print('out datas', our_datas)

    ef = []
    tot = []
    unique = []
    hit = []
    top_k_each = []
    df_append = None

    # for i, d_path in enumerate(our_datas):
    if 'morld_' in d_path:
        COLUMNS = ['SMILES', 'DOCKING', 'SA', 'QED'] # morld
    elif 'rei_' in d_path:
        COLUMNS = ['ITER', 'SMILES', 'DOCKING', 'd', 'r'] # REINVENT
    df = pd.read_csv(d_path, names = COLUMNS)
    # df['DOCKING'] = -df['DOCKING'] # ReLEASE
    # df = df.loc[df['ITER']>2000].loc[df['ITER']<30000] # remark for morld
    if ('_rei' not in d_path) and ('_morld' not in d_path):
        df = df.loc[df['ITER']>4000].loc[df['ITER']<20000] # remark for morld
    # df = df.loc[df['ITER']>4000] # remark for morld
    df = df.head(3000)
    n_total_smi = len(df)
    # tot.append(n_total_smi)
    print('Total molecules : ', n_total_smi)

    df = df.drop_duplicates(subset=['SMILES'])
    df['MOL'] = df['SMILES'].apply(Chem.MolFromSmiles)
    df = df.dropna(subset=['MOL'])

    n_unique_smi = len(df)
    print('Unique molecules : ', n_unique_smi)
    # if df_append is None:
    #     df_append = df
    # else:
    #     df_append = df_append.append(df)
    
    # unique.append(n_unique_smi)
    if 'fa7_' in d_path:
        df_hit = df.loc[df['DOCKING']>8.5] # fa7
    elif 'parp1_' in d_path:
        df_hit = df.loc[df['DOCKING']>10.] # parp1
    elif '5ht1b_' in d_path:
        df_hit = df.loc[df['DOCKING']>8.7845] # 5ht1b
    # df_hit = df.loc[df['DOCKING']>10.3] # braf
    n_hit = len(df_hit)
    print('Hit molecules : ', n_hit)
    # hit.append(n_hit)
    # ef.append(float(n_hit/n_total_smi))
    print('Hit ratio : ', float(n_hit/n_total_smi))

    idx_tmp = int(len(df)*.05)
    if len(df)<20:
        # top_k_each.append(df.sort_values(by='DOCKING', ascending=False).loc[:,'DOCKING'].iloc[0]
        top_5_score = df.sort_values(by='DOCKING', ascending=False).loc[:,'DOCKING'].iloc[0]
        print('Top 5% score : ', top_5_score)
    else: 
        # top_k_each.append(df.sort_values(by='DOCKING', ascending=False).loc[:,'DOCKING'].iloc[:idx_tmp].mean())
        top_5_score = df.sort_values(by='DOCKING', ascending=False).loc[:,'DOCKING'].iloc[:idx_tmp].mean()
        print('Top 5% score : ', top_5_score)


    # df_append = df_append.sort_values(by='DOCKING', ascending=False)
    # print('df append', len(df_append))
    # top_k = 5
    # idx = int(len(df_append)*(top_k/100))
    # top_k_score = df_append.loc[:, 'DOCKING'].iloc[:idx].mean()

    # print("tot", tot)
    # print("unique", unique)
    # print("hit", hit)
    # print("ef", ef)
    # print("top 5% each", top_k_each)
    # print(np.mean(np.array(ef)), np.std(np.array(ef)))
    # print("top "+str(top_k)+" score ",np.mean(np.array(top_k_each)), np.std(np.array(top_k_each)))
