import sys
import functools
import numpy as np
from rdkit import Chem
from rdkit import DataStructs

from rdkit.Chem import AllChem

from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.rdMolDescriptors import CalcTPSA
from rdkit.Chem.rdMolDescriptors import CalcExactMolWt
from rdkit.Chem.rdMolDescriptors import CalcNumHBD
from rdkit.Chem.rdMolDescriptors import CalcNumHBA
from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds
from rdkit.Chem.Lipinski import NumAromaticRings
from rdkit.Chem.Fragments import fr_nitro

from rdkit.ML.Cluster import Butina
from openbabel import pybel

import pandas as pd

from rdkit.Chem.SaltRemover import SaltRemover

def read_df(path):
    format_ = path.split('.')[-1].strip()
    if format_ == 'csv':
        return pd.read_csv(path)
    if format_ == 'tsv':
        return pd.read_csv(path, sep='\t')
    elif format_ == 'pkl':
        return pd.read_pickle(path)

def smi_to_fingerprint(smi, fp_type='FP2'):
    remover = SaltRemover(
        defnData='[Cl,Br,I,Na,K]'
    )
    mol = Chem.MolFromSmiles(smi)
    if '.' in smi:
        res = remover.StripMol(mol, dontRemoveEverything=True)
        smi = Chem.MolToSmiles(res, canonical=True)
    else:
        smi = Chem.MolToSmiles(mol, canonical=True)
    m = pybel.readstring("smi", smi)
    return m.calcfp(fp_type).bits

def calc_tanimoto(fp1, fp2):
    intersection = set(fp1).intersection(set(fp2))
    union = set(fp1).union(set(fp2))
    return len(intersection)/len(union)

def cluster_filtering(
        mol_list, 
        fp_list,
        num_mols_per_cluster=10, 
        dist_cutoff=0.4, 
        idx=0, 
        num_unit=1,
    ):
    num_fp = len(fp_list)

    dists = []
    for i in range(1, num_fp):
        sims = DataStructs.BulkTanimotoSimilarity(fp_list[i], fp_list[:i])
        dists.extend([1-x for x in sims])
    
    clusters = Butina.ClusterData(dists, num_fp, dist_cutoff, isDistData=True)
    print ("Number of clusters =", len(clusters))

    accepted_list = []
    for cluster in clusters:
        tmp_list = []
        cluster = cluster[:num_mols_per_cluster]
        for i in cluster:
            accepted_list.append(i+idx*num_unit)
    
    return accepted_list


def similarity_filtering(
        fp, 
        ref_fp_list, 
        threshold
    ):
    sim_list = [calc_tanimoto(fp, ref_fp) for ref_fp in ref_fp_list]
    if max(sim_list) < threshold:
        return True
    else:
        return False


def lipinski_rule(
        mol,
        logp_min,
        logp_max,
        tpsa_max,
        mw_min,
        mw_max,
        hbd_max,
        hba_max,
        nrb_max,
    ):
    logp = MolLogP(mol)
    logp_criteria = (logp >= logp_min and logp <= logp_max)

    tpsa = CalcTPSA(mol)
    tpsa_criteria = (tpsa <= tpsa_max)

    molwt = CalcExactMolWt(mol)
    molwt_criteria = (molwt >= mw_min and molwt <= mw_max)

    hbd_criteria = (CalcNumHBD(mol) <= hbd_max)
    hba_criteria = (CalcNumHBA(mol) <= hba_max)

    num_rot_bonds = CalcNumRotatableBonds(mol)
    nrb_criteria = (num_rot_bonds <= nrb_max)

    if logp_criteria and \
       tpsa_criteria and \
       molwt_criteria and \
       hbd_criteria and \
       hba_criteria and \
       nrb_criteria:
           return True
    else:
           return False


def trivial_filtering(
        mol, 
        num_aromatic_rings_max,
        num_fluorines_max,
        num_chlorines_max,
        num_bromines_max,
        num_nitro_max,
    ):

    undesired_atoms = [
        'Si', 'Co', 'P', 'As', 
    ]
    atom_list = [atom.GetSymbol() for atom in mol.GetAtoms()]
    
    num_fluorines = atom_list.count('F')
    num_chlorines = atom_list.count('Cl')
    num_bromines = atom_list.count('Br')

    exist_undesirable = False
    for atom in atom_list:
        if atom in undesired_atoms:
            exist_undesriable = True

    num_aromatic_rings = NumAromaticRings(mol)
    num_nitro = fr_nitro(mol)
    if (num_aromatic_rings <= num_aromatic_rings_max) and \
       (num_fluorines <= num_fluorines_max) and \
       (num_chlorines <= num_chlorines_max) and \
       (num_bromines <= num_bromines_max) and \
       (exist_undesirable == False) and \
       (num_nitro <= num_nitro_max):
        return True
    else:
        return False


def alert_filtering(
        mol, 
        smarts,
    ):
    if mol == None:
        return False
    else:
        accepted = True
        for k in smarts:
            subs = Chem.MolFromSmarts(k)
            if subs != None:
                if mol.HasSubstructMatch(subs):
                    accepted = False
                    pass

        return accepted


def get_filter_fn_list(
        prioritization_filters, 
        args
    ):
    table_path = 'alert_collection.csv'
    alert_table = read_df(table_path)
    filter_fn_list = []

    for filter_name in prioritization_filters:
        if filter_name == 'Lipinski':
            filter_fn_list.append(
                functools.partial(
                    lipinski_rule,
                    logp_min=args.logp_min,
                    logp_max=args.logp_max,
                    tpsa_max=args.tpsa_max,
                    mw_min=args.mw_min,
                    mw_max=args.mw_max,
                    hbd_max=args.hbd_max,
                    hba_max=args.hba_max,
                    nrb_max=args.nrb_max,
                )
            )

        elif filter_name == 'Trivial':
            filter_fn_list.append(
                functools.partial(
                    trivial_filtering,
                    num_aromatic_rings_max=args.num_aromatic_rings_max,
                    num_fluorines_max=args.num_fluorines_max,
                    num_chlorines_max=args.num_chlorines_max,
                    num_bromines_max=args.num_bromines_max,
                    num_nitro_max=args.num_nitro_max,
                )
            )

        elif filter_name == 'Glaxo':
            smarts = list(alert_table[alert_table['rule_set_name']=='Glaxo']['smarts'])
            filter_fn_list.append(
                functools.partial(
                    alert_filtering,
                    smarts=smarts
                )
            )

        elif filter_name == 'BMS':
            smarts = list(alert_table[alert_table['rule_set_name']=='BMS']['smarts'])
            filter_fn_list.append(
                functools.partial(
                    alert_filtering,
                    smarts=smarts
                )
            )


        elif filter_name == 'PAINS':
            smarts = list(alert_table[alert_table['rule_set_name']=='PAINS']['smarts'])
            filter_fn_list.append(
                functools.partial(
                    alert_filtering,
                    smarts=smarts
                )
            )

        elif filter_name == 'SureChEMBL':
            smarts = list(alert_table[alert_table['rule_set_name']=='SureChEMBL']['smarts'])
            filter_fn_list.append(
                functools.partial(
                    alert_filtering,
                    smarts=smarts
                )
            )

        elif filter_name == 'Similarity':
            if args.ref_df_path is not None:
                df_ref = read_df(args.ref_df_path)
                smi_list = list(df_ref['smiles'])
                ref_fp_list = [smi_to_fingerprint(smi) for smi in smi_list]
                filter_fn_list.append(
                    functools.partial(
                        similarity_filtering,
                        ref_fp_list=ref_fp_list,
                        threshold=args.sim_threshold
                    )
                )

        else:
            raise ValueError("This filering necessitates an absolute path of the file of SMILES will be used for similarity comparison")

    return filter_fn_list


if __name__ == '__main__': 
    import argparse
    import glob
    import pandas as pd

    parser = argparse.ArgumentParser()

    # For Lipinski's Rulee of Five
    parser.add_argument('--logp_min', type=float, default=-5.0,
                        help='')
    parser.add_argument('--logp_max', type=float, default=5.0,
                        help='')
    parser.add_argument('--tpsa_max', type=float, default=140.0,
                        help='')
    parser.add_argument('--mw_min', type=float, default=200.0,
                        help='')
    parser.add_argument('--mw_max', type=float, default=750.0,
                        help='')
    parser.add_argument('--hbd_max', type=int, default=5,
                        help='')
    parser.add_argument('--hba_max', type=int, default=10,
                        help='')
    parser.add_argument('--nrb_max', type=int, default=10,
                        help='')

    # For trivial filtering
    parser.add_argument('--num_aromatic_rings_max', type=float, default=5,
                        help='')
    parser.add_argument('--num_fluorines_max', type=float, default=6,
                        help='')
    parser.add_argument('--num_chlorines_max', type=float, default=3,
                        help='')
    parser.add_argument('--num_bromines_max', type=float, default=2,
                        help='')
    parser.add_argument('--num_nitro_max', type=float, default=1,
                        help='')

    # For similarity filtering
    parser.add_argument('--ref_df_path', type=str, required=False,
                        help='Absolute path where the file of smiles \
                              will be used for similarity comparison')
    parser.add_argument('--sim_threshold', type=float, default=0.6,
                        help='')

    parser.add_argument('--data', type=str, required=True,
                        help='')

    args = parser.parse_args()

    COLUMNS = ['SMILES', 'DOCKING', 'ITER']
    # COLUMNS = ['ITER', 'SMILES', 'DOCKING']
    # COLUMNS = ['SMILES', 'a1', 'a2', 'a3', 'Docking', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10', 'a11', 'a12']
    # COLUMNS = ['ITER', 'SMILES', 'DOCKING', 'd', 'r']
    # COLUMNS = ['ITER', 'SMILES', 'DOCKING', 'QED']
    # COLUMNS = ['SMILES', 'DOCKING', 'r']

    # our_data_name = args.data
    # our_datas = glob.glob(f'/home/crystal/rl_final_results/molecule_gen/{our_data_name}*')
    d_path = args.data

    # filter_fn_list = get_filter_fn_list(['PAINS'], args)
    
    # table_path = '/nfs/romanoff/ext01/shared/DD_data/csv_files/etc/alert_collection.csv'
    table_path = 'alert_collection.csv'
    alert_table = read_df(table_path)
    PAINS_smarts = list(alert_table[alert_table['rule_set_name']=='PAINS']['smarts'])
    Glaxo_smarts = list(alert_table[alert_table['rule_set_name']=='Glaxo']['smarts'])
    SureChEMBL_smarts = list(alert_table[alert_table['rule_set_name']=='SureChEMBL']['smarts'])
    print(len(PAINS_smarts))
    print(len(Glaxo_smarts))
    print(len(SureChEMBL_smarts))
    # exit(-1)
    print("loaded alert table")
    glaxo, sure, pains = [], [], []
    # for i, d_path in enumerate(our_datas):
    #     if 'morld_' in d_path:
    #         COLUMNS = ['SMILES', 'DOCKING', 'SA', 'QED'] # morld
    #     elif 'rei_' in d_path:
    #         COLUMNS = ['ITER', 'SMILES', 'DOCKING', 'd', 'r'] # REINVENT
    #     else:
    #         COLUMNS = ['SMILES', 'DOCKING', 'ITER'] # SAC
    #     df = pd.read_csv(d_path, names = COLUMNS)[['ITER', 'SMILES', 'DOCKING']]
    #     df = df.loc[df['ITER']>4000]
    #     df = df.head(3000)
    #     print("total molecules", len(df))
    #     tot = len(df)
    #     # df = df.iloc[1000:]
    #     df['MOL'] = df['SMILES'].apply(Chem.MolFromSmiles)
    #     # print(df['MOL'].head(100))
    #     # df = df.loc[df['MOL']!=None]
    #     df = df.dropna(subset=['MOL'])   
    #     print(len(df))
    #     # print(len(df_))
    #     # print(list(df['SMILES'].head(50)))
    #     # exit(-1)
    #     print("finished converting smiles to mol")

    #     df['PAINS'] = df['MOL'].apply(lambda m: alert_filtering(m, PAINS_smarts))
    #     df['Glaxo'] = df['MOL'].apply(lambda m: alert_filtering(m, Glaxo_smarts))
    #     df['SureChEMBL'] = df['MOL'].apply(lambda m: alert_filtering(m, SureChEMBL_smarts))

    #     # print("PAINS", float(len(df.loc[df['PAINS']==True])/len(df)))
    #     pains.append(float(len(df.loc[df['PAINS']==True])/tot))
    #     glaxo.append(float(len(df.loc[df['Glaxo']==True])/tot))
    #     sure.append(float(len(df.loc[df['SureChEMBL']==True])/tot))

    if 'morld_' in d_path:
        COLUMNS = ['SMILES', 'DOCKING', 'SA', 'QED'] # morld
    elif 'rei_' in d_path:
        COLUMNS = ['ITER', 'SMILES', 'DOCKING', 'd', 'r'] # REINVENT
    else:
        COLUMNS = ['SMILES', 'DOCKING', 'ITER'] # SAC
    df = pd.read_csv(d_path, names = COLUMNS)[['ITER', 'SMILES', 'DOCKING']]
    df = df.loc[df['ITER']>4000]
    df = df.head(3000)
    print("total molecules", len(df))
    tot = len(df)
    # df = df.iloc[1000:]
    df['MOL'] = df['SMILES'].apply(Chem.MolFromSmiles)
    # print(df['MOL'].head(100))
    # df = df.loc[df['MOL']!=None]
    df = df.dropna(subset=['MOL'])   
    print(len(df))
    # print(len(df_))
    # print(list(df['SMILES'].head(50)))
    # exit(-1)
    print("finished converting smiles to mol")

    df['PAINS'] = df['MOL'].apply(lambda m: alert_filtering(m, PAINS_smarts))
    df['Glaxo'] = df['MOL'].apply(lambda m: alert_filtering(m, Glaxo_smarts))
    df['SureChEMBL'] = df['MOL'].apply(lambda m: alert_filtering(m, SureChEMBL_smarts))
    print("PAINS : ", float(len(df.loc[df['PAINS']==True])/tot))
    print("Glaxo : ", float(len(df.loc[df['Glaxo']==True])/tot))
    print("SureChEMBL : ", float(len(df.loc[df['SureChEMBL']==True])/tot))

    # # print("PAINS", float(len(df.loc[df['PAINS']==True])/len(df)))
    # pains.append(float(len(df.loc[df['PAINS']==True])/tot))
    # glaxo.append(float(len(df.loc[df['Glaxo']==True])/tot))
    # sure.append(float(len(df.loc[df['SureChEMBL']==True])/tot))
    # print(glaxo)
    # print(sure)
    # print(pains)
    # print(np.mean(np.array(glaxo)))
    # print(np.std(np.array(glaxo)))
    # print(np.mean(np.array(sure)))
    # print(np.std(np.array(sure)))
    # print(np.mean(np.array(pains)))
    # print(np.std(np.array(pains)))