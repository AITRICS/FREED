import os
import copy
import csv
import numpy as np
from rdkit import Chem

import selfies as sf

def get_att_points(mol):
    att_points = []
    for a in mol.GetAtoms(): 
        if a.GetSymbol() == '*':
            att_points.append(a.GetIdx())

    return att_points

# ATOM_VOCAB = ['C', 'N', 'O', 'S', 'P', 'F', 'I', 'Cl','Br']
ATOM_VOCAB = ['C', 'N', 'O', 'S', 'P', 'F', 'I', 'Cl','Br', '*']

# SFS_VOCAB = open('/home/crystal/rl_gcpn/gym_molecule/dataset/motifs_graph.txt','r').readlines()
# SFS_VOCAB = open('gym_molecule/dataset/motifs_graph.txt','r').readlines() # romanoff
# SFS_VOCAB = open('gym_molecule/dataset/motifs_new.txt','r').readlines() # active
# SFS_VOCAB = open('gym_molecule/dataset/motifs_new_rand.txt','r').readlines() # random
SFS_VOCAB = open('gym_molecule/dataset/motifs_zinc_random_92.txt','r').readlines() # random
# SFS_VOCAB = open('gym_molecule/dataset/motif_cleaned.txt','r').readlines() # random
# SFS_VOCAB = open('gym_molecule/dataset/motif_cleaned2.txt','r').readlines() # random
# SFS_VOCAB = open('gym_molecule/dataset/motifs_zinc_random1_cleaned.txt','r').readlines() # random clean
# SFS_VOCAB = open('gym_molecule/dataset/motifs_imitation_tgfr1_atom0.txt','r').readlines() # romanoff
SFS_VOCAB = [s.strip('\n').split(',') for s in SFS_VOCAB] 
SFS_VOCAB_MOL = [Chem.MolFromSmiles(s[0]) for s in SFS_VOCAB]
SFS_VOCAB_ATT = [get_att_points(m) for m in SFS_VOCAB_MOL]

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: float(x == s), allowable_set))

def edge_feature(bond):
    bt = bond.GetBondType()
    return np.asarray([
        bt == Chem.rdchem.BondType.SINGLE,
        bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE,
        bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()])

def atom_feature(atom, use_atom_meta):
    if use_atom_meta == False:
        return np.asarray(
            one_of_k_encoding_unk(atom.GetSymbol(), ATOM_VOCAB) 
            )
    else:
        return np.asarray(
            one_of_k_encoding_unk(atom.GetSymbol(), ATOM_VOCAB) +
            one_of_k_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) +
            one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
            one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5]) +
            [atom.GetIsAromatic()])

# TODO(Bowen): check, esp if input is not radical
def convert_radical_electrons_to_hydrogens(mol):
    """
    Converts radical electrons in a molecule into bonds to hydrogens. Only
    use this if molecule is valid. Results a new mol object
    :param mol: rdkit mol object
    :return: rdkit mol object
    """
    m = copy.deepcopy(mol)
    if Chem.Descriptors.NumRadicalElectrons(m) == 0:  # not a radical
        return m
    else:  # a radical
        for a in m.GetAtoms():
            num_radical_e = a.GetNumRadicalElectrons()
            if num_radical_e > 0:
                a.SetNumRadicalElectrons(0)
                a.SetNumExplicitHs(num_radical_e)
    return m


def load_scaffold():
    cwd = os.path.dirname(__file__)
    path = os.path.join(os.path.dirname(cwd), 'dataset',
                       'vocab.txt')  # gdb 13
    with open(path, 'r') as fp:
        reader = csv.reader(fp, delimiter=',', quotechar='"')
        data = [Chem.MolFromSmiles(row[0]) for row in reader]
        data = [mol for mol in data if mol.GetRingInfo().NumRings() == 1 and (mol.GetRingInfo().IsAtomInRingOfSize(0, 5) or mol.GetRingInfo().IsAtomInRingOfSize(0, 6))]
        for mol in data:
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        print('num of scaffolds:', len(data))
        return data


def load_conditional(type='low'):
    if type=='low':
        cwd = os.path.dirname(__file__)
        path = os.path.join(os.path.dirname(cwd), 'dataset',
                            'opt.test.logP-SA')
        import csv
        with open(path, 'r') as fp:
            reader = csv.reader(fp, delimiter=' ', quotechar='"')
            data = [row+[id] for id,row in enumerate(reader)]
    elif type=='high':
        cwd = os.path.dirname(__file__)
        path = os.path.join(os.path.dirname(cwd), 'dataset',
                            'zinc_plogp_sorted.csv')
        import csv
        with open(path, 'r') as fp:
            reader = csv.reader(fp, delimiter=',', quotechar='"')
            data = [[row[1], row[0],id] for id, row in enumerate(reader)]
            data = data[0:800]
    return data
