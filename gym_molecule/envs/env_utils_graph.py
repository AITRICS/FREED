import os
import copy
import csv
import numpy as np
from rdkit import Chem

def get_att_points(mol):
    att_points = []
    for a in mol.GetAtoms(): 
        if a.GetSymbol() == '*':
            att_points.append(a.GetIdx())

    return att_points

ATOM_VOCAB = ['C', 'N', 'O', 'S', 'P', 'F', 'I', 'Cl','Br', '*']

FRAG_VOCAB = open('gym_molecule/dataset/motifs_zinc_random_92.txt','r').readlines() # n=91
# FRAG_VOCAB = open('gym_molecule/dataset/motif_cleaned.txt','r').readlines() # cleaned

FRAG_VOCAB = [s.strip('\n').split(',') for s in FRAG_VOCAB] 
FRAG_VOCAB_MOL = [Chem.MolFromSmiles(s[0]) for s in FRAG_VOCAB]
FRAG_VOCAB_ATT = [get_att_points(m) for m in FRAG_VOCAB_MOL]

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

# From GCPN
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
