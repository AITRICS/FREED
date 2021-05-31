import copy
import itertools

import numpy as np
import networkx as nx

from rdkit import Chem  # TODO(Bowen): remove and just use AllChem
from rdkit.Chem import AllChem
from rdkit.Chem.Descriptors import qed, MolLogP
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalogParams, FilterCatalog

from gym_molecule.envs.sascorer import calculateScore

### SUBSTRUCTURE FILTERS ###
# avoid_substructs.append(Chem.MolFromSmarts('[$([*R2;$(*=*)]([*R2])([*R]))]'))
# C=[ND2]
avoids = ['*=*=*', '*~1~*~*=*~1', '*~1=*~*~1', '*~1:*~*:*~1', '*:*:*', '*~1:*~*~1',
        '[Br;$(*#*)]', '[I;$(*#*)]', '[Cl;$(*#*)]', '[S;$(*#*)]', '[R3]', '[R4]', '[R5]']
avoid_substructs = [Chem.MolFromSmarts(sm) for sm in avoids]

def avoid_filter(mol):
    for struct in avoid_substructs:
        if mol.HasSubstructMatch(struct):
            return False
    return True

def avoid_filter_2(mol):
    num_struct = 0 
    for struct in avoid_substructs:
        num_struct += len(mol.GetSubstructMatches(struct))
    return num_struct

goods = ['[r6]1[a][a][a][a][r6]1', '[r5]1[a][a][a][r5]1']
good_substructs = [Chem.MolFromSmarts(s) for s in goods]
def good_filter(mol):
    for struct in good_substructs:
        if mol.HasSubstructMatch(struct):
            return True
    return False

def good_filter_2(mol):
    num_struct = 0 
    for struct in good_substructs:
        num_struct += len(mol.GetSubstructMatches(struct))
    return num_struct

### TARGET VALUE REWARDS ###
def reward_vina(smis, predictor, reward_vina_min=0):
    reward = - np.array(predictor.predict(smis))
    reward = np.clip(reward, reward_vina_min, None)
    return reward

def reward_target(mol, target, ratio, val_max, val_min, func):
    x = func(mol)
    reward = max(-1*np.abs((x-target)/ratio) + val_max,val_min)
    return reward

def reward_target_new(mol, func,r_max1=4,r_max2=2.25,r_mid=2,r_min=-2,x_start=500, x_mid=525):
    x = func(mol)
    return max((r_max1-r_mid)/(x_start-x_mid)*np.abs(x-x_mid)+r_max1, (r_max2-r_mid)/(x_start-x_mid)*np.abs(x-x_mid)+r_max2,r_min)

def reward_target_logp(mol, target,ratio=0.5,max=5):
    """
    Reward for a target log p
    :param mol: rdkit mol object
    :param target: float
    :return: float (-inf, max]
    """
    x = MolLogP(mol)
    reward = -1 * np.abs((x - target)/ratio) + max
    return reward

def reward_target_penalizelogp(mol, target,ratio=3,max=4):
    """
    Reward for a target log p
    :param mol: rdkit mol object
    :param target: float
    :return: float (-inf, max]
    """
    x = reward_penalized_log_p(mol)
    reward = -1 * np.abs((x - target)/ratio) + max
    return reward

def reward_target_qed(mol, target,ratio=0.1,max=4):
    """
    Reward for a target log p
    :param mol: rdkit mol object
    :param target: float
    :return: float (-inf, max]
    """
    x = qed(mol)
    reward = -1 * np.abs((x - target)/ratio) + max
    return reward

def reward_target_mw(mol, target,ratio=40,max=4):
    """
    Reward for a target molecular weight
    :param mol: rdkit mol object
    :param target: float
    :return: float (-inf, max]
    """
    x = rdMolDescriptors.CalcExactMolWt(mol)
    reward = -1 * np.abs((x - target)/ratio) + max
    return reward

# TODO(Bowen): num rings is a discrete variable, so what is the best way to
# calculate the reward?
def reward_target_num_rings(mol, target):
    """
    Reward for a target number of rings
    :param mol: rdkit mol object
    :param target: int
    :return: float (-inf, 1]
    """
    x = rdMolDescriptors.CalcNumRings(mol)
    reward = -1 * (x - target)**2 + 1
    return reward

# TODO(Bowen): more efficient if we precalculate the target fingerprint
from rdkit import DataStructs
def reward_target_molecule_similarity(mol, target, radius=2, nBits=2048,
                                      useChirality=True):
    """
    Reward for a target molecule similarity, based on tanimoto similarity
    between the ECFP fingerprints of the x molecule and target molecule
    :param mol: rdkit mol object
    :param target: rdkit mol object
    :return: float, [0.0, 1.0]
    """
    x = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=radius,
                                                        nBits=nBits,
                                                        useChirality=useChirality)
    target = rdMolDescriptors.GetMorganFingerprintAsBitVect(target,
                                                            radius=radius,
                                                        nBits=nBits,
                                                        useChirality=useChirality)
    return DataStructs.TanimotoSimilarity(x, target)

### CRYSTAL: made range filter, should check if it's okay
def reward_range_log_p(mol, v_min=-1.99, v_max=2.19):
    '''
    According to Ghose filter
    v_min = -0.4
    v_max = 5.6
    normalized_v_min = -1.99
    normalized_v_max = 2.19 
    '''
    logP_mean = 2.4570953396190123
    logP_std = 1.434324401111988
    x = MolLogP(mol)
    normalized_x = (x - logP_mean) / logP_std
    return min(normalized_x, v_max)

# incomplete... do not use
def reward_range_mw(mol, v_min, v_max):
    '''
    According to Ghose filter
    v_min = 180
    v_max = 480
    normalized_v_min = -1.99
    normalized_v_max = 2.19 
    '''
    mw_mean = 0
    mw_std = 0
    x = rdMolDescriptors.CalcExactMolWt(mol)
    normalized_x = (x - logP_mean) / logP_std
    return min(normalized_x, v_max)


### TERMINAL VALUE REWARDS ###
def reward_penalized_log_p(mol):
    """
    Reward that consists of log p penalized by SA and # long cycles,
    as described in (Kusner et al. 2017). Scores are normalized based on the
    statistics of 250k_rndm_zinc_drugs_clean.smi dataset
    :param mol: rdkit mol object
    :return: float
    """
    # normalization constants, statistics from 250k_rndm_zinc_drugs_clean.smi
    logP_mean = 2.4570953396190123
    logP_std = 1.434324401111988
    SA_mean = -3.0525811293166134
    SA_std = 0.8335207024513095
    cycle_mean = -0.0485696876403053
    cycle_std = 0.2860212110245455

    log_p = MolLogP(mol)
    SA = -calculateScore(mol)

    # cycle score
    cycle_list = nx.cycle_basis(nx.Graph(
        Chem.rdmolops.GetAdjacencyMatrix(mol)))
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([len(j) for j in cycle_list])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    cycle_score = -cycle_length

    normalized_log_p = (log_p - logP_mean) / logP_std
    normalized_SA = (SA - SA_mean) / SA_std
    normalized_cycle = (cycle_score - cycle_mean) / cycle_std

    return normalized_log_p + normalized_SA + normalized_cycle


def get_normalized_values():
    fname = '/home/bowen/pycharm_deployment_directory/rl_graph_generation/gym-molecule/gym_molecule/dataset/250k_rndm_zinc_drugs_clean.smi'
    with open(fname) as f:
        smiles = f.readlines()

    for i in range(len(smiles)):
        smiles[i] = smiles[i].strip()
    smiles_rdkit = []

    for i in range(len(smiles)):
        smiles_rdkit.append(Chem.MolToSmiles(Chem.MolFromSmiles(smiles[i])))
    print(i)

    logP_values = []
    for i in range(len(smiles)):
        logP_values.append(MolLogP(Chem.MolFromSmiles(smiles_rdkit[i])))
    print(i)

    SA_scores = []
    for i in range(len(smiles)):
        SA_scores.append(
            -calculateScore(Chem.MolFromSmiles(smiles_rdkit[i])))
    print(i)

    cycle_scores = []
    for i in range(len(smiles)):
        cycle_list = nx.cycle_basis(nx.Graph(
            Chem.rdmolops.GetAdjacencyMatrix(Chem.MolFromSmiles(smiles_rdkit[
                                                                  i]))))
        if len(cycle_list) == 0:
            cycle_length = 0
        else:
            cycle_length = max([len(j) for j in cycle_list])
        if cycle_length <= 6:
            cycle_length = 0
        else:
            cycle_length = cycle_length - 6
        cycle_scores.append(-cycle_length)
    print(i)

    SA_scores_normalized = (np.array(SA_scores) - np.mean(SA_scores)) / np.std(
        SA_scores)
    logP_values_normalized = (np.array(logP_values) - np.mean(
        logP_values)) / np.std(logP_values)
    cycle_scores_normalized = (np.array(cycle_scores) - np.mean(
        cycle_scores)) / np.std(cycle_scores)

    return np.mean(SA_scores), np.std(SA_scores), np.mean(
        logP_values), np.std(logP_values), np.mean(
        cycle_scores), np.std(cycle_scores)


### YES/NO filters ###
def zinc_molecule_filter(mol):
    """
    Flags molecules based on problematic functional groups as
    provided set of ZINC rules from
    http://blaster.docking.org/filtering/rules_default.txt.
    :param mol: rdkit mol object
    :return: Returns True if molecule is okay (ie does not match any of
    therules), False if otherwise
    """
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.ZINC)
    catalog = FilterCatalog(params)
    return not catalog.HasMatch(mol)


# TODO(Bowen): check
def steric_strain_filter(mol, cutoff=0.82,
                         max_attempts_embed=20,
                         max_num_iters=200):
    """
    Flags molecules based on a steric energy cutoff after max_num_iters
    iterations of MMFF94 forcefield minimization. Cutoff is based on average
    angle bend strain energy of molecule
    :param mol: rdkit mol object
    :param cutoff: kcal/mol per angle . If minimized energy is above this
    threshold, then molecule fails the steric strain filter
    :param max_attempts_embed: number of attempts to generate initial 3d
    coordinates
    :param max_num_iters: number of iterations of forcefield minimization
    :return: True if molecule could be successfully minimized, and resulting
    energy is below cutoff, otherwise False
    """
    # check for the trivial cases of a single atom or only 2 atoms, in which
    # case there is no angle bend strain energy (as there are no angles!)
    if mol.GetNumAtoms() <= 2:
        return True

    # make copy of input mol and add hydrogens
    m = copy.deepcopy(mol)
    m_h = Chem.AddHs(m)

    # generate an initial 3d conformer
    try:
        flag = AllChem.EmbedMolecule(m_h, maxAttempts=max_attempts_embed)
        if flag == -1:
            # print("Unable to generate 3d conformer")
            return False
    except: # to catch error caused by molecules such as C=[SH]1=C2OC21ON(N)OC(=O)NO
        # print("Unable to generate 3d conformer")
        return False

    # set up the forcefield
    AllChem.MMFFSanitizeMolecule(m_h)
    if AllChem.MMFFHasAllMoleculeParams(m_h):
        mmff_props = AllChem.MMFFGetMoleculeProperties(m_h)
        try:    # to deal with molecules such as CNN1NS23(=C4C5=C2C(=C53)N4Cl)S1
            ff = AllChem.MMFFGetMoleculeForceField(m_h, mmff_props)
        except:
            # print("Unable to get forcefield or sanitization error")
            return False
    else:
        # print("Unrecognized atom type")
        return False

    # minimize steric energy
    try:
        ff.Minimize(maxIts=max_num_iters)
    except:
        # print("Minimization error")
        return False

    # get the angle bend term contribution to the total molecule strain energy
    mmff_props.SetMMFFBondTerm(False)
    mmff_props.SetMMFFAngleTerm(True)
    mmff_props.SetMMFFStretchBendTerm(False)
    mmff_props.SetMMFFOopTerm(False)
    mmff_props.SetMMFFTorsionTerm(False)
    mmff_props.SetMMFFVdWTerm(False)
    mmff_props.SetMMFFEleTerm(False)

    ff = AllChem.MMFFGetMoleculeForceField(m_h, mmff_props)

    min_angle_e = ff.CalcEnergy()
    # print("Minimized angle bend energy: {}".format(min_angle_e))

    # find number of angles in molecule
    # TODO(Bowen): there must be a better way to get a list of all angles
    # from molecule... This is too hacky
    num_atoms = m_h.GetNumAtoms()
    atom_indices = range(num_atoms)
    angle_atom_triplets = itertools.permutations(atom_indices, 3)  # get all
    # possible 3 atom indices groups. Currently, each angle is represented by
    #  2 duplicate groups. Should remove duplicates here to be more efficient
    double_num_angles = 0
    for triplet in list(angle_atom_triplets):
        if mmff_props.GetMMFFAngleBendParams(m_h, *triplet):
            double_num_angles += 1
    num_angles = double_num_angles / 2  # account for duplicate angles
    # print("Num atoms: {}".format(num_atoms))
    # print("Number of angles: {}".format(num_angles))

    avr_angle_e = min_angle_e / num_angles

    # print("Average minimized angle bend energy: {}".format(avr_angle_e))

    if avr_angle_e < cutoff:
        return True
    else:
        return False