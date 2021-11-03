import os
from rdkit import Chem
from rdkit.Chem import Descriptors, MolToSmiles, AllChem
from gym_molecule.envs.env_utils_graph import convert_radical_electrons_to_hydrogens
from rdkit.Chem.rdmolops import FastFindRings

current_path = os.path.dirname(os.path.abspath(__file__))
descriptors_list = []

with open('gym_molecule/dataset/descriptors_list.txt', "r") as f:
    for line in f:
        descriptors_list.append(line.strip())

descriptors_dict = dict(Descriptors.descList)

def get_final_motif(mol):
    m = Chem.DeleteSubstructs(mol, Chem.MolFromSmiles("*"))
    m.UpdatePropertyCache()
    FastFindRings(m)
    return m

def featurize_molecule(mol, features):
    features_list = []
    for feature in features:
        features_list.extend(feat_dict[feature['type']](mol, feature))
    return features_list

def ecfp(molecule):
    molecule = get_final_motif(molecule)
    return [x for x in AllChem.GetMorganFingerprintAsBitVect(
        molecule, 2, 1024)]

def rdkit_headers():
    headers = [x[0] for x in Descriptors.descList]
    return headers


def fingerprint_headers(options):
    return ['{}{}_{}'.format(options['type'], options['radius'], x) for x in range(options['length'])]


def rdkit_descriptors(molecule, options=None):
    molecule = get_final_motif(molecule)
    descriptors = []
    for desc_name in descriptors_list:
        try:
            desc = descriptors_dict[desc_name]
            bin_value = desc(molecule)
        except (ValueError, TypeError, ZeroDivisionError) as exception:
            print(
                'Calculation of the Descriptor {} failed for a molecule {} due to {}'.format(
                    str(desc_name), str(MolToSmiles(molecule)), str(exception))
            )
            bin_value = 'NaN'

        descriptors.append(bin_value)

    return descriptors


feat_dict = {"ECFP": ecfp, "DESCS": rdkit_descriptors}

if __name__ == "__main__":
    
        # return convert_radical_electrons_to_hydrogens(m)

    FRAG_VOCAB = open('gym_molecule/dataset/motifs_new.txt','r').readlines() # romanoff
    FRAG_VOCAB = [s.strip('\n').split(',') for s in FRAG_VOCAB] 
    FRAG_VOCAB_MOL = [Chem.MolFromSmiles(s[0]) for s in FRAG_VOCAB]
    FRAG_VOCAB_MOL = [get_final_mol(s) for s in FRAG_VOCAB_MOL]

    for i, mol in enumerate(FRAG_VOCAB_MOL):
        desc = ecfp
        print(desc(mol))
    