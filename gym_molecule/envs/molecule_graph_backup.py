import sys, os
import random
import time
import csv
import itertools
from contextlib import contextmanager
import copy

import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import dgl
import dgl.function as fn
import networkx as nx
import pickle
import pandas as pd 

from rdkit import Chem  
from rdkit.Chem import AllChem
from rdkit.Chem.Descriptors import qed, MolLogP
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalogParams, FilterCatalog

import gym
from gym_molecule.envs.sascorer import calculateScore
from gym_molecule.dataset.dataset_utils import gdb_dataset,mol_to_nx 
from gym_molecule.envs.rewards import *
from gym_molecule.envs.docking_simple import DockingVina
from gym_molecule.envs.env_utils_graph import *


import torch

# block std out
@contextmanager
def nostdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def adj2sparse(adj):
    """
        adj: [3, 47, 47] float numpy array
        return: a tuple of 2 lists
    """
    adj = [x*(i+1) for i, x in enumerate(adj)]
    adj = [sparse.dok_matrix(x) for x in adj]
    
    if not all([adj_i is None for adj_i in adj]):
        adj = sparse.dok_matrix(np.sum(adj))
        adj.setdiag(0)   

        all_edges = list(adj.items())
        e_types = np.array([edge[1]-1 for edge in all_edges], dtype=int)
        e = np.transpose(np.array([list(edge[0]) for edge in all_edges]))

        n_edges = len(all_edges)

        e_x = np.zeros((n_edges, 4))
        e_x[np.arange(n_edges),e_types] = 1
        e_x = torch.Tensor(e_x)
        return e, e_x
    else:
        return None

def get_att_points(mol):
    att_points = []
    for a in mol.GetAtoms(): 
        if a.GetSymbol() == '*':
            att_points.append(a.GetIdx())

    return att_points

def map_idx(idx, idx_list, mol):
    abs_id = idx_list[idx]
    neigh_idx = mol.GetAtomWithIdx(abs_id).GetNeighbors()[0].GetIdx()
    return neigh_idx 
  
class MoleculeEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def init(self, docking_config=dict(), data_type='zinc',ratios=dict(),reward_step_total=1,is_normalize=0,reward_type='gan',reward_target=0.5,has_scaffold=False, has_feature=False,is_conditional=False,conditional='low',max_action=128,min_action=20,force_final=False):
        '''
        own init function, since gym does not support passing argument
        '''
        self.is_normalize = bool(is_normalize)
        self.has_feature = has_feature

        # init smi
        # self.smi = 'C([*:1])([*:2])[*:3]'
        self.smi = 'c1([*:1])c([*:2])ccc([*:3])c1'
        # self.smi = 'C([*:1])c1cccc(-c2[nH]c([*:2])nc2-c2cc([*:3])c3ncc([*:4])cc3c2)n1'
        # self.smi = '[*:1]-C1\CCc2cc(-c3nc([*:2])[nH]c3-c3ccncc3)ccc21'
        # self.smi = '[*:1]c1ccc(-c2ccnc(Nc3cccc([*:2])c3)n2)cc1'
        # self.smi = 'Cc1cccc(-c2[nH]c([*:1])nc2-c2ccc3ncccc3c2)n1' 

        self.mol = Chem.MolFromSmiles(self.smi)
        self.smile_list = []

        possible_atoms = ATOM_VOCAB
        possible_motifs = SFS_VOCAB
        possible_bonds = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                          Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
        self.atom_type_num = len(possible_atoms)
        self.motif_type_num = len(possible_motifs)
        self.possible_atom_types = np.array(possible_atoms)
        self.possible_motif_types = np.array(possible_motifs)

        self.possible_bond_types = np.array(possible_bonds, dtype=object)

        self.d_n = len(self.possible_atom_types)+18 

        self.max_action = max_action
        self.min_action = min_action

        self.max_atom = 150

        self.reward_type = reward_type
        self.reward_target = reward_target
        self.logp_ratio = ratios['logp']
        self.qed_ratio = ratios['qed']
        self.sa_ratio = ratios['sa']
        self.mw_ratio = ratios['mw']
        self.filter_ratio = ratios['filter']
        self.docking_ratio = ratios['docking']

        self.action_space = gym.spaces.MultiDiscrete([2, 20, len(SFS_VOCAB), 20])
        # stop, where to atom1, which atom2, where to atom 
        ## load expert data
        cwd = os.path.dirname(__file__)
        if data_type=='gdb':
            path = os.path.join(os.path.dirname(cwd), 'dataset',
                                'gdb13.rand1M.smi.gz')  # gdb 13
        elif data_type=='zinc':
            path = os.path.join(os.path.dirname(cwd), 'dataset',
                                '250k_rndm_zinc_drugs_clean_sorted.smi')  # ZINC
        elif data_type=='docking':
            path = os.path.join(os.path.dirname(cwd), 'dataset', 
                                'docking_selected.csv')
        self.dataset = gdb_dataset(path)

        self.counter = 0
        self.level = 0 # for curriculum learning, level starts with 0, and increase afterwards

        self.predictor = DockingVina(docking_config)

        self.attach_point = Chem.MolFromSmiles('*')
        self.Na = Chem.MolFromSmiles('[Na+]')
        self.K = Chem.MolFromSmiles('[K+]')
        self.H = Chem.MolFromSmiles('[H]')

        # ## load active data for discriminator
        # self.active_smi_path = os.path.join(os.path.dirname(os.path.dirname(cwd)), 'actives.smi')     # smiles
        # self.active_pkl_path = os.path.join(os.path.dirname(os.path.dirname(cwd)), 'actives.pickle')  # observations

        # if not os.path.exists(self.active_pkl_path):
        #     self.make_active_obs()

        # with open(self.active_pkl_path, 'rb') as f:
        #     self.active_dataset = np.array(pickle.load(f))
        #     print('number of active molecules:', self.active_dataset.shape[0])

    def level_up(self):
        self.level += 1

    def seed(self,seed):
        np.random.seed(seed=seed)
        random.seed(seed)

    def normalize_adj(self,adj):
        degrees = np.sum(adj,axis=2)
        # print('degrees',degrees)
        D = np.zeros((adj.shape[0],adj.shape[1],adj.shape[2]))
        for i in range(D.shape[0]):
            D[i,:,:] = np.diag(np.power(degrees[i,:],-0.5))
        adj_normal = D@adj@D
        adj_normal[np.isnan(adj_normal)]=0
        return adj_normal
    
    def reset_batch(self):
        self.smile_list = []
    
    def reward_batch(self):
        # reward = []
        # print('smiles list', self.smile_list)
        # for s in self.smile_list:
        #     # print('smiles', Chem.MolFromSmiles(s))
        #     q = qed(Chem.MolFromSmiles(s))
        #     # print(s)
        #     # q = 1
        #     reward.append(q)
        # return np.array(reward)
        return reward_vina(self.smile_list, self.predictor)

    def step(self, ac):
        """
        Perform a given action
        :param action:
        :param action_type:
        :return: reward of 1 if resulting molecule graph does not exceed valency,
        -1 if otherwise
        """
        # print(ac)
        ac = ac[0]
         
        ### init
        info = {}  # info we care about
        self.mol_old = copy.deepcopy(self.mol) # keep old mol
        
        stop = False    
        new = False
        
        # if (self.counter >= self.max_action) or ac[0]==1 or get_att_points(self.mol) == []:
        if (self.counter >= self.max_action) or get_att_points(self.mol) == []:
            # stop = True
            new = True
        else:
            self._add_motif(ac) # problems here

        reward_step = 0.05
        if self.mol.GetNumAtoms() > self.mol_old.GetNumAtoms():
            reward_step += 0.005
        self.counter += 1

        if new:            
            reward = 0
            # Only store for obs if attachment point exists in o2
            if get_att_points(self.mol) != []:
                mol_no_att = self.get_final_mol() 
                Chem.SanitizeMol(mol_no_att, sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                smi_no_att = Chem.MolToSmiles(mol_no_att)
                info['smile'] = smi_no_att
                print("smi:", smi_no_att)
                self.smile_list.append(smi_no_att)
                stop = True
            else:
                stop = False
            self.counter = 0      

        ### use stepwise reward
        else:
            reward = reward_step

        info['stop'] = stop

        # get observation
        ob = self.get_observation()
        return ob,reward,new,info

    def reset(self,smile=None):
        '''
        to avoid error, assume an atom already exists
        :return: ob
        '''
        if smile is not None:
            self.mol = Chem.RWMol(Chem.MolFromSmiles(smile))
            Chem.SanitizeMol(self.mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        else:
            # init smi
            # smi = 'C([*:1])([*:2])[*:3]'
            self.smi = 'c1([*:1])c([*:2])ccc([*:3])c1'
            # smi = 'C([*:1])c1cccc(-c2[nH]c([*:2])nc2-c2cc([*:3])c3ncc([*:4])cc3c2)n1' 
            # smi = 'Cc1cccc(-c2[nH]c([*:1])nc2-c2ccc3ncccc3c2)n1' 
            # smi = '[*:1]-C1\CCc2cc(-c3nc([*:2])[nH]c3-c3ccncc3)ccc21'
            # smi = '[*:1]c1ccc(-c2ccnc(Nc3cccc([*:2])c3)n2)cc1'
            self.mol = Chem.MolFromSmiles(self.smi) 
        self.counter = 0
        ob = self.get_observation()
        return ob

    def render(self, mode='human', close=False):
        return

    def sample_motif(self):
        go_on = True
        while go_on:
            cur_mol_atts = get_att_points(self.mol)
            ac1 = np.random.randint(len(cur_mol_atts))
            ac2 = np.random.randint(self.motif_type_num)
            motif = SFS_VOCAB_MOL[ac2]
            ac3 = np.random.randint(len(SFS_VOCAB_ATT[ac2]))
            a = self.action_space.sample()
            
            a[1] = ac1
            a[2] = ac2
            a[3] = ac3

            go_on = False

        return a

    def _add_motif(self, ac):
        # ac[0]: stop, ac[1]: where on cur_mol to attach, ac[2]: what to attach, ac[3]: where on motif to attach  
        cur_mol = Chem.ReplaceSubstructs(self.mol, self.attach_point, self.Na)[ac[1]]
        motif = SFS_VOCAB_MOL[ac[2]]
        att_point = SFS_VOCAB_ATT[ac[2]]
        motif_atom = map_idx(ac[3], att_point, motif) 
        motif = Chem.ReplaceSubstructs(motif, self.attach_point, self.K)[ac[3]]
        motif = Chem.DeleteSubstructs(motif, self.K)
        next_mol = Chem.ReplaceSubstructs(cur_mol, self.Na, motif, replacementConnectionPoint=motif_atom)[0]
        self.mol = next_mol

    def get_final_smiles_mol(self):
        """
        Returns a SMILES of the final molecule. Converts any radical
        electrons into hydrogens. Works only if molecule is valid
        :return: SMILES
        """
        m = Chem.DeleteSubstructs(self.mol, Chem.MolFromSmiles("*"))
        m = convert_radical_electrons_to_hydrogens(m)
        return m, Chem.MolToSmiles(m, isomericSmiles=True)

    def get_final_mol(self):
        """
        Returns a rdkit mol object of the final molecule. Converts any radical
        electrons into hydrogens. Works only if molecule is valid
        :return: SMILES
        """
        m = Chem.DeleteSubstructs(self.mol, Chem.MolFromSmiles("*"))
        return m

    def get_observation(self, expert_smi=None):
        """
        ob['adj']:d_e*n*n --- 'E'
        ob['node']:1*n*d_n --- 'F'
        n = atom_num + atom_type_num
        """
        ob = {}

        if expert_smi:
            mol = Chem.MolFromSmiles(expert_smi)
        else:
            ob['att'] = get_att_points(self.mol)
            mol = copy.deepcopy(self.mol)
        
        try:
            Chem.SanitizeMol(mol)
        except:
            pass

        smi = Chem.MolToSmiles(mol)

        n = mol.GetNumAtoms()
        F = np.zeros((1, self.max_atom, self.d_n))

        for a in mol.GetAtoms():
            atom_idx = a.GetIdx()
            
            atom_symbol = a.GetSymbol()
            if self.has_feature:
                float_array = atom_feature(a, use_atom_meta=True)
            else:
                float_array = (atom_symbol == self.possible_atom_types).astype(float)

            F[0, atom_idx, :] = float_array

        d_e = len(self.possible_bond_types)
        E = np.zeros((d_e, self.max_atom, self.max_atom))

        for b in mol.GetBonds(): 
            begin_idx = b.GetBeginAtomIdx()
            end_idx = b.GetEndAtomIdx()
            bond_type = b.GetBondType()
            float_array = (bond_type == self.possible_bond_types).astype(float)
            try:
                assert float_array.sum() != 0
            except:
                print('error',bond_type)
            E[:, begin_idx, end_idx] = float_array
        
        if self.is_normalize:
            E = self.normalize_adj(E)
        
        ob_adj = adj2sparse(E.squeeze())
        ob_node = torch.Tensor(F)
        g = dgl.DGLGraph()

        ob_len = torch.sum(torch.sum(ob_node, dim=-1).bool().float().squeeze(-2), dim=-1)
        g.add_nodes(ob_len)
        if ob_adj is not None and len(ob_adj[0])>0 :
            g.add_edges(ob_adj[0][0], ob_adj[0][1], {'x': ob_adj[1]})
        g.ndata['x'] = ob_node[:, :int(ob_len),:].squeeze(0)
        
        ob['g'] = g
        ob['smi'] = smi
        
        return ob

    def make_active_obs(self):
        active_smi = pd.read_csv(self.active_smi_path, header=None, names=['smiles'])
        num_active = active_smi.shape[0]

        active_obs = []
        for idx in range(num_active):
            smi = active_smi['smiles'][idx]
            ob = self.get_observation(expert_smi=smi)
            active_obs.append(ob)

        with open(self.active_pkl_path, 'wb') as f:
            pickle.dump(active_obs, f)

    def get_active_batch(self, batch_size):
        dataset_len = self.active_dataset.shape[0]
        idxs = np.random.randint(0, dataset_len, size=batch_size)

        obs_batch = self.active_dataset[idxs]
        return obs_batch

    def get_observation_mol(self,mol):
        """
        ob['adj']:d_e*n*n --- 'E'
        ob['node']:1*n*d_n --- 'F'
        n = atom_num + atom_type_num
        """
        ob = {}

        ob['att'] = get_att_points(mol)
        
        try:
            Chem.SanitizeMol(mol)
        except:
            pass

        smi = Chem.MolToSmiles(mol)

        n = mol.GetNumAtoms()
        F = np.zeros((1, self.max_atom, self.d_n))

        for a in mol.GetAtoms():
            atom_idx = a.GetIdx()
            
            atom_symbol = a.GetSymbol()
            if self.has_feature:
                float_array = atom_feature(a, use_atom_meta=True)
            else:
                float_array = (atom_symbol == self.possible_atom_types).astype(float)

            F[0, atom_idx, :] = float_array

        d_e = len(self.possible_bond_types)
        E = np.zeros((d_e, self.max_atom, self.max_atom))

        for b in mol.GetBonds(): 

            begin_idx = b.GetBeginAtomIdx()
            end_idx = b.GetEndAtomIdx()
            bond_type = b.GetBondType()
            float_array = (bond_type == self.possible_bond_types).astype(float)

            try:
                assert float_array.sum() != 0
            except:
                print('error',bond_type)
            E[:, begin_idx, end_idx] = float_array
        
        if self.is_normalize:
            E = self.normalize_adj(E)
        
        ob_adj = adj2sparse(E.squeeze())
        ob_node = torch.Tensor(F)
        g = dgl.DGLGraph()

        ob_len = torch.sum(torch.sum(ob_node, dim=-1).bool().float().squeeze(-2), dim=-1)
        g.add_nodes(ob_len)
        if ob_adj is not None and len(ob_adj[0])>0 :
            g.add_edges(ob_adj[0][0], ob_adj[0][1], {'x': ob_adj[1]})
        g.ndata['x'] = ob_node[:, :int(ob_len),:].squeeze(0)
        
        ob['g'] = g
        ob['smi'] = smi
        return ob

if __name__ == '__main__':
    env = gym.make('molecule-v0') # in gym format

    box_center = (37.4,36.1,12.0)
    box_size = (15.9,26.7,15.0)
    box_parameter = (box_center, box_size)

    docking_config = dict()

    docking_config['vina_program'] = 'qvina02'
    docking_config['receptor_file'] = '../ReLeaSE_Vina/docking/pdb/2oh4A_receptor.pdbqt'
    docking_config['box_parameter'] = box_parameter
    docking_config['temp_dir'] = 'tmp'
    docking_config['exhaustiveness'] = 1
    docking_config['num_sub_proc'] = 10
    # docking_config['num_cpu_dock'] = 1
    docking_config['num_cpu_dock'] = 10
    docking_config['num_modes'] = 10
    docking_config['timeout_gen3d'] = 10
    docking_config['timeout_dock'] = 100

    m_env = MoleculeEnv(docking_config)
    m_env.init(data_type='zinc',has_feature=True)

