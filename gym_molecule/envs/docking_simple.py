#!/usr/bin/env python
import sys
import os
from multiprocessing import Manager
from multiprocessing import Process
from multiprocessing import Queue
import subprocess
from openbabel import pybel
# import pybel

class DockingVina(object):
    """
    python module for Vina
    """
    def __init__(self, docking_params):
        """
            Construction Docking object
            put parameters with dictionary
            docking_params['vina_program']
            docking_params['receptor_file']
            docking_params['box_parameter']
            (self.box_center, self.box_size) = box_parameter
            docking_params['temp_dir']
            docking_params['exhaustiveness']
            docking_params['num_sub_proc']
            docking_params['num_cpu_dock']
            docking_params['num_modes']
            docking_params['timeout_gen3d']
            docking_params['timeout_dock']
        """
        super(DockingVina, self).__init__()
        self.vina_program = docking_params['vina_program']
        self.receptor_file = docking_params['receptor_file']
        box_parameter = docking_params['box_parameter']
        (self.box_center, self.box_size) = box_parameter
        self.temp_dir = docking_params['temp_dir']
        self.exhaustiveness = docking_params['exhaustiveness']
        self.num_sub_proc = docking_params['num_sub_proc']
        self.num_cpu_dock = docking_params['num_cpu_dock']
        self.num_modes = docking_params['num_modes']
        self.timeout_gen3d = docking_params['timeout_gen3d']
        self.timeout_dock = docking_params['timeout_dock']

        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)

    def gen_3d(self, smi, ligand_mol_file):
        """
            generate initial 3d conformation from SMILES
            input :
                SMILES string
                ligand_mol_file (output file)
        """
        run_line = 'obabel -:%s --gen3D -O %s' % (smi, ligand_mol_file)
        result = subprocess.check_output(run_line.split(),
                                         stderr=subprocess.STDOUT,
                                         timeout=self.timeout_gen3d, universal_newlines=True)

    def docking(self, receptor_file, ligand_mol_file, ligand_pdbqt_file, docking_pdbqt_file):
        """
            run_docking program using subprocess
            input :
                receptor_file
                ligand_mol_file
                ligand_pdbqt_file
                docking_pdbqt_file
            output :
                affinity list for a input molecule
        """
        ms = list(pybel.readfile("mol", ligand_mol_file))
        m = ms[0]
        m.write("pdbqt", ligand_pdbqt_file, overwrite=True)
        run_line = '%s --receptor %s --ligand %s --out %s' % (self.vina_program,
                                                              receptor_file, ligand_pdbqt_file, docking_pdbqt_file)
        run_line += ' --center_x %s --center_y %s --center_z %s' %(self.box_center)
        run_line += ' --size_x %s --size_y %s --size_z %s' %(self.box_size)
        run_line += ' --cpu %d' % (self.num_cpu_dock)
        run_line += ' --num_modes %d' % (self.num_modes)
        run_line += ' --exhaustiveness %d ' % (self.exhaustiveness)
        result = subprocess.check_output(run_line.split(),
                                         stderr=subprocess.STDOUT,
                                         timeout=self.timeout_dock, universal_newlines=True)
        result_lines = result.split('\n')

        check_result = False
        affinity_list = list()
        for result_line in result_lines:
            if result_line.startswith('-----+'):
                check_result = True
                continue
            if not check_result:
                continue
            if result_line.startswith('Writing output'):
                break
            if result_line.startswith('Refine time'):
                break
            lis = result_line.strip().split()
            if not lis[0].isdigit():
                break
#            mode = int(lis[0])
            affinity = float(lis[1])
            affinity_list += [affinity]
        return affinity_list

    def creator(self, q, data, num_sub_proc):
        """
            put data to queue
            input: queue
                data = [(idx1,smi1), (idx2,smi2), ...]
                num_sub_proc (for end signal)
        """
        for d in data:
            idx = d[0]
            dd = d[1]
            q.put((idx, dd))

        for i in range(0, num_sub_proc):
            q.put('DONE')

    def docking_subprocess(self, q, return_dict, sub_id=0):
        """
            generate subprocess for docking
            input
                q (queue)
                return_dict
                sub_id: subprocess index for temp file
        """
        while True:
            qqq = q.get()
            if qqq == 'DONE':
                break
            (idx, smi) = qqq
            # print(smi)
            receptor_file = self.receptor_file
            ligand_mol_file = '%s/ligand_%s.mol' % (self.temp_dir, sub_id)
            ligand_pdbqt_file = '%s/ligand_%s.pdbqt' % (self.temp_dir, sub_id)
            docking_pdbqt_file = '%s/dock_%s.pdbqt' % (self.temp_dir, sub_id)
            try:
                self.gen_3d(smi, ligand_mol_file)
            except Exception as e:
                print(e)
                print("gen_3d unexpected error:", sys.exc_info())
                print("smiles: ", smi)
                return_dict[idx] = 99.9
                continue
            try:
                affinity_list = self.docking(receptor_file, ligand_mol_file,
                                             ligand_pdbqt_file, docking_pdbqt_file)
            except Exception as e:
                print(e)
                print("docking unexpected error:", sys.exc_info())
                print("smiles: ", smi)
                return_dict[idx] = 99.9
                continue
            if len(affinity_list)==0:
                affinity_list.append(99.9)
            
            affinity = affinity_list[0]
            return_dict[idx] = affinity

    def predict(self, smiles_list):
        """
            input SMILES list
            output affinity list corresponding to the SMILES list
            if docking is fail, docking score is 99.9
        """
        data = list(enumerate(smiles_list))
        q1 = Queue()
        manager = Manager()
        return_dict = manager.dict()
        proc_master = Process(target=self.creator,
                              args=(q1, data, self.num_sub_proc))
        proc_master.start()

        # create slave process
        procs = []
        for sub_id in range(0, self.num_sub_proc):
            proc = Process(target=self.docking_subprocess,
                           args=(q1, return_dict, sub_id))
            procs.append(proc)
            proc.start()

        q1.close()
        q1.join_thread()
        proc_master.join()
        for proc in procs:
            proc.join()
        keys = sorted(return_dict.keys())
        affinity_list = list()
        for key in keys:
            affinity = return_dict[key]
            affinity_list += [affinity]
        return affinity_list
