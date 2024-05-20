#!/bin/usr/env python
import os, sys
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root)
import copy
from rdkit import Chem
from rdkit.Chem import BondStereo, BondDir
import itertools
import tempfile
import shutil
import pandas as pd
from typing import Tuple, List, Optional, Dict
from Auto3D.auto3D import options, main
from Auto3D.ASE.thermo import calc_thermo
from Auto3D.utils import hartree2kcalpermol


class BreakRing(object):
    """Break single bonds that are formed by two non-substituted SP3 carbons 
    in rings. Restricting bond conditions by using larger depth value (which check 
    bond neighboring atoms to be SP3 carbon)."""
    def __init__(self, smi, depth=1, force=False,
                 atom_types=[1, 6, 7, 8, 9, 16, 17, 5, 14, 15, 33, 34, 35, 53]):
        # the last 7 elements are for AIMNet2
        self.smi = smi
        self.depth = depth
        self.mol = Chem.MolFromSmiles(self.smi)
        self.atoms = set(atom_types)
        self.force = force

    def bad_molecule(self) -> bool:
        """Check if a ring can be handled by ANI/AIMNet2"""
        atom_nums = []
        for atom in self.mol.GetAtoms():
            atom_num = atom.GetAtomicNum()
            atom_nums.append(atom_num)
        if set(atom_nums).issubset(self.atoms):
            return False
        return True

    def detect_rings(self) -> Tuple[Tuple[int]]:
        """Find all ring system in the molecule, return a list of bond indices"""
        ring = self.mol.GetRingInfo()
        ring_bonds = ring.BondRings()
        return ring_bonds

    def atom_neighbors(self, idx: int, depth: Optional[int]=None) -> List[int]:
        """Return the neighbors of an atom with depth range"""
        if depth is None:
            depth = self.depth
        search_range = [idx]
        searched = []
        cumulative = []
        while depth > 0:
            search_range_ = []
            for idx in search_range:
                searched.append(idx)
                atom = self.mol.GetAtomWithIdx(idx)
                neighbors_i = [x.GetIdx() for x in atom.GetNeighbors()]
                neighbors_i2 = [val for val in neighbors_i if val not in searched]
                cumulative += neighbors_i2
                search_range_ += neighbors_i2
            search_range = search_range_
            depth -= 1
        neighbors = list(set(cumulative))
        return neighbors

    def check_bond_neighbors(self, bond_idx) -> bool:
        """Return True if the bond and its neighboring atoms are SP3 carbonds
        
        Arguments:
            bond_idx: bond index"""
        bond = self.mol.GetBondWithIdx(bond_idx)
        atom1_idx = bond.GetBeginAtomIdx()
        atom2_idx = bond.GetEndAtomIdx()
        atom1_num = self.mol.GetAtomWithIdx(atom1_idx).GetAtomicNum()
        atom2_num = self.mol.GetAtomWithIdx(atom2_idx).GetAtomicNum()
        if atom1_num != 6 or (atom2_num != 6):
            return False
        if self.force == True:
            return True
        else:
            neighbor_idxes1 = self.atom_neighbors(atom1_idx)
            neighbor_idxes2 = self.atom_neighbors(atom2_idx)
            neighbor_idxes = list(set(neighbor_idxes1 + neighbor_idxes2))
            for idx in neighbor_idxes:
                atom = self.mol.GetAtomWithIdx(idx)
                atomic_num = atom.GetAtomicNum()
                if (atomic_num != 6) or (str(atom.GetHybridization()) != 'SP3'):
                    return False
            return True

    def collate_bonds(self) -> List[int]:
        """Return all bond indices to be cut"""
        results = []
        ring_bonds0 = list(itertools.chain(*self.ring_bonds))
        ring_bonds = [bond for bond in ring_bonds0 if self.check_bond_neighbors(bond)]
        for idx in ring_bonds:
            bond = self.mol.GetBondWithIdx(idx)
            atom1 = bond.GetBeginAtom()
            atom2 = bond.GetEndAtom()
            b_type = str(bond.GetBondType())
            if self.force ==True:
                if (b_type == 'SINGLE'):
                    results.append(idx)
            else:
                if ((b_type == 'SINGLE') and (str(atom1.GetHybridization()) == 'SP3') and 
                (str(atom2.GetHybridization()) == 'SP3') and (atom1.GetNumImplicitHs() == 2)
                and (atom2.GetNumImplicitHs() == 2)):
                    results.append(idx)
        return results

    def db_ring_sub_helper1(self, mol, idx1: int, idx2: int, ring: List[int]) -> int:
        """mol: RDKit mol/rwmol object
        idx1: double_bond_atom_idx1
        idx2: double_bond_atom_idx2
        ring: ring atom indices
        For atom with idx1, return its neighbor in the ring.
        """
        neighbors = mol.GetAtomWithIdx(idx1).GetNeighbors()
        neighbors = [x.GetIdx() for x in neighbors]
        neighbors = [val for val in neighbors if val != idx2]
        if len(neighbors) == 1:
            stereo_atom = neighbors[0]
        else:
            if neighbors[0] in ring:
                stereo_atom = neighbors[0]
            else:
                stereo_atom = neighbors[1]
        return stereo_atom

    def db_ring_substituents(self, mol, idx1: int, idx2: int, ring: List[int]) -> List[int]:
        """mol: RDKit mol/rwmol object
        idx1: db_begin_atom_idx
        idx2: db_end_atom_idx
        ring: ring atom indices
        Return the neighbors of the double bond that are in the ring.
        Return list length is 2
        """
        stereo_atom1 = self.db_ring_sub_helper1(mol, idx1, idx2, ring)
        stereo_atom2 = self.db_ring_sub_helper1(mol, idx2, idx1, ring)
        return [stereo_atom1, stereo_atom2]

    def get_parent_ring(self, index: int) -> Tuple[int]:
        """check ring that contains the bond.
        Assuming this bond is in only one ring, since we do not cut bonds in fused rings."""
        for ring in self.ring_bonds:
            if index in ring:
                return ring
    
    def get_ring_atom_idxes(self, ring: Tuple[int]) -> List[int]:
        """Get the atom indexes for the ring"""
        ring_atom_idxes = []
        for bond_idx in ring:
            bond = self.mol.GetBondWithIdx(bond_idx)
            ring_atom_idxes.append(bond.GetBeginAtomIdx())
            ring_atom_idxes.append(bond.GetEndAtomIdx())
        ring_atom_idxes = list(set(ring_atom_idxes))
        return ring_atom_idxes

    def cut_sp3_helper1(self, mol, db_begin_atom_idx:int, neighbor_atom_idx1:int,
                        db_end_atom_idx: int, neighbor_atom_idx2:int):
        """Helper function for set the stereo atoms and bond directions for ring size >= 4
        db_begin_atom_idx: double bond begin atom index
        neighbor_atom_idx1: neighbor atom index of db_begin_atom that is in the ring
        db_end_atom_idx: double bond end atom index
        neighbor_atom_idx2: neighbor atom index of db_end_atom that is in the ring"""
        mol.GetBondBetweenAtoms(db_begin_atom_idx, db_end_atom_idx).SetStereoAtoms(neighbor_atom_idx1, neighbor_atom_idx2)
        # if both double bond atoms are the Begin/End atom of the substituents of the double bond
        db_substituent1 = mol.GetBondBetweenAtoms(db_begin_atom_idx, neighbor_atom_idx1)
        db_substituent1_begin_idx = db_substituent1.GetBeginAtomIdx()
        db_substituent1_end_idx = db_substituent1.GetEndAtomIdx()

        db_substituent2 = mol.GetBondBetweenAtoms(db_end_atom_idx, neighbor_atom_idx2)
        db_substituent2_begin_idx = db_substituent2.GetBeginAtomIdx()
        db_substituent2_end_idx = db_substituent2.GetEndAtomIdx()

        both_begin = (db_substituent1_begin_idx == db_begin_atom_idx) and (db_substituent2_begin_idx == db_end_atom_idx)
        both_end = (db_substituent1_end_idx == db_begin_atom_idx) and (db_substituent2_end_idx == db_end_atom_idx)
        if both_begin or both_end:
            mol.GetBondBetweenAtoms(db_begin_atom_idx, neighbor_atom_idx1).SetBondDir(BondDir.ENDUPRIGHT)
            mol.GetBondBetweenAtoms(db_end_atom_idx, neighbor_atom_idx2).SetBondDir(BondDir.ENDUPRIGHT)
        else:
            mol.GetBondBetweenAtoms(db_begin_atom_idx, neighbor_atom_idx1).SetBondDir(BondDir.ENDDOWNRIGHT)
            mol.GetBondBetweenAtoms(db_end_atom_idx, neighbor_atom_idx2).SetBondDir(BondDir.ENDUPRIGHT)

    def cut_sp3_helper2(self, mol, db_begin_atom_idx:int, db_end_atom_idx: int,
                        cut_bond_begin_idx: int, cut_bond_end_idx: int,
                        atom1_idx: int, atom2_idx: int):
        """
        Helper function for set the stereo atoms and bond directions for ring size = 3
        db_begin_atom_idx: double bond begin atom index
        db_end_atom_idx: double bond end atom index
        cut_bond_begin_idx: begin atom index of the bond to be cut
        cut_bond_end_idx: end atom index of the bond to be cut
        atom1_idx: new atom 1 index
        atom2_idx: new atom 2 index
        """
        if str(mol.GetAtomWithIdx(cut_bond_begin_idx).GetHybridization()) == 'SP3':
            assert(str(mol.GetAtomWithIdx(cut_bond_end_idx).GetHybridization()) == 'SP2')
            stereo_atom_idx = cut_bond_begin_idx
            stereo_atom_idx2 = atom2_idx
        else:
            assert(str(mol.GetAtomWithIdx(cut_bond_end_idx).GetHybridization()) == 'SP3')
            stereo_atom_idx = cut_bond_end_idx
            stereo_atom_idx2 = atom1_idx
        mol.GetBondBetweenAtoms(db_begin_atom_idx, db_end_atom_idx).SetStereoAtoms(stereo_atom_idx, stereo_atom_idx2)

        # check bond direction 
        db_substituent1 = mol.GetBondBetweenAtoms(stereo_atom_idx, db_begin_atom_idx)
        db_substituent2 = mol.GetBondBetweenAtoms(stereo_atom_idx2, db_end_atom_idx)
        if db_substituent1 is None:
            db_substituent1 = mol.GetBondBetweenAtoms(stereo_atom_idx, db_end_atom_idx)
            db_substituent2 = mol.GetBondBetweenAtoms(stereo_atom_idx2, db_begin_atom_idx)
        db_substituent1_begin_idx = db_substituent1.GetBeginAtomIdx()
        db_substituent1_end_idx = db_substituent1.GetEndAtomIdx()

        if db_substituent1_begin_idx != stereo_atom_idx2:
            # both substituents' begin atoms are the double bond end atom
            db_substituent1.SetBondDir(BondDir.ENDUPRIGHT)
            db_substituent2.SetBondDir(BondDir.ENDUPRIGHT)
        else:
            db_substituent1.SetBondDir(BondDir.ENDDOWNRIGHT)
            db_substituent2.SetBondDir(BondDir.ENDUPRIGHT)

    def is_carbond_double_bond(self, dond) -> bool:
        """Return True if a bond is a double bond between two carbons"""
        atom1 = dond.GetBeginAtom()
        atom2 = dond.GetEndAtom()
        b_type = str(dond.GetBondType())
        if (b_type == 'DOUBLE') and (atom1.GetAtomicNum() == 6) and (atom2.GetAtomicNum() == 6):
            return True
        return False

    def cut_sp3(self, index) -> Chem.Mol:
        """Cut a molecule based on the bond index, 
           return resulting molecule with two extra CH3"""
        enforce_stereo = False
        ring = self.get_parent_ring(index)
        ring_size = len(ring)
        ring_atom_idxes = self.get_ring_atom_idxes(ring)

        # identify the double bond index if the ring size is at most 7
        # for ring size >=8, the double bond stereo configuration should be defined in the input SMILES
        if ring_size <= 7:
            for bond_idx in ring:
                bond = self.mol.GetBondWithIdx(bond_idx)
                if self.is_carbond_double_bond(bond):
                    db_begin_atom_idx = bond.GetBeginAtom().GetIdx()
                    db_end_atom_idx = bond.GetEndAtom().GetIdx()
                    enforce_stereo = True
                    break  # assuming only 1 double bond in the ring with at most 7 atoms

        # cut the bond
        rwmol = Chem.RWMol(copy.deepcopy(self.mol))
        b = rwmol.GetBondWithIdx(index)
        b_type = b.GetBondType()
        bond_atom1 = b.GetBeginAtomIdx()
        bond_atom2 = b.GetEndAtomIdx()
        rwmol.RemoveBond(bond_atom1, bond_atom2)

        # Adding C at both ends
        atom1_idx = rwmol.AddAtom(Chem.Atom('C'))
        atom2_idx = rwmol.AddAtom(Chem.Atom('C'))
        rwmol.AddBond(bond_atom1, atom1_idx, b_type)
        rwmol.AddBond(bond_atom2, atom2_idx, b_type)
        mol,  = Chem.GetMolFrags(rwmol, asMols=True)

        # set the stereochemistry of the double bond
        if enforce_stereo:
            db_ring_neighbors = self.db_ring_substituents(mol, db_begin_atom_idx, db_end_atom_idx, ring_atom_idxes)
            double_bond = mol.GetBondBetweenAtoms(db_begin_atom_idx, db_end_atom_idx)
            double_bond.SetStereo(BondStereo.STEREOZ)
            if len(db_ring_neighbors) == 2:
                self.cut_sp3_helper1(mol, db_begin_atom_idx, db_ring_neighbors[0], db_end_atom_idx, db_ring_neighbors[1])
            else:
                self.cut_sp3_helper2(mol, db_begin_atom_idx, db_end_atom_idx, bond_atom1, bond_atom2, atom1_idx, atom2_idx)

        Chem.SanitizeMol(mol)
        return mol

    def get_broken_ring(self):
        """break sp3 bonds in the input smi, return result pairs (og, broken)"""
        if self.bad_molecule():
            print("Input %s contain complex atoms" % (self.smi))
            return []
        # get ring bond indices
        self.ring_bonds = self.detect_rings()
        
        results = []
        bonds2cut = self.collate_bonds()
        for bond in bonds2cut:
            mol = self.cut_sp3(bond)
            smi = Chem.MolToSmiles(mol, doRandom=False, isomericSmiles=True)
            results.append(smi)
        results = sorted(list(set(results)))
        return results


# class SmiProcessor(object):
def process_smi(path: str, depth: int=1):
    '''Process the input smi file, 
    return a path to the output file where the rings and idx are written in smi format.
    For normal and relaxed cases, only keep 1 broken ring.
    For forced cases, keep all broken rings.'''
    # self.path = path
    # self.depth = depth

    # Read input file into a dictionary
    with open(path, 'r') as f:
        data = f.readlines()
    dct = {}
    name_id_dict = {}
    for i, line in enumerate(data):
        smi, name = line.strip().split()
        dct[str(i)] = smi
        name_id_dict[name] = str(i)
    id_name_dict = {v: k for k, v in name_id_dict.items()}

# def process(self) -> str:
    normal, relaxed, forced, failed, valid = {}, {}, {}, {}, {}
    for idx, smi in dct.items():
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            br = BreakRing(smi, depth)
            broken_rings = br.get_broken_ring()

            if len(broken_rings) == 0:
                depth_ = depth
                while ((depth_ >= 1) and (len(broken_rings) ==0)):
                    depth_ -= 1
                    br_ = BreakRing(smi, depth_)
                    broken_rings = br_.get_broken_ring()
                if len(broken_rings) == 0:
                    print(f"Trying force breaking for {smi} {id_name_dict[idx]}... ", end="")
                    br_ = BreakRing(smi, 0, force=True)
                    broken_rings = br_.get_broken_ring()
                    if len(broken_rings) == 0:
                        print("Input ring cannot be broken")
                        failed[idx] = [smi]
                    else:
                        print("Success!")
                        forced[idx + 'b'] = broken_rings
                        valid[idx] = [smi]
                else:
                    print(f"Relaxed ring breaking restrictions for Input {smi} {id_name_dict[idx]}")
                    relaxed[idx + 'b'] = broken_rings
                    valid[idx] = [smi]
            else:
                print(f"Normal ring breaking for Input {smi} {id_name_dict[idx]}")
                normal[idx + 'b'] = broken_rings
                valid[idx] = [smi]
        else:
            print(f"Invaid SMILES: {smi} for {id_name_dict[idx]}")

    print('Total number of input SMILES: ', len(dct))
    print('Number of normal broken SMILES: ', len(normal))
    print('Number of relaxed broken SMILES: ', len(relaxed))
    print('Number of forced broken SMILES: ', len(forced))
    print('Number of failed broken SMILES: ', len(failed))
    print('Number of valid rings: ', len(valid))

    print('Writing the rings and broken rings into smi file...')
    folder = os.path.dirname(os.path.abspath(path))
    basename = os.path.basename(path).split('.')[0]
    out_path = os.path.join(folder, f'{basename}_rings_and_broken_rings.smi')
    name_id_path = os.path.join(folder, 'name_id_dict.txt')
    # only keep 1 broken ring for normal and relaxed cases
    normal = {k: v[:1] for k, v in normal.items()}
    relaxed = {k: v[:1] for k, v in relaxed.items()}

    # combine all dictionaries into one
    all_data = {**valid, **normal, **relaxed, **forced}
    all_data_sorted = sorted([(k, v) for k, v in all_data.items()], key=lambda x: x[0])

    with open(out_path, 'w') as f:
        for id, smiles in all_data_sorted:
            if len(smiles) == 1:
                f.write(f"{smiles[0]} {id}\n")
            else:
                for i, smi in enumerate(smiles):
                    f.write(f"{smi} {id}_{i}\n")

    with open(name_id_path, 'w') as f:
        for k, v in name_id_dict.items():
            f.write(f"{k} {v}\n")
    
    print(f'Output file is saved at {out_path}')
    print(f'Name ID dictionary is saved at {name_id_path}')
    return out_path, name_id_path


def _keep_lowest_energy(path: str):
    # only keep the lowest energy conformer for each molecule
    mols = {}
    supp = Chem.SDMolSupplier(path, removeHs=False)
    for mol in supp:
        id = mol.GetProp('_Name')
        if '_' in id:
            id = id.split('_')[0]
            mol.SetProp('_Name', id)
            if id in mols:
                prev_e = float(mols[id].GetProp('E_tot'))
                e = float(mol.GetProp('E_tot'))
                if e < prev_e:
                    mols[id] = mol
            else:
                mols[id] = mol
        else:
            mols[id] = mol
    
    os.remove(path)
    with Chem.SDWriter(path) as w:
        for mol in mols.values():
            w.write(mol)
    return path

def sdf2rse(path: str):
    '''Given an SDF file containing conformers
    (each conformer ID has to be digits or digits + 'b', for exampple, 1 and 1b)
    return a csv file containing the RSE values'''
    # path = path
    # AIMNet2 ethane
    print('Using AIMNet2 ethane as reference...')
    ethane_h = -79.79468799014762
    folder = os.path.dirname(os.path.abspath(path))
    basename = os.path.basename(path).split('.')[0]
    rse_path = os.path.join(folder, f'{basename}_rse.csv')

    # only keep the lowest energy conformer for each molecule
    mols = {}
    supp = Chem.SDMolSupplier(path, removeHs=True)
    for mol in supp:
        id = mol.GetProp('_Name')
        mols[id] = mol

    # compute RSE
    data = []
    for id, mol in mols.items():
        if id.isdigit() and ((id + 'b') in mols):
            mol_b = mols[id + 'b']
            h = float(mol.GetProp('H_hartree'))
            h_b = float(mol_b.GetProp('H_hartree'))
            rse = round((h + ethane_h - h_b) * hartree2kcalpermol, 2)

            ring_smi = Chem.MolToSmiles(mol, doRandom=False, isomericSmiles=True)
            broken_smi = Chem.MolToSmiles(mol_b, doRandom=False, isomericSmiles=True)
            data.append([id, ring_smi, rse, broken_smi])
    
    df = pd.DataFrame(data, columns=['ID', 'Ring', 'RSE (kcal/mol)', 'Broken Ring'])
    df.to_csv(rse_path, index=False)
    return rse_path


def add_names_to_rse(rse_path: str, name_id_path: str):
    '''Given the rse csv file and name_id_dict.txt file,
    return a csv file with names'''
    df = pd.read_csv(rse_path)
    id_name_dict = {}
    with open(name_id_path, 'r') as f:
        data = f.readlines()
    for line in data:
        name, id = line.strip().split()
        id_name_dict[id] = name
    df['Name'] = df['ID'].apply(lambda x: id_name_dict[str(x)])
    df = df[['Name', 'Ring', 'RSE (kcal/mol)', 'Broken Ring']]
    df.to_csv(rse_path, index=False)
    return rse_path


def compute_rse(smi: str, gpu_idx: Optional[int]=False):
    real_sdf_path = smi.replace('.smi', '.sdf')
    real_rse_path = smi.replace('.smi', '.csv')

    with tempfile.TemporaryDirectory() as temp_dir:
        path = shutil.copy(smi, temp_dir)
        print('Step 1: Breaking rings...')
        path, name_id_path = process_smi(path)
        print()

        print('Step 2: Conformer search with Auto3D...')
        if gpu_idx >= 0:
            sdf = main(options(path, k=1, gpu_idx=gpu_idx))
        else:
            sdf = main(options(path, k=1, use_gpu=False))
        sdf = _keep_lowest_energy(sdf)
        print()

        print('Step 3: Compute thermodynamical properties with Auto3D...')
        if gpu_idx >= 0:
            sdf = calc_thermo(sdf, model_name='AIMNET', gpu_idx=gpu_idx)
        else:
            sdf = calc_thermo(sdf, model_name='AIMNET')
        print(sdf)
        shutil.copy(sdf, real_sdf_path)
        print()

        print('Step 4: Compute RSE and save in csv file...')
        rse_path = add_names_to_rse(sdf2rse(sdf), name_id_path)
        shutil.copy(rse_path, real_rse_path)
        print(real_rse_path)
    return real_rse_path


def compute_rse_cli():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help='Path to the input SMI file')
    parser.add_argument('--gpu_idx', nargs='?', default=False, const=True, type=int,
                        help='GPU index')
    args = parser.parse_args()

    compute_rse(args.path, args.gpu_idx)


if __name__ == '__main__':
    compute_rse_cli()
