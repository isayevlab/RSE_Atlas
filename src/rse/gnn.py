import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import HybridizationType
import os
import glob
import networkx as nx
from typing import List
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.loader import DataLoader
from torch_geometric.nn import NNConv
from torch_geometric.nn.aggr import SumAggregation
RDLogger.DisableLog('rdApp.*')
from typing import Optional


class SMILES2Graph(object):
    """Converting  a SMILES into a 2D Graph"""
    def __init__(self, smi: str, id=0):
        self.smi = smi
        self.mol = Chem.MolFromSmiles(smi)
        ring_atom_idxes = self.mol.GetRingInfo().AtomRings()
        self.ring_atom_idxes = set(ring_atom_idxes[0])
        self.bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}
        self.types = {'H': 0, 'B': 1, 'C': 2, 'N': 3, 'O': 4, 'Si': 5, 'P': 6,
                      'S': 7, 'As': 8, 'Se': 9, 'I': 10}
        self.chirals = {'CHI_UNSPECIFIED': 0,
                       'CHI_TETRAHEDRAL_CW': 1,
                       'CHI_TETRAHEDRAL_CCW': 2,
                       'CHI_OTHER': 3,
                       'CHI_TETRAHEDRAL': 4,
                       'CHI_ALLENE': 5,
                       'CHI_SQUAREPLANAR': 6,
                       'CHI_TRIGONALBIPYRAMIDAL': 7,
                       'CHI_OCTAHEDRAL': 8}
        self.bond_stereo = {
            'STEREOANY': 0,
            'STEREOATROPCCW': 1,
            'STEREOATROPCW': 2,
            'STEREOCIS': 3,
            'STEREOE': 4,
            'STEREONONE': 5,
            'STEREOTRANS': 6,
            'STEREOZ': 7
        }
        # Initialize UFF bond radii (Rappe et al. JACS 1992)
        # Units of angstroms 
        # These radii neglect the bond-order and electronegativity corrections in the original paper. Where several values exist for the same atom, the largest was used. 
        self.radii = {1:0.354, 
                      5:0.838, 6:0.757, 7:0.700,  8:0.658,  9:0.668,
                      14:1.117, 15:1.117, 16:1.064, 17:1.044,
                      32: 1.197, 33:1.211, 34:1.190, 35:1.192,
                      51:1.407, 52:1.386,  53:1.382}
        self.id = id


    def edge_index_attr(self):
        """mol: RDKit mol object
        Return [2, #edges] and [#edges, #edge_feature_dimension]"""
        N = self.mol.GetNumAtoms()
        row, col, edge_type = [], [], []
        aromatic = []
        conjugated = []
        stereo = []
        in_ring = []

        for bond in self.mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            edge_type += 2 * [self.bonds[bond.GetBondType()]]
            aromatic += 2 * [1 if bond.GetIsAromatic() else 0]
            conjugated += 2 * [1 if bond.GetIsConjugated() else 0]
            in_ring += 2 * [1 if bond.IsInRing() else 0]
            stereo += 2 * [self.bond_stereo[str(bond.GetStereo())]]
        edge_index = torch.tensor([row, col], dtype=torch.long)
        bond_type = F.one_hot(torch.tensor(edge_type), num_classes=len(self.bonds))  #4
        bond_stereo = F.one_hot(torch.tensor(stereo), num_classes=len(self.bond_stereo))  #8
        bond_info = torch.tensor([aromatic, conjugated, in_ring]).t()  #3
        edge_attr = torch.concatenate([bond_type, bond_stereo, bond_info], dim=-1).to(torch.float).contiguous()

        #Sort the edge_index and edge_attr
        perm = (edge_index[0] * N + edge_index[1]).argsort()
        edge_index = edge_index[:, perm]
        # edge_type = edge_type[perm]
        edge_attr = edge_attr[perm] #nx15

        return (edge_index, edge_attr)


    def node_features(self):
        """
        Return [#nodes, #no_feature_dimension]"""
        type_idx = []
        periods = []
        groups = []
        chirals = []

        in_ring = []
        aromatic = []
        sp = []
        sp2 = []
        sp3 = []
        allow_implicit_h = []
        num_hs = []
        degrees = []
        total_degrees = []
        valances = []
        implicit_valances = []
        total_valances = []
        charges = []
        radical_electrons = []
        atom_radii = []
        masses = []
        for atom in self.mol.GetAtoms():
            atom_idx = atom.GetIdx()
            atom_symbol = atom.GetSymbol()
            chiral_tag = str(atom.GetChiralTag())

            type_idx.append(self.types[atom_symbol])
            chirals.append(self.chirals[chiral_tag])
            if atom_idx in self.ring_atom_idxes:
                in_ring.append(1)
            else:
                in_ring.append(0)

            if atom_symbol in {'H'}:
                periods.append(0)
            elif atom_symbol in {'B', 'C', 'N', 'O', 'F'}:
                periods.append(1)
            elif atom_symbol in {'Si', 'P', 'S', 'Cl'}:
                periods.append(2)
            elif atom_symbol in {'As', 'Se', 'Br'}:
                periods.append(3)
            elif atom_symbol in {'I'}:
                periods.append(4)
            else:
                raise ValueError(f"Unknown atom type {atom_symbol}")

            if atom_symbol in {'H'}:
                groups.append(0)
            elif atom_symbol in {'B'}:
                groups.append(1)
            elif atom_symbol in {'C', 'Si'}:
                groups.append(2)
            elif atom_symbol in {'N', 'P', 'As'}:
                groups.append(3)
            elif atom_symbol in {'O', 'S', 'Se'}:
                groups.append(4)
            elif atom_symbol in {'F', 'Cl', 'Br', 'I'}:
                groups.append(5)

            aromatic.append(1 if atom.GetIsAromatic() else 0)
            hybridization = atom.GetHybridization()
            sp.append(1 if hybridization == HybridizationType.SP else 0)
            sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
            sp3.append(1 if hybridization == HybridizationType.SP3 else 0)
            allow_implicit_h.append(1 if atom.GetNoImplicit() else 0)

            hs1 = atom.GetNumExplicitHs()
            hs2 = atom.GetNumImplicitHs()
            num_hs.append((hs1 + hs2))
            degrees.append(atom.GetDegree())
            total_degrees.append(atom.GetTotalDegree())
            valances.append(atom.GetExplicitValence())
            implicit_valances.append(atom.GetImplicitValence())
            total_valances.append(atom.GetTotalValence())
            charges.append(atom.GetFormalCharge())
            radical_electrons.append(atom.GetNumRadicalElectrons())
            atom_radii.append(self.radii[atom.GetAtomicNum()])
            masses.append(atom.GetMass())

        x1 = F.one_hot(torch.tensor(type_idx), num_classes=len(self.types))  #nx11
        x2 = F.one_hot(torch.tensor(periods), num_classes=5)  #nx5
        x3 = F.one_hot(torch.tensor(groups), num_classes=6) #nx6
        x4 = F.one_hot(torch.tensor(chirals), num_classes=9) #nx9
        x5 = torch.tensor([in_ring, aromatic, sp, sp2, sp3,
                           allow_implicit_h,
                           num_hs, degrees, total_degrees, valances,
                           implicit_valances, total_valances, charges,
                           radical_electrons, masses, atom_radii],
                            dtype=torch.float).t()  #nx16
        
        x = torch.cat([x1, x2, x3, x4, x5], dim=-1)  #nx47
        x = x.to(torch.float).contiguous()
        return x


    def get_graph_components(self):
        """Return x, edge_index, edge_attr and id components.
        These components can be used to form a Pytorch Geometric Graph object
        or be used to build a h5py 2D graph dataset"""
        x = self.node_features()
        edge_index, edge_attr = self.edge_index_attr()
        return (x.numpy(), edge_index.numpy(), edge_attr.numpy(), self.id)


    def get_graph(self):
        """Return a Pytorch Graph for the SMILES"""
        x, edge_index, edge_attr, id = self.get_graph_components()
        data = Data(x=torch.tensor(x, dtype=torch.double),
                    edge_index=torch.tensor(edge_index, dtype=torch.long),
                    edge_attr=torch.tensor(edge_attr, dtype=torch.double),
                    name=id)
        return data


    def visualize_graph(self):
        """Return a networkx Graph object for the graph"""
        x, edge_index, edge_attr, id = self.get_graph_components()
        G = nx.Graph(id=id)
        #add nodes x
        m, n = x.shape
        assert(n == 29)
        for i in range(m):
            G.add_node(i, one_hot=x[i, :12])
        #add edges and their attributes
        d1, d2 = edge_index.shape
        assert(d1 == 2)
        for i in range(d2):
            G.add_edge(edge_index[0, i], edge_index[1, i], bond_type=edge_attr[i, :])
        return G


class GNN(torch.nn.Module):
    def __init__(self, in_channel=47, out_channel=47, edge_dim=15,hidden_dim=39,
                 layers=6, layer_dim=154):
        super().__init__()
        # define the message passing network
        nn = Sequential(Linear(edge_dim, hidden_dim),
                        ReLU(),
                        Linear(hidden_dim, in_channel * out_channel))
        self.conv = NNConv(in_channel, out_channel, nn, aggr='sum')
        nn2 = Sequential(Linear(edge_dim, hidden_dim),
                        ReLU(),
                        Linear(hidden_dim, in_channel * out_channel))
        self.conv2 = NNConv(in_channel, out_channel, nn2, aggr='sum')
        self.agg = SumAggregation()
        # define the feed forward network
        self.lin0 = Linear(out_channel, layer_dim)
        self.layers = torch.nn.ModuleList([
           Sequential(
            #    Dropout(),
               Linear(layer_dim, layer_dim),
               ReLU(),
               Linear(layer_dim, layer_dim)
           )
           for _ in range(layers)
        ])
        #output layer
        self.size = Linear(layer_dim, 7)  # probability over 7 ring sizes
        self.rse = Linear(layer_dim, 1)

    def forward(self, data):
        out = self.conv(data.x, data.edge_index, data.edge_attr)
        out = self.conv2(out, data.edge_index, data.edge_attr)
        out = self.agg(out, data.batch)
        out = self.lin0(out)
        for layer in self.layers:
            out = layer(out) + out
        # out = self.lin1(out)
        size_logits = self.size(out)
        rse = self.rse(out)
        return size_logits, rse


class GnnEnsemble(torch.nn.Module):
    def __init__(self, paths: List[str], device=torch.device('cpu')):
        super().__init__()
        self.models = []
        for path in paths:
            model = GNN().double().to(device)
            model.load_state_dict(torch.load(path, map_location=device)['model'])
            model.eval()
            self.models.append(model)
        self.device = device

    def forward(self, data):
        '''data is a torch geometric graph object'''
        rse_preds = []
        for model in self.models:
            size_logits, rse = model(data)  # rse shape (batch_size, 1)
            # size = torch.argmax(size_logits, dim=1) + 3  # size shape (batch_size,)
            rse_preds.append(rse)

        rses = torch.stack(rse_preds)  # shape (5, batch_size, 1)
        rse_avg = torch.mean(rses, dim=0)  # shape (batch_size, 1)
        uncertainties = torch.std(rses, dim=0)  # shape (batch_size, 1)
        return rse_avg, uncertainties


class GnnDataset(Dataset):
    def __init__(self, smi: str):
        super().__init__()
        # read SMILES
        with open(smi, 'r') as f:
            data = f.readlines()
        smiles, names = [], []
        for line in data:
            smi, name = line.strip().split()
            smiles.append(smi)
            names.append(name)
        self.smiles = smiles
        self.names = names

    def len(self):
        return len(self.smiles)
    
    def get(self, idx):
        smi = self.smiles[idx]
        name = self.names[idx]
        graph = SMILES2Graph(smi, id=name)
        data = graph.get_graph()
        return data


def predict_rse(smi: str, gpu_idx: Optional[int]=False):
    '''Predict the ring strain energy of a molecule'''
    # prepare the dataset
    dataset = GnnDataset(smi)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    # verify the device
    if (gpu_idx is not False) and (gpu_idx >= 0):
        device = torch.device(f'cuda:{gpu_idx}')
    else:
        device = torch.device('cpu')

    # load the model
    root = os.path.dirname(os.path.abspath(__file__))
    model_paths = os.path.join(root, 'models', '*.tar.gz')
    model_files = sorted(glob.glob(model_paths))
    model_ensemble = GnnEnsemble(model_files, device=device)

    # make predictions
    all_names = dataset.names
    all_smiles = dataset.smiles
    rse_preds_all, uncertainties = [], []
    for data in loader:
        data.to(device)
        rses, uncertainty = model_ensemble(data)
        rse_preds_all.append(rses)
        uncertainties.append(uncertainty)
    rse_preds_all = torch.cat(rse_preds_all, dim=0).view(-1).detach().cpu().numpy()
    uncertainties = torch.cat(uncertainties, dim=0).view(-1).detach().cpu().numpy()
    results = pd.DataFrame({'name': all_names, 'smiles': all_smiles,
                            'RSE (kcal/mol)': np.round(rse_preds_all, 2),
                            'uncertainty (kcal/mol)': np.round(uncertainties, 2)})

    # write the output
    out_path = smi.replace('.smi', '_rse_prediction.csv')
    results.to_csv(out_path, index=False)
    return out_path


def predict_rse_cli():
    import argparse

    parser = argparse.ArgumentParser(description='Predict the ring strain energy of a molecule')
    parser.add_argument('smi', type=str, help='Path to the smi file')
    parser.add_argument('--gpu_idx', type=int, default=False, nargs='?', help='GPU index')
    args = parser.parse_args()

    out = predict_rse(args.smi, args.gpu_idx)
    return out


if __name__ == '__main__':
    predict_rse_cli()
