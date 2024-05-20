import os
import torch
import pandas as pd
from rse.workflow import compute_rse


if torch.cuda.is_available():
    gpu_idx = 0
else:
    gpu_idx = False

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ring_path = os.path.join(root, 'examples', 'files', 'rings.smi')

def test_compute_rse():
    rse_path = compute_rse(ring_path, gpu_idx)
    df = pd.read_csv(rse_path, index_col=0)
    assert len(df) == 3
    assert (df.loc['smi1', 'RSE (kcal/mol)'] - 27.0) < 2
    assert (df.loc['smi2', 'RSE (kcal/mol)'] - 26.0) < 2
    assert (df.loc['smi3', 'RSE (kcal/mol)'] - 17.0) < 2
