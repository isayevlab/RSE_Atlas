# Ring_Atlas

This repository contains the public code and data for the paper "Ring strain energy prediction with machine learning and the application in strain-promoted reactions". Specifically, it contains the following components:
- The *AIMNet2 RSE Workflow* for ring strain energy (RSE) computation
- The link to Ring Atlas


## RSE prediction methods
**Please note that the above methods only apply to single-ring molecules**

### Installation
To run the above methods, the following packages are required:

1. Python >= 3.7
2. PyTorch >= 2.1.0
3. [Auto3D](https://pypi.org/project/Auto3D/) >= 2.2.11

The code can be installed following the steps below:
```
git clone https://github.com/isayevlab/RSE_Atlas.git
cd Ring_Atlas
pip install .
```

### Compute RSE using the workflow
Open any command line interface and run the following command:
```
compute_rse "path_to_smiles_file.smi" --gpu_idx=0
```
This will run the workflow on the SMILES file and output the computed RSE to the same directory in a CSV file. 
The workflow also outputs the final conformers for the rings and broken rings (methyl groups attached to each end) in the `.sdf` file. The CSV and SDF file have the same file name as the input `smi` file., but different extensions 

## Ring Atlas

A interactive visualization for the Ring Atlas is available at https://rseatlas.isayevlab.org. This website visualzies about 10% of the Ring Atlas, and provides a search bar for querying the database. The full dataset can be downloaded as a CSV file [here](https://github.com/isayevlab/RSE_Atlas/blob/08ef10efbd74b6d027b5c1d6312d8604eebb9d16/rse_atlas.csv).
