# Ring_Atlas

This repository contains the public code and data for the paper "Ring strain energy prediction with machine learning and the application in strain-promoted reactions". Specifically, it contains the following components:
- Workflow for ring strain energy (RSE) computation
- Pre-trained GNN models for RSE prediction
- links to Ring Atlas


## RSE prediction methods
### Comparision between different methods for obtaining RSE
![Comparision between different works](./figures/intro-part2-2.png)

In addition to experimental methods and QM methods, this work provides 2 additional methods for RSE prediction: 
- The workflow is a physics-based method. For each ring, the workflow firstly constructs an alchemical reaction, then generates low-energy conformers and computes the RSE as the energy difference between the ring and the counterparts. The confomer search and energy calculation are performed using [Auto3D](https://github.com/isayevlab/Auto3D_pkg) with [AIMNet2](https://github.com/isayevlab/AIMNet2) backend.
- The GNN method directly predicts the RSE using only SMILES as the input. The GNN model is trained on the dataset generated by the workflow.

**Please note that the above methods only apply to single-ring molecules**

### Installation
To run the above methods, the following packages are required:

1. Python >= 3.7
2. PyTorch >= 2.1.0
3. [Auto3D](https://pypi.org/project/Auto3D/) >= 2.2.11

The code can be installed following the steps below:
```
https://github.com/isayevlab/RSE_Atlas.git
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

### Predict RSE using pre-trained GNN models
Open any command line interface and run the following command:
```
predict_rse "path_to_smiles_file.smi" --gpu_idx=0
```
This will run the GNN model on the SMILES file and output the predicted RSE in a CSV file with the following tailing name: `_rse_prediction.csv`. 

For both methods, the `--gpu_idx` argument is optional and specifies the GPU index to run the model. If not specified, the model will run on the CPU. It is highly recommended to run the workflow on a GPU, because it is at least 10 times faster than running on a CPU.

## Ring Atlas

A interactive visualization for the Ring Atlas is available at https://zhen.pythonanywhere.com/. This website visualzies about 10% of the Ring Atlas, and provides a search bar for querying the database. The full dataset can be downloaded as a CSV file from the paper Supporting Information.
