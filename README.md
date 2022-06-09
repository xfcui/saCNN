# saCNN
The [saCNN](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9669781) is a protein-ligand prediction tool based on 3D convolutional neural network with spatial attention mechanisms, to encourage spatial feature learning. It can focus more on the voxels near interaction centers. You can quickly get started with the saCNN tool according to the following instructions.

## Installation
Create virtual environment and install packages:
```bash
conda create -n saCNN htmd=2.0.6 -c acellera -c conda-forge
conda activate saCNN
pip install torch
```

## Quick start

### Git clone
Clone this repository by:
```bash
git clone https://github.com/xfcui/saCNN.git
```

### Data preparation
After creating a virtual environment, you need to prepare data and trained model. We provide a sample data in the `data/dataset/3jvr` directory, which contains the files of protein (`3jvr_protein.pdb`) and ligand (`3jvr_ligand.mol2`). We also provide the trained model under the `checkpoint/model.pkl`.


### Data processing
Run the following command to complete the characterization of protein and ligand. The file path of protein, ligand and feature generation are set in `data.sh` file.  

```bash
bash src/data.sh
```


### Model inference
Run the following command to complete the affinity prediction of protein and ligand. The file path of feature and model checkpoint are set in `inference.sh` file.

```bash
bash src/inference.sh
```

## Usage
If you want to run our model on your own data, you need to provide the protein (`.pdb`) file and ligands (`.mol2`) file.

```
Authors: Yuxiao Wang, Zongzhao Qiu, Qihong Jiao, Cheng Chen, Zhaoxu Meng and Xuefeng Cui*
Contact: xfcui@email.sdu.edu.cn
```