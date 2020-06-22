# proj_fsnet_exp
Additional experiments for the paper: FsNet: Feature Selection Network on High-dimensional Biological Data


## Experiments
### Download datasets
Datasets were taken on this webpage: http://featureselection.asu.edu/datasets.php

Download them in the data folder, or just copy paste these commands in a terminal:

```bash
mkdir data
cd data
wget http://featureselection.asu.edu/files/datasets/ALLAML.mat
wget http://featureselection.asu.edu/files/datasets/CLL_SUB_111.mat
wget http://featureselection.asu.edu/files/datasets/GLI_85.mat
wget http://featureselection.asu.edu/files/datasets/GLIOMA.mat
wget http://featureselection.asu.edu/files/datasets/Prostate_GE.mat
wget http://featureselection.asu.edu/files/datasets/SMK_CAN_187.mat
```

### Installation
This installation have been tested with conda and Python 3.6, and should work by running these commands:  

```bash
conda create --name fsnet python=3.6
conda activate fsnet
pip install numpy
pip install Cython
pip install pymrmr
pip install pyHSICLasso
pip install click
pip install tqdm
pip install sklearn
```


### Run the experiments

```bash
python main.py [ALLAML|CLL_SUB|GLI_85|GLIOMA|Prostate_GE|SMK_CAN]
```
