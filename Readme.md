# Sampling-guided Heterogeneous Graph Neural Network with Temporal Smoothing for Scalable Longitudinal Data Imputation

## Installation

```
conda create --name SHT-GNN python=3.7.16
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install torch-geometric
```



## Data Resources

The GLOBEM dataset mentioned in the paper can be accessed at https://physionet.org/content/globem/.

The ADNI dataset referenced in the paper can be accessed at https://ida.loni.usc.edu.

## Data Preprocessing

After obtaining the data, we first use the script `Response_generate_16.py` to simulate response values based on the observed covariates.

Once the complete data is acquired, we need to generate relevant auxiliary variables for longitudinal data, such as the longitudinal edge index, time decay weight, Jaccard distance, etc.



## Experiment

After data preprocessing, the main script (main.py) can be used to train and evaluate covariate imputation and response prediction on different datasets under various settings.

All training and evaluation results, including validation loss, training loss, and the saved model parameter files, will be output to the target folder.

