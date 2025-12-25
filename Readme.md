# SHT-GNN

**Sampling-guided Heterogeneous Graph Neural Network with Temporal Smoothing for Scalable Longitudinal Data Imputation**

This project implements a graph neural network-based framework for longitudinal data imputation and prediction, specifically designed to handle heterogeneous graph data with temporal dimensions. The project supports two main tasks: covariate imputation and response prediction.

---

## 📋 Table of Contents

- [Installation](#installation)
- [Data Resources](#data-resources)
- [Project Structure](#project-structure)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Core Modules](#core-modules)
- [Usage Examples](#usage-examples)

---

## 🔧 Installation

### Create Virtual Environment

```bash
conda create --name SHT-GNN python=3.7.16
conda activate SHT-GNN
```

### Install Dependencies

```bash
# Install PyTorch (CUDA 11.7)
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu117

# Install PyTorch Geometric
pip install torch-geometric

# Install other dependencies
pip install numpy pandas scipy matplotlib seaborn scikit-learn
```

---

## 📊 Data Resources

This project supports the following public datasets:

- **GLOBEM Dataset**: https://physionet.org/content/globem/
- **ADNI Dataset**: https://ida.loni.usc.edu

Please apply and download according to each dataset's usage agreement.

---

## 📁 Project Structure

The organized project structure is as follows:

```text
SHT-GNN/
├── README.md                     
├── main.py                       
│
├── scripts/                      
│   ├── preprocess/              
│   │   ├── Data_preprocessing.py     
│   │   ├── Response_gerenate_16.py   
│   │   ├── Edge_index_generate.py    
│   │   ├── IndiceGenerate.py         
│   │   ├── TimeConvert.py            
│   │   ├── Time_Decay_weight.py      
│   │   ├── Normalize_Matrix.py       
│   │   └── JaccardDistance.py        
│   └── baselines/                
│       └── linear_regression_y_baseline.py  
│
├── data_process/                 
│   ├── __init__.py
│   ├── data_load.py              
│   └── data_subparser.py         
│
├── models/                      
│   ├── __init__.py
│   ├── gnn_model.py              
│   ├── egcn.py                   
│   ├── egsage.py                 
│   ├── longitudinal_network.py   
│   └── prediction_model.py       
│
├── training/                     
│   ├── __init__.py
│   ├── gnn_y.py                  
│   ├── baseline.py               
│   ├── linear_regression.py      
│   ├── subject_mapping.py        
│   └── WeightGraph.py            
│
└── utils/                        
    ├── __init__.py
    ├── utils.py                  
    └── plot_utils.py             
```



---

## 🔄 Data Preprocessing

### Step 1: Generate Response Variables (Simulation Experiments)

```bash
python scripts/preprocess/Response_gerenate_16.py \
    --input_data path/to/raw_data \
    --output_path path/to/processed_data
```

This script simulates and generates response values based on observed covariates for experimental evaluation.

### Step 2: Data Cleaning and Preprocessing

```bash
python scripts/preprocess/Data_preprocessing.py \
    --input_path path/to/raw_data \
    --output_path path/to/processed_data
```

### Step 3: Generate Longitudinal Data Auxiliary Variables

These variables are crucial for graph neural network modeling:

```bash
# Generate indices
python scripts/preprocess/IndiceGenerate.py --data_dir path/to/processed_data

# Generate longitudinal edge index
python scripts/preprocess/Edge_index_generate.py --data_dir path/to/processed_data

# Time conversion
python scripts/preprocess/TimeConvert.py --data_dir path/to/processed_data

# Calculate time decay weights
python scripts/preprocess/Time_Decay_weight.py --data_dir path/to/processed_data

# Calculate Jaccard distance (for building similarity graph)
python scripts/preprocess/JaccardDistance.py --data_dir path/to/processed_data

# Matrix normalization
python scripts/preprocess/Normalize_Matrix.py --data_dir path/to/processed_data
```

### Quick Start: One-Command Preprocessing

You can create a `preprocess_all.sh` script to automatically execute all steps:

```bash
#!/bin/bash
DATA_DIR="path/to/processed_data"

python scripts/preprocess/Response_gerenate_16.py --data_dir $DATA_DIR
python scripts/preprocess/Data_preprocessing.py --data_dir $DATA_DIR
python scripts/preprocess/IndiceGenerate.py --data_dir $DATA_DIR
python scripts/preprocess/Edge_index_generate.py --data_dir $DATA_DIR
python scripts/preprocess/TimeConvert.py --data_dir $DATA_DIR
python scripts/preprocess/Time_Decay_weight.py --data_dir $DATA_DIR
python scripts/preprocess/JaccardDistance.py --data_dir $DATA_DIR
python scripts/preprocess/Normalize_Matrix.py --data_dir $DATA_DIR
```

---

## 🚀 Model Training

### Training with Main Script

`main.py` is the unified entry point for training and evaluation, supporting various configurations:

```bash
python main.py \
    --task imputation \
    --model egcn \
    --data_dir data/processed \
    --epochs 200 \
    --lr 0.001 \
    --hidden_dim 64 \
    --batch_size 32 \
    --device cuda:0 \
    --output_dir results/experiment_1
```

### Main Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--task` | Task type (imputation/prediction) | imputation |
| `--model` | Model type (egcn/egsage) | egcn |
| `--data_dir` | Preprocessed data directory | data/processed |
| `--epochs` | Number of training epochs | 200 |
| `--lr` | Learning rate | 0.001 |
| `--hidden_dim` | Hidden layer dimension | 64 |
| `--batch_size` | Batch size | 32 |
| `--device` | Computing device (cuda:0/cpu) | cuda:0 |
| `--output_dir` | Output directory (save models and results) | results |

### Training Output

During training, the following will be saved in the specified `output_dir`:

- **Training Log**: Records train loss and validation loss for each epoch
- **Best Model**: `best_model.pt` - Model with best performance on validation set
- **Checkpoints**: `checkpoint_epoch_*.pt` - Periodically saved checkpoints
- **Evaluation Results**: Final evaluation metrics on test set

---

## 📚 Core Modules

### 1. Data Processing Module (`data_process/`)

- **`data_load.py`**: 
  - Load data from preprocessed files
  - Build PyTorch Geometric data objects
  - Split train/validation/test sets
  
- **`data_subparser.py`**: 
  - Data-related command line argument management

### 2. Model Module (`models/`)

- **`gnn_model.py`**: 
  - General GNN model wrapper
  - Support multi-layer GNN stack
  - Integrate temporal information and edge features

- **`egcn.py`**: 
  - Edge-based Graph Convolutional Network
  - Enhanced edge feature modeling on standard GCN

- **`egsage.py`**: 
  - Edge-based GraphSAGE
  - Sampling aggregation strategy, suitable for large-scale graphs

- **`longitudinal_network.py`**: 
  - Specifically designed for longitudinal data
  - Information propagation between time steps
  - Inter-subject relationship modeling

- **`prediction_model.py`**: 
  - Prediction head wrapper
  - Support both covariate imputation and response prediction tasks

### 3. Training Module (`training/`)

- **`gnn_y.py`**: 
  - GNN training main pipeline
  - Loss calculation, gradient updates
  - Early stopping mechanism

- **`baseline.py`**: 
  - Non-GNN baseline methods

- **`linear_regression.py`**: 
  - Linear regression baseline implementation

- **`subject_mapping.py`**: 
  - Subject ID to node ID mapping

- **`WeightGraph.py`**: 
  - Construct weighted graph based on similarity

### 4. Utility Module (`utils/`)

- **`utils.py`**: 
  - Logging
  - Evaluation metric calculation (MSE, MAE, R², etc.)
  - File path management

- **`plot_utils.py`**: 
  - Training curve visualization
  - Prediction result visualization

---

## 💡 Usage Examples

### Example 1: Covariate Imputation Task

```bash
# 1. Preprocess data (first run)
bash preprocess_all.sh

# 2. Train GNN imputation model
python main.py \
    --task imputation \
    --model egcn \
    --data_dir data/GLOBEM_processed \
    --epochs 200 \
    --lr 0.001 \
    --output_dir results/imputation_egcn

# 3. Evaluate model
python main.py \
    --task imputation \
    --mode eval \
    --model egcn \
    --checkpoint results/imputation_egcn/best_model.pt \
    --data_dir data/GLOBEM_processed
```

### Example 2: Response Prediction Task

```bash
python main.py \
    --task prediction \
    --model egsage \
    --data_dir data/ADNI_processed \
    --epochs 200 \
    --lr 0.001 \
    --output_dir results/prediction_egsage
```

### Example 3: Baseline Model Comparison

```bash
# Linear regression baseline
python scripts/baselines/linear_regression_y_baseline.py \
    --data_dir data/processed \
    --output_dir results/baseline_lr
```

---

## 📈 Experimental Results

After training, results will be saved in the specified output directory:

```text
results/experiment_1/
├── train_log.txt          # Training log
├── best_model.pt          # Best model parameters
├── metrics.json           # Evaluation metrics
├── loss_curve.png         # Loss curve plot
└── predictions.png        # Prediction result visualization
```

### Evaluation Metrics

- **Imputation Task**: MSE, RMSE, MAE, R²
- **Prediction Task**: MSE, RMSE, MAE, R², AUC (if applicable)

---

## 🔨 File Reorganization

If you haven't organized the file structure yet, you can use the following script:

Create `reorganize_structure.sh` in the `SHT-GNN` directory:

```bash
#!/bin/bash

echo "Reorganizing SHT-GNN file structure..."

# Create directories
mkdir -p scripts/preprocess
mkdir -p scripts/baselines

# Move preprocessing scripts
mv Data_preprocessing.py scripts/preprocess/ 2>/dev/null
mv Edge_index_generate.py scripts/preprocess/ 2>/dev/null
mv IndiceGenerate.py scripts/preprocess/ 2>/dev/null
mv TimeConvert.py scripts/preprocess/ 2>/dev/null
mv Time_Decay_weight.py scripts/preprocess/ 2>/dev/null
mv Normalize_Matrix.py scripts/preprocess/ 2>/dev/null
mv Response_gerenate_16.py scripts/preprocess/ 2>/dev/null
mv JaccardDistance.py scripts/preprocess/ 2>/dev/null

# Move baseline scripts
mv linear_regression_y_baseline.py scripts/baselines/ 2>/dev/null

echo "File reorganization complete!"
echo ""
echo "New structure:"
echo "scripts/"
echo "  ├── preprocess/    (Data preprocessing scripts)"
echo "  └── baselines/     (Baseline model scripts)"
```

Run the script:

```bash
cd /path/to/SHT-GNN
chmod +x reorganize_structure.sh
./reorganize_structure.sh
```

---

## 🎯 Key Features

1. **Temporal Modeling**: Integrates time decay weights to capture temporal dependencies
2. **Heterogeneous Graphs**: Supports multiple node and edge types
3. **Scalability**: Sampling strategy adapts to large-scale data
4. **Flexibility**: Supports various GNN architectures and tasks
5. **Complete Pipeline**: End-to-end workflow from data preprocessing to model evaluation

---

## 📖 Citation

If this project helps your research, please cite:

```bibtex
@article{sht-gnn-2025,
  title={Sampling-guided Heterogeneous Graph Neural Network with Temporal Smoothing for Scalable Longitudinal Data Imputation},
  author={Your Name},
  journal={Conference/Journal Name},
  year={2025}
}
```

---

## 📧 Contact

For questions or suggestions, please contact:

- Email: zzhaoy@[your-domain]
- GitHub Issues: [Project Issues Page]

---

## 📄 License

MIT License


