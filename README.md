# SEMADDA

### Drug–Disease Association Prediction Based on Enhanced Graph Neural Networks

SEMADDA is a research-oriented deep learning project for drug–disease association prediction using graph neural networks (GNNs).

The project implements an enhanced Subgraph-based Link Graph Convolutional Network (SLGCN) framework to investigate potential associations between drugs and diseases. It integrates heterogeneous graph construction, similarity-based feature representations, subgraph extraction, line graph transformation, and graph representation learning.

## Overview

Predicting potential drug–disease associations is an important task in computational biology and drug repurposing.

SEMADDA models drug–disease association prediction as a graph-based link prediction problem. The framework incorporates drug similarity information, disease similarity information, and known drug–disease associations to construct a heterogeneous network.

The implementation explores enhanced graph learning architectures and training strategies for identifying potential drug–disease associations.

## Key Features

* **Heterogeneous Graph Construction:** Integrates drug–drug similarities, disease–disease similarities, and drug–disease associations.
* **Similarity-Based Filtering:** Applies Top-k filtering to similarity matrices for graph construction.
* **Subgraph-Based Link Prediction:** Extracts enclosing subgraphs around candidate associations.
* **Line Graph Transformation:** Converts extracted subgraphs into line graph representations.
* **Enhanced Graph Neural Networks:** Incorporates GraphSAGE-based graph convolution and structure-enhanced message aggregation.
* **Attention-Based Representation Learning:** Includes attention pooling for graph-level feature aggregation.
* **Improved Training Strategies:** Supports residual connections, layer normalization, dropout, label smoothing, gradient accumulation, and learning-rate scheduling.
* **Cross-Validation:** Implements 10-fold cross-validation for model evaluation.

## Project Structure

```text
SEMADDA/
├── Main.py
├── model.py
├── config.py
├── data_loader_integrated.py
├── 1.py
├── dataset/
│   └── B-dataset/
│       ├── drug_drug.csv
│       ├── disease_disease.csv
│       └── drug_disease.csv
└── README.md
```

### File Descriptions

| File                        | Description                                               |
| --------------------------- | --------------------------------------------------------- |
| `Main.py`                   | Main entry point for model training and evaluation        |
| `model.py`                  | Graph neural network architectures and model components   |
| `config.py`                 | Configuration parameters                                  |
| `data_loader_integrated.py` | Integrated data loading and graph preprocessing utilities |
| `1.py`                      | Utility for inspecting MATLAB `.mat` files                |

**Note:** The directory structure above illustrates the expected CSV dataset layout. The main script also imports `util_functions.py`, which must be provided for the training pipeline.

## Requirements

The project is implemented in Python using the following major libraries:

* Python
* PyTorch
* PyTorch Geometric
* NumPy
* SciPy
* Pandas
* Scikit-learn

Install the main dependencies using:

```bash
pip install torch torch-geometric numpy scipy pandas scikit-learn
```

Ensure that the installed PyTorch and PyTorch Geometric versions are compatible with your computing environment.

## Dataset

The framework supports CSV and MATLAB (`.mat`) data formats.

### CSV Format

Place the dataset files in:

```text
dataset/B-dataset/
```

Required files:

* `drug_drug.csv`: Drug–drug similarity matrix
* `disease_disease.csv`: Disease–disease similarity matrix
* `drug_disease.csv`: Known drug–disease association matrix

### MATLAB Format

The MATLAB data loader expects files following these naming conventions:

```text
data/
├── B-dataset-A.mat
├── B-datasetdataset.mat
└── B-dataset-1.mat
```

The dataset name can be changed through the configuration parameters.

## Model Architecture

The implemented framework contains the following components:

1. **Data Preprocessing:** Load similarity and association matrices and construct heterogeneous graph representations.
2. **Subgraph Extraction:** Extract local enclosing subgraphs for candidate drug–disease links.
3. **Line Graph Construction:** Transform extracted subgraphs into line graphs.
4. **Graph Representation Learning:** Apply graph convolution and structure-enhanced message aggregation.
5. **Feature Aggregation:** Combine learned graph features using attention-based or other supported aggregation mechanisms.
6. **Association Prediction:** Predict candidate drug–disease association labels through a classification layer.

The implementation provides several model variants, including:

* Original SLGCN
* Hierarchical SLGCN
* Enhanced SLGCN
* Simplified Enhanced SLGCN

## Usage

### 1. Prepare the Dataset

Place the corresponding input files in the expected dataset directory.

### 2. Run the Model

The main training entry point is:

```bash
python Main.py
```

To specify a dataset and training parameters:

```bash
python Main.py --data-name B-dataset --data-source csv --model-type enhanced --epoch 30
```

To select the hierarchical model:

```bash
python Main.py --data-name B-dataset --model-type hierarchical --use-hierarchy
```

**Note:** These commands describe the intended entry points. The complete training pipeline requires the supporting utility functions and compatible input datasets.

## Evaluation Metrics

The implementation evaluates model performance using:

* **Accuracy (ACC):** Classification accuracy.
* **Area Under the ROC Curve (AUC):** Discrimination ability across classification thresholds.
* **Area Under the Precision–Recall Curve (AUPR):** Precision–recall performance for association prediction.
* **Loss:** Optimization and evaluation loss.

The training pipeline supports 10-fold cross-validation and reports performance statistics across folds.

## Output

During training, experiment outputs are saved to the `runtime/` directory.

Typical output files include:

```text
runtime/
├── config.txt
├── results.npy
├── model_fold_0_best.pth
├── model_fold_1_best.pth
└── ...
```

These files contain experimental configurations, evaluation results, and saved model parameters.

## Development Status

This repository contains a research implementation under active development.

The code is intended for experimentation and further investigation of graph-based drug–disease association prediction methods.

## Authors

CDMBlab

## Contact

For questions or further information, please open an issue in this GitHub repository.

## Disclaimer

This project is intended for academic research purposes only. Predicted drug–disease associations should not be interpreted as clinically validated findings or medical recommendations.
