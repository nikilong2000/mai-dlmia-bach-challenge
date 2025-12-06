# BACH Challenge - Breast Cancer Histology Image Classification

This project implements a deep learning solution for the **ICIAR 2018 BACH Challenge** (Part A), focusing on the automated classification of breast cancer histology images. The system classifies microscopy images into four distinct classes: **Normal**, **Benign**, **In Situ Carcinoma**, and **Invasive Carcinoma**.

## 📌 Project Overview

The solution leverages a **Grand Ensemble** approach, combining multiple deep learning architectures (ResNet101, DenseNet161) trained using **5-Fold Cross-Validation**. The final predictions are generated via a **Soft Voting** mechanism to ensure robustness and high accuracy.

### Key Features

- **Models:** ResNet101 and DenseNet161 (Pre-trained on ImageNet).
- **Normalization:** Comparison of ImageNet statistics vs. custom BACH dataset statistics.
- **Augmentation:** Extensive geometric and photometric augmentations to handle stain variability.
- **Ensemble Strategy:** "Grand Ensemble" of 15 models (3 configurations × 5 folds) using Soft Voting.
- **Resolution:** Analysis of performance at 256x256 vs 512x512 input resolutions.

## 📂 Project Structure

```
bach_challenge/
├── data/                   # Dataset directory (ICIAR 2018)
├── figures/                # Generated plots and confusion matrices
├── results/                # Trained model weights and training history
├── src/                    # Source code
│   ├── config.yaml         # Configuration for hyperparameters and paths
│   ├── dataset.py          # Data loading and splitting logic
│   ├── model.py            # Model definitions (ResNet, DenseNet)
│   ├── train.py            # Training loop implementation
│   ├── ensemble.py         # Ensemble evaluation logic
│   ├── augmentations.py    # Data augmentation pipelines
│   └── utils.py            # Utility functions
├── submission/             # Final report and submission notebooks
│   └── submission.ipynb    # Comprehensive project report
├── evaluate_all_folds.py   # Script to evaluate all ensemble models
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## 🚀 Installation

1.  **Clone the repository:**

    ```bash
    git clone <repository-url>
    cd bach_challenge
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.

    ```bash
    pip install -r requirements.txt
    ```

    _Note: This project uses PyTorch. Ensure you have the correct version installed for your hardware (CUDA/MPS/CPU)._

## 📊 Usage

### 1. Configuration

Modify `src/config.yaml` to adjust hyperparameters, paths, and model settings.

### 2. Training

To train a model (or a set of models for Cross-Validation), use the training script or the provided notebooks.

```python
from src.train import train_k_fold
# Example: Train 5 folds for ResNet101
train_k_fold(model_name="resnet101", normalisation_scheme="imagenet")
```

### 3. Evaluation

To evaluate the trained models and generate confusion matrices:

```bash
python evaluate_all_folds.py
```

To generate summary figures (Confusion Matrices grids):

```bash
python src/create_cm_summary_figure.py
```

### 4. Notebooks

- **`submission/submission.ipynb`**: The main report containing the methodology, experiments, and final results.
- **`test.ipynb`**: Playground for testing individual components.

## 📈 Results

The final **Grand Ensemble** achieved an accuracy of **95.00%** on the stratified hold-out test set.

| Model Configuration        | Mean Accuracy (5-Fold) | Std Dev |
| :------------------------- | :--------------------: | :-----: |
| **ResNet101 (ImageNet)**   |         87.00%         | ± 2.74% |
| **DenseNet161 (ImageNet)** |         91.00%         | ± 3.79% |
| **DenseNet161 (BACH)**     |         92.00%         | ± 4.81% |

## 📝 Author

**Niklas Long Schiefelbein**
DLMIA Course - Semester 3
