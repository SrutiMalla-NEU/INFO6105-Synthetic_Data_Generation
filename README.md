# DistVAE: Distributional Learning VAE for Synthetic Tabular Data

This project implements the **"Distributional Learning of Variational AutoEncoder"** approach for generating high-quality synthetic tabular data, based on the paper by An and Jeon (NeurIPS 2023).

## Project Overview

Traditional synthetic data generation approaches often struggle with complex real-world data distributions. DistVAE overcomes these limitations by:

- Directly estimating conditional cumulative distribution functions (CDFs)
- Using spline-based quantile functions to model complex distributions
- Balancing statistical accuracy with strong privacy guarantees

Our implementation supports multiple datasets and provides comprehensive evaluation of synthetic data quality.

## Requirements

- `torch >= 1.8.0`
- `numpy >= 1.19.0`
- `pandas >= 1.0.0`
- `matplotlib >= 3.3.0`
- `scikit-learn >= 0.24.0`
- `statsmodels >= 0.12.0`
- `tqdm >= 4.50.0`

Install the required libraries:

```bash
pip install -r requirements.txt
```

## Setup

1. Create directories for results:

```bash
mkdir -p modules
mkdir -p data
mkdir -p assets
mkdir -p privacy
```

2. Download and process datasets:

Example for the Adult dataset:

```bash
python data_preparation.py --dataset adult
```

## Project Structure

- `main.py`: Main training script (without WandB)
- `simple_train.py`: Simplified training script
- `simple_inference.py`: Evaluates the quality of estimated distributions
- `simple_calibration.py`: Implements calibration for ordinal variables
- `simple_synthesize.py`: Evaluates synthetic data quality
- `shadow_data.py`, `shadow_main.py`, `shadow_attack.py`: Privacy evaluation pipeline

### modules/ (Core Modules)

- `model.py`: Implementation of the DistVAE model
- `train.py`: Training loop
- `evaluation.py`: Quality metrics and evaluation functions
- `*_datasets.py`: Dataset-specific processing modules

## Running the Pipeline

### 1. Train the model and generate synthetic data

```bash
python simple_train.py --dataset adult --epochs 10 --batch_size 256 --latent_dim 2
```

### 2. Evaluate distribution estimation

```bash
python simple_inference.py --dataset adult
```

### 3. Apply calibration for ordinal variables

```bash
python simple_calibration.py --dataset adult
```

### 4. Evaluate synthetic data quality

```bash
python simple_synthesize.py --dataset adult
```

### 5. Run privacy evaluation

```bash
python shadow_data.py --dataset adult
python shadow_main.py --dataset adult
python shadow_attack.py --dataset adult
```

## Supported Datasets

- Adult Census Income
- Forest Cover Type
- Home Credit Default Risk
- Personal Loan Modeling
- Taxi Pricing
- King County House Sales

## Evaluation Metrics

The synthetic data is evaluated on:

- **Statistical Similarity**:  
  - K-S statistic
  - Wasserstein distance
  - Correlation structure
- **Machine Learning Utility**:  
  - Regression (Mean Absolute Relative Error - MARE)
  - Classification (F1 Score)
- **Privacy Metrics**:  
  - DCR (Disclosure Control Risk)
  - Attribute disclosure risk
  - Membership inference attack resistance

## Results

Model outputs are saved in:

- `assets/DistVAE_{dataset}.pth`: Trained model weights
- `assets/synthetic_{dataset}.csv`: Generated synthetic data
- `assets/{dataset}/`: Visualizations and metrics
- `assets/results/`: Summary evaluation results
- `privacy/results/`: Privacy evaluation metrics


## Citation

An, S., & Jeon, J. (2023). Distributional Learning of Variational AutoEncoder: Application to Synthetic Data Generation. NeurIPS 2023.
Would you like me to show that too? 🎨 (makes the top part look fancy)
