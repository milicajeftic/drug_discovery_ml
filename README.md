# Predictive Modeling for Drug Discovery via Virtual Screening

## Project Overview

This project applies machine learning models to predict whether chemical compounds are biologically active or inactive against a target. The task is formulated as a binary classification problem:

- `1` = active compound
- `0` = inactive compound

The project simulates a virtual screening workflow used in early-stage drug discovery. The goal is to compare several machine learning algorithms and evaluate which model performs best for prioritizing potentially active compounds before expensive laboratory testing.

## Dataset

Source: Kaggle - Drug Discovery Virtual Screening Dataset  
Link: https://www.kaggle.com/datasets/shahriarkabir/drug-discovery-virtual-screening-dataset

The dataset contains 2,000 compounds described by molecular descriptors, protein-related features, binding affinity measurements, and a binary activity label.

The raw dataset is not stored in this repository because `data/raw/` is ignored by git. To reproduce the project, download the dataset from Kaggle and place it at:

```text
data/raw/drug_discovery_virtual_screening.csv
```

## Methods

The analysis is organized into six Jupyter notebooks:

```text
notebooks/
|-- 00_setup_and_data_check.ipynb
|-- 01_eda_and_preprocessing.ipynb
|-- 02_baseline_logistic_regression.ipynb
|-- 03_tree_models_rf_gb.ipynb
|-- 04_neural_network.ipynb
`-- 05_model_comparison_and_error_analysis.ipynb
```

Models trained and evaluated:

- Logistic Regression baseline
- Random Forest
- Gradient Boosting
- Feed-forward Neural Network using `sklearn.neural_network.MLPClassifier`

All models use the same train/validation/test split prepared in notebook 01. The split is stratified, and feature scaling is fitted only on the training set to avoid data leakage.

## Results

Final test-set performance:

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9891 | 0.9651 | 1.0000 | 0.9822 | 0.9998 |
| Random Forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Gradient Boosting | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Neural Network | 0.9854 | 0.9647 | 0.9880 | 0.9762 | 0.9993 |

The tree-based models achieved the best results on this dataset. Random Forest and Gradient Boosting made no errors on the held-out test set.

## Interpretation

Feature importance analysis showed that `binding_affinity` is the dominant predictor, especially for Gradient Boosting. This suggests that the `active` label is strongly related to binding affinity and associated descriptors.

Because the dataset is synthetic and highly separable, the perfect performance of the tree-based models should be interpreted carefully. In real drug discovery workflows, binding affinity may not always be available before prediction. If the target label was derived from binding affinity, including it as a model input can make the classification task artificially easy.

## Reproducing the Project

1. Clone the repository:

```bash
git clone https://github.com/milicajeftic/drug_discovery_ml.git
cd drug_discovery_ml
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download the dataset from Kaggle and place it in:

```text
data/raw/drug_discovery_virtual_screening.csv
```

4. Run the notebooks in order:

```text
00_setup_and_data_check.ipynb
01_eda_and_preprocessing.ipynb
02_baseline_logistic_regression.ipynb
03_tree_models_rf_gb.ipynb
04_neural_network.ipynb
05_model_comparison_and_error_analysis.ipynb
```

Generated files are saved locally in:

```text
data/processed/
models/
results/
```

These directories are ignored by git because they contain generated data, model files, and result artifacts.

## Project Structure

```text
drug-discovery-virtual-screening/
|-- data/
|   |-- raw/             # Local raw dataset, ignored by git
|   `-- processed/       # Generated processed splits, ignored by git
|-- models/              # Generated trained models, ignored by git
|-- notebooks/           # Main project workflow
|-- results/             # Generated metrics and figures, ignored by git
|-- src/                 # Supporting source-code folder
|-- README.md
|-- requirements.txt
`-- LICENSE
```

## Author

Milica Jeftic  
Bioinformatics student  
University of Primorska - FAMNIT  
Student ID: 89211255

## License

This project is released under the MIT License.
