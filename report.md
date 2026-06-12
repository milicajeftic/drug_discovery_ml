# Predictive Modeling for Drug Discovery via Virtual Screening

## 1. Introduction

The objective of this project is to build and evaluate machine learning models that predict whether a chemical compound is biologically active or inactive against a target. The task is formulated as a binary classification problem, where `1` represents an active compound and `0` represents an inactive compound.

This type of prediction is relevant for virtual screening in drug discovery. In a real workflow, computational models can help prioritize promising compounds before laboratory testing. The goal of this project is not to claim that a new drug candidate has been discovered, but to compare different machine learning models for activity prediction and analyze their behavior.

## 2. Dataset

The dataset used in this project is the Kaggle Drug Discovery Virtual Screening Dataset:

https://www.kaggle.com/datasets/shahriarkabir/drug-discovery-virtual-screening-dataset

The project code repository is available at:

https://github.com/milicajeftic/drug_discovery_ml

The raw dataset contains 2,000 compounds and 17 columns. These include:

- 14 numeric modeling features
- 2 identifier columns: `compound_id` and `protein_id`
- 1 target column: `active`

The target variable is moderately imbalanced:

- Inactive compounds: approximately 70%
- Active compounds: approximately 30%

Three columns contain missing values:

- `logp`
- `polar_surface_area`
- `hydrophobicity`

Each of these columns has 60 missing values. Because the missing values do not occur in exactly the same rows, removing rows with any missing value leaves 1,826 complete samples.

## 3. Preprocessing

The preprocessing workflow is implemented in `01_eda_and_preprocessing.ipynb`.

The main preprocessing steps are:

1. Remove rows with missing values.
2. Remove identifier columns from the feature matrix.
3. Separate features from the `active` target variable.
4. Create stratified train/validation/test splits using a 70/15/15 ratio.
5. Fit `StandardScaler` only on the training set.
6. Transform validation and test sets using the training-fitted scaler.

Fitting the scaler only on the training set avoids data leakage. The processed data is saved locally in `data/processed/`, and the fitted scaler is saved in `models/standard_scaler.joblib`.

## 4. Models

Four model types were trained and evaluated.

### Logistic Regression

Logistic Regression was used as the baseline model. It is simple, interpretable, and provides a useful reference point for more complex models. Balanced class weights were used to account for the moderate class imbalance.

### Random Forest

Random Forest was used as a non-linear ensemble model. It can capture feature interactions and provides feature importance scores. Balanced class weights were used during training.

### Gradient Boosting

Gradient Boosting was used as another tree-based ensemble method. Balanced sample weights were used because the model does not directly support `class_weight`.

### Neural Network

A feed-forward neural network was implemented using `sklearn.neural_network.MLPClassifier`. This model uses two hidden layers, ReLU activation, L2 regularization, early stopping, and balanced sample weights. This satisfies the neural network component proposed in the project while remaining compatible with the project `.conda` environment.

## 5. Evaluation Metrics

All models were evaluated using the same metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

Recall is especially important in virtual screening because false negatives represent active compounds that the model fails to identify.

Learning curves were also analyzed using F1-score to compare training and cross-validation performance as the training set size increases. They show that Random Forest and Gradient Boosting remain consistently strong, Logistic Regression improves gradually with more data, and the Neural Network is more variable on smaller training subsets.

## 6. Results

Final test-set performance:

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9891 | 0.9651 | 1.0000 | 0.9822 | 0.9998 |
| Random Forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Gradient Boosting | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Neural Network | 0.9854 | 0.9647 | 0.9880 | 0.9762 | 0.9993 |

The best-performing models are Random Forest and Gradient Boosting. Both achieved perfect validation and test performance on the current processed dataset.

## 7. Error Analysis

The test-set error analysis shows:

| Model | False Positives | False Negatives | Total Errors |
|---|---:|---:|---:|
| Logistic Regression | 3 | 0 | 3 |
| Random Forest | 0 | 0 | 0 |
| Gradient Boosting | 0 | 0 | 0 |
| Neural Network | 3 | 1 | 4 |

Random Forest and Gradient Boosting made no test-set errors. Logistic Regression made three false positive errors but no false negatives, which is favorable in a virtual screening context because it did not miss active compounds. The Neural Network made four errors, including one false negative.

## 8. Interpretability

Feature importance analysis was performed for the tree-based models.

The most important Random Forest features were:

1. `binding_affinity`
2. `logp_pi_interaction`
3. `logp`

For Gradient Boosting, `binding_affinity` dominated the feature importance almost completely.

This suggests that the activity label is strongly related to binding affinity and closely related descriptors. This also explains why the tree-based models achieved perfect performance.

## 9. Limitations

The main limitation of this project is related to the dataset. The dataset appears to be synthetic and highly separable. Since `binding_affinity` is included as an input feature and is strongly associated with the target label, the classification task may be easier than a real-world virtual screening problem.

In real drug discovery, binding affinity may not always be known before prediction. If the activity label was derived from binding affinity, using binding affinity as an input feature can overestimate model performance.

Therefore, the results should be interpreted as a comparison of model behavior on this dataset, not as proof that the same performance would be achieved on real experimental drug discovery data.

Another limitation is that only a limited hyperparameter search was performed. The model parameters were selected manually and kept fixed during the final comparison. Since the models already achieved very high validation and test performance, an extensive grid search was not necessary for the main goal of comparing model families, but it could be added in future work.

## 10. Conclusion

This project successfully developed and evaluated a machine learning pipeline for predicting compound activity in a virtual screening setting. The workflow includes data checking, preprocessing, baseline modeling, tree-based modeling, neural network modeling, model comparison, and error analysis.

The tree-based models performed best, with Random Forest and Gradient Boosting achieving perfect test-set results. Logistic Regression also performed very strongly and served as a reliable baseline. The Neural Network achieved high performance but did not outperform the tree-based models.

The final conclusion is that tree-based ensemble models are the best choice for this dataset. However, the dominant role of `binding_affinity` should be discussed as an important limitation when interpreting the results.
