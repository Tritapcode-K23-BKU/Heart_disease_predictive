# Heart Disease Prediction

A machine learning project that predicts the likelihood of heart disease using three classification algorithms: **Decision Tree**, **Logistic Regression**, and **Random Forest**. The pipeline covers end-to-end data preprocessing, class balancing, feature selection, hyperparameter tuning, and model evaluation.

---

## Project Structure

```
Heart_disease_predictive-main/
├── data/
│   ├── heart_train_cleaned.csv     # Preprocessed training set (814 samples, post-SMOTE)
│   └── heart_test_cleaned.csv      # Preprocessed test set (185 samples)
├── eda_plots/
│   ├── 1_target_distribution.png
│   ├── 2_numeric_boxplots.png
│   ├── 3_numeric_histograms.png
│   ├── 4_correlation_heatmap.png
│   ├── 5_target_before_after_smote.png
│   └── 6_pca_space_comparison.png
├── models/
│   ├── preprocessor.pkl            # Fitted sklearn ColumnTransformer pipeline
│   ├── selector.pkl                # Fitted SelectKBest feature selector
│   ├── decision_tree_model.pkl
│   ├── logistic_regression_model.pkl
│   └── rf_model.pkl
├── data_preprocess.py              # Full EDA + preprocessing pipeline
├── decision_tree.py                # Decision Tree training + tuning
├── Logistic_Regression.py          # Logistic Regression training
└── Random_Forest.py                # Random Forest training + GridSearchCV
```

---

## Pipeline Overview

### 1. Data Preprocessing (`data_preprocess.py`)

- **Source**: Raw dataset loaded from Google Drive (UCI Heart Disease dataset)
- **Target binarization**: `num > 0 → 1` (disease), `num = 0 → 0` (no disease)
- **EDA**: Descriptive statistics, boxplots, histograms, correlation heatmap
- **Missing value imputation**: `KNNImputer` (numeric), `SimpleImputer` (categorical)
- **Scaling**: `RobustScaler` (resistant to outliers)
- **Encoding**: `OneHotEncoder` with `drop='first'` for categorical features
- **Class balancing**: `SMOTE` to address minority class imbalance
- **Feature selection**: `SelectKBest` with `mutual_info_classif` (top 15 features)
- **Correlation filter**: Drop features with Pearson correlation > 0.9

### 2. Decision Tree (`decision_tree.py`)

- Baseline model evaluation (unconstrained tree → detects overfitting)
- **Coarse tuning**: `RandomizedSearchCV` over 200 combinations (CV=10, F1 scoring)
- **Fine tuning**: `GridSearchCV` in a narrow grid around the best parameters
- **Threshold tuning**: Iterates thresholds 0.25–0.70, optimizing 50% F1 + 50% Accuracy
- Outputs: Confusion Matrix, ROC Curve, Feature Importance, Tree visualization (3 levels)

### 3. Logistic Regression (`Logistic_Regression.py`)

- Baseline `LogisticRegression` with `max_iter=1000`
- Outputs: Confusion Matrix, ROC Curve, AUC score

### 4. Random Forest (`Random_Forest.py`)

- `GridSearchCV` over: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features` (CV=10)
- 10-fold cross-validation on training set
- Outputs: Confusion Matrix, ROC Curve, Top-15 Feature Importance

---

## 🔑 Selected Features (Top 15 via Mutual Information)

| Feature | Description |
|---|---|
| `num__ca` | Number of major vessels colored by fluoroscopy |
| `cat__thal_reversable defect` | Thalassemia type |
| `cat__cp_atypical angina` | Chest pain type |
| `cat__exang_True` | Exercise-induced angina |
| `num__oldpeak` | ST depression induced by exercise |
| `cat__slope_flat` | Slope of the peak exercise ST segment |
| `num__thalch` | Maximum heart rate achieved |
| `cat__sex_Male` | Sex |
| `num__age` | Age |
| `cat__cp_non-anginal` | Non-anginal chest pain |
| `num__chol` | Serum cholesterol |
| `cat__slope_upsloping` | Upsloping ST segment |
| `cat__thal_normal` | Normal thalassemia |
| `cat__fbs_True` | Fasting blood sugar > 120 mg/dl |
| `cat__restecg_st-t abnormality` | Resting ECG result |

---

##  Dataset

| Split | Samples | Notes |
|---|---|---|
| Train | 814 | After SMOTE balancing |
| Test | 185 | Original distribution (no SMOTE) |

- **15 features** retained after preprocessing and feature selection
- Raw data sourced from the [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+disease)

---

##  Getting Started

### Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn joblib
```

### Run Order

Steps must be executed sequentially:

```bash
# Step 1 — Preprocess data and generate cleaned CSVs + saved artifacts
python data_preprocess.py

# Step 2 — Train and evaluate models (any order)
python decision_tree.py
python Logistic_Regression.py
python Random_Forest.py
```

> **Note:** `data_preprocess.py` loads raw data from Google Drive. Update the `file_id` variable if using a different dataset source.

### Load a Saved Model

```python
import joblib
import pandas as pd

# Load artifacts
preprocessor = joblib.load('models/preprocessor.pkl')
selector      = joblib.load('models/selector.pkl')
model         = joblib.load('models/rf_model.pkl')

# Predict on new raw data
X_new_pre   = preprocessor.transform(X_new)
X_new_final = selector.transform(X_new_pre)
predictions  = model.predict(X_new_final)
```

---

##  Evaluation Metrics

Each model is evaluated on:

- **Accuracy** — Overall correctness
- **ROC-AUC** — Ability to discriminate between classes
- **Sensitivity (Recall)** — Correctly identified disease cases *(critical for medical diagnosis)*
- **Specificity** — Correctly identified healthy cases
- **F1-Score** — Balance between Precision and Recall
- **Confusion Matrix** — Breakdown of TP, TN, FP, FN

---

##  Tech Stack

- **Language**: Python 3
- **ML Framework**: scikit-learn
- **Class Balancing**: imbalanced-learn (SMOTE)
- **Data**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Model Persistence**: joblib
- **Environment**: Google Colab / Local Python

---

##  Report

A full technical report (`REPORT_BTL_N10_L03.pdf`) is included with detailed methodology, experimental results, and analysis.
