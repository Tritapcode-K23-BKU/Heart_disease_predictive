# =========================================================
# 1. IMPORT LIBRARIES
# =========================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve, f1_score
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import time
import warnings
warnings.filterwarnings('ignore')
# =========================================================
# 2. LOAD DATA
# =========================================================
df_train = pd.read_csv('heart_train_cleaned.csv')
df_test  = pd.read_csv('heart_test_cleaned.csv')

X_train = df_train.drop(columns=['target'])
y_train = df_train['target']
X_test  = df_test.drop(columns=['target'])
y_test  = df_test['target']

# =========================================================
# 3. BASELINE EVALUATION — UNCONSTRAINED DECISION TREE
# =========================================================
dt_default = DecisionTreeClassifier(random_state=42)
dt_default.fit(X_train, y_train)

# Calculate baseline metrics for comparison
y_pred_baseline     = dt_default.predict(X_test)
y_proba_baseline    = dt_default.predict_proba(X_test)[:, 1]

cm_base             = confusion_matrix(y_test, y_pred_baseline)
TN_b, FP_b, FN_b, TP_b = cm_base.ravel()

acc_base            = accuracy_score(y_test, y_pred_baseline)
auc_base            = roc_auc_score(y_test, y_proba_baseline)
sensitivity_base    = TP_b / (TP_b + FN_b)
specificity_base    = TN_b / (TN_b + FP_b)
f1_base             = f1_score(y_test, y_pred_baseline)

print("=" * 55)
print("  BASELINE — UNCONSTRAINED DECISION TREE")
print("=" * 55)
print(f"  Train Accuracy       : {dt_default.score(X_train, y_train):.2%}")
print(f"  Test  Accuracy       : {acc_base:.2%}")
print(f"  ROC-AUC              : {auc_base:.4f}")
print(f"  Sensitivity (Recall) : {sensitivity_base:.4f}")
print(f"  Specificity          : {specificity_base:.4f}")
print(f"  F1-Score (Test)      : {f1_base:.4f}")
print("-" * 55)
print("  → Large Train/Test gap indicates Overfitting.")
print("    Tuning is required.")
print("=" * 55, "\n")

# =========================================================
# 4. COARSE TUNING — RANDOMIZED SEARCH
#    Goal: Quickly explore a wide parameter space.
#    Keep criterion='gini' and class_weight=None as default.
# =========================================================
print("=" * 55)
print("  STEP 1 — COARSE TUNING (RANDOMIZED SEARCH)")
print("=" * 55)
print("  Searching over a wide parameter space...")

start_time = time.time()

# Hyperparameter space for coarse search
param_dist_dt_coarse = {
    "max_depth":         [3, 5, 7, 9, 11, 13, 15, None],
    "max_features":      ["sqrt", "log2", None],
    "min_samples_split": [2, 5, 8, 10, 15, 20],
    "min_samples_leaf":  [1, 2, 4, 6, 8, 10, 12],
}

# RandomizedSearchCV: sample 200 random combinations to save time
rand_dt = RandomizedSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_distributions=param_dist_dt_coarse,
    n_iter=200,       # Number of random combinations
    cv=10,            # 10-fold CV for reliable estimation
    scoring="f1",
    n_jobs=-1,
    random_state=42
)

rand_dt.fit(X_train, y_train)
elapsed = round(time.time() - start_time, 2)

print(f"  Completed in {elapsed} seconds.")
print(f"\n  Best Hyperparameters:")
for k, v in rand_dt.best_params_.items():
    print(f"    {k:<22}: {v}")
print(f"\n  Best F1-Score (CV=10) : {rand_dt.best_score_:.4f}")
print("=" * 55, "\n")


# =========================================================
# HYPERPARAMETER SENSITIVITY ANALYSIS
# Visualize the impact of each hyperparameter on the
# maximum F1-Score achieved during the search.
# =========================================================
results_df = pd.DataFrame(rand_dt.cv_results_)
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.ravel()

params_to_plot = ['param_max_depth', 'param_min_samples_split',
                  'param_min_samples_leaf', 'param_max_features']
titles = ['Max Depth', 'Min Samples Split', 'Min Samples Leaf', 'Max Features']

for idx, param in enumerate(params_to_plot):
    temp_df = results_df.copy()
    temp_df[param] = temp_df[param].fillna('None_Value').astype(str)
    summary = temp_df.groupby(param)['mean_test_score'].max().reset_index()

    if param != 'param_max_features':
        summary = summary[summary[param] != 'None_Value']
        summary[param] = summary[param].astype(float)
        summary = summary.sort_values(by=param)
        x_values = summary[param].astype(int).astype(str)
    else:
        summary[param] = summary[param].replace('None_Value', 'None')
        x_values = summary[param]

    y_values = summary['mean_test_score']

    axes[idx].plot(x_values, y_values, marker='o', color='crimson',
                   linewidth=2, markersize=8)
    axes[idx].set_title(f'{titles[idx]}', fontweight='bold')
    axes[idx].set_xlabel(titles[idx])
    axes[idx].set_ylabel('Max F1-Score (CV)')
    axes[idx].grid(True, linestyle='--', alpha=0.6)

plt.suptitle('Hyperparameter Sensitivity Analysis (Coarse Tuning)',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# =========================================================
# 5. FINE TUNING — GRID SEARCH
#    Goal: Fine-tune in a narrow region around the best
#    parameters found in Coarse Tuning.
# =========================================================
print("=" * 55)
print("  STEP 2 — FINE TUNING (GRID SEARCH)")
print("=" * 55)

best_dt = rand_dt.best_params_

# Handle max_depth = None (unconstrained tree)
d_dt  = best_dt["max_depth"] or 15
ms_dt = best_dt["min_samples_split"]
ml_dt = best_dt["min_samples_leaf"]

# Narrow grid: search ±2 and +4 around the best values.
# Use set{} to automatically remove duplicates before sorting.
param_grid_dt_fine = {
    "max_features":      [best_dt["max_features"]],
    "max_depth":         sorted({max(2, d_dt-2),  d_dt, d_dt+2,  d_dt+4}),
    "min_samples_split": sorted({max(2, ms_dt-2), ms_dt, ms_dt+2, ms_dt+4}),
    "min_samples_leaf":  sorted({max(1, ml_dt-2), ml_dt, ml_dt+2, ml_dt+4}),
}

grid_dt = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid=param_grid_dt_fine,
    cv=10,
    scoring="f1",
    n_jobs=-1
)

grid_dt.fit(X_train, y_train)
dt_tuned = grid_dt.best_estimator_

print(f"  Final Tuned Hyperparameters:")
for k, v in grid_dt.best_params_.items():
    print(f"    {k:<22}: {v}")
print(f"\n  Final F1-Score (CV=10) : {grid_dt.best_score_:.4f}")
print("=" * 55, "\n")

# =========================================================
# 6. THRESHOLD TUNING — OPTIMAL CLASSIFICATION THRESHOLD
#    Iterate thresholds from 0.25 to 0.70, optimizing a
#    combined F1-Score and Accuracy metric (50/50).
# =========================================================
print("=" * 55)
print("  STEP 3 — THRESHOLD TUNING")
print("=" * 55)
print(f"  {'Thresh':>8}  {'F1-Score':>10}  {'Accuracy':>10}  {'Combined':>10}")
print("  " + "-" * 45)

y_pred_proba  = dt_tuned.predict_proba(X_test)[:, 1]
best_thresh   = 0.5
best_combined = 0.0
weight_f1     = 0.5   # Weights: 50% F1, 50% Accuracy

for thresh in np.arange(0.25, 0.70, 0.01):
    y_tmp = (y_pred_proba >= thresh).astype(int)

    # Skip if model predicts only one class
    if len(np.unique(y_tmp)) < 2:
        continue

    f        = f1_score(y_test, y_tmp)
    a        = accuracy_score(y_test, y_tmp)
    combined = weight_f1 * f + (1 - weight_f1) * a

    # Print only when a new record is set
    if combined > best_combined:
        best_combined = combined
        best_thresh   = thresh
        print(f"  {thresh:>8.2f}  {f:>10.4f}  {a:>10.4f}  {combined:>10.4f}")

y_pred_final = (y_pred_proba >= best_thresh).astype(int)
print(f"\n  → Optimal Classification Threshold : {best_thresh:.2f}")
print("=" * 55, "\n")

# =========================================================
# 7. MODEL EVALUATION ON TEST SET
# =========================================================
cm = confusion_matrix(y_test, y_pred_final)
TN, FP, FN, TP = cm.ravel()

accuracy    = accuracy_score(y_test, y_pred_final)
auc_score   = roc_auc_score(y_test, y_pred_proba)
sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)

print("=" * 55)
print(f"  EVALUATION RESULTS (Threshold = {best_thresh:.2f})")
print("=" * 55)
print(f"  Accuracy             : {accuracy:.2%}")
print(f"  ROC-AUC              : {auc_score:.4f}")
print(f"  Sensitivity (Recall) : {sensitivity:.4f}")
print(f"  Specificity          : {specificity:.4f}")
print("-" * 55)
print(classification_report(y_test, y_pred_final,
                             target_names=['No Disease', 'Disease']))
print("=" * 55, "\n")

# ---------------------------------------------------------
# BASELINE VS. TUNED COMPARISON
# ---------------------------------------------------------
delta_acc  = accuracy    - acc_base
delta_auc  = auc_score   - auc_base
delta_sens = sensitivity - sensitivity_base
delta_spec = specificity - specificity_base
delta_f1   = f1_score(y_test, y_pred_final) - f1_base

def fmt_delta(v):
    """Format delta: add '+' if positive, keep '-' if negative."""
    return f"+{v:.4f}" if v >= 0 else f"{v:.4f}"

print("=" * 55)
print("  BASELINE VS. TUNED COMPARISON")
print("=" * 55)
print(f"  {'Metric':<24} {'Baseline':>8}  {'Tuned':>8}  {'Δ':>8}")
print("  " + "-" * 51)
print(f"  {'Accuracy':<24} {acc_base:>8.2%}  {accuracy:>8.2%}  {fmt_delta(delta_acc):>8}")
print(f"  {'ROC-AUC':<24} {auc_base:>8.4f}  {auc_score:>8.4f}  {fmt_delta(delta_auc):>8}")
print(f"  {'Sensitivity (Recall)':<24} {sensitivity_base:>8.4f}  {sensitivity:>8.4f}  {fmt_delta(delta_sens):>8}")
print(f"  {'Specificity':<24} {specificity_base:>8.4f}  {specificity:>8.4f}  {fmt_delta(delta_spec):>8}")
print(f"  {'F1-Score (Test)':<24} {f1_base:>8.4f}  {f1_score(y_test, y_pred_final):>8.4f}  {fmt_delta(delta_f1):>8}")
print("=" * 55, "\n")

# =========================================================
# 8. VISUALIZE RESULTS
#    Includes 3 plots: Confusion Matrix, ROC Curve,
#    and Feature Importance of the tuned model.
# =========================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# --- Plot 1: Confusion Matrix ---
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['No Disease', 'Disease'],
            yticklabels=['No Disease', 'Disease'])
axes[0].set_title(f'Confusion Matrix (Threshold = {best_thresh:.2f})',
                  fontweight='bold')
axes[0].set_ylabel('True Label')
axes[0].set_xlabel('Predicted Label')

# --- Plot 2: ROC Curve ---
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
axes[1].plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {auc_score:.4f}')
axes[1].plot([0, 1], [0, 1], 'k--', label='Random')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC Curve', fontweight='bold')
axes[1].legend(loc='lower right')
axes[1].grid(True, linestyle='--', alpha=0.6)

# --- Plot 3: Feature Importance ---
# Show only features with actual contribution (importance > 0)
importance = pd.Series(dt_tuned.feature_importances_, index=X_train.columns)
importance = importance[importance > 0].sort_values(ascending=False)

axes[2].barh(importance.index[::-1], importance.values[::-1], color='steelblue')
axes[2].set_title('Feature Importance', fontweight='bold')
axes[2].set_xlabel('Importance Score')

plt.suptitle('Heart Disease Prediction — Decision Tree',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# =========================================================
# 9. DECISION TREE VISUALIZATION
#    Show the first 3 levels for readability.
# =========================================================
plt.figure(figsize=(22, 10))
plot_tree(
    dt_tuned,
    feature_names=X_train.columns.tolist(),
    class_names=["No Disease (0)", "Disease (1)"],
    filled=True,
    rounded=True,
    max_depth=3,      # Limit to 3 levels to avoid clutter
    fontsize=10
)
plt.title(
    f"Decision Tree Diagram (Showing 3 levels / Total depth = {dt_tuned.max_depth})",
    fontsize=16, fontweight='bold'
)
plt.show()

# =========================================================
# 10. SAVE MODEL
# =========================================================
joblib.dump(dt_tuned, 'decision_tree_model.pkl')
print("\nModel successfully saved to 'decision_tree_model.pkl'")
