import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.model_selection import GridSearchCV, cross_val_score

# LOAD PREPROCESSED DATA

df_train = pd.read_csv('heart_train_cleaned.csv')
df_test  = pd.read_csv('heart_test_cleaned.csv')

X_train = df_train.drop(columns=['target'])
y_train = df_train['target']

X_test = df_test.drop(columns=['target'])
y_test = df_test['target']

print(f"Train: {X_train.shape} | Test: {X_test.shape}")

# HYPERPARAMETER TUNING
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=10,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train) # Train model

best_params = grid_search.best_params_
rf_model    = grid_search.best_estimator_

print(f"Bộ tham số tốt nhất: {best_params}")
print(f"Model tốt nhất: {rf_model}")

joblib.dump(rf_model, 'rf_model.pkl') # Save file

# PREDICT & EVALUATE
y_pred       = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

cm = confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = cm.ravel()

accuracy    = accuracy_score(y_test, y_pred)
auc         = roc_auc_score(y_test, y_pred_proba)
sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)

print(f"\n{'='*45}")
print("KẾT QUẢ ĐÁNH GIÁ")
print(f"{'='*45}")
print(f"Accuracy:             {round(accuracy * 100, 2)}%")
print(f"ROC-AUC:              {round(auc, 4)}")
print(f"Sensitivity (Recall): {round(sensitivity, 4)}")
print(f"Specificity:          {round(specificity, 4)}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred,
      target_names=['Không bệnh', 'Có bệnh']))

cv_scores = cross_val_score(rf_model, X_train, y_train, cv=10, scoring='accuracy')

print(f"\n{'='*45}")
print("K-FOLD CROSS-VALIDATION (K=10)")
print(f"{'='*45}")
print(f"Các mức Accuracy từng Fold: {cv_scores}")
print(f"Accuracy trung bình (Mean): {round(cv_scores.mean() * 100, 2)}%")
print(f"Độ lệch chuẩn (Std Dev):    {round(cv_scores.std() * 100, 2)}%")

# VISUALIZATION

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# --- Confusion Matrix ---
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Không bệnh', 'Có bệnh'],
            yticklabels=['Không bệnh', 'Có bệnh'])
axes[0].set_title('Confusion Matrix', fontweight='bold')
axes[0].set_ylabel('Thực tế')
axes[0].set_xlabel('Dự đoán')

# --- ROC Curve ---
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
axes[1].plot(fpr, tpr, color='blue', lw=2,
             label=f'AUC = {round(auc, 4)}')
axes[1].plot([0, 1], [0, 1], 'k--', label='Random')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC Curve', fontweight='bold')
axes[1].legend()

# --- Feature Importance ---
feat_imp = pd.Series(rf_model.feature_importances_,
                     index=X_train.columns) \
             .sort_values(ascending=False).head(15)

axes[2].barh(feat_imp.index[::-1], feat_imp.values[::-1], color='steelblue')
axes[2].set_title('Top 15 Feature Importance', fontweight='bold')
axes[2].set_xlabel('Importance Score')

plt.suptitle('Heart Disease Prediction — Random Forest',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('rf_evaluation.png', bbox_inches='tight', dpi=150)
plt.show()
plt.close()
