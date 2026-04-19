import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve
)

# =========================================================
# 1. ĐỌC DỮ LIỆU ĐÃ PREPROCESSING
# =========================================================
df_train = pd.read_csv('heart_train_cleaned.csv')
df_test  = pd.read_csv('heart_test_cleaned.csv')

X_train = df_train.drop(columns=['target'])
y_train = df_train['target']
X_test  = df_test.drop(columns=['target'])
y_test  = df_test['target']

print(f"Kích thước tập Train: {X_train.shape} | Tập Test: {X_test.shape}")

# =========================================================
# 2. KHỞI TẠO VÀ HUẤN LUYỆN MODEL
# =========================================================
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)

# =========================================================
# 3. DỰ ĐOÁN VÀ ĐÁNH GIÁ TRÊN TẬP TEST
# =========================================================
y_pred      = lr_model.predict(X_test)
y_pred_prob = lr_model.predict_proba(X_test)[:, 1]  # xác suất class 1 (có bệnh)

print("--- KẾT QUẢ ĐÁNH GIÁ LOGISTIC REGRESSION ---")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("ROC-AUC  :", roc_auc_score(y_test, y_pred_prob))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# =========================================================
# 4. TRỰC QUAN HÓA CONFUSION MATRIX
# =========================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Không bệnh (0)', 'Có bệnh (1)'],
            yticklabels=['Không bệnh (0)', 'Có bệnh (1)'],
            ax=axes[0])
axes[0].set_title('Confusion Matrix — Logistic Regression')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

# --- ROC Curve ---
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
auc_score   = roc_auc_score(y_test, y_pred_prob)

axes[1].plot(fpr, tpr, color='steelblue', lw=2,
             label=f'ROC Curve (AUC = {auc_score:.3f})')
axes[1].plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Classifier')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC Curve — Logistic Regression')
axes[1].legend(loc='lower right')

plt.tight_layout()
plt.show()

joblib.dump(lr_model, 'logistic_regression_model.pkl')
print("\nModel successfully saved to 'logistic_regression_model.pkl'")
