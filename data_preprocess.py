# =========================================================
# 1. IMPORT
# =========================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import math

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif
from sklearn.decomposition import PCA

from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")

pd.set_option('display.max_columns', None)

# =========================================================
# 2. LOAD DATA
# =========================================================
file_id = '1rTFaaNhlieVoe08twl_kQltsxnpc7Qa8'
url = f'https://drive.google.com/uc?id={file_id}'
df = pd.read_csv(url)

print("="*60)
print("THÔNG TIN DỮ LIỆU BAN ĐẦU")
print("="*60)
print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\n5 dòng đầu:")
print(df.head())

# =========================================================
# 3. XỬ LÝ NHÃN + DROP
# =========================================================
df['num'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
df = df.drop(columns=['id', 'dataset'], errors='ignore')

# =========================================================
# 4. KHÁM PHÁ DỮ LIỆU (EDA) 
# =========================================================
print("\n" + "="*60)
print("EDA - KHÁM PHÁ VÀ THỐNG KÊ DỮ LIỆU")
print("="*60)

# 4.1. Thống kê mô tả
eda_summary = df.describe().T
print(eda_summary)
eda_summary.to_csv('eda_summary_statistics.csv')

# Tạo thư mục lưu hình EDA
if not os.path.exists("eda_plots"):
    os.makedirs("eda_plots")

# 4.2. Vẽ phân bố Target
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='num', palette='Set2')
plt.title("Phân bố biến mục tiêu (Target Distribution)")
plt.savefig("eda_plots/1_target_distribution.png")
plt.close()

# Tách riêng cột số để vẽ đồ thị
numeric_cols_eda = df.select_dtypes(include=[np.number]).columns.drop('num', errors='ignore')
n_cols = len(numeric_cols_eda)
cols_plot = 4
rows_plot = math.ceil(n_cols / cols_plot)

# 4.3. Vẽ Boxplot kiểm tra Outlier
plt.figure(figsize=(15, rows_plot * 3))
for i, col in enumerate(numeric_cols_eda, 1):
    plt.subplot(rows_plot, cols_plot, i)
    sns.boxplot(y=df[col], color='skyblue')
    plt.title(f"Boxplot: {col}")
    plt.tight_layout()
plt.savefig("eda_plots/2_numeric_boxplots.png")
plt.close()

# 4.4. Vẽ Histogram kiểm tra phân phối (Skewness)
plt.figure(figsize=(15, rows_plot * 3))
for i, col in enumerate(numeric_cols_eda, 1):
    plt.subplot(rows_plot, cols_plot, i)
    sns.histplot(df[col], kde=True, color='salmon')
    plt.title(f"Histogram: {col}")
    plt.tight_layout()
plt.savefig("eda_plots/3_numeric_histograms.png")
plt.close()

print("\nĐã xuất các bảng thống kê và biểu đồ EDA vào thư mục 'eda_plots/'!")

# =========================================================
# 5. KIỂM TRA MISSING
# =========================================================
print("\n" + "="*60)
print("MISSING VALUES")
print("="*60)

missing = df.isnull().sum()
missing_percent = (missing / len(df)) * 100

missing_df = pd.DataFrame({
    'Missing': missing,
    'Percent': missing_percent
})

print(missing_df[missing_df['Missing'] > 0])

# =========================================================
# 6. LỌC TƯƠNG QUAN
# =========================================================
numeric_cols = df.drop(columns=['num']).select_dtypes(include=[np.number]).columns
corr_matrix = df[numeric_cols].corr().abs()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, cmap='RdBu_r')
plt.title("Correlation Heatmap")
plt.savefig("eda_plots/4_correlation_heatmap.png")
plt.close()

upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > 0.9)]

df = df.drop(columns=to_drop)

print("\nCORRELATION FILTER")
print("Drop:", to_drop if len(to_drop) > 0 else "Không có")

# =========================================================
# 7. TÁCH X, y
# =========================================================
X = df.drop(columns=['num'])
y = df['num']

# =========================================================
# 8. CHIA TRAIN / TEST
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTRAIN TEST SPLIT")
print("Train:", X_train.shape)
print("Test:", X_test.shape)

# =========================================================
# 9. XÁC ĐỊNH FEATURE TYPE
# =========================================================
numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

# =========================================================
# 10. PIPELINE TIỀN XỬ LÝ
# =========================================================
numeric_transformer = Pipeline([
    ('imputer', KNNImputer(n_neighbors=5)),
    ('scaler', RobustScaler()),
    ('variance', VarianceThreshold(0.01))
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# =========================================================
# 11. TRANSFORM DATA
# =========================================================
preprocessor.fit(X_train)

X_train_pre = preprocessor.transform(X_train)
X_test_pre = preprocessor.transform(X_test)
feature_names = preprocessor.get_feature_names_out()

# =========================================================
# 12. CÂN BẰNG DỮ LIỆU (SMOTE)
# =========================================================
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_pre, y_train)

# =========================================================
# 13. FEATURE SELECTION
# =========================================================
k = min(15, len(feature_names))

selector = SelectKBest(score_func=mutual_info_classif, k=k)
np.random.seed(42)
X_train_final = selector.fit_transform(X_train_bal, y_train_bal)
X_test_final = selector.transform(X_test_pre)

mask = selector.get_support()
selected_features = feature_names[mask]

feature_score_df = pd.DataFrame({
    'Feature': feature_names,
    'Score': selector.scores_
}).sort_values(by='Score', ascending=False)

print("\n" + "="*60)
print("TOP 10 FEATURE ĐƯỢC CHỌN")
print("="*60)
print(feature_score_df.head(10))

# =========================================================
# 14. DATAFRAME
# =========================================================
df_train_cleaned = pd.DataFrame(X_train_final, columns=selected_features)
df_train_cleaned['target'] = y_train_bal.values

df_test_cleaned = pd.DataFrame(X_test_final, columns=selected_features)
df_test_cleaned['target'] = y_test.values

# =========================================================
# 15. SO SÁNH TRƯỚC VÀ SAU XỬ LÝ
# =========================================================
print("\n" + "="*60)
print("SO SÁNH DỮ LIỆU TRƯỚC & SAU XỬ LÝ")
print("="*60)

# 15.1 So sánh nhãn Target trước và sau SMOTE
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.countplot(x=y_train, ax=axes[0], palette='Set2')
axes[0].set_title('Mất cân bằng Target (Trước SMOTE)')
sns.countplot(x=y_train_bal, ax=axes[1], palette='Set2')
axes[1].set_title('Cân bằng Target (Sau SMOTE)')
plt.tight_layout()
plt.savefig('eda_plots/5_target_before_after_smote.png')
plt.close()

# 15.2 Vẽ biểu đồ không gian PCA so sánh cấu trúc dữ liệu
pca = PCA(n_components=2, random_state=42)
# Khớp (Fit) PCA trên dữ liệu TRƯỚC SMOTE
X_pca_before = pca.fit_transform(X_train_pre)
# Biến đổi (Transform) dữ liệu SAU SMOTE dựa trên cùng hệ trục tọa độ
X_pca_after = pca.transform(X_train_bal)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Đồ thị trước SMOTE
scatter_before = axes[0].scatter(X_pca_before[:, 0], X_pca_before[:, 1], c=y_train,
                                 cmap='coolwarm', alpha=0.6, edgecolors='k')
axes[0].set_title('Không gian Dữ liệu TRƯỚC SMOTE (PCA 2D)', fontsize=14)
axes[0].set_xlabel('Thành phần chính 1')
axes[0].set_ylabel('Thành phần chính 2')

# Đồ thị sau SMOTE
scatter_after = axes[1].scatter(X_pca_after[:, 0], X_pca_after[:, 1], c=y_train_bal,
                                cmap='coolwarm', alpha=0.6, edgecolors='k')
axes[1].set_title('Không gian Dữ liệu SAU SMOTE (PCA 2D)', fontsize=14)
axes[1].set_xlabel('Thành phần chính 1')

handles, labels = scatter_before.legend_elements()
fig.legend(handles, ['Không bệnh (0)', 'Có bệnh (1)'], loc='upper center', ncol=2, fontsize=12)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('eda_plots/6_pca_space_comparison.png')
plt.close()

print("-> Đã lưu các biểu đồ so sánh vào thư mục 'eda_plots/'")

# =========================================================
# 16. SAVE FILE
# =========================================================
df_train_cleaned.to_csv('heart_train_cleaned.csv', index=False)
df_test_cleaned.to_csv('heart_test_cleaned.csv', index=False)

joblib.dump(preprocessor, 'preprocessor.pkl')
joblib.dump(selector, 'selector.pkl')

# =========================================================
# 17. TỔNG KẾT
# =========================================================
print("\n" + "="*60)
print("TỔNG KẾT QUÁ TRÌNH")
print("="*60)
print("Train:", df_train_cleaned.shape)
print("Test:", df_test_cleaned.shape)
print("Số features giữ lại:", len(selected_features))
