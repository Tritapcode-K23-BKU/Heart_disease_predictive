# Dự đoán Bệnh lý Tim mạch (Heart Disease Prediction)

Dự án áp dụng các thuật toán Machine Learning (Decision Tree, Logistic Regression, Random Forest) để dự đoán khả năng mắc bệnh tim mạch. Mã nguồn được viết và thực thi trên môi trường Google Colab.

## 1. Cấu trúc file Notebook
File `heart_desease_L03.ipynb` được chia làm 4 phần chính, yêu cầu thực thi tuần tự:
* **Phần 1 - Tiền xử lý & EDA:** Tải dữ liệu tự động từ Google Drive, làm sạch, cân bằng dữ liệu bằng SMOTE, chọn lọc đặc trưng (Feature Selection) và lưu tập dữ liệu đã xử lý.
* **Phần 2 - Decision Tree:** Huấn luyện, dò tìm siêu tham số (Coarse/Fine Tuning), tối ưu ngưỡng (Threshold Tuning) và trực quan hóa cây.
* **Phần 3 - Logistic Regression:** Huấn luyện baseline và đánh giá.
* **Phần 4 - Random Forest:** Huấn luyện bằng GridSearchCV, đánh giá và trực quan hóa Feature Importance.
