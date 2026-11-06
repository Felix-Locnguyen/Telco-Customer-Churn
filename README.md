# Telco-Customer-Churn
End-to-end Machine Learning pipeline for Telco Customer Churn analysis, feature engineering, predictive modeling, and model explainability.

Link data: https://www.kaggle.com/datasets/blastchar/telco-customer-churn 


# Cấu trúc dự án 
```text
Telco-Customer-Churn/
├── config/
│   └── config.yaml             # Cấu hình tham số, đường dẫn
├── data/
│   ├── raw/                    # Dữ liệu gốc (.csv)
│   └── processed/              # Dữ liệu sau tiền xử lý
├── docs/
│   └── ERD.png                 # Sơ đồ thực thể liên kết (ERD)
├── models/                     # Lưu file mô hình & pipeline (.pkl / .joblib)
├── notebooks/                  # Jupyter Notebooks phục vụ EDA và thử nghiệm
│   ├── 01_exploratory_data_analysis.ipynb  # Phân tích EDA
│   ├── 02_data_preprocessing.ipynb         # Làm sạch dữ liệu, xử lý missing values
│   ├── 03_feature_engineering.ipynb        # Tạo và lựa chọn đặc trưng mới
│   ├── 04_data_encoding.ipynb              # Mã hóa các biến categorical (One-Hot, Label, v.v.)
│   ├── 05_model_training.ipynb             # Huấn luyện mô hình
│   └── 06_model_evaluation_explain.ipynb   # Đánh giá & SHAP Explainability
├── src/                        # Source code chính
│   ├── preprocess.py           # Xử lý missing values, encoding, scaling
│   ├── train.py                # Huấn luyện mô hình (Train/Test)
│   ├── evaluate.py             # Đánh giá mô hình (Recall, ROC-AUC)
│   └── pipeline.py             # Đóng gói Sklearn/Custom Pipeline cho Inference
├── app/                        # Giao diện Demo (Streamlit / FastAPI)
│   └── app.py                  # Load pipeline từ src/pipeline.py để dự đoán
├── requirements.txt            # Thư viện Python cần thiết
└── README.md                   # Hướng dẫn sử dụng & mô tả dự án
```

# Road map 
```text
 ┌───────────────────────────┐
 │        1. ERD             │  (Thiết kế cơ sở dữ liệu / quan hệ thực thể)
 └─────────────┬─────────────┘
               │
               ▼
 ┌───────────────────────────┐
 │     2. PREPROCESS         │  (Làm sạch dữ liệu, xử lý khoảng trắng & missing)
 └─────────────┬─────────────┘
               │
               ▼
 ┌───────────────────────────┐
 │      3. FEATURE           │  (Kỹ thuật đặc trưng: tạo biến mới & lọc chọn)
 └─────────────┬─────────────┘
               │
               ▼
 ┌───────────────────────────┐
 │      4. ENCODER           │  (Mã hóa biến phân loại & chuẩn hóa dữ liệu số)
 └─────────────┬─────────────┘
               │
               ▼
 ┌───────────────────────────┐
 │     5. TRAIN MODEL        │  (Huấn luyện thuật toán, tối ưu Recall & Threshold)
 └─────────────┬─────────────┘
               │
               ▼
 ┌───────────────────────────┐
 │        6. EVAL            │  (Đánh giá hiệu suất: ROC-AUC, F1-Score & SHAP)
 └─────────────┬─────────────┘
               │
               ▼
 ┌───────────────────────────┐
 │      7. PIPELINE          │  (Đóng gói toàn bộ quy trình vào file .pkl)
 └─────────────┬─────────────┘
               │
               ▼
 ┌───────────────────────────┐
 │    8. DEMO (GRADIO)       │  (Xây dựng ứng dụng web giao diện người dùng)
 └───────────────────────────┘
 ```


# Ý nhĩa các cột trong data
| Cột | Ý nghĩa |
|---|---|
| customerID | Mã khách hàng |
| gender | Giới tính |
| SeniorCitizen | Người cao tuổi (0 = Không, 1 = Có) |
| Partner | Có vợ/chồng |
| Dependents | Có người phụ thuộc |
| tenure | Số tháng khách hàng ở lại |
| PhoneService | Có dùng điện thoại |
| MultipleLines | Nhiều đường điện thoại |
| InternetService | Loại internet (DSL, Fiber optic, No) |
| OnlineSecurity | Bảo mật trực tuyến |
| OnlineBackup | Sao lưu trực tuyến |
| DeviceProtection | Bảo vệ thiết bị |
| TechSupport | Hỗ trợ kỹ thuật |
| StreamingTV | Truyền hình trực tuyến |
| StreamingMovies | Phim trực tuyến |
| Contract | Loại hợp đồng |
| PaperlessBilling | Hóa đơn điện tử |
| PaymentMethod | Cách thanh toán |
| MonthlyCharges | Phí mỗi tháng |
| TotalCharges | Tổng phí đã trả |
| **Churn** | **Khách rời đi hay không (target)** |
