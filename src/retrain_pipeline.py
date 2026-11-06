import os
import joblib
import pandas as pd

from preprocess import TelcoPreprocessor
from train import TelcoModelTrainer


# Đường dẫn tuyệt đối theo vị trí file -> chạy từ thư mục nào cũng được
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(
    ROOT, 'data', 'raw', 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
)
MODEL_DIR = os.path.join(ROOT, 'models')


# 1. Đọc dữ liệu
df = pd.read_csv(DATA_PATH)


# 2. Tách Features và Target
X = df.drop(columns=['Churn', 'customerID'])
y = df['Churn'].map({'No': 0, 'Yes': 1})


# 3. Preprocessing trên TOÀN BỘ dữ liệu
# (model cuối cùng dùng hết dữ liệu, không giữ tập test riêng)
preprocessor = TelcoPreprocessor()
X_processed = preprocessor.fit_transform(X)


# 4. Huấn luyện model cuối
train_data = X_processed.copy()
train_data['Churn'] = y.to_numpy()

trainer = TelcoModelTrainer(
    output_path=os.path.join(MODEL_DIR, 'model_final.pkl')
)

model = trainer.train(
    train_data,
    threshold=0.4
)


# 5. Lưu artifact đầy đủ cho app: model + preprocessor + threshold
# (cùng format với model.pkl -> app chỉ cần đọc 1 kiểu)
artifact = joblib.load(os.path.join(MODEL_DIR, 'model_final.pkl'))
artifact['preprocessor'] = preprocessor

joblib.dump(
    artifact,
    os.path.join(MODEL_DIR, 'telco_churn_final.pkl')
)

print('Đã lưu models/telco_churn_final.pkl')
