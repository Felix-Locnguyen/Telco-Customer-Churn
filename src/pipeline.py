import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

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


# 3. Chia Train / Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 4. Preprocessing
# fit trên Train, transform cả 2 -> tránh data leakage
# TelcoPreprocessor đã làm: thiếu -> outlier -> 14 đặc trưng -> One-Hot
preprocessor = TelcoPreprocessor()

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)


# 5. Ghép nhãn vào Train Data (trainer cần cột Churn)
train_data = X_train_processed.copy()
train_data['Churn'] = y_train.to_numpy()


# 6. Huấn luyện (trainer tự chia tập validation để đánh giá)
trainer = TelcoModelTrainer(
    output_path=os.path.join(MODEL_DIR, 'model.pkl')
)

model = trainer.train(
    train_data,
    threshold=0.4
)


# 7. Lưu Preprocessor (dùng cho app khi dự đoán khách mới)
joblib.dump(
    preprocessor,
    os.path.join(MODEL_DIR, 'preprocessor.pkl')
)


# 8. Đánh giá trên Test (một lần, chưa từng được model nhìn thấy)
artifact = joblib.load(os.path.join(MODEL_DIR, 'model.pkl'))

probabilities = artifact['model'].predict_proba(
    X_test_processed
)[:, 1]

predictions = (
    probabilities >= artifact['threshold']
).astype(int)


print("Classification Report:")
print(classification_report(y_test, predictions))

print("ROC-AUC:")
print(roc_auc_score(y_test, probabilities))
