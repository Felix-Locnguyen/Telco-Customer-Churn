import os
import joblib
import numpy as np
import xgboost as xgb

from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split


# Đường dẫn mặc định: <project>/models/model.pkl (đúng vị trí file này)
DEFAULT_OUTPUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'models', 'model.pkl'
)


class TelcoModelTrainer:
    """
    Quản lý huấn luyện, đánh giá và lưu mô hình XGBoost.
    """

    def __init__(self, output_path=DEFAULT_OUTPUT):
        self.output_path = output_path

    def _prepare_data(self, df):
        """Tách Features và Target."""

        if 'Churn' not in df.columns:
            raise ValueError("Không tìm thấy cột Churn!")

        X = df.drop(columns=['Churn'])

        y = (
            df['Churn']
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(['yes', '1', 'true'])
            .astype(int)
        )

        return X, y

    def _evaluate(self, model, X_val, y_val, threshold):
        """Đánh giá mô hình."""

        y_proba = model.predict_proba(X_val)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)

        print('\n--- CLASSIFICATION REPORT ---')
        print(classification_report(y_val, y_pred, zero_division=0))

        auc = roc_auc_score(y_val, y_proba)
        print(f'ROC-AUC Score: {auc:.4f}')

        return auc

    def train(
        self,
        df,
        params=None,
        threshold=0.5,
        test_size=0.2
    ):
        """Huấn luyện và đánh giá mô hình."""

        X, y = self._prepare_data(df)

        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42,
            stratify=y
        )

        # Xử lý mất cân bằng dữ liệu
        neg = (y_train == 0).sum()
        pos = (y_train == 1).sum()

        scale_pos_weight = neg / pos if pos > 0 else 1.0

        model_params = {
            'n_estimators': 300,
            'max_depth': 4,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'tree_method': 'hist',
            'device': 'cpu',
            'random_state': 42,
            'n_jobs': -1,
            'scale_pos_weight': scale_pos_weight
        }

        if params:
            model_params.update(params)

        print('--- ĐANG HUẤN LUYỆN MÔ HÌNH ---')

        model = xgb.XGBClassifier(**model_params)

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        # Đánh giá
        auc = self._evaluate(
            model,
            X_val,
            y_val,
            threshold
        )

        # Lưu mô hình
        os.makedirs(
            os.path.dirname(self.output_path) or '.',
            exist_ok=True
        )

        joblib.dump(
            {
                'model': model,
                'threshold': threshold,
                'features': X.columns.tolist(),
                'params': model_params,
                'roc_auc': auc
            },
            self.output_path
        )

        print(f'\nĐã lưu mô hình tại: {self.output_path}')

        return model

    def retrain(self, df, additional_rounds=50):
        """Tiếp tục huấn luyện mô hình với dữ liệu mới.

        Lưu ý: df phải đã qua TelcoPreprocessor (đều là cột số),
        và có cột 'Churn'.
        """

        if not os.path.exists(self.output_path):
            raise FileNotFoundError("Không tìm thấy mô hình đã lưu!")

        X_new, y_new = self._prepare_data(df)

        artifact = joblib.load(self.output_path)

        model = artifact['model']
        old_features = artifact['features']

        # Đảm bảo thứ tự Features giống mô hình cũ
        X_new = X_new.reindex(
            columns=old_features,
            fill_value=0
        )

        print('--- ĐANG TIẾP TỤC HUẤN LUYỆN ---')

        model_params = model.get_params()
        model_params['n_estimators'] = additional_rounds

        new_model = xgb.XGBClassifier(**model_params)

        new_model.fit(
            X_new,
            y_new,
            xgb_model=model.get_booster(),
            verbose=False
        )

        artifact['model'] = new_model

        joblib.dump(artifact, self.output_path)

        print('Đã cập nhật mô hình thành công!')

        return new_model