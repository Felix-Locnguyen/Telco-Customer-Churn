import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


class TelcoPreprocessor(BaseEstimator, TransformerMixin):
    """
    Tiền xử lý dữ liệu Telco Churn.

    Chức năng:
    1. Xử lý dữ liệu thiếu (mean/median với số, mode với chữ).
    2. Xử lý Outlier bằng cách CLIP (giữ nguyên số dòng).
    3. Feature Engineering gồm 14 đặc trưng mới.
    4. Mã hóa cột chữ thành One-Hot (0/1) để model đọc được.
    """

    def __init__(self, skew_threshold=1.0, iqr_threshold=1.5):
        self.skew_threshold = skew_threshold
        self.iqr_threshold = iqr_threshold

    # ==========================================
    # 1. FIT - học tham số từ dữ liệu TRAIN
    # ==========================================

    def fit(self, X, y=None):
        df = self._prepare(X.copy())

        # Giá trị thay thế dữ liệu thiếu
        self.num_fill_ = {}
        self.cat_fill_ = {}

        for col in df.select_dtypes(include=np.number).columns:

            if col == 'SeniorCitizen':
                continue

            skew = df[col].skew()

            if abs(skew) > self.skew_threshold:
                self.num_fill_[col] = df[col].median()
            else:
                self.num_fill_[col] = df[col].mean()

        for col in df.select_dtypes(include=['object', 'category']).columns:

            mode = df[col].mode()

            self.cat_fill_[col] = (
                mode.iloc[0] if not mode.empty else 'Missing'
            )

        # Ngưỡng Outlier (IQR) lấy từ Train
        self.outlier_bounds_ = {}

        for col in ['tenure', 'MonthlyCharges', 'TotalCharges']:

            if col not in df.columns:
                continue

            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1

            self.outlier_bounds_[col] = (
                q1 - self.iqr_threshold * iqr,
                q3 + self.iqr_threshold * iqr
            )

        # Chạy thử 1 lần để biết danh sách cột đầu ra
        # (sau feature engineering + mã hóa One-Hot)
        self.feature_cols_ = self._process(df).columns.tolist()

        return self

    # ==========================================
    # 2. TRANSFORM
    # ==========================================

    def transform(self, X):
        df = self._process(self._prepare(X.copy()))

        # Cột nào không có trong lúc fit (vd tập test thiếu 1 category)
        # thì thêm vào với giá trị 0
        return df.reindex(columns=self.feature_cols_, fill_value=0)

    # ==========================================
    # 3. CÁC BƯỚC DÙNG CHUNG
    # ==========================================

    def _prepare(self, df):
        """Chuyển TotalCharges sang số + bỏ cột không cần thiết."""

        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(
                df['TotalCharges'], errors='coerce'
            )

        # customerID: mã định danh, không học được
        # Churn: nhãn target, không được đưa vào X
        df = df.drop(columns=['customerID', 'Churn'], errors='ignore')

        return df

    def _process(self, df):
        """Thiếu -> Outlier -> 14 đặc trưng -> Mã hóa One-Hot."""

        # Xử lý dữ liệu thiếu
        for col, value in self.num_fill_.items():
            if col in df.columns:
                df[col] = df[col].fillna(value)

        for col, value in self.cat_fill_.items():
            if col in df.columns:
                df[col] = df[col].fillna(value)

        # Xử lý Outlier bằng CLIP (không xóa dòng).
        # Dữ liệu hiện tại có 0 outlier - clip chỉ phòng giá trị bất thường
        # xuất hiện ở dữ liệu mới.
        for col, (lower, upper) in self.outlier_bounds_.items():
            if col in df.columns:
                df[col] = df[col].clip(lower, upper)

        # Feature Engineering (14 đặc trưng)
        df = self._create_charge_features(df)
        df = self._create_service_features(df)
        df = self._create_tenure_features(df)
        df = self._create_payment_features(df)
        df = self._create_demographic_features(df)

        # Mã hóa: cột chữ (gender, Contract, TenureGroup...) -> One-Hot 0/1
        df = pd.get_dummies(df, dtype=float)

        return df

    # ==========================================
    # 4. FEATURE ENGINEERING
    # ==========================================

    def _create_charge_features(self, df):
        """Nhóm 1: Tài chính và giá trị khách hàng."""

        df['AverageMonthlyCharges'] = np.where(
            df['tenure'] > 0,
            df['TotalCharges'] / df['tenure'],
            df['MonthlyCharges']
        )

        df['ChargeRatio'] = (
            df['MonthlyCharges'] /
            df['AverageMonthlyCharges'].replace(0, np.nan)
        ).round(2).fillna(0)

        df['LifetimeValue_Proxy'] = (
            df['MonthlyCharges'] * df['tenure']
        )

        return df

    def _create_service_features(self, df):
        """Nhóm 2: Sử dụng và mức độ gắn kết dịch vụ."""

        services = [
            'OnlineSecurity',
            'OnlineBackup',
            'DeviceProtection',
            'TechSupport',
            'StreamingTV',
            'StreamingMovies'
        ]

        security_services = [
            'OnlineSecurity',
            'DeviceProtection',
            'TechSupport'
        ]

        df['TotalServicesCount'] = (
            df[services] == 'Yes'
        ).sum(axis=1)

        df['HasSecuritySuite'] = (
            (df[security_services] == 'Yes').sum(axis=1) >= 2
        ).astype(int)

        df['HasStreamingSuite'] = (
            (df['StreamingTV'] == 'Yes') &
            (df['StreamingMovies'] == 'Yes')
        ).astype(int)

        return df

    def _create_tenure_features(self, df):
        """Nhóm 3: Phân đoạn và giai đoạn khách hàng."""

        df['TenureGroup'] = pd.cut(
            df['tenure'],
            bins=[-1, 12, 24, 48, np.inf],
            labels=['0-12', '12-24', '24-48', '>48']
        )

        df['IsNewCustomer'] = (
            df['tenure'] <= 6
        ).astype(int)

        contract_months = df['Contract'].map({
            'Month-to-month': 0,
            'One year': 12,
            'Two year': 24
        })

        df['Contract_Tenure_Ratio'] = np.where(
            contract_months > 0,
            (df['tenure'] / contract_months).clip(upper=1),
            1.0
        )

        df['IsNearContractEnd'] = (
            (contract_months > 0) &
            (df['tenure'] >= contract_months - 2)
        ).astype(int)

        return df

    def _create_payment_features(self, df):
        """Nhóm 4: Rủi ro thanh toán và hợp đồng."""

        df['IsHighRiskPayment'] = (
            (df['PaymentMethod'] == 'Electronic check') &
            (df['Contract'] == 'Month-to-month')
        ).astype(int)

        df['IsAutomaticPayment'] = (
            df['PaymentMethod']
            .str.contains('automatic', case=False, na=False)
        ).astype(int)

        return df

    def _create_demographic_features(self, df):
        """Nhóm 5: Nhân khẩu học và gia đình."""

        df['HouseholdSize_Proxy'] = (
            (df['Partner'] == 'Yes').astype(int) +
            (df['Dependents'] == 'Yes').astype(int)
        )

        df['IsSingleSenior'] = (
            (df['SeniorCitizen'] == 1) &
            (df['Partner'] == 'No') &
            (df['Dependents'] == 'No')
        ).astype(int)

        return df
