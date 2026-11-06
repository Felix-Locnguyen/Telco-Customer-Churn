import os
import sys
from datetime import datetime

import gradio as gr
import joblib
import matplotlib.pyplot as plt
import pandas as pd

# Đưa src/ vào đường dẫn import để load được TelcoPreprocessor
sys.path.append(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src')
)

# ==========================================
# 1. LOAD MODEL
# ==========================================

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

artifact = joblib.load(os.path.join(ROOT, 'models', 'model.pkl'))
model = artifact['model']
threshold = artifact['threshold']
preprocessor = joblib.load(os.path.join(ROOT, 'models', 'preprocessor.pkl'))

# 21 cột nhập liệu (giống dữ liệu gốc)
COLS = [
    'customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
    'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
    'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn'
]

NUM_COLS = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']

# 14 đặc trưng mới do pipeline tạo (TenureGroup bắt đầu bằng 'TenureGroup_')
NEW_FEATURES = [
    'AverageMonthlyCharges', 'ChargeRatio', 'LifetimeValue_Proxy',
    'TotalServicesCount', 'HasSecuritySuite', 'HasStreamingSuite',
    'TenureGroup_', 'IsNewCustomer', 'Contract_Tenure_Ratio',
    'IsNearContractEnd', 'IsHighRiskPayment', 'IsAutomaticPayment',
    'HouseholdSize_Proxy', 'IsSingleSenior'
]

YES_NO = ['Yes', 'No']
CHOICES = {
    'gender': ['Male', 'Female'],
    'SeniorCitizen': [0, 1],
    'Partner': YES_NO,
    'Dependents': YES_NO,
    'PhoneService': YES_NO,
    'MultipleLines': ['Yes', 'No', 'No phone service'],
    'InternetService': ['DSL', 'Fiber optic', 'No'],
    'OnlineSecurity': ['Yes', 'No', 'No internet service'],
    'OnlineBackup': ['Yes', 'No', 'No internet service'],
    'DeviceProtection': ['Yes', 'No', 'No internet service'],
    'TechSupport': ['Yes', 'No', 'No internet service'],
    'StreamingTV': ['Yes', 'No', 'No internet service'],
    'StreamingMovies': ['Yes', 'No', 'No internet service'],
    'Contract': ['Month-to-month', 'One year', 'Two year'],
    'PaperlessBilling': YES_NO,
    'PaymentMethod': [
        'Electronic check', 'Mailed check',
        'Bank transfer (automatic)', 'Credit card (automatic)'
    ],
    'Churn': YES_NO
}

# Bộ nhớ tạm: kết quả lần dự đoán gần nhất (dùng cho nút Lưu)
LAST_RAW = None   # df gốc người dùng nhập
LAST_OUT = None   # df đã có đặc trưng mới + dự đoán


# ==========================================
# 2. CÁC HÀM XỬ LÝ
# ==========================================
def format_table(out, n=10):
    """Làm gọn bảng hiển thị: 2 cột kết quả lên đầu, làm tròn số."""
    view = out.head(n).copy()
    first = ['Churn_Probability', 'Predicted_Churn']
    rest = [c for c in view.columns if c not in first]
    view = view[first + rest]

    for c in view.select_dtypes(include='float').columns:
        view[c] = view[c].round(3)
    return view

def predict(df):
    """Dự đoán -> df có đặc trưng mới (từ pipeline) + 2 cột dự đoán."""
    processed = preprocessor.transform(df)
    proba = model.predict_proba(processed)[:, 1]

    out = processed.copy()
    out['Churn_Probability'] = proba.round(3)
    out['Predicted_Churn'] = ['Yes' if p >= threshold else 'No' for p in proba]
    return out


def make_pie(out):
    """Biểu đồ tròn tỷ lệ Churn / No Churn."""
    counts = out['Predicted_Churn'].value_counts()
    counts = counts.reindex(['No', 'Yes']).fillna(0)

    fig, ax = plt.subplots()
    ax.pie(counts.values, labels=['No Churn', 'Churn'],
           autopct='%1.1f%%', colors=['#8ecae6', '#ff6b6b'],
           startangle=90)
    ax.set_title('Tỷ lệ dự đoán Churn / No Churn')
    return fig


def make_top5():
    """Biểu đồ 5 feature ảnh hưởng nhiều nhất."""
    imp = pd.Series(model.feature_importances_, index=artifact['features'])
    imp = imp.nlargest(5).sort_values()

    fig, ax = plt.subplots()
    imp.plot(kind='barh', ax=ax, color='salmon')
    ax.set_title('5 feature ảnh hưởng nhiều nhất')
    ax.set_xlabel('Mức quan trọng')
    return fig


def build_result_df():
    """Mã hóa ngược: df gốc (Yes/No giữ nguyên) + 14 đặc trưng + dự đoán."""
    out = LAST_RAW.copy()

    for name in NEW_FEATURES:
        if name == 'TenureGroup_':
            # 4 cột dummy -> 1 cột nhãn '0-12', '12-24', '24-48', '>48'
            dummies = [c for c in LAST_OUT.columns if c.startswith(name)]
            out['TenureGroup'] = (
                LAST_OUT[dummies].idxmax(axis=1).str.replace(name, '')
            )
        else:
            # Đặc trưng số / cờ 0-1: giữ nguyên, không cần giải mã
            for c in [c for c in LAST_OUT.columns if c.startswith(name)]:
                out[c] = LAST_OUT[c].values

    out['Churn_Probability'] = LAST_OUT['Churn_Probability'].values
    out['Predicted_Churn'] = LAST_OUT['Predicted_Churn'].values
    return out


def show(df):
    """Dự đoán + trả về 3 đồ thị/hội thoại cho giao diện."""
    global LAST_RAW, LAST_OUT
    plt.close('all')  # dọn figure của lần chạy trước

    LAST_RAW = df.copy()
    LAST_OUT = predict(df)

    # Bảng hiển thị: cột gốc Yes/No đọc được + đặc trưng mới + dự đoán
    return make_pie(LAST_OUT), format_table(build_result_df()), make_top5()

def on_manual(*args):
    """Nhập tay 1 khách hàng -> DataFrame."""
    df = pd.DataFrame([dict(zip(COLS, args))], columns=COLS)

    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    return show(df)


def on_file(file_path):
    """Tải file csv/excel -> DataFrame."""
    if file_path is None:
        raise gr.Error('Hãy chọn file trước!')

    path = file_path if isinstance(file_path, str) else file_path.name

    if path.endswith(('.xlsx', '.xls')):
        try:
            df = pd.read_excel(path)
        except ImportError:
            raise gr.Error('Đọc file Excel cần cài: pip install openpyxl')
    else:
        df = pd.read_csv(path)  # mặc định csv

    missing = [c for c in COLS if c not in df.columns]
    if missing:
        raise gr.Error(f'File thiếu cột: {missing}')

    return show(df[COLS])


def on_save(with_new_features):
    """Lưu kết quả. Tích = lưu kèm đặc trưng mới, không tích = cột gốc."""
    if LAST_RAW is None:
        return '⚠️ Chưa có kết quả nào để lưu. Hãy dự đoán trước.'

    # Luôn bắt đầu từ cột gốc (đủ 21 cột, có customerID)
    if with_new_features:
        # Cột gốc + 14 đặc trưng (đã mã hóa ngược) + dự đoán
        out = build_result_df()
    else:
        # Chỉ các cột gốc + kết quả dự đoán
        out = LAST_RAW.copy()
        out['Churn_Probability'] = LAST_OUT['Churn_Probability'].values
        out['Predicted_Churn'] = LAST_OUT['Predicted_Churn'].values

    folder = os.path.join(ROOT, 'data', 'output')
    os.makedirs(folder, exist_ok=True)

    path = os.path.join(
        folder,
        f"predict_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.csv"
    )
    out.to_csv(path, index=False)

    return f'✅ Đã lưu {len(out)} dòng -> {path}'


# ==========================================
# 3. GIAO DIỆN
# ==========================================

CSS = """
.gradio-container {max-width: 1280px !important; margin: auto;}
#header {text-align: center; padding: 18px 0 6px;}
#header h1 {margin-bottom: 4px;}
#header p {color: #64748b; margin: 0;}
.panel {border: 1px solid #e2e8f0; border-radius: 14px; padding: 8px;}
#btn-predict button, .predict-btn {font-weight: 600; font-size: 16px;}


"""
CSS += """
#result-table {overflow-x: auto;}
#result-table table {width: max-content; min-width: 100%;}
#result-table th, #result-table td {
    white-space: nowrap !important;
    padding: 6px 12px !important;
    font-size: 13px;
}
#result-table th {position: sticky; top: 0; background: #f1f5f9; z-index: 1;}
"""
# Chia 21 cột thành 3 nhóm cho dễ nhìn (chỉ để hiển thị)
GROUPS = [
    ('👤 Thông tin khách hàng',
     ['customerID', 'gender', 'SeniorCitizen', 'Partner',
      'Dependents', 'tenure']),
    ('📡 Dịch vụ sử dụng',
     ['PhoneService', 'MultipleLines', 'InternetService',
      'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
      'TechSupport', 'StreamingTV', 'StreamingMovies']),
    ('💳 Hợp đồng & thanh toán',
     ['Contract', 'PaperlessBilling', 'PaymentMethod',
      'MonthlyCharges', 'TotalCharges', 'Churn']),
]


def make_input(c):
    if c in CHOICES:
        return gr.Dropdown(choices=CHOICES[c], value=CHOICES[c][0], label=c)
    if c in NUM_COLS:
        return gr.Number(label=c, value=0)
    return gr.Textbox(label=c, placeholder='VD: 7590-VHVEG')


theme = gr.themes.Soft(
    primary_hue='indigo',
    secondary_hue='rose',
    neutral_hue='slate',
    radius_size='lg',
)

with gr.Blocks(title='Telco Churn Demo') as demo:
    gr.HTML(
        f"""
        <div id="header">
            <h1>📊 Dự đoán Telco Customer Churn</h1>
            <p>Nhập thông tin một khách hàng hoặc tải file csv/excel để dự đoán
            &nbsp;·&nbsp; Threshold: <b>{threshold}</b></p>
        </div>
        """
    )

    with gr.Row(equal_height=False):
        # ===== Cột trái: nhập liệu + lưu =====
        with gr.Column(scale=5, elem_classes='panel'):
            comp = {}  # tên cột -> component

            with gr.Tabs():
                # --- Tab 1: nhập tay ---
                with gr.Tab('✍️ Nhập liệu'):
                    for title, cols in GROUPS:
                        with gr.Accordion(title, open=(title == GROUPS[0][0])):
                            for i in range(0, len(cols), 2):
                                with gr.Row():
                                    for c in cols[i:i + 2]:
                                        comp[c] = make_input(c)

                    btn_manual = gr.Button(
                        '🔍 Dự đoán', variant='primary', size='lg'
                    )

                # --- Tab 2: tải file ---
                with gr.Tab('📁 Tải file'):
                    gr.Markdown(
                        'File cần đủ **21 cột** như dữ liệu gốc '
                        '(csv, xlsx, xls).'
                    )
                    file_in = gr.File(
                        label='Kéo thả hoặc chọn file',
                        file_types=['.csv', '.xlsx', '.xls']
                    )
                    btn_file = gr.Button(
                        '🔍 Dự đoán', variant='primary', size='lg'
                    )

            # --- Lưu kết quả ---
            with gr.Accordion('💾 Lưu kết quả', open=False):
                ck_new = gr.Checkbox(
                    label='Lưu kèm 14 đặc trưng mới từ pipeline',
                    value=False
                )
                btn_save = gr.Button('Xác nhận lưu', variant='secondary')
                save_msg = gr.Markdown()

        # ===== Cột phải: kết quả =====
        with gr.Column(scale=7, elem_classes='panel'):
            gr.Markdown('### 📈 Kết quả dự đoán')
            with gr.Tabs():
                with gr.Tab('🥧 Tỷ lệ Churn'):
                    pie_out = gr.Plot(show_label=False)
                with gr.Tab('📋 Bảng dữ liệu'):
                    top10_out = gr.Dataframe(
                        label='10 dòng đầu (có đặc trưng mới)',
                        wrap=False,            # không xuống dòng -> không vỡ ô
                        interactive=False,     # chỉ xem
                        max_height=420,        # cuộn dọc khi dài
                        show_label=False,
                        elem_id='result-table',
                    )
                with gr.Tab('⭐ Top 5 feature'):
                    top5_out = gr.Plot(show_label=False)

    # ----- Gắn sự kiện (giữ nguyên logic) -----
    inputs = [comp[c] for c in COLS]  # đúng thứ tự COLS

    btn_manual.click(
        on_manual,
        inputs=inputs,
        outputs=[pie_out, top10_out, top5_out]
    )
    btn_file.click(
        on_file,
        inputs=file_in,
        outputs=[pie_out, top10_out, top5_out]
    )
    btn_save.click(
        on_save,
        inputs=ck_new,
        outputs=save_msg
    )


if __name__ == '__main__':
    demo.launch(theme=theme, css=CSS)