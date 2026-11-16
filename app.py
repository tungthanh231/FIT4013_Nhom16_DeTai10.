import os
import __main__
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin

EUR_TO_VND = 27000  # Tỷ giá quy đổi ngầm để khớp với dữ liệu gốc (EUR) của mô hình

# Khai báo lại lớp Feature Engineering để joblib nạp đúng mô hình đã lưu
class BankFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()
        X_out["was_contacted_before"] = (X_out["pdays"] != -1).astype(int)
        X_out["has_any_loan"] = ((X_out["housing"] == "yes") | (X_out["loan"] == "yes")).astype(int)
        X_out["age_group"] = pd.cut(
            X_out["age"], bins=[0, 25, 60, 120], labels=["young", "adult", "senior"]
        ).astype(str)
        return X_out

__main__.BankFeatureEngineer = BankFeatureEngineer

st.set_page_config(
    page_title="Hệ thống Dự đoán Tiền gửi Ngân hàng",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Fintech Dashboard hiện đại
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .stApp {
        background-color: #f4f7fb;
    }
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2.5rem;
        max-width: 1280px;
    }
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    .hero-banner {
        background: linear-gradient(135deg, #091e3a 0%, #1d4ed8 55%, #0ea5e9 100%);
        padding: 1.8rem 2.2rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 12px 28px rgba(15, 23, 42, 0.14);
    }
    .hero-title { font-size: 1.85rem; font-weight: 800; margin: 0; letter-spacing: -0.02em; }
    .hero-sub { font-size: 0.96rem; color: #e0f2fe; margin-top: 0.4rem; font-weight: 400; }
    .badge-tag {
        display: inline-block;
        background: rgba(255,255,255,0.14);
        backdrop-filter: blur(6px);
        border: 1px solid rgba(255,255,255,0.28);
        padding: 0.28rem 0.8rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.5rem;
        margin-top: 0.8rem;
    }
    /* Tab dạng Pill hiện đại */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #e2e8f0;
        padding: 5px;
        border-radius: 12px;
        border-bottom: none;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 9px;
        padding: 8px 18px;
        font-weight: 600;
        color: #475569;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        color: #1d4ed8 !important;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.08);
    }
    .stTabs [data-baseweb="tab-highlight"] {
        display: none;
    }
    /* Form & Card trắng nổi bật */
    div[data-testid="stForm"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.4rem;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.04);
    }
    .kpi-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.1rem 1.2rem;
        text-align: center;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04);
    }
    .kpi-val { font-size: 1.65rem; font-weight: 800; color: #0f172a; }
    .kpi-lbl { font-size: 0.82rem; color: #64748b; margin-top: 0.25rem; font-weight: 500; }
    .panel-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.3rem 1.4rem;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.04);
        margin-bottom: 1rem;
    }
    .result-card-yes {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        border: 1px solid #a7f3d0;
        border-left: 6px solid #059669;
        padding: 1.3rem 1.5rem;
        border-radius: 14px;
        margin-bottom: 1rem;
    }
    .result-card-no {
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        border: 1px solid #fde68a;
        border-left: 6px solid #d97706;
        padding: 1.3rem 1.5rem;
        border-radius: 14px;
        margin-bottom: 1rem;
    }
    .prob-number { font-size: 2.6rem; font-weight: 800; line-height: 1.1; margin: 0.25rem 0; }
    button[kind="primary"] {
        border-radius: 10px !important;
        font-weight: 600 !important;
        background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 100%) !important;
        border: none !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25) !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load("models/best_bank_model.pkl")

@st.cache_data
def load_raw_data():
    path = os.path.join("data", "bank-full.csv")
    if os.path.exists(path):
        return pd.read_csv(path, sep=";")
    return None

model = load_model()
raw_df = load_raw_data()

if "history" not in st.session_state:
    st.session_state.history = []

# Từ điển ánh xạ tiếng Việt <-> giá trị gốc cho mô hình
JOB_MAP = {
    "Quản lý": "management",
    "Kỹ thuật viên": "technician",
    "Lao động phổ thông": "blue-collar",
    "Hành chính / Văn phòng": "admin.",
    "Dịch vụ": "services",
    "Đã nghỉ hưu": "retired",
    "Tự kinh doanh": "self-employed",
    "Doanh nhân": "entrepreneur",
    "Chưa có việc làm": "unemployed",
    "Giúp việc": "housemaid",
    "Sinh viên": "student",
    "Không rõ": "unknown"
}
MARITAL_MAP = {"Đã kết hôn": "married", "Độc thân": "single", "Ly hôn / Góa": "divorced"}
EDU_MAP = {"Đại học / Sau đại học": "tertiary", "Trung học": "secondary", "Tiểu học": "primary", "Không rõ": "unknown"}
YN_MAP = {"Không": "no", "Có": "yes"}
CONTACT_MAP = {"Di động": "cellular", "Điện thoại bàn": "telephone", "Không rõ": "unknown"}
POUTCOME_MAP = {"Chưa rõ": "unknown", "Thành công": "success", "Thất bại": "failure", "Khác": "other"}
MONTH_MAP = {f"Tháng {i+1}": m for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}

REV_JOB = {v: k for k, v in JOB_MAP.items()}
REV_MARITAL = {v: k for k, v in MARITAL_MAP.items()}
REV_EDU = {v: k for k, v in EDU_MAP.items()}
REV_YN = {v: k for k, v in YN_MAP.items()}
REV_CONTACT = {v: k for k, v in CONTACT_MAP.items()}
REV_POUT = {v: k for k, v in POUTCOME_MAP.items()}
REV_MONTH = {v: k for k, v in MONTH_MAP.items()}

COL_VI_MAP = {
    "age": "Tuổi", "job": "Nghề nghiệp", "marital": "Hôn nhân", "education": "Học vấn",
    "default": "Nợ xấu", "balance": "Số dư (VNĐ)", "housing": "Vay mua nhà", "loan": "Vay tiêu dùng",
    "contact": "Kênh liên hệ", "day": "Ngày", "month": "Tháng", "campaign": "Số lần gọi",
    "pdays": "Ngày từ lần trước", "previous": "Số lần trước đó", "poutcome": "Kết quả trước",
    "Actual_y": "Thực tế", "Predicted_y": "Dự đoán"
}

def translate_df_to_vi(df_in):
    df_out = df_in.copy()
    if "balance" in df_out.columns:
        df_out["balance"] = (df_out["balance"] * EUR_TO_VND).apply(lambda x: f"{x:,.0f} đ")
    if "job" in df_out.columns: df_out["job"] = df_out["job"].map(REV_JOB).fillna(df_out["job"])
    if "marital" in df_out.columns: df_out["marital"] = df_out["marital"].map(REV_MARITAL).fillna(df_out["marital"])
    if "education" in df_out.columns: df_out["education"] = df_out["education"].map(REV_EDU).fillna(df_out["education"])
    for c in ["default", "housing", "loan"]:
        if c in df_out.columns: df_out[c] = df_out[c].map(REV_YN).fillna(df_out[c])
    if "contact" in df_out.columns: df_out["contact"] = df_out["contact"].map(REV_CONTACT).fillna(df_out["contact"])
    if "poutcome" in df_out.columns: df_out["poutcome"] = df_out["poutcome"].map(REV_POUT).fillna(df_out["poutcome"])
    if "month" in df_out.columns: df_out["month"] = df_out["month"].map(REV_MONTH).fillna(df_out["month"])
    if "Actual_y" in df_out.columns: df_out["Actual_y"] = df_out["Actual_y"].map({0: "Không gửi", 1: "Có gửi"}).fillna(df_out["Actual_y"])
    if "Predicted_y" in df_out.columns: df_out["Predicted_y"] = df_out["Predicted_y"].map({0: "Không gửi", 1: "Có gửi"}).fillna(df_out["Predicted_y"])
    return df_out.rename(columns=COL_VI_MAP)

# THANH BÊN (SIDEBAR)
with st.sidebar:
    st.markdown("#### ⚡ Kịch bản mẫu")
    preset = st.radio(
        "Điền nhanh hồ sơ:",
        ["Tùy chỉnh thủ công", "Khách hàng tiềm năng cao", "Khách hàng ít tiềm năng"],
        index=0
    )

    st.divider()
    st.markdown("#### 🎚️ Ngưỡng quyết định (Threshold)")
    threshold = st.slider(
        "Ngưỡng xác suất tối thiểu để gọi điện:",
        min_value=0.10, max_value=0.90, value=0.50, step=0.05,
        help="Giảm ngưỡng (<0.50) để tăng Recall (ít bỏ sót khách). Tăng ngưỡng (>0.50) để tăng Precision (tiết kiệm chi phí gọi)."
    )

    st.divider()
    st.success("**Mô hình đang kích hoạt:**\nRandom Forest (Luồng B)\n\n• F1-Score: `0.4577`\n• ROC-AUC: `0.7971`")

if preset == "Khách hàng tiềm năng cao":
    d_age, d_job, d_mar, d_edu = 64, 5, 0, 0
    d_bal_vnd, d_def, d_hou, d_loa = 120000000, 0, 0, 0
    d_con, d_mon, d_day, d_cam, d_pdays, d_prev, d_pout = 0, 9, 12, 1, 90, 2, 1
elif preset == "Khách hàng ít tiềm năng":
    d_age, d_job, d_mar, d_edu = 33, 2, 0, 1
    d_bal_vnd, d_def, d_hou, d_loa = -4000000, 0, 1, 1
    d_con, d_mon, d_day, d_cam, d_pdays, d_prev, d_pout = 2, 4, 20, 6, -1, 0, 0
else:
    d_age, d_job, d_mar, d_edu = 42, 0, 0, 0
    d_bal_vnd, d_def, d_hou, d_loa = 50000000, 0, 0, 0
    d_con, d_mon, d_day, d_cam, d_pdays, d_prev, d_pout = 0, 4, 15, 1, -1, 0, 0

# BANNER ĐẦU TRANG
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">🏦 Hệ thống Hỗ trợ Quyết định Tiếp thị Tiền gửi Ngân hàng</div>
    <div class="hero-sub">Phân tích xác suất đăng ký tiền gửi có kỳ hạn theo thời gian thực & chấm điểm danh sách khách hàng</div>
    <div>
        <span class="badge-tag">📊 45.211 mẫu UCI Bank Marketing</span>
        <span class="badge-tag">🌲 Random Forest + Feature Engineering</span>
        <span class="badge-tag">🛡️ Chống Data Leakage (Bỏ duration)</span>
    </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Dự đoán Cá nhân",
    "📂 Dự đoán Hàng loạt",
    "📊 Khám phá Dữ liệu (EDA)",
    "🏆 Đối sánh & Phân tích Lỗi"
])

# ==================== TAB 1: DỰ ĐOÁN CÁ NHÂN ====================
with tab1:
    col_left, col_right = st.columns([1.28, 1], gap="large")

    with col_left:
        with st.form("crm_form"):
            st.markdown("#### 1. Thông tin nhân khẩu học & Tài chính")
            c1, c2 = st.columns(2)
            with c1:
                age = st.number_input("Tuổi", 18, 95, d_age)
                job_label = st.selectbox("Nghề nghiệp", list(JOB_MAP.keys()), index=d_job)
                marital_label = st.selectbox("Tình trạng hôn nhân", list(MARITAL_MAP.keys()), index=d_mar)
                edu_label = st.selectbox("Trình độ học vấn", list(EDU_MAP.keys()), index=d_edu)
            with c2:
                balance_str = st.text_input(
                    "Số dư bình quân năm (VNĐ)",
                    value=f"{d_bal_vnd:,}",
                    key=f"bal_{preset}",
                    help="Bạn có thể nhập số có hoặc không có dấu phẩy (ví dụ: 120,000,000). Tự động quy đổi 1 EUR = 27.000 VNĐ."
                )
                try:
                    balance_vnd = int(balance_str.replace(",", "").replace(".", "").replace(" ", ""))
                except ValueError:
                    st.error("Vui lòng chỉ nhập số hợp lệ (ví dụ: 120,000,000).")
                    balance_vnd = d_bal_vnd
                default_label = st.selectbox("Tình trạng nợ xấu", list(YN_MAP.keys()), index=d_def)
                housing_label = st.selectbox("Đang vay mua nhà?", list(YN_MAP.keys()), index=d_hou)
                loan_label = st.selectbox("Đang vay tiêu dùng?", list(YN_MAP.keys()), index=d_loa)

            st.markdown("#### 2. Lịch sử & Kế hoạch tiếp thị")
            c3, c4 = st.columns(2)
            with c3:
                contact_label = st.selectbox("Kênh liên hệ", list(CONTACT_MAP.keys()), index=d_con)
                month_label = st.selectbox("Tháng dự kiến gọi", list(MONTH_MAP.keys()), index=d_mon)
                day = st.slider("Ngày trong tháng", 1, 31, d_day)
            with c4:
                campaign = st.number_input("Số lần gọi chiến dịch này", 1, 50, d_cam)
                pdays = st.number_input("Số ngày từ lần gọi trước (-1 nếu chưa)", -1, 900, d_pdays)
                previous = st.number_input("Số lần liên hệ trước đó", 0, 100, d_prev)
                poutcome_label = st.selectbox("Kết quả chiến dịch trước", list(POUTCOME_MAP.keys()), index=d_pout)

            submitted = st.form_submit_button("🚀 Phân tích & Lưu kết quả", type="primary", use_container_width=True)

    balance_eur = balance_vnd / EUR_TO_VND
    input_df = pd.DataFrame([{
        "age": age, "job": JOB_MAP[job_label], "marital": MARITAL_MAP[marital_label],
        "education": EDU_MAP[edu_label], "default": YN_MAP[default_label],
        "balance": balance_eur, "housing": YN_MAP[housing_label], "loan": YN_MAP[loan_label],
        "contact": CONTACT_MAP[contact_label], "day": day, "month": MONTH_MAP[month_label],
        "campaign": campaign, "pdays": pdays, "previous": previous,
        "poutcome": POUTCOME_MAP[poutcome_label]
    }])

    prob = float(model.predict_proba(input_df)[0][1])
    pred = 1 if prob >= threshold else 0

    if submitted:
        st.session_state.history.insert(0, {
            "Tuổi": age, "Nghề nghiệp": job_label, "Số dư (VNĐ)": f"{balance_vnd:,} đ",
            "Vay nợ": "Có" if (YN_MAP[housing_label] == "yes" or YN_MAP[loan_label] == "yes") else "Không",
            "Kết quả trước": poutcome_label,
            "Xác suất (%)": round(prob * 100, 1),
            "Ngưỡng": threshold,
            "Quyết định": "Nên gọi" if pred == 1 else "Bỏ qua"
        })

    with col_right:
        st.markdown("#### 📋 Kết quả phân tích")
        if pred == 1:
            st.markdown(f"""
            <div class="result-card-yes">
                <div style="font-weight:700; color:#065f46;">✅ KHÁCH HÀNG TIỀM NĂNG (Ngưỡng ≥ {threshold:.2f})</div>
                <div class="prob-number" style="color:#047857;">{prob*100:.1f}%</div>
                <div style="color:#065f46; font-size:0.92rem;">
                    <b>Khuyến nghị:</b> Đưa vào danh sách ưu tiên gọi điện tư vấn gửi tiết kiệm.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card-no">
                <div style="font-weight:700; color:#92400e;">⚠️ ÍT TIỀM NĂNG (Ngưỡng ≥ {threshold:.2f})</div>
                <div class="prob-number" style="color:#b45309;">{prob*100:.1f}%</div>
                <div style="color:#92400e; font-size:0.92rem;">
                    <b>Khuyến nghị:</b> Tỷ lệ chuyển đổi thấp, ưu tiên phân bổ nhân sự cho nhóm khác.
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Biểu đồ Donut Gauge hiện đại
        main_color = "#059669" if pred == 1 else "#d97706"
        fig, ax = plt.subplots(figsize=(4.2, 2.3))
        fig.patch.set_alpha(0.0)
        wedges, _ = ax.pie(
            [prob * 100, (1 - prob) * 100],
            startangle=90,
            counterclock=False,
            colors=[main_color, "#e2e8f0"],
            wedgeprops=dict(width=0.32, edgecolor="white", linewidth=2)
        )
        ax.text(0, 0.05, f"{prob*100:.1f}%", ha="center", va="center", fontsize=20, fontweight="bold", color="#0f172a")
        ax.text(0, -0.25, f"Ngưỡng: {threshold*100:.0f}%", ha="center", va="center", fontsize=9, color="#64748b")
        ax.axis("equal")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown("<div class='panel-card'><b>🔎 Tín hiệu giải thích từ hồ sơ:</b><br>", unsafe_allow_html=True)
        signals = []
        if YN_MAP[housing_label] == "no" and YN_MAP[loan_label] == "no":
            signals.append("🟢 Không có khoản vay nhà ở hay tiêu dùng (dư địa tài chính tốt).")
        else:
            signals.append("🟠 Đang có khoản vay nợ (giảm khả năng gửi tiết kiệm mới).")
        if pdays != -1:
            signals.append(f"🟢 Đã từng tương tác trước đây (cách {pdays} ngày, kết quả: **{poutcome_label}**).")
        else:
            signals.append("⚪ Khách hàng mới chưa từng tiếp cận ở chiến dịch trước.")
        if age > 60 or age < 26:
            signals.append("🟢 Nhóm tuổi đặc thù (trên 60 hoặc dưới 26) thường có tỷ lệ gửi tiết kiệm cao hơn trung bình.")
        if campaign > 4:
            signals.append("🔴 Tần suất gọi cao (> 4 lần) có xu hướng làm giảm tỷ lệ đồng ý.")
        for s in signals:
            st.write(s)
        st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.history:
        st.divider()
        h_col1, h_col2 = st.columns([4, 1])
        h_col1.markdown("#### 🕒 Lịch sử dự đoán trong phiên")
        hist_df = pd.DataFrame(st.session_state.history)
        h_col2.download_button(
            "⬇️ Tải lịch sử CSV",
            hist_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="lich_su_du_doan.csv",
            mime="text/csv",
            use_container_width=True
        )
        st.dataframe(hist_df, use_container_width=True, hide_index=True)

# ==================== TAB 2: DỰ ĐOÁN HÀNG LOẠT ====================
with tab2:
    st.markdown("#### Chấm điểm hàng loạt danh sách khách hàng")
    b_col1, b_col2 = st.columns([2, 1])
    uploaded_file = b_col1.file_uploader("Tải lên tệp dữ liệu khách hàng (CSV, hỗ trợ dấu `,` hoặc `;`)", type=["csv"])
    run_sample = b_col2.button("⚡ Chạy thử 200 mẫu từ bank-full.csv", use_container_width=True, type="primary")

    if uploaded_file is not None:
        try:
            st.session_state.batch_df = pd.read_csv(uploaded_file, sep=None, engine="python")
        except Exception as e:
            st.error(f"Không đọc được tệp CSV: {e}")
    elif run_sample:
        if raw_df is not None:
            st.session_state.batch_df = (
                raw_df.sample(200, random_state=42)
                .drop(columns=["y", "duration"], errors="ignore")
                .reset_index(drop=True)
            )
        else:
            st.warning("Chưa tìm thấy tệp data/bank-full.csv.")

    batch_df = st.session_state.get("batch_df", None)
    if batch_df is not None:
        required_cols = ["age", "job", "marital", "education", "default", "balance", "housing",
                         "loan", "contact", "day", "month", "campaign", "pdays", "previous", "poutcome"]
        missing = [c for c in required_cols if c not in batch_df.columns]
        if missing:
            st.error(f"Tệp CSV thiếu các cột bắt buộc: {missing}")
        else:
            probs = model.predict_proba(batch_df[required_cols])[:, 1]
            preds = (probs >= threshold).astype(int)

            out_df = translate_df_to_vi(batch_df[required_cols])
            out_df.insert(0, "Khuyến nghị", np.where(preds == 1, "✅ Nên gọi", "Bỏ qua"))
            out_df.insert(1, "Xác suất (%)", (probs * 100).round(2))

            k1, k2, k3, k4 = st.columns(4)
            k1.markdown(f"<div class='kpi-card'><div class='kpi-val'>{len(out_df)}</div><div class='kpi-lbl'>Tổng số khách hàng</div></div>", unsafe_allow_html=True)
            k2.markdown(f"<div class='kpi-card'><div class='kpi-val' style='color:#059669;'>{(preds==1).sum()}</div><div class='kpi-lbl'>Khách hàng nên gọi</div></div>", unsafe_allow_html=True)
            k3.markdown(f"<div class='kpi-card'><div class='kpi-val'>{(preds==1).mean()*100:.1f}%</div><div class='kpi-lbl'>Tỷ lệ đề xuất gọi</div></div>", unsafe_allow_html=True)
            k4.markdown(f"<div class='kpi-card'><div class='kpi-val'>{probs.mean()*100:.1f}%</div><div class='kpi-lbl'>Xác suất trung bình</div></div>", unsafe_allow_html=True)

            st.write("")
            only_yes = st.checkbox("Chỉ hiển thị danh sách khách hàng tiềm năng (Nên gọi)", value=False)
            view_df = out_df[preds == 1] if only_yes else out_df
            st.dataframe(view_df.sort_values("Xác suất (%)", ascending=False), use_container_width=True, hide_index=True)

            st.download_button(
                "⬇️ Tải xuống toàn bộ kết quả chấm điểm (.csv)",
                out_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="ket_qua_cham_diem_tien_gui.csv",
                mime="text/csv"
            )

# ==================== TAB 3: KHÁM PHÁ DỮ LIỆU (EDA) ====================
with tab3:
    st.markdown("#### Khám phá Tỷ lệ Đăng ký Tiền gửi theo từng nhóm đặc trưng")
    if raw_df is not None:
        eda_options = {
            "Nghề nghiệp": "job",
            "Trình độ học vấn": "education",
            "Tình trạng hôn nhân": "marital",
            "Kênh liên hệ": "contact",
            "Kết quả chiến dịch trước": "poutcome",
            "Tháng liên hệ": "month",
            "Vay mua nhà": "housing",
            "Vay tiêu dùng": "loan"
        }
        eda_label = st.selectbox("Chọn thuộc tính phân loại muốn phân tích:", list(eda_options.keys()))
        eda_col = eda_options[eda_label]

        val_map = {
            "job": REV_JOB, "education": REV_EDU, "marital": REV_MARITAL,
            "contact": REV_CONTACT, "poutcome": REV_POUT, "month": REV_MONTH,
            "housing": REV_YN, "loan": REV_YN
        }[eda_col]

        conv_df = (
            raw_df.assign(
                y_num=(raw_df["y"] == "yes").astype(int),
                nhom=raw_df[eda_col].map(val_map).fillna(raw_df[eda_col])
            )
            .groupby("nhom")["y_num"]
            .agg(Số_mẫu="count", Tỷ_lệ_đăng_ký=lambda s: s.mean() * 100)
            .sort_values("Tỷ_lệ_đăng_ký", ascending=True)
            .reset_index()
            .rename(columns={"nhom": eda_label, "Số_mẫu": "Số lượng mẫu", "Tỷ_lệ_đăng_ký": "Tỷ lệ đăng ký (%)"})
        )

        e1, e2 = st.columns([1.4, 1], gap="large")
        with e1:
            fig, ax = plt.subplots(figsize=(7, 4.4))
            fig.patch.set_alpha(0.0)
            ax.set_facecolor("#ffffff")
            colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(conv_df)))
            bars = ax.barh(conv_df[eda_label], conv_df["Tỷ lệ đăng ký (%)"], color=colors, height=0.65)
            for bar in bars:
                w = bar.get_width()
                ax.text(w + 0.5, bar.get_y() + bar.get_height()/2, f"{w:.1f}%", va="center", fontsize=9, fontweight="bold", color="#1e293b")
            ax.set_xlabel("Tỷ lệ đăng ký (%)", fontsize=10, color="#475569")
            ax.set_title(f"Tỷ lệ khách hàng đăng ký tiền gửi theo {eda_label}", fontsize=12, fontweight="bold", pad=12, color="#0f172a")
            ax.set_xlim(0, conv_df["Tỷ lệ đăng ký (%)"].max() * 1.18)
            ax.grid(axis="x", linestyle="--", alpha=0.35)
            ax.set_axisbelow(True)
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        with e2:
            st.dataframe(conv_df.sort_values("Tỷ lệ đăng ký (%)", ascending=False).round(2), use_container_width=True, hide_index=True)
    else:
        st.info("Chưa tìm thấy data/bank-full.csv để vẽ biểu đồ tương tác.")

# ==================== TAB 4: ĐỐI SÁNH MÔ HÌNH & PHÂN TÍCH LỖI ====================
with tab4:
    st.markdown("#### Bảng tổng hợp kết quả thực nghiệm (Luồng A vs. Luồng B)")
    comp_path = os.path.join("results", "model_comparison.csv")
    if os.path.exists(comp_path):
        comp_df = pd.read_csv(comp_path)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        img1, img2 = st.columns(2)
        if os.path.exists("results/model_comparison_f1.png"):
            img1.image("results/model_comparison_f1.png", caption="So sánh F1-Score giữa Luồng A và Luồng B")
        if os.path.exists("results/feature_importance.png"):
            img2.image("results/feature_importance.png", caption="Top 10 đặc trưng quan trọng nhất")
        if os.path.exists("results/confusion_matrices.png"):
            st.image("results/confusion_matrices.png", caption="Ma trận nhầm lẫn các mô hình tiêu biểu")

        st.divider()
        st.markdown("#### 🔍 Xem mẫu dự đoán sai (Error Analysis)")
        err_type = st.radio("Chọn nhóm lỗi:", ["False Positives (Dự đoán gọi nhầm)", "False Negatives (Bỏ sót khách tiềm năng)"], horizontal=True)
        err_file = "results/error_false_positives_sample.csv" if "False Positives" in err_type else "results/error_false_negatives_sample.csv"
        if os.path.exists(err_file):
            st.dataframe(translate_df_to_vi(pd.read_csv(err_file)), use_container_width=True, hide_index=True)
    else:
        st.info("Hãy chạy `python src/train_evaluate.py` trước để tạo dữ liệu đối sánh.")