# Đề tài 10: Dự đoán khách hàng đăng ký tiền gửi ngân hàng (FIT4013)

**Nhóm thực hiện:** Nhóm 16  
**Học phần:** Nhập môn Học máy (FIT4013)

## 1. Nguồn dữ liệu
- **Bộ dữ liệu:** Bank Marketing Dataset (`bank-full.csv` — 45.211 mẫu, 17 thuộc tính).
- **Liên kết tải chính thức:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- **Cách chuẩn bị:** Tải `bank+marketing.zip`, giải nén tệp `bank.zip` bên trong và đặt `bank-full.csv` vào thư mục `data/`.

## 2. Cấu trúc thư mục
- `data/`: Chứa dữ liệu đầu vào `bank-full.csv`.
- `src/eda.py`: Khám phá dữ liệu, kiểm tra Data Leakage và xuất 5 biểu đồ EDA vào `results/`.
- `src/train_evaluate.py`: Huấn luyện, đối sánh **Luồng A (Cơ bản)** và **Luồng B (Cải tiến)** qua `GridSearchCV` (K-NN, Naive Bayes, Decision Tree, Random Forest), lưu mô hình tốt nhất vào `models/best_bank_model.pkl`.
- `app.py`: Ứng dụng Web Demo (Streamlit) gồm 4 tab: Dự đoán Cá nhân (VNĐ, chỉnh Threshold), Dự đoán Hàng loạt (CSV), Khám phá Dữ liệu (EDA) và Đối sánh / Error Analysis.
- `results/`: Chứa toàn bộ biểu đồ (`.png`), bảng kết quả (`model_comparison.csv`) và mẫu lỗi (`FP`/`FN`).
- `report/`: Chứa mã nguồn báo cáo LaTeX (`main.tex`).

## 3. Kết quả thực nghiệm nổi bật
- Loại bỏ thuộc tính `duration` để tránh Data Leakage trước khi gọi điện.
- Mô hình tốt nhất: **Random Forest (Luồng B - Cải tiến)** đạt **F1-Score = 0.4577**, **Recall = 0.5652**, **ROC-AUC = 0.7971**.

## 4. Hướng dẫn cài đặt và chạy lại từ đầu
```powershell
python -m venv venv
.\venv\Scripts\activate
python -m pip install -r requirements.txt

# Bước 1: Khám phá dữ liệu và xuất biểu đồ EDA
python src/eda.py

# Bước 2: Huấn luyện, đánh giá và lưu mô hình
python src/train_evaluate.py

# Bước 3: Khởi chạy ứng dụng Demo
python -m streamlit run app.py
```