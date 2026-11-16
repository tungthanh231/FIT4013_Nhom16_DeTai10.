import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Cấu hình giao diện biểu đồ rõ nét để đưa vào báo cáo LaTeX
sns.set_theme(style="whitegrid", font_scale=1.05)
os.makedirs("results", exist_ok=True)

# 1. Đọc dữ liệu
df = pd.read_csv(os.path.join("data", "bank-full.csv"), sep=";")

print("=== 1. TỔNG QUAN DỮ LIỆU ===")
print(f"Số dòng: {df.shape[0]} | Số cột: {df.shape[1]}")
print(f"Số dòng trùng lặp hoàn toàn: {df.duplicated().sum()}")
print(f"Số giá trị NaN chuẩn: {df.isna().sum().sum()}")

print("\n=== 2. PHÂN PHỐI BIẾN MỤC TIÊU (y) ===")
target_counts = df["y"].value_counts()
target_pct = df["y"].value_counts(normalize=True) * 100
for k in target_counts.index:
    print(f"  - {k}: {target_counts[k]} mẫu ({target_pct[k]:.2f}%)")

print("\n=== 3. THỐNG KÊ GIÁ TRỊ 'unknown' & ĐẶC BIỆT ===")
for col in df.select_dtypes(include=["object", "string"]).columns:
    unk = (df[col] == "unknown").sum()
    if unk > 0:
        print(f"  - {col}: {unk} mẫu 'unknown' ({unk / len(df) * 100:.2f}%)")
print(f"  - pdays == -1 (chưa từng liên hệ trước đây): {(df['pdays'] == -1).sum()} mẫu ({(df['pdays'] == -1).mean() * 100:.2f}%)")
print(f"  - duration == 0: {(df['duration'] == 0).sum()} mẫu")

print("\n=== 4. THỐNG KÊ MÔ TẢ CÁC BIẾN SỐ ===")
num_cols = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
print(df[num_cols].describe().round(2).to_string())

# --- BIỂU ĐỒ 1: PHÂN PHỐI BIẾN MỤC TIÊU ---
plt.figure(figsize=(6, 4.5))
ax = sns.countplot(data=df, x="y", hue="y", palette=["#4C72B0", "#DD8452"], legend=False)
for p in ax.patches:
    h = p.get_height()
    ax.annotate(f"{int(h):,}\n({h/len(df)*100:.1f}%)",
                (p.get_x() + p.get_width() / 2., h),
                ha="center", va="bottom", fontsize=10, fontweight="bold")
plt.title("Phân phối biến mục tiêu y (Đăng ký tiền gửi)", pad=14)
plt.xlabel("Khách hàng đăng ký (y)")
plt.ylabel("Số lượng mẫu")
plt.ylim(0, target_counts.max() * 1.15)
plt.tight_layout()
plt.savefig("results/eda_target_distribution.png", dpi=200)
plt.close()

# --- BIỂU ĐỒ 2: HISTOGRAM CÁC ĐẶC TRƯNG SỐ ---
fig, axes = plt.subplots(2, 4, figsize=(15, 7.5))
axes = axes.flatten()
for i, col in enumerate(num_cols):
    sns.histplot(df[col], bins=35, kde=True, ax=axes[i], color="#2b6cb0")
    axes[i].set_title(f"Phân phối {col}")
    axes[i].set_xlabel("")
axes[-1].axis("off")
plt.suptitle("Histogram các đặc trưng số (bank-full.csv)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("results/eda_numeric_histograms.png", dpi=200)
plt.close()

# --- BIỂU ĐỒ 3: BOXPLOT PHÁT HIỆN NGOẠI LỆ ---
fig, axes = plt.subplots(2, 4, figsize=(15, 7))
axes = axes.flatten()
for i, col in enumerate(num_cols):
    sns.boxplot(x=df[col], ax=axes[i], color="#63b3ed")
    axes[i].set_title(f"Boxplot {col}")
    axes[i].set_xlabel("")
axes[-1].axis("off")
plt.suptitle("Phát hiện giá trị ngoại lệ (Outliers) trên các biến số", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("results/eda_numeric_boxplots.png", dpi=200)
plt.close()

# --- BIỂU ĐỒ 4: BIẾN PHÂN LOẠI QUAN TRỌNG THEO NHÃN y ---
cat_focus = ["job", "education", "contact", "poutcome"]
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
axes = axes.flatten()
for i, col in enumerate(cat_focus):
    order = df[col].value_counts().index
    sns.countplot(data=df, y=col, hue="y", order=order, ax=axes[i], palette=["#4C72B0", "#DD8452"])
    axes[i].set_title(f"Tần suất '{col}' theo nhãn y")
    axes[i].set_ylabel("")
plt.tight_layout()
plt.savefig("results/eda_categorical_features.png", dpi=200)
plt.close()

# --- BIỂU ĐỒ 5: TƯƠNG QUAN & CHỨNG MINH DATA LEAKAGE (duration) ---
df_corr = df[num_cols].copy()
df_corr["target_y"] = (df["y"] == "yes").astype(int)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.heatmap(df_corr.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=axes[0])
axes[0].set_title("Ma trận tương quan các biến số & target_y")

sns.boxplot(data=df, x="y", y="duration", hue="y", palette=["#4C72B0", "#DD8452"], legend=False, ax=axes[1])
axes[1].set_title("Thời lượng cuộc gọi (duration) theo nhãn y (Data Leakage)")
axes[1].set_xlabel("Nhãn mục tiêu (y)")
axes[1].set_ylabel("Thời lượng cuộc gọi (giây)")
plt.tight_layout()
plt.savefig("results/eda_correlation_and_leakage.png", dpi=200)
plt.close()

print("\n[HOÀN TẤT] Đã lưu 5 biểu đồ EDA vào thư mục results/!")