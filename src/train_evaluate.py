import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

RANDOM_STATE = 42
os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)
sns.set_theme(style="whitegrid", font_scale=1.0)

# 1. ĐỌC DỮ LIỆU & CHIA TẬP TRAIN/TEST (BỎ CỘT duration GÂY DATA LEAKAGE)
df = pd.read_csv(os.path.join("data", "bank-full.csv"), sep=";")
X = df.drop(columns=["y", "duration"])
y = (df["y"] == "yes").astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
print(f"Tập Train: {X_train.shape[0]} mẫu | Tập Test: {X_test.shape[0]} mẫu")

# 2. LUỒNG A: PIPELINE CƠ BẢN (BASELINE)
num_cols_A = ["age", "balance", "day", "campaign", "pdays", "previous"]
cat_cols_A = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"]

preprocessor_A = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols_A),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols_A),
    ]
)

# 3. LUỒNG B: PIPELINE CẢI TIẾN (FEATURE ENGINEERING + ROBUST SCALER)
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

num_cols_B = ["age", "balance", "day", "campaign", "previous", "was_contacted_before", "has_any_loan"]
ord_cols_B = ["education"]
nom_cols_B = ["job", "marital", "default", "housing", "loan", "contact", "month", "poutcome", "age_group"]

preprocessor_B = Pipeline(steps=[
    ("feat_eng", BankFeatureEngineer()),
    ("col_trans", ColumnTransformer(
        transformers=[
            ("num", RobustScaler(), num_cols_B),
            ("ord", OrdinalEncoder(
                categories=[["unknown", "primary", "secondary", "tertiary"]],
                handle_unknown="use_encoded_value", unknown_value=-1
            ), ord_cols_B),
            ("nom", OneHotEncoder(handle_unknown="ignore", sparse_output=False), nom_cols_B),
        ]
    ))
])

# 4. CẤU HÌNH MÔ HÌNH & LƯỚI THAM SỐ
models_config = {
    "K-NN": (
        KNeighborsClassifier(),
        {"clf__n_neighbors": [5, 11, 21], "clf__weights": ["uniform", "distance"]}
    ),
    "Naive Bayes": (
        GaussianNB(),
        {"clf__var_smoothing": [1e-9, 1e-7, 1e-5]}
    ),
    "Decision Tree": (
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        {
            "clf__criterion": ["gini", "entropy"],
            "clf__max_depth": [5, 8, 12],
            "clf__class_weight": [None, "balanced"]
        }
    ),
    "Random Forest (Mở rộng)": (
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        {"clf__n_estimators": [100], "clf__max_depth": [10, 15], "clf__class_weight": ["balanced"]}
    )
}

pipelines_to_test = {
    "Luồng A (Cơ bản)": preprocessor_A,
    "Luồng B (Cải tiến)": preprocessor_B
}

# 5. HUẤN LUYỆN & ĐÁNH GIÁ
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
results = []
trained_models = {}
best_f1 = -1.0
best_pipeline = None
best_name = ""

for flow_name, preproc in pipelines_to_test.items():
    for model_name, (clf, param_grid) in models_config.items():
        print(f"Đang huấn luyện: {flow_name} - {model_name}...")
        t0 = time.time()
        pipe = Pipeline(steps=[("prep", preproc), ("clf", clf)])
        grid = GridSearchCV(pipe, param_grid=param_grid, cv=cv, scoring="f1", n_jobs=-1)
        grid.fit(X_train, y_train)
        elapsed = round(time.time() - t0, 1)

        best_est = grid.best_estimator_
        y_pred = best_est.predict(X_test)
        y_prob = best_est.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)

        results.append({
            "Luồng xử lý": flow_name,
            "Thuật toán": model_name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
            "ROC-AUC": round(auc, 4),
            "Thời gian (s)": elapsed,
            "Tham số tốt nhất": str(grid.best_params_)
        })
        trained_models[f"{flow_name} | {model_name}"] = (best_est, y_pred)

        if f1 > best_f1:
            best_f1 = f1
            best_pipeline = best_est
            best_name = f"{flow_name} - {model_name}"

results_df = pd.DataFrame(results)
print("\n=== BẢNG TỔNG HỢP KẾT QUẢ TRÊN TẬP TEST ===")
print(results_df.drop(columns=["Tham số tốt nhất"]).to_string(index=False))
results_df.to_csv("results/model_comparison.csv", index=False)

joblib.dump(best_pipeline, "models/best_bank_model.pkl")
print(f"\n[LƯU MÔ HÌNH] Mô hình tốt nhất: {best_name} (F1 = {best_f1:.4f}) -> models/best_bank_model.pkl")

# 6. VẼ BIỂU ĐỒ SO SÁNH F1-SCORE & CONFUSION MATRIX
plt.figure(figsize=(10, 5.5))
ax = sns.barplot(data=results_df, x="Thuật toán", y="F1-Score", hue="Luồng xử lý", palette=["#4C72B0", "#DD8452"])
for p in ax.patches:
    h = p.get_height()
    if h > 0:
        ax.annotate(f"{h:.3f}", (p.get_x() + p.get_width() / 2., h),
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
plt.title("So sánh F1-Score (lớp yes) giữa Luồng A và Luồng B", pad=12)
plt.ylim(0, results_df["F1-Score"].max() * 1.2)
plt.tight_layout()
plt.savefig("results/model_comparison_f1.png", dpi=200)
plt.close()

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
cm_keys = [
    "Luồng A (Cơ bản) | Decision Tree",
    "Luồng B (Cải tiến) | Decision Tree",
    "Luồng B (Cải tiến) | Random Forest (Mở rộng)"
]
for i, k in enumerate(cm_keys):
    _, pred_k = trained_models[k]
    cm = confusion_matrix(y_test, pred_k)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[i],
                xticklabels=["No (0)", "Yes (1)"], yticklabels=["No (0)", "Yes (1)"])
    axes[i].set_title(k.replace(" | ", "\n"), fontsize=10)
    axes[i].set_xlabel("Dự đoán")
    axes[i].set_ylabel("Thực tế")
plt.tight_layout()
plt.savefig("results/confusion_matrices.png", dpi=200)
plt.close()

# 7. TRÍCH XUẤT TOP ĐẶC TRƯNG QUAN TRỌNG (NẾU LÀ CÂY QUYẾT ĐỊNH / RANDOM FOREST)
clf_best = best_pipeline.named_steps["clf"]
if hasattr(clf_best, "feature_importances_"):
    feat_names = best_pipeline.named_steps["prep"].named_steps["col_trans"].get_feature_names_out() if "Luồng B" in best_name else best_pipeline.named_steps["prep"].get_feature_names_out()
    imp_df = pd.DataFrame({"Feature": feat_names, "Importance": clf_best.feature_importances_})
    imp_df = imp_df.sort_values("Importance", ascending=False).head(10)
    plt.figure(figsize=(8, 5))
    sns.barplot(data=imp_df, x="Importance", y="Feature", hue="Feature", palette="viridis", legend=False)
    plt.title(f"Top 10 đặc trưng quan trọng nhất ({best_name})")
    plt.tight_layout()
    plt.savefig("results/feature_importance.png", dpi=200)
    plt.close()

# 8. PHÂN TÍCH LỖI (ERROR ANALYSIS)
_, y_pred_best = trained_models[best_name.replace(" - ", " | ")]
error_df = X_test.copy()
error_df["Actual_y"] = y_test.values
error_df["Predicted_y"] = y_pred_best

fp = error_df[(error_df["Actual_y"] == 0) & (error_df["Predicted_y"] == 1)]
fn = error_df[(error_df["Actual_y"] == 1) & (error_df["Predicted_y"] == 0)]

print(f"\n=== PHÂN TÍCH LỖI TRÊN {best_name} ===")
print(f"- False Positive (Dự đoán đăng ký nhưng thực tế từ chối): {len(fp)} mẫu")
print(f"- False Negative (Bỏ sót khách hàng thực tế có đăng ký): {len(fn)} mẫu")
fp.head(15).to_csv("results/error_false_positives_sample.csv", index=False)
fn.head(15).to_csv("results/error_false_negatives_sample.csv", index=False)
print("[HOÀN TẤT] Đã lưu toàn bộ kết quả và biểu đồ vào thư mục results/!")