from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import os

# 取得 ML 的 feature_vector
features_path = "../data/keyword/machine_learning"
features_vector = np.load(f"{features_path}/feature_vector.npy")
with open(f"{features_path}/feature_name.json", "r", encoding="utf-8-sig") as jsonfile:
    features_name = json.load(jsonfile)

# 取得 price 的 csv 檔
price_path = "../data/coin_price"
df = pd.read_csv(f"{price_path}/current_tweet_price_output.csv")
df['date'] = pd.to_datetime(df['date'], format="%Y/%m/%d")

# 取得 price_diff 的 npy 檔
price_diff = np.load(f"{price_path}/price_diff.npy")

# 建立 target label
labels = (price_diff > 0).astype(int)

# 把當天沒有抓到推文的日期存起來
unprocessed_dates = []
for i in range(len(df)):
    if not df.loc[i, "has_tweet"]:
        unprocessed_dates.append(df.loc[i, "date"].strftime("%Y/%m/%d"))

# 過濾 features_vector 和 labels
X = features_vector
Y = labels

# 資料拆分 60% train / 20% val / 20% test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, Y, test_size=0.4, random_state=42, stratify=Y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# ================= SVM =================
model = LinearSVC(
    penalty='l1',      # 使用 L1 正則化
    dual=False,        # L1 必須 dual=False
    max_iter=10000000,
    tol=1e-4,
    verbose=1
)

model.fit(X_train, y_train)

# 計算準確率
train_acc = accuracy_score(y_train, model.predict(X_train))
val_acc = accuracy_score(y_val, model.predict(X_val))
test_acc = accuracy_score(y_test, model.predict(X_test))

print(f"Train 準確率: {train_acc:.4f}")
print(f"Validation 準確率: {val_acc:.4f}")
print(f"Test 準確率: {test_acc:.4f}")

# 測試集詳細報告
print("\n分類報告 (Test set):")
print(classification_report(y_test, model.predict(X_test)))

# ========== 畫 overfitting 圖 ==========
output_dir = "../outputs/figures/ml/classification"
os.makedirs(output_dir, exist_ok=True)

plt.figure(figsize=(6,4))
plt.bar(["Train", "Validation"], [train_acc, val_acc], color=["skyblue", "orange"])
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("Train vs Validation Accuracy (SVM)")
plt.savefig(os.path.join(output_dir, "svm_overfitting_check.png"))
plt.close()

# ========== 關鍵字係數輸出 ==========
coefficients = pd.Series(model.coef_[0], index=features_name).sort_values(ascending=False)
coeff_dict = coefficients.to_dict()

output_file = "../data/ml/classification"
os.makedirs(output_file, exist_ok=True)
output_path = f"{output_file}/svm_keyword_coefficients.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(coeff_dict, f, ensure_ascii=False, indent=4)

print(f"已存成 JSON：{output_path}")
print("\n被排除的日期（沒有推文或無法計算價格變化）:")
print(unprocessed_dates)
