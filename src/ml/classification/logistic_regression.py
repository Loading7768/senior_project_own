from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import loguniform  # 用來隨機抽取 C 值（對數分布）
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import os
from scipy import sparse
from sklearn.preprocessing import StandardScaler
import argparse
from collections import defaultdict

# === 匯入 config ===
from pathlib import Path
import sys
parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))
from config import JSON_DICT_NAME, COIN_SHORT_NAME

# === utils for FS ===
from ml.utils.feature_selection import make_selector


# 取得 ML 的 X, Y
X_train = sparse.load_npz(f"../data/ml/dataset/X_train.npz")
X_test = sparse.load_npz(f"../data/ml/dataset/X_test.npz")
y_train = np.load(f"../data/ml/dataset/Y_train.npy")
y_test = np.load(f"../data/ml/dataset/Y_test.npy")

scaler = StandardScaler(with_mean=False)  # 適合 sparse matrix
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 建立 target label：上漲為 1，否則為 0（二元分類）
y_train = (y_train > 0).astype(int)
y_test = (y_test > 0).astype(int)

# 取得 all_keywords(features_name)
with open(f"../data/keyword/machine_learning/all_keywords.json", "r", encoding="utf-8-sig") as jsonfile:
    features_name = json.load(jsonfile)


# 取得 price 的 csv 檔
price_path = "../data/coin_price"
df = pd.read_csv(f"{price_path}/{COIN_SHORT_NAME}_current_tweet_price_output.csv")
df['date'] = pd.to_datetime(df['date'], format="%Y/%m/%d")  # 把 date 欄位轉成日期格式

# 把當天沒有抓到推文的日期存起來
unprocessed_dates = []
for i in range(len(df)):
    if df.loc[i, "has_tweet"] == False:
        unprocessed_dates.append(df.loc[i, "date"].strftime("%Y/%m/%d"))

        

parser = argparse.ArgumentParser()
parser.add_argument("--fs", type=str, default="none", help="Feature selection method")
parser.add_argument("--k", type=int, default=600, help="Top k features")
args = parser.parse_args()

# === 特徵選擇 ===
selector = make_selector(task="clf", method=args.fs, k=args.k)
if selector is not None:
    X_train = selector.fit_transform(X_train, y_train)
    X_test = selector.transform(X_test)
    print(f"[INFO] Feature selection ({args.fs}) done, X_train shape = {X_train.shape}")


# 定義模型
log_reg = LogisticRegression(solver='saga', max_iter=100000, verbose=1, penalty='l1', n_jobs=-1)

# 定義參數分布（隨機抽樣）
param_dist = {
    'C': loguniform(1e-3, 1e3),   # C 值在 [0.001, 1000] 範圍隨機抽
    'penalty': ['l1', 'l2']       # L1 / L2 正則化
}

# 隨機搜尋
random_search = RandomizedSearchCV(
    estimator=log_reg,
    param_distributions=param_dist,
    n_iter=5,             # 隨機挑 5 組（你可以改成 10 或更多）
    scoring='accuracy',   # 評估方式
    cv=3,                 # 3 折交叉驗證
    verbose=2,
    random_state=42,
    n_jobs=-1             # 多核心加速
)

# 開始訓練
random_search.fit(X_train, y_train)

print("最佳參數:", random_search.best_params_)
print("最佳交叉驗證準確率:", random_search.best_score_)

best_model = random_search.best_estimator_

# 在 train/test 評估
train_acc = accuracy_score(y_train, best_model.predict(X_train))
test_acc = accuracy_score(y_test, best_model.predict(X_test))

print(f"Train 準確率: {train_acc:.4f}")
print(f"Test 準確率: {test_acc:.4f}")

print("\n分類報告 (Test set):")
print(classification_report(y_test, best_model.predict(X_test)))


# 關鍵字係數
coefficients = pd.Series(best_model.coef_[0], index=features_name).sort_values(ascending=False)
coeff_dict = coefficients.to_dict()

output_file = "../data/ml/classification"
os.makedirs(output_file, exist_ok=True)
output_path = f"{output_file}/logistic_regression_keyword_coefficients.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(coeff_dict, f, ensure_ascii=False, indent=4)

print(f"已存成 JSON：{output_path}")

print("\n被排除的日期（沒有推文或無法計算價格變化）:")
print(unprocessed_dates)

# 最後一筆也無法計算（因為沒「明天」）
# unprocessed_dates.append(df.loc[len(df)-1, "date"].strftime("%Y/%m/%d"))




# === 預測所有樣本 ===
y_pred_train = best_model.predict(X_train)
y_pred_test = best_model.predict(X_test)

# === 載入日期對應表 (假設你有存每筆樣本的日期資訊) ===
# 例如：每筆推文在 dataset 的時候就有對應日期，存成 npy
train_dates = np.load("../data/ml/dataset/train_dates.npy")  # shape = (len(y_train),)
test_dates  = np.load("../data/ml/dataset/test_dates.npy")   # shape = (len(y_test),)

# === 逐日彙總 ===
def evaluate_by_day(dates, y_true, y_pred):
    day_results = defaultdict(list)

    for d, t, p in zip(dates, y_true, y_pred):
        day_results[d].append((t, p))  # 收集當天所有推文的 (真實標籤, 預測)

    daily_summary = {}
    for d, results in day_results.items():
        truths, preds = zip(*results)
        # 當天真實漲跌（應該所有推文都是同一個 label）
        true_label = truths[0]

        # 看當天的推文預測結果有幾個正確
        correct = sum(t == p for t, p in results)
        total = len(results)
        acc = correct / total

        # 也可以算「當天整體預測」：用多數決決定一天的結果
        majority_pred = int(np.mean(preds) >= 0.5)
        majority_correct = (majority_pred == true_label)

        daily_summary[d] = {
            "true_label": int(true_label),
            "tweets_total": total,
            "tweets_correct": correct,
            "tweets_acc": acc,
            "majority_pred": majority_pred,
            "majority_correct": int(majority_correct)
        }

    return daily_summary


train_daily = evaluate_by_day(train_dates, y_train, y_pred_train)
test_daily  = evaluate_by_day(test_dates,  y_test,  y_pred_test)

# === 存成 JSON ===
output_file = "../data/ml/classification"
os.makedirs(output_file, exist_ok=True)

with open(f"{output_file}/logreg_train_daily_results.json", "w", encoding="utf-8") as f:
    json.dump(train_daily, f, ensure_ascii=False, indent=4)

with open(f"{output_file}/logreg_test_daily_results.json", "w", encoding="utf-8") as f:
    json.dump(test_daily, f, ensure_ascii=False, indent=4)

print("已輸出逐日預測結果：")
print(f"- train: {output_file}/logreg_train_daily_results.json")
print(f"- test:  {output_file}/logreg_test_daily_results.json")