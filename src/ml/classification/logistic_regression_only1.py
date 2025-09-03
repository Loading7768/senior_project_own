from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import loguniform  # 用來隨機抽取 C 值（對數分布）
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import os
from scipy import sparse
from sklearn.preprocessing import StandardScaler
import argparse
from collections import defaultdict
import joblib
import pickle
import gc


# === 匯入 config ===
from pathlib import Path
import sys
parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))
from config import JSON_DICT_NAME, COIN_SHORT_NAME

# === utils for FS ===
from ml.utils.feature_selection import make_selector


'''可修改參數'''
N_SAMPLES = 0  # 設定 random sampling 要取多少樣本數  (0: 取所有樣本)

N_RUNS = 10  # 設定 random sampling 要跑幾次

IS_TRAIN = True  # 看是否要訓練

INPUT_PATH = "../data/ml/dataset"

OUTPUT_PATH = "../data/ml/classification/logistic_regression"

SAVE_MODEL_PATH = "../data/ml/models/classification"
'''可修改參數'''

os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(SAVE_MODEL_PATH, exist_ok=True)







def get_random_samples_sparse(X: csr_matrix, y: np.ndarray, seed: int = 42):
    global N_SAMPLES
    n_total = X.shape[0]

    if N_SAMPLES == 0:
        print(f"已將樣本數設為最大值: {n_total} 筆")
        N_SAMPLES = n_total

    if N_SAMPLES > n_total:
        print(f"樣本數過多！最大只能 {n_total} 筆")
        N_SAMPLES = n_total

    samples = []
    for run in range(N_RUNS):
        np.random.seed(seed + run)
        indices = np.random.choice(n_total, N_SAMPLES, replace=False)
        X_sample = X[indices]             # 保持 sparse CSR matrix
        y_sample = y[indices]
        samples.append((X_sample, y_sample))

        print(f"[INFO] Run {run}: 抽樣後 X_train={X_sample.shape}, y_train={y_sample.shape}")
    
    return samples



# === 逐日逐幣種彙總 ===
def evaluate_by_coin_date(ids, y_true, y_pred):
    """
    ids: list/array of (coin, date, idx)
    y_true: 對應的真實標籤
    y_pred: 對應的預測結果
    """
    results = defaultdict(list)

    # 依照 (coin, date) 聚合
    for (coin, date, _), t, p in zip(ids, y_true, y_pred):
        results[(coin, date)].append((t, p))

    daily_summary = {}
    for (coin, date), samples in results.items():
        truths, preds = zip(*samples)
        true_label = truths[0]  # 同一幣種當天的漲幅應該是一樣的  因此取第一個元素即可

        up = sum(p == 1 for p in preds)
        down = sum(p == 0 for p in preds)
        total = len(preds)
        up_ratio = up / total * 100
        down_ratio = down / total * 100
        majority_pred = 1 if up >= down else 0
        majority_correct = (majority_pred == true_label)

        daily_summary.setdefault(coin, {})
        daily_summary[coin][date] = {
            "true_label": int(true_label),
            "up": up,
            "down": down,
            "total": total,
            "up_ratio": up_ratio,
            "down_ratio": down_ratio,
            "majority_pred": int(majority_pred),
            "majority_correct": int(majority_correct)
        }

    return daily_summary



# --- 訓練用函式 ---
def train_function(X_train, X_test, y_train, y_test, scaler, features_name, unprocessed_dates, selector, pipeline_path, i):
    
    all_results = []  # 儲存所有訓練結果
    best_test_acc = -1
    best_run_info = None

    # --- 分層隨機抽樣 50 萬 ---
    train_sample = get_random_samples_sparse(X_train, y_train)  # 裡面存[(X_sample), (y_sample), ...]

    # 定義模型
    base_model = LogisticRegression(
        solver='saga', 
        max_iter=100000, 
        verbose=1, 
        penalty='l2', 
        C = 0.001, 
        n_jobs=-1)
    
    model = OneVsRestClassifier(base_model, n_jobs=-1)

    # 定義參數分布（隨機抽樣）
    # param_dist = {
    #     'C': 0.001,   # C 值在 [0.001, 1000] 範圍隨機抽
    # }

    

    for run in range(N_RUNS):  # 總共訓練 N_RUNS 次
        # 隨機搜尋
        # random_search = RandomizedSearchCV(
        #     estimator=log_reg,
        #     param_distributions=param_dist,
        #     n_iter=1,             # 隨機挑 10 組
        #     scoring='accuracy',   # 評估方式
        #     cv=3,                 # 3 折交叉驗證
        #     verbose=2,
        #     random_state=42 + run,
        #     n_jobs=1             # 不使用多核心
        # )

        # 開始訓練
        X_train_sample, y_train_sample = train_sample[run]
        model.fit(X_train_sample, y_train_sample)

        # print("Random search 最佳參數:", random_search.best_params_)
        # print("Random search 最佳交叉驗證準確率:", random_search.best_score_)

        # best_model = random_search.best_estimator_

        # --- 評估 ---
        train_acc = accuracy_score(y_train_sample, model.predict(X_train_sample))
        test_acc = accuracy_score(y_test, model.predict(X_test))

        print(f"[RUN {run}] Train acc={train_acc:.4f}, Test acc={test_acc:.4f}")

        # --- 保存結果 ---
        all_results.append({
            "run": run,
            "train_acc": train_acc,
            "test_acc": test_acc,
            # "best_params": random_search.best_params_
        })

        # --- 更新最佳模型 ---
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_run_info = {
                "run": run,
                "model": model,
                "scaler": scaler,
                "selector": selector,
                "train_acc": train_acc,
                "test_acc": test_acc,
                # "params": random_search.best_params_
            }
        
        # === 強制清理 ===
        # del random_search
        # del best_model
        # gc.collect()


    # --- 全部結果輸出 ---
    results_df = pd.DataFrame(all_results)
    print("\n=== 所有 Run 的結果 ===")
    print(results_df)
    results_df.to_csv(f"{OUTPUT_PATH}/logreg_sampling_results{i}.csv", index=False)


    # --- 儲存最佳模型 ---
    joblib.dump({
        "model": best_run_info["model"],
        "scaler": best_run_info["scaler"],
        "selector": best_run_info["selector"]
    }, pipeline_path)

    print("\n=== 最佳模型 ===")
    print(f"Run {best_run_info['run']} | Train acc={best_run_info['train_acc']:.4f}, Test acc={best_run_info['test_acc']:.4f}")
    # print(f"最佳參數: {best_run_info['params']}")
    print(f"已儲存最佳 pipeline 到 {pipeline_path}")

    print("\n分類報告 (Test set):")
    print(classification_report(y_test, best_run_info["model"].predict(X_test)))



    # === 用最佳模型做輸出和預測 ===
    most_best_model = best_run_info["model"]


    # 關鍵字係數
    # coefficients = pd.Series(most_best_model.coef_[0], index=features_name).sort_values(ascending=False)
    # coeff_dict = coefficients.to_dict()

    # coeff_path = f"{OUTPUT_PATH}/logistic_regression_keyword_coefficients.json"
    # with open(coeff_path, "w", encoding="utf-8") as f:
    #     json.dump(coeff_dict, f, ensure_ascii=False, indent=4)

    # print(f"關鍵詞係數已存成 JSON：{coeff_path}")

    print("\n被排除的日期（沒有推文或無法計算價格變化）:")
    print(unprocessed_dates)

    # 最後一筆也無法計算（因為沒「明天」）
    # unprocessed_dates.append(df.loc[len(df)-1, "date"].strftime("%Y/%m/%d"))



# --- 預測用函式 ---
def predict_function(X_train, X_test, y_train, y_test, pipeline_path, i):
     # === 載入最佳模型 ===
    pipeline = joblib.load(pipeline_path)
    model = pipeline["model"]
    
    # === 預測所有樣本 ===
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # === 載入推文 ID 對應表 ===
    with open(f"{INPUT_PATH}/ids_train.pkl", "rb") as f:   # rb = read binary
        ids_train = pickle.load(f)
    with open(f"{INPUT_PATH}/ids_test.pkl", "rb") as f:   # rb = read binary
        ids_test = pickle.load(f)

    # 將 ids 轉成 np.array 方便接下來的處理
    ids_train = np.array(ids_train)
    ids_test = np.array(ids_test)

    


    # === 套用在 train / test ===
    train_daily = evaluate_by_coin_date(ids_train, y_train, y_pred_train)
    test_daily  = evaluate_by_coin_date(ids_test,  y_test,  y_pred_test)

    # === 存成 JSON ===
    with open(f"{OUTPUT_PATH}/logreg_train_daily_results{i}.json", "w", encoding="utf-8") as f:
        json.dump(train_daily, f, ensure_ascii=False, indent=4, default=int)

    with open(f"{OUTPUT_PATH}/logreg_test_daily_results{i}.json", "w", encoding="utf-8") as f:
        json.dump(test_daily, f, ensure_ascii=False, indent=4, default=int)

    print("已輸出逐日預測結果：")
    print(f"- train: {OUTPUT_PATH}/logreg_train_daily_results{i}.json")
    print(f"- test:  {OUTPUT_PATH}/logreg_test_daily_results{i}.json")

    # === 合併 train + test ===
    combined_daily = {}
    for coin, daily in train_daily.items():
        combined_daily.setdefault(coin, {}).update(daily)
    for coin, daily in test_daily.items():
        combined_daily.setdefault(coin, {}).update(daily)

    # === 存成合併後的 TXT ===
    txt_path = f"{OUTPUT_PATH}/logreg_combined_results{i}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        for coin, daily in combined_daily.items():
            f.write(f"\n=== {coin} ===\n")
            # 依日期排序
            for date, stats in sorted(daily.items()):
                true_label = "up" if stats["true_label"] == 1 else "down"
                pred_label = "up" if stats["majority_pred"] == 1 else "down"
                correct_mark = "✅" if stats["majority_correct"] == 1 else "❌"
                line = (
                    f"{date} → 👍 {stats['up']}  👎 {stats['down']}  📊 {stats['total']}  "
                    f"👍比: {stats['up_ratio']:.2f}%  👎比: {stats['down_ratio']:.2f}%  "
                    f"多數: {pred_label} | 真實: {true_label} {correct_mark}"
                )
                f.write(line + "\n")

    print(f"\n合併後的人類可讀版結果已輸出到：{txt_path}")

    # === 同時印到 console 預覽 ===
    with open(txt_path, "r", encoding="utf-8") as f:
        print("\n=== 輸出檔案內容預覽 ===\n")
        print(f.read())


def load_and_preprocess(y_train, y_test):
    # 取得 ML 的 X
    X_train = sparse.load_npz(f"{INPUT_PATH}/X_train_filtered.npz")
    X_test = sparse.load_npz(f"{INPUT_PATH}/X_test.npz")


    scaler = StandardScaler(with_mean=False)  # 適合 sparse matrix
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)


    # 取得 all_keywords(features_name)
    with open(f"{INPUT_PATH}/keyword/filtered_keywords.json", "r", encoding="utf-8-sig") as jsonfile:
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



    # === 特徵選擇 ===
    parser = argparse.ArgumentParser()
    parser.add_argument("--fs", type=str, default="none", help="Feature selection method")
    parser.add_argument("--k", type=int, default=600, help="Top k features")
    args = parser.parse_args()

    selector = make_selector(task="clf", method=args.fs, k=args.k)
    if selector is not None:
        X_train = selector.fit_transform(X_train, y_train)
        X_test = selector.transform(X_test)
        features_name = selector.get_feature_names_out(features_name)  # 更新 features_name
        print(f"[INFO] Feature selection ({args.fs}) done, X_train shape = {X_train.shape}")


    return X_train, X_test, y_train, y_test, scaler, features_name, unprocessed_dates, selector



def main():

    for i in range(5):  # 5 種 Y
        print(f"\n=== Training for Y set {i} ===")

        y_train_all = np.load(f"{INPUT_PATH}/Y_train_filtered.npz")
        y_train_all = y_train_all['Y']
        y_test_all = np.load(f"{INPUT_PATH}/Y_test.npz")
        y_test_all = y_test_all['Y']
        
        # 建立 target label：上漲為 1，否則為 0（二元分類）
        y_train = (y_train_all[:, i] >= 0).astype(int)
        y_test = (y_test_all[:, i] >= 0).astype(int)

        # --- 載入資料 ---
        X_train, X_test, y_train, y_test, scaler, features_name, unprocessed_dates, selector = load_and_preprocess(y_train, y_test)

        pipeline_path = f"{SAVE_MODEL_PATH}/logreg_best_pipeline{i}.joblib"  # 儲存訓練模型的位置

        if IS_TRAIN:
            # --- 訓練模型 --- 
            train_function(X_train, X_test, y_train, y_test, scaler, features_name, unprocessed_dates, selector, pipeline_path, i)

            # --- 預測模型 ---
            predict_function(X_train, X_test, y_train, y_test, pipeline_path, i)
        else:
            if not os.path.exists(pipeline_path):
                print("找不到已訓練好的模型，請先將 IS_TRAIN 設為 True")

            # --- 預測模型 ---
            predict_function(X_train, X_test, y_train, y_test, pipeline_path, i)



if __name__ == "__main__":
    main()