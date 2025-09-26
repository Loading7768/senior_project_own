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
from tqdm import tqdm


# === utils for FS ===
# from ml.utils.feature_selection import make_selector


'''可修改參數'''
N_SAMPLES = 1_000_000  # 設定 random sampling 要取多少樣本數  (0: 取所有樣本 且 不做 random sampling)

N_RUNS = 10  # 設定 random sampling 要跑幾次

# LABELS = 5  # 有多少標籤

# C = 0.18360757138767084
C = 0.01

T1 = 0.0590 # 0.1

T2 = 0.0102 # 0.00125

T3 = 0.0060

T4 = 0.0657

PRICE_CSV_PATH = "../data/coin_price"

INPUT_PATH = "../data/ml/dataset"

OUTPUT_PATH = "../data/ml/classification/logistic_regression"

SAVE_MODEL_PATH = "../data/ml/models/classification"

MODEL_NAME = "logreg"  # 第二個分類器目前輸入的模型名字

RUN_FIRST_CLASSIFIER = True  # 是否要跑第一個分類器

RUN_SECOND_CLASSIFIER = False  # 是否要跑第二個分類器

IS_GROUPED_CV = False  # 是否要跑第二個分類器的交叉驗證

IS_TRAIN = True  # 看是否要訓練

IS_FILTERED = True  # 看是否有分 normal 與 bot

IS_RUN_AUGUST = False  # 看現在是不是要跑 2025/08 的資料
'''可修改參數'''

os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(SAVE_MODEL_PATH, exist_ok=True)

SUFFIX_FILTERED = "" if IS_FILTERED else "_non_filtered"
SUFFIX_AUGUST   = "_202508" if IS_RUN_AUGUST else ""



def get_random_samples_sparse_stratified(X: csr_matrix, y: np.ndarray, seed: int = 42):
    """
    X: csr_matrix
    y: np.ndarray, shape=(N,)  多類別標籤
    """
    global N_SAMPLES
    # global ENABLE_SAMPLING
    n_total = X.shape[0]

    if N_SAMPLES == 0:
        print(f"[INFO] 不做 random sampling，使用所有樣本數: {n_total} 筆")
        # ENABLE_SAMPLING = False
        return [(X, y)]  # 回傳一個原始數量的 (X, y) tuple

    classes = np.unique(y)
    n_classes = len(classes)
    if N_SAMPLES < n_classes:
        raise ValueError(f"樣本數 {N_SAMPLES} 太少，無法平均分配到每個類別 ({n_classes})")
    
    samples_per_class = N_SAMPLES // n_classes

    # 建立索引字典
    class_indices = defaultdict(list)
    for idx, label in enumerate(y):
        class_indices[label].append(idx)

    samples = []
    for run in range(N_RUNS):
        np.random.seed(seed + run)
        selected_indices = []

        for c in classes:
            idx_list = class_indices[c]
            if len(idx_list) <= samples_per_class:
                # 如果該類別數量不夠，就全部拿
                selected_indices.extend(idx_list)
            else:
                selected_indices.extend(np.random.choice(idx_list, samples_per_class, replace=False))

        # 如果總數少於 N_SAMPLES，從剩餘樣本補足
        if len(selected_indices) < N_SAMPLES:
            # set(range(n_total)) 是所有樣本索引（0 ~ n_total-1）   set(selected_indices) 是已被選過的索引集合
            remaining_idx = list(set(range(n_total)) - set(selected_indices))
            remaining_needed = N_SAMPLES - len(selected_indices)
            selected_indices.extend(np.random.choice(remaining_idx, remaining_needed, replace=False))

        np.random.shuffle(selected_indices)  # 打亂順序
        X_sample = X[selected_indices]
        y_sample = y[selected_indices]
        samples.append((X_sample, y_sample))

        # === 新增：統計類別數量與比例 ===
        unique, counts = np.unique(y_sample, return_counts=True)
        total = len(y_sample)
        print(f"\n[INFO] Run {run}: Stratified sample X_train={X_sample.shape}, y_train={y_sample.shape}")
        for cls, cnt in zip(unique, counts):
            pct = cnt / total * 100
            print(f"   Class {cls}: {cnt} samples ({pct:.2f}%)")

    return samples



# def stratified_train_test_balance(X, y, ids, max_per_class=None, seed=42):
#     labels, counts = np.unique(y, return_counts=True)
#     print("\n各類別數量（np.unique）:")
#     for label, count in zip(labels, counts):
#         print(f"類別 {label}: {count} 筆")

#     # 儲存 index
#     train_idx = []
#     test_idx = []

#     # 找出每個類別的所有 index
#     class_indices = defaultdict(list)
#     for idx, label in enumerate(y):
#         class_indices[label].append(idx)

#     rng = np.random.default_rng(seed)  # 建立固定 seed 的隨機生成器

#     for label, indices in class_indices.items():
#         indices = np.array(indices)
#         rng.shuffle(indices)  # 使用 rng.shuffle 代替 np.random.shuffle
#         train_samples = indices[:max_per_class]
#         test_samples = indices[max_per_class:]

#         train_idx.extend(train_samples)
#         test_idx.extend(test_samples)

#     # 依照 index 取出資料
#     X_train = X[train_idx]
#     y_train = y[train_idx]
#     ids_train = ids[train_idx]

#     X_test = X[test_idx]
#     y_test = y[test_idx]
#     ids_test = ids[test_idx]

#     return X_train, X_test, y_train, y_test, ids_train, ids_test



# 新增一個函式來平衡訓練集
# def balance_train_data(X_train, y_train, ids_train):
    classes = np.unique(y_train)
    class_indices = defaultdict(list)
    for idx, label in enumerate(y_train):
        class_indices[label].append(idx)
    
    min_class_count = min(len(indices) for indices in class_indices.values())
    
    balanced_indices = []
    for c in classes:
        idx_list = class_indices[c]
        np.random.shuffle(idx_list)
        balanced_indices.extend(idx_list[:min_class_count])
    
    np.random.shuffle(balanced_indices)
    
    X_train_balanced = X_train[balanced_indices]
    y_train_balanced = y_train[balanced_indices]
    ids_train_balanced = [ids_train[i] for i in balanced_indices]
    
    return X_train_balanced, y_train_balanced, ids_train_balanced



# def evaluate_by_coin_date_2_category(ids, y_true, y_pred):
    """
    ids: list/array of (coin, date, idx)
    y_true: shape (N, num_labels) 或 (N,) 對應真實標籤
    y_pred: shape (N, num_labels) 或 (N,) 對應預測結果
    """
    results = defaultdict(list)

    # 將樣本依照 (coin, date) 聚合
    for (coin, date, _), t, p in zip(ids, y_true, y_pred):
        results[(coin, date)].append((t, p))

    daily_summary = {}
    for (coin, date), samples in results.items():
        truths, preds = zip(*samples)
        truths = np.array(truths)
        preds  = np.array(preds)

        # 如果是單標籤，轉成 2D 方便統一處理
        if truths.ndim == 1:
            truths = truths[:, None]
        if preds.ndim == 1:
            preds = preds[:, None]

        num_labels = truths.shape[1]
        majority_pred = []

        for i in range(num_labels):
            up   = np.sum(preds[:, i] == 1)
            down = np.sum(preds[:, i] == 0)
            majority_pred.append(1 if up >= down else 0)

        majority_pred = np.array(majority_pred)
        # 同一天的真實標籤取第一個樣本 (假設每天同幣種漲跌相同)
        true_label = truths[0]

        # 計算每個標籤是否正確
        majority_correct = majority_pred == true_label

        # --- 轉換成符號 ---
        pred_symbols = ["🟢" if p == 1 else "🔴" for p in majority_pred]
        true_symbols = ["✅" if c else "❌" for c in majority_correct]

        daily_summary.setdefault(coin, {})
        daily_summary[coin][date] = {
            "true_label": true_label.tolist(),
            "majority_pred": majority_pred.tolist(),
            "majority_correct": majority_correct.tolist(),
            "up_counts": np.sum(preds == 1, axis=0).tolist(),
            "down_counts": np.sum(preds == 0, axis=0).tolist(),
            "total_counts": len(preds),
            "pred_symbols": pred_symbols,
            "true_symbols": true_symbols,
        }

    return daily_summary, num_labels



def evaluate_by_coin_date(ids, y_true, y_pred):
    LABEL_SYMBOLS = {
        0: "🔴",  # 大跌
        1: "🟠",    # 跌
        2: "⚪",    # 持平
        3: "🟡",    # 漲
        4: "🟢"   # 大漲
    }

    if RUN_FIRST_CLASSIFIER:
        results = defaultdict(list)

        # 聚合
        for (coin, date, _), t, p in zip(ids, y_true, y_pred):
            results[(coin, date)].append((t, p))

        daily_summary = {}
        for (coin, date), samples in results.items():
            truths, preds = zip(*samples)
            truths = np.array(truths)
            preds  = np.array(preds)

            # 多數決
            values, counts = np.unique(preds, return_counts=True)
            majority_pred = values[np.argmax(counts)]

            true_label = truths[0]  # 假設同一天真實標籤一致
            correct = (majority_pred == true_label)

            # 計算每個標籤是否正確
            # majority_correct = majority_pred == true_label


            daily_summary.setdefault(coin, {})

            # 將各類別出現次數轉成 list（保持原本 up_counts/down_counts 的感覺）
            class_counts = [np.sum(preds == i) for i in range(5)]  # 0~4 五類
            pred_symbols = [LABEL_SYMBOLS[majority_pred]]           # 單一預測符號

            # majority_pred = int(majority_pred)  # 轉成 Python scalar
            # true_label = int(true_label)        # 轉成 Python scalar
            # true_symbols = ["✅" if majority_pred == true_label else "❌"]

            true_symbols   = [LABEL_SYMBOLS[int(true_label)]]   # 真實符號
            result_symbols = ["✅" if correct else "❌"]         # 對錯符號


            daily_summary[coin][date] = {
                "true_label": int(true_label),
                "majority_pred": int(majority_pred),
                "majority_correct": bool(correct),
                "class_counts": class_counts,    # 替代 up_counts/down_counts
                "total_counts": len(preds),      # 原本 total_counts
                "pred_symbols": pred_symbols,
                "true_symbols": true_symbols,     # 真實符號
                "result_symbols": result_symbols  # 對錯符號
            }

        return daily_summary, len(np.unique(y_true))
    
    # --- 未完成 ---
    elif RUN_SECOND_CLASSIFIER:
        daily_summary = {}

        for (coin, date), t, p in zip(ids, y_true, y_pred):
            correct = (p == t)

            # 各類別計數 (這裡因為只有一筆，只有一個類別會是 1，其餘都是 0)
            class_counts = [1 if p == i else 0 for i in range(5)]

            daily_summary.setdefault(coin, {})
            daily_summary[coin][date] = {
                "true_label": int(t),
                "majority_pred": int(p),
                "majority_correct": bool(correct),
                "class_counts": class_counts,
                "total_counts": 1,
                "pred_symbols": [LABEL_SYMBOLS[int(p)]],
                "true_symbols": [LABEL_SYMBOLS[int(t)]],
                "result_symbols": ["✅" if correct else "❌"]
            }

        return daily_summary, len(np.unique(y_true))



# --- 訓練用函式 ---
def train_function(X_train, X_test, y_train, y_test, pipeline_path, scaler = None, features_name = None):

    # print(f"\n=== Training label column {count} ===")
    # y_train = y_train[:, count]
    # y_test  = y_test[:, count]

        
    all_results = []  # 儲存所有訓練結果
    best_test_acc = -1
    best_run_info = None

    # 定義模型
    log_reg = LogisticRegression(
        solver='saga', 
        max_iter=100000, 
        verbose=1, 
        penalty='l2', 
        C = C, 
        n_jobs=-1,
        # tol=1e-6  # 收斂容忍度 (越小越嚴格，訓練可能更久)
    )
    
    # model = OneVsRestClassifier(log_reg, n_jobs=-1)

    # 定義參數分布（隨機抽樣）
    # param_dist = {
    #     'C': 0.001,   # C 值在 [0.001, 1000] 範圍隨機抽
    # }

    
    if RUN_FIRST_CLASSIFIER:
        # --- 分層隨機抽樣 ---
        train_sample = get_random_samples_sparse_stratified(X_train, y_train)  # 裡面存[(X_sample, y_sample), ...]

        # 總共訓練 N_RUNS 次 (但是以 train_sample 的長度判斷，所以若沒有用 random sampling 跑一次便會執行完成)
        run_count = len(train_sample)

    elif RUN_SECOND_CLASSIFIER:
        train_sample = [(X_train, y_train)]
        run_count = 1

    else:
        raise ValueError("請設定 RUN_FIRST_CLASSIFIER 或 RUN_SECOND_CLASSIFIER")

    # --- 執行 N_RUNS 次 Random Sampling (無則執行一次) ---
    for run in range(run_count):  
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
        log_reg.fit(X_train_sample, y_train_sample)

        # print("Random search 最佳參數:", random_search.best_params_)
        # print("Random search 最佳交叉驗證準確率:", random_search.best_score_)

        # best_model = random_search.best_estimator_

        # --- 評估 ---
        train_acc = accuracy_score(y_train_sample, log_reg.predict(X_train_sample))
        test_acc = accuracy_score(y_test, log_reg.predict(X_test))

        print(f"[RUN {run}] Train acc={train_acc:.4f}, Test acc={test_acc:.4f}")

        # --- 保存結果 ---
        all_results.append({
            "run": run,
            "train_acc": train_acc,
            "test_acc": test_acc,
            # "best_params": random_search.best_params_
        })

        # --- 更新最佳模型 ---
        if (RUN_FIRST_CLASSIFIER and test_acc > best_test_acc) or RUN_SECOND_CLASSIFIER:
            best_test_acc = test_acc
            best_run_info = {
                "run": run,
                "model": log_reg,
                "scaler": scaler,
                "train_acc": train_acc,
                "test_acc": test_acc
                # "params": random_search.best_params_
            }


    # --- 全部結果輸出 ---
    results_df = pd.DataFrame(all_results)
    print("\n=== 所有 Run 的結果 ===")
    print(results_df)

    if RUN_FIRST_CLASSIFIER:
        results_df.to_csv(f"{OUTPUT_PATH}/logreg_sampling_results_{N_SAMPLES}{SUFFIX_FILTERED}.csv", index=False)
    elif RUN_SECOND_CLASSIFIER:
        results_df.to_csv(f"{OUTPUT_PATH}/logreg_classifier_2_results{SUFFIX_FILTERED}{SUFFIX_AUGUST}.csv", index=False)

    # 儲存最佳模型
    model_dict = {"model": best_run_info["model"]}
    if scaler is not None:
        model_dict["scaler"] = scaler
    joblib.dump(model_dict, pipeline_path)

    print("\n=== 最佳模型 ===")
    print(f"Run {best_run_info['run']} | Train acc={best_run_info['train_acc']:.4f}, Test acc={best_run_info['test_acc']:.4f}")
    # print(f"最佳參數: {best_run_info['params']}")
    print(f"已儲存最佳 pipeline 到 {pipeline_path}")

    # Test 集分類報告
    print("\n分類報告 (Test set):")
    print(classification_report(y_test, best_run_info["model"].predict(X_test)))



    # === 用最佳模型做輸出和預測 ===
    # most_best_model = best_run_info["model"]


    # 關鍵字係數
    # coefficients = pd.Series(most_best_model.coef_[0], index=features_name).sort_values(ascending=False)
    # coeff_dict = coefficients.to_dict()

    # coeff_path = f"{OUTPUT_PATH}/logistic_regression_keyword_coefficients.json"
    # with open(coeff_path, "w", encoding="utf-8") as f:
    #     json.dump(coeff_dict, f, ensure_ascii=False, indent=4)

    # print(f"關鍵詞係數已存成 JSON：{coeff_path}")

    # print("\n被排除的日期（沒有推文或無法計算價格變化）:")
    # print(unprocessed_dates)

    # 最後一筆也無法計算（因為沒「明天」）
    # unprocessed_dates.append(df.loc[len(df)-1, "date"].strftime("%Y/%m/%d"))



def coin_month_cv(X, y, ids, C):
    # ids_classifier_2 是 list/array，格式 [(coin, date), ...]
    ids_array = np.array(ids)
    coins, dates = ids_array[:,0], ids_array[:,1]
    dates = pd.to_datetime(dates)  # 轉 datetime
    months = dates.to_period("M")   # 取得月份，如 2025-01

    # 生成 (coin, month) 標籤
    coin_month_labels = np.array([f"{c}_{m}" for c, m in zip(coins, months)])

    unique_groups = np.unique(coin_month_labels)  # 所有幣種每月組合
    results_all = []

    for group in unique_groups:
        # 留出當前幣種月份
        test_mask = coin_month_labels == group
        train_mask = ~test_mask

        X_train_cv, X_test_cv = X[train_mask], X[test_mask]
        y_train_cv, y_test_cv = y[train_mask], y[test_mask]
        ids_train_cv, ids_test_cv = [ids[i] for i in range(len(ids)) if train_mask[i]], [ids[i] for i in range(len(ids)) if test_mask[i]]

        # 訓練 Logistic Regression
        model = LogisticRegression(
            solver='saga', 
            max_iter=100000, 
            penalty='l2', 
            C=C, 
            n_jobs=-1
        )
        model.fit(X_train_cv, y_train_cv)

        # 評估
        y_pred = model.predict(X_test_cv)
        acc = accuracy_score(y_test_cv, y_pred)
        print(f"[CV] Group {group} | Test acc: {acc:.4f}")

        results_all.append({
            "group": group,
            "test_acc": acc,
            "y_true": y_test_cv,
            "y_pred": y_pred,
            "ids_test": ids_test_cv
        })

    all_accs = [r['test_acc'] for r in results_all]
    print(f"\nAverage CV accuracy: {np.mean(all_accs):.4f}")

    return results_all



# --- 預測用函式 ---
def predict_function(X_train, X_test, y_train, y_test, ids_train, ids_test, pipeline_path):

    # y_train = y_train[:, count]
    # y_test  = y_test[:, count]

    # === 載入最佳模型 ===
    pipeline = joblib.load(pipeline_path)
    model = pipeline["model"]
    
    # === 預測所有樣本 ===
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    print(y_pred_train.shape)
    print(y_pred_test.shape)

    # # === 載入推文 ID 對應表 ===
    # with open(f"{INPUT_PATH}/ids_train.pkl", "rb") as f:   # rb = read binary
    #     ids_train = pickle.load(f)
    # with open(f"{INPUT_PATH}/ids_test.pkl", "rb") as f:   # rb = read binary
    #     ids_test = pickle.load(f)

    # 將 ids 轉成 np.array 方便接下來的處理
    ids_train = np.array(ids_train)
    ids_test = np.array(ids_test)

    
    print("\n分類報告 (Test set):")
    print(classification_report(y_test, model.predict(X_test), zero_division=0))


    # === 套用在 train / test ===
    train_daily, _ = evaluate_by_coin_date(ids_train, y_train, y_pred_train)
    test_daily, _  = evaluate_by_coin_date(ids_test,  y_test,  y_pred_test)

    if RUN_FIRST_CLASSIFIER:

        # === 存成 JSON ===
        with open(f"{OUTPUT_PATH}/logreg_train_daily_results_{N_SAMPLES}{SUFFIX_FILTERED}{SUFFIX_AUGUST}.json", "w", encoding="utf-8") as f:
            json.dump(train_daily, f, ensure_ascii=False, indent=4, default=int)

        with open(f"{OUTPUT_PATH}/logreg_test_daily_results_{N_SAMPLES}{SUFFIX_FILTERED}{SUFFIX_AUGUST}.json", "w", encoding="utf-8") as f:
            json.dump(test_daily, f, ensure_ascii=False, indent=4, default=int)

        print("已輸出逐日預測結果：")
        print(f"- train: {OUTPUT_PATH}/logreg_train_daily_results_{N_SAMPLES}{SUFFIX_FILTERED}{SUFFIX_AUGUST}.json")
        print(f"- test:  {OUTPUT_PATH}/logreg_test_daily_results_{N_SAMPLES}{SUFFIX_FILTERED}{SUFFIX_AUGUST}.json")

        # === 合併 train + test ===
        combined_daily = {}
        for coin, daily in train_daily.items():
            combined_daily.setdefault(coin, {}).update(daily)
        for coin, daily in test_daily.items():
            combined_daily.setdefault(coin, {}).update(daily)

        # === 存成合併後的 TXT ===
        txt_path = f"{OUTPUT_PATH}/logreg_combined_results_{N_SAMPLES}{SUFFIX_FILTERED}{SUFFIX_AUGUST}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            # === 初始化統計器 ===
            label_correct = np.zeros(1, dtype=int)
            label_total   = np.zeros(1, dtype=int)

            for coin, daily in combined_daily.items():
                f.write(f"\n=== {coin} ===\n")

                # 用來存放每天的 (date, pred_class)
                records = []

                for date, stats in sorted(daily.items()):
                    # --- 每日輸出到 TXT ---
                    class_str = " ".join(f"{x:5d}" for x in stats['class_counts'])
                    line = (
                        f"{date} → 📊 {class_str}  "
                        f"總數: {stats['total_counts']:5d}  "
                        f"預測: {''.join(stats['pred_symbols'])}  "
                        f"真實: {''.join(stats['true_symbols'])}  "
                        f"結果: {''.join(stats['result_symbols'])}\n"
                    )
                    f.write(line)

                    # --- 更新累積準確率 ---
                    label_total[0] += 1
                    if stats["majority_correct"]:
                        label_correct[0] += 1

                    # --- 取當天預測類別 (class_counts 最大的 index) ---
                    pred_class = int(np.argmax(stats["class_counts"]))
                    records.append((date, pred_class))

                # --- 輸出整體準確率 (百分比) ---
                accuracy_summary = " ".join(
                    f"{(c / t * 100):.2f}%" if t > 0 else "N/A"
                    for c, t in zip(label_correct, label_total)
                )
                f.write(f"\n整體準確率: {accuracy_summary}\n")

                # === 存成 .npy (每日預測結果，依日期排序) ===
                if records:
                    records.sort(key=lambda x: x[0])
                    _, preds = zip(*records)
                    preds = np.array(preds, dtype=np.int32)

                    npy_path = f"{OUTPUT_PATH}/{coin}_logreg_classifier_1_result{SUFFIX_FILTERED}{SUFFIX_AUGUST}.npy"
                    np.save(npy_path, preds)
                    print(preds[:50])
                    print(f"{coin} → {npy_path} 已完成, shape={preds.shape}")


        print(f"\n合併後的人類可讀版結果已輸出到：{txt_path}")

    elif RUN_SECOND_CLASSIFIER:
        # === 存成 JSON ===
        with open(f"{OUTPUT_PATH}/logreg_train_daily_classifier_2_results{SUFFIX_FILTERED}{SUFFIX_AUGUST}.json", "w", encoding="utf-8") as f:
            json.dump(train_daily, f, ensure_ascii=False, indent=4, default=int)

        with open(f"{OUTPUT_PATH}/logreg_test_daily_classifier_2_results{SUFFIX_FILTERED}{SUFFIX_AUGUST}.json", "w", encoding="utf-8") as f:
            json.dump(test_daily, f, ensure_ascii=False, indent=4, default=int)

        print("已輸出逐日預測結果：")
        print(f"- train: {OUTPUT_PATH}/logreg_train_daily_classifier_2_results{SUFFIX_FILTERED}{SUFFIX_AUGUST}.json")
        print(f"- test:  {OUTPUT_PATH}/logreg_test_daily_classifier_2_results{SUFFIX_FILTERED}{SUFFIX_AUGUST}.json")

        # === 合併 train + test ===
        combined_daily = {}
        for coin, daily in train_daily.items():
            combined_daily.setdefault(coin, {}).update(daily)
        for coin, daily in test_daily.items():
            combined_daily.setdefault(coin, {}).update(daily)

        # === 存成合併後的 TXT ===
        txt_path = f"{OUTPUT_PATH}/logreg_combined_classifier_2_results{SUFFIX_FILTERED}{SUFFIX_AUGUST}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            label_correct = 0
            label_total = 0

            for coin, daily in combined_daily.items():
                f.write(f"\n=== {coin} ===\n")

                records = []
                for date, stats in sorted(daily.items()):
                    # --- 每日輸出到 TXT ---
                    line = (
                        f"{date} → "
                        f"預測: {''.join(stats['pred_symbols'])}  "
                        f"真實: {''.join(stats['true_symbols'])}  "
                        f"結果: {''.join(stats['result_symbols'])}\n"
                    )
                    f.write(line)

                    # --- 更新累積準確率 ---
                    label_total += 1
                    if stats["majority_correct"]:
                        label_correct += 1

                    # --- 保存每日預測類別 ---
                    records.append((date, stats["majority_pred"]))

                # --- 輸出整體準確率 ---
                acc = (label_correct / label_total * 100) if label_total > 0 else 0
                f.write(f"\n整體準確率: {acc:.2f}%\n")

        print(f"\n合併後的人類可讀版結果已輸出到：{txt_path}")



def predict_august_function(pipeline_path):
    combined_daily = {}  # 用來放 合併 三種幣種 的資料 ===

    # --- 載入資料 ---
    for coin_short_name in ['DOGE', 'PEPE', 'TRUMP']:
        if RUN_FIRST_CLASSIFIER:
            X_august = sparse.load_npz(f'{INPUT_PATH}/keyword/{coin_short_name}_X_sparse{SUFFIX_FILTERED}{SUFFIX_AUGUST}.npz')
            y_august = np.load(f'{INPUT_PATH}/coin_price/{coin_short_name}_price_diff{SUFFIX_FILTERED}{SUFFIX_AUGUST}.npy')
            with open(f'{INPUT_PATH}/keyword/{coin_short_name}_ids{SUFFIX_FILTERED}{SUFFIX_AUGUST}.pkl', 'rb') as file:
                ids_august = pickle.load(file)

        elif RUN_SECOND_CLASSIFIER:
            X_august = np.load(f"{INPUT_PATH}/keyword/{coin_short_name}_{MODEL_NAME}_X_classifier_2{SUFFIX_FILTERED}{SUFFIX_AUGUST}.npy")
            y_august = np.load(f"{INPUT_PATH}/coin_price/{coin_short_name}_price_diff_original{SUFFIX_FILTERED}{SUFFIX_AUGUST}.npy")
            with open(f"{INPUT_PATH}/keyword/{coin_short_name}_{MODEL_NAME}_ids_classifier_2{SUFFIX_FILTERED}{SUFFIX_AUGUST}.pkl", 'rb') as file:
                ids_august = pickle.load(file)

        y_august_categorized = categorize_array_multi(y_august, T1, T2, T3, T4)

        # === 載入最佳模型 ===
        pipeline = joblib.load(pipeline_path)
        model = pipeline["model"]
        
        # === 預測所有樣本 ===
        y_pred_august = model.predict(X_august)
        print(y_pred_august.shape)

        # 將 ids 轉成 np.array 方便接下來的處理
        ids_august = np.array(ids_august)

        
        print(f"\n分類報告 ({coin_short_name} August set):")
        print(classification_report(y_august_categorized, y_pred_august, zero_division=0))

        # august_score = knn.score(X_august, Y_august)
        print(f'{coin_short_name} August accuracy')  

        print("ids_august[:5]", ids_august[:5])

        august_daily, _ = evaluate_by_coin_date(ids_august, y_august_categorized, y_pred_august)

        if RUN_FIRST_CLASSIFIER:
            # === 存成 JSON ===
            with open(f"{OUTPUT_PATH}/{coin_short_name}_logreg_august_daily_results_{N_SAMPLES}{SUFFIX_FILTERED}{SUFFIX_AUGUST}.json", "w", encoding="utf-8") as f:
                json.dump(august_daily, f, ensure_ascii=False, indent=4, default=int)

            print("已輸出逐日預測結果：")
            print(f"- august: {OUTPUT_PATH}/{coin_short_name}_logreg_august_daily_results_{N_SAMPLES}{SUFFIX_FILTERED}{SUFFIX_AUGUST}.json")

            # === 合併 三種幣種 ===
            for coin, daily in august_daily.items():
                combined_daily.setdefault(coin, {}).update(daily)

            # === 存成合併後的 TXT ===
            txt_path = f"{OUTPUT_PATH}/logreg_combined_results_{N_SAMPLES}{SUFFIX_FILTERED}{SUFFIX_AUGUST}.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                # === 初始化統計器 ===
                label_correct = np.zeros(1, dtype=int)
                label_total   = np.zeros(1, dtype=int)

                for coin, daily in combined_daily.items():
                    f.write(f"\n=== {coin} ===\n")

                    # 用來存放每天的 (date, pred_class)
                    records = []

                    for date, stats in sorted(daily.items()):
                        # --- 每日輸出到 TXT ---
                        class_str = " ".join(f"{x:5d}" for x in stats['class_counts'])
                        line = (
                            f"{date} → 📊 {class_str}  "
                            f"總數: {stats['total_counts']:5d}  "
                            f"預測: {''.join(stats['pred_symbols'])}  "
                            f"真實: {''.join(stats['true_symbols'])}  "
                            f"結果: {''.join(stats['result_symbols'])}\n"
                        )
                        f.write(line)

                        # --- 更新累積準確率 ---
                        label_total[0] += 1
                        if stats["majority_correct"]:
                            label_correct[0] += 1

                        # --- 取當天預測類別 (class_counts 最大的 index) ---
                        pred_class = int(np.argmax(stats["class_counts"]))
                        records.append((date, pred_class))

                    # --- 輸出整體準確率 (百分比) ---
                    accuracy_summary = " ".join(
                        f"{(c / t * 100):.2f}%" if t > 0 else "N/A"
                        for c, t in zip(label_correct, label_total)
                    )
                    f.write(f"\n整體準確率: {accuracy_summary}\n")

                    # === 存成 .npy (每日預測結果，依日期排序) ===
                    if records:
                        records.sort(key=lambda x: x[0])
                        _, preds = zip(*records)
                        preds = np.array(preds, dtype=np.int32)

                        npy_path = f"{OUTPUT_PATH}/{coin}_logreg_classifier_1_result{SUFFIX_FILTERED}{SUFFIX_AUGUST}.npy"
                        np.save(npy_path, preds)
                        print(preds[:50])
                        print(f"{coin} → {npy_path} 已完成, shape={preds.shape}")


            print(f"\n合併後的人類可讀版結果已輸出到：{txt_path}")

        elif RUN_SECOND_CLASSIFIER:

            # === 存成 JSON ===
            with open(f"{OUTPUT_PATH}/{coin_short_name}_logreg_train_daily_classifier_2_results{SUFFIX_FILTERED}{SUFFIX_AUGUST}.json", "w", encoding="utf-8") as f:
                json.dump(august_daily, f, ensure_ascii=False, indent=4, default=int)

            print("已輸出逐日預測結果：")
            print(f"- august: {OUTPUT_PATH}/{coin_short_name}_logreg_train_daily_classifier_2_results{SUFFIX_FILTERED}{SUFFIX_AUGUST}.json")

            # === 合併 三種幣種 ===
            for coin, daily in august_daily.items():
                combined_daily.setdefault(coin, {}).update(daily)

            # === 存成合併後的 TXT ===
            txt_path = f"{OUTPUT_PATH}/logreg_combined_classifier_2_results{SUFFIX_FILTERED}{SUFFIX_AUGUST}.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                label_correct = 0
                label_total = 0

                for coin, daily in combined_daily.items():
                    f.write(f"\n=== {coin} ===\n")

                    records = []
                    for date, stats in sorted(daily.items()):
                        # --- 每日輸出到 TXT ---
                        line = (
                            f"{date} → "
                            f"預測: {''.join(stats['pred_symbols'])}  "
                            f"真實: {''.join(stats['true_symbols'])}  "
                            f"結果: {''.join(stats['result_symbols'])}\n"
                        )
                        f.write(line)

                        # --- 更新累積準確率 ---
                        label_total += 1
                        if stats["majority_correct"]:
                            label_correct += 1

                        # --- 保存每日預測類別 ---
                        records.append((date, stats["majority_pred"]))

                    # --- 輸出整體準確率 ---
                    acc = (label_correct / label_total * 100) if label_total > 0 else 0
                    f.write(f"\n整體準確率: {acc:.2f}%\n")

            print(f"\n合併後的人類可讀版結果已輸出到：{txt_path}")



def categorize_array_multi(Y, t1, t2, t3, t4, ids=None):
    """
    Y: np.ndarray, shape = (num_labels,), 價格變化率
    t1, t2: 五元分類閾值，百分比
    """

    print(Y.shape)
    # print(len(ids))

    # 五元分類
    labels = np.full_like(Y, 2, dtype=int)  # 預設持平
    labels[Y <= -t1] = 0  # 大跌
    labels[(Y > -t1) & (Y <= -t2)] = 1  # 跌
    labels[(Y >= t3) & (Y < t4)] = 3  # 漲
    labels[Y >= t4] = 4  # 大漲

    if ids is not None:
        # 找出 Y==0 的索引
        zero_idx = np.where(Y == 0)[0]
        # 只取對應的 ids
        dates_is_0 = set((ids[i][0], ids[i][1]) for i in zero_idx)
        if len(dates_is_0) > 0:
            print(f"共有 {len(dates_is_0)} 天 Y==0")
            for id in sorted(dates_is_0):
                print(id)

    if np.any(Y == 0):  # 檢查是否有任何元素等於 0
        count = np.sum(Y == 0)
        print(f"共有 {count} 個 Y == 0")
        labels[Y == 0] = 4  # 為了校正 TRUMP 前兩天的價格相同 第一天設為大漲

    return labels



def load_and_preprocess():
    if RUN_FIRST_CLASSIFIER:
        # 取得 ML 的 X
        X_train = sparse.load_npz(f"{INPUT_PATH}/X_train{SUFFIX_FILTERED}.npz")
        X_test = sparse.load_npz(f"{INPUT_PATH}/X_test{SUFFIX_FILTERED}.npz")

        print(X_train.shape)

        # 匯入 Y
        y_train = np.load(f"{INPUT_PATH}/Y_train{SUFFIX_FILTERED}.npz")
        y_train = y_train['Y']
        y_test = np.load(f"{INPUT_PATH}/Y_test{SUFFIX_FILTERED}.npz")
        y_test = y_test['Y']

        print(y_train.shape)

        with open(f"{INPUT_PATH}/ids_train{SUFFIX_FILTERED}.pkl", 'rb') as file:
            ids_train = pickle.load(file)
        with open(f"{INPUT_PATH}/ids_test{SUFFIX_FILTERED}.pkl", 'rb') as file:
            ids_test = pickle.load(file)

        scaler = StandardScaler(with_mean=False)  # 適合 sparse matrix
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # 取得 all_keywords(features_name)
        with open(f"{INPUT_PATH}/keyword/filtered_keywords{SUFFIX_FILTERED}.json", "r", encoding="utf-8-sig") as jsonfile:
            features_name = json.load(jsonfile)


        # # 取得 price 的 csv 檔
        # price_path = "../data/coin_price"
        # df = pd.read_csv(f"{price_path}/{COIN_SHORT_NAME}_current_tweet_price_output.csv")
        # df['date'] = pd.to_datetime(df['date'], format="%Y/%m/%d")  # 把 date 欄位轉成日期格式

        # # 把當天沒有抓到推文的日期存起來
        # unprocessed_dates = []
        # for i in range(len(df)):
        #     if df.loc[i, "has_tweet"] == False:
        #         unprocessed_dates.append(df.loc[i, "date"].strftime("%Y/%m/%d"))



        # === 特徵選擇 ===
        # parser = argparse.ArgumentParser()
        # parser.add_argument("--fs", type=str, default="none", help="Feature selection method")
        # parser.add_argument("--k", type=int, default=600, help="Top k features")
        # args = parser.parse_args()

        # selector = make_selector(task="clf", method=args.fs, k=args.k)
        # if selector is not None:
        #     X_train = selector.fit_transform(X_train, y_train_categorized)
        #     X_test = selector.transform(X_test)
        #     features_name = selector.get_feature_names_out(features_name)  # 更新 features_name
        #     print(f"[INFO] Feature selection ({args.fs}) done, X_train shape = {X_train.shape}")
    
    elif RUN_SECOND_CLASSIFIER:
        # 取得資料
        X = np.load(f"{INPUT_PATH}/{MODEL_NAME}_X_classifier_2{SUFFIX_FILTERED}{SUFFIX_AUGUST}.npy")
        y = np.load(f"{INPUT_PATH}/{MODEL_NAME}_Y_classifier_2{SUFFIX_FILTERED}{SUFFIX_AUGUST}.npy")
        with open(f"{INPUT_PATH}/{MODEL_NAME}_ids_classifier_2{SUFFIX_FILTERED}{SUFFIX_AUGUST}.pkl", 'rb') as file:
            ids = pickle.load(file)

        X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
            X, y, ids, test_size=0.2, random_state=42, shuffle=True
        )

        # X_train, y_train, ids_train = balance_train_data(X_train, y_train, ids_train)

        # # 建立 target label：五元分類
        # y_categorized = categorize_array_multi(y, ids, T1, T2, T3, T4)  # shape (N,)
        # print("已成功分類別")

        # # 分割成 Train / Test
        # X_train, X_test, y_train_categorized, y_test_categorized, ids_train, ids_test = stratified_train_test_balance(
        #     X, y_categorized, ids, max_per_class=303
        # )

        print("X_test shape:", X_test.shape)
        print("y_test shape:", y_test.shape)


        scaler = None
        features_name = None

        print("X_train shape:", X_train.shape)
        print("X_test shape:", X_test.shape)
        print("y_train shape:", y_train.shape)
        print("y_test shape:", y_test.shape)
        print("Train IDs count:", len(ids_train))
        print("Test IDs count:", len(ids_test))

    else:
        raise ValueError("必須指定 run_first_classifier 或 run_second_classifier")
    
    # 建立 target label：五元分類
    y_train_categorized = categorize_array_multi(y_train, T1, T2, T3, T4, ids_train)  # shape (N,)
    y_test_categorized  = categorize_array_multi(y_test, T1, T2, T3, T4, ids_test)   # shape (N,)
    print("已成功分類別")

    # 統計每個類別數量
    print(f"大跌：-{T1 * 100:.2f}%以下, 跌：-{T1 * 100:.2f}% ~ -{T2 * 100}%, 持平：-{T2 * 100}% ~ {T3 * 100}%, 漲：{T3 * 100}% ~ {T4 * 100:.2f}%, 大漲：{T4 * 100:.2f}%以上")
    train_total_row = y_train_categorized.shape[0]
    test_total_row = y_test_categorized.shape[0]
    # for col in range(y_train_categorized.shape[1]):
    counts = np.bincount(y_train_categorized, minlength=5)
    percentages = counts / train_total_row * 100
    percentages_str = " ".join([f"{p:.2f}%" for p in percentages])
    print(f"[TRAIN] column 類別: {percentages_str}")

    counts = np.bincount(y_test_categorized, minlength=5)
    percentages = counts / test_total_row * 100
    percentages_str = " ".join([f"{p:.2f}%" for p in percentages])
    print(f"[TEST]  column 類別: {percentages_str}\n")

    input("pasue...")

    return X_train, X_test, y_train_categorized, y_test_categorized, ids_train, ids_test, scaler, features_name



def main():

    if RUN_FIRST_CLASSIFIER:

        pipeline_path = f"{SAVE_MODEL_PATH}/logreg_best_pipeline_{N_SAMPLES}{SUFFIX_FILTERED}.joblib"  # 儲存訓練模型的位置

        if not IS_RUN_AUGUST:
            # --- 載入資料 ---
            X_train, X_test, y_train, y_test, ids_train, ids_test, scaler, features_name = load_and_preprocess()

            # for count in range(LABELS):

            if IS_TRAIN:
                # --- 訓練模型 --- 
                train_function(X_train, X_test, y_train, y_test, pipeline_path, scaler, features_name)

                # --- 預測模型 ---
                predict_function(X_train, X_test, y_train, y_test, ids_train, ids_test, pipeline_path)
            else:
                if not os.path.exists(pipeline_path):
                    print("找不到已訓練好的 第一個分類器 模型，請先將 IS_TRAIN 設為 True")

                # --- 預測模型 ---
                predict_function(X_train, X_test, y_train, y_test, ids_train, ids_test, pipeline_path)

        else:
            # --- 預測 2025-08 ---
            predict_august_function(pipeline_path)

    elif RUN_SECOND_CLASSIFIER:

        pipeline_path = f"{SAVE_MODEL_PATH}/logreg_classifier_2{SUFFIX_FILTERED}.joblib"  # 儲存訓練模型的位置

        if not IS_RUN_AUGUST:
            if IS_GROUPED_CV == False:
                # --- 載入資料 ---
                X_train, X_test, y_train, y_test, ids_train, ids_test, _, _ = load_and_preprocess()

                if IS_TRAIN:
                    # --- 訓練模型 --- 
                    train_function(X_train, X_test, y_train, y_test, pipeline_path)

                    # --- 預測模型 ---
                    predict_function(X_train, X_test, y_train, y_test, ids_train, ids_test, pipeline_path)
                else:
                    if not os.path.exists(pipeline_path):
                        print("找不到已訓練好的 第二個分類器 模型，請先將 IS_TRAIN 設為 True")

                    # --- 預測模型 ---
                    predict_function(X_train, X_test, y_train, y_test, ids_train, ids_test, pipeline_path)

            else:
                # 取得資料
                X = np.load(f"{INPUT_PATH}/{MODEL_NAME}_X_classifier_2{SUFFIX_FILTERED}{SUFFIX_AUGUST}.npy")
                y = np.load(f"{INPUT_PATH}/{MODEL_NAME}_Y_classifier_2{SUFFIX_FILTERED}{SUFFIX_AUGUST}.npy")
                with open(f"{INPUT_PATH}/{MODEL_NAME}_ids_classifier_2{SUFFIX_FILTERED}{SUFFIX_AUGUST}.pkl", 'rb') as file:
                    ids = pickle.load(file)

                y_categorized = categorize_array_multi(y, ids, T1, T2, T3, T4)  # shape (N,)

                results_all = coin_month_cv(X, y_categorized, ids, C=C)

        else:
            # --- 預測 2025-08 ---
            predict_august_function(pipeline_path)  



if __name__ == "__main__":
    main()