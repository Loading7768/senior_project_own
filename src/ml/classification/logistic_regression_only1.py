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
from ml.utils.feature_selection import make_selector


'''可修改參數'''
N_SAMPLES = 1_000_000  # 設定 random sampling 要取多少樣本數  (0: 取所有樣本 且 不做 random sampling)

N_RUNS = 10  # 設定 random sampling 要跑幾次

LABELS = 5  # 有多少標籤

IS_TRAIN = True  # 看是否要訓練

# C = 0.18360757138767084
C = 0.01

T1 = 0.1

T2 = 0.00125

PRICE_CSV_PATH = "../data/coin_price"

INPUT_PATH = "../data/ml/dataset"

OUTPUT_PATH = "../data/ml/classification/logistic_regression"

SAVE_MODEL_PATH = "../data/ml/models/classification"
'''可修改參數'''

os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(SAVE_MODEL_PATH, exist_ok=True)

ENABLE_SAMPLING = True





# def get_random_samples_sparse(X: csr_matrix, y: np.ndarray, seed: int = 42):
#     global N_SAMPLES, ENABLE_SAMPLING
#     n_total = X.shape[0]

#     if N_SAMPLES == 0:
#         print(f"[INFO] 不做 random sampling，使用所有樣本數: {n_total} 筆")
#         ENABLE_SAMPLING = False
#         return [(X, y)] * N_RUNS   # 直接複製 N_RUNS 份，保持迴圈結構一致

#     if N_SAMPLES > n_total:
#         raise ValueError(f"樣本數過多！最大只能 {n_total} 筆")

#     samples = []
#     for run in range(N_RUNS):
#         np.random.seed(seed + run)
#         indices = np.random.choice(n_total, N_SAMPLES, replace=False)
#         X_sample = X[indices]             # 保持 sparse CSR matrix
#         y_sample = y[indices]
#         samples.append((X_sample, y_sample))

#         print(f"[INFO] Run {run}: 抽樣後 X_train={X_sample.shape}, y_train={y_sample.shape}")
    
#     return samples


def get_random_samples_sparse_stratified(X: csr_matrix, y: np.ndarray, seed: int = 42):
    """
    X: csr_matrix
    y: np.ndarray, shape=(N,)  多類別標籤
    """
    global N_SAMPLES, ENABLE_SAMPLING
    n_total = X.shape[0]

    if N_SAMPLES == 0:
        print(f"[INFO] 不做 random sampling，使用所有樣本數: {n_total} 筆")
        ENABLE_SAMPLING = False
        return [(X, y)] * N_RUNS

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





# def evaluate_by_coin_date(ids, y_true, y_pred):
#     """
#     ids: list/array of (coin, date, idx)
#     y_true: shape (N, num_labels) 或 (N,) 對應真實標籤
#     y_pred: shape (N, num_labels) 或 (N,) 對應預測結果
#     """
#     results = defaultdict(list)

#     # 將樣本依照 (coin, date) 聚合
#     for (coin, date, _), t, p in zip(ids, y_true, y_pred):
#         results[(coin, date)].append((t, p))

#     daily_summary = {}
#     for (coin, date), samples in results.items():
#         truths, preds = zip(*samples)
#         truths = np.array(truths)
#         preds  = np.array(preds)

#         # 如果是單標籤，轉成 2D 方便統一處理
#         if truths.ndim == 1:
#             truths = truths[:, None]
#         if preds.ndim == 1:
#             preds = preds[:, None]

#         num_labels = truths.shape[1]
#         majority_pred = []

#         for i in range(num_labels):
#             up   = np.sum(preds[:, i] == 1)
#             down = np.sum(preds[:, i] == 0)
#             majority_pred.append(1 if up >= down else 0)

#         majority_pred = np.array(majority_pred)
#         # 同一天的真實標籤取第一個樣本 (假設每天同幣種漲跌相同)
#         true_label = truths[0]

#         # 計算每個標籤是否正確
#         majority_correct = majority_pred == true_label

#         # --- 轉換成符號 ---
#         pred_symbols = ["🟢" if p == 1 else "🔴" for p in majority_pred]
#         true_symbols = ["✅" if c else "❌" for c in majority_correct]

#         daily_summary.setdefault(coin, {})
#         daily_summary[coin][date] = {
#             "true_label": true_label.tolist(),
#             "majority_pred": majority_pred.tolist(),
#             "majority_correct": majority_correct.tolist(),
#             "up_counts": np.sum(preds == 1, axis=0).tolist(),
#             "down_counts": np.sum(preds == 0, axis=0).tolist(),
#             "total_counts": len(preds),
#             "pred_symbols": pred_symbols,
#             "true_symbols": true_symbols,
#         }

#     return daily_summary, num_labels



def evaluate_by_coin_date(ids, y_true, y_pred):
    results = defaultdict(list)

    LABEL_SYMBOLS = {
        0: "🔴",  # 大跌
        1: "🟠",    # 跌
        2: "⚪",    # 持平
        3: "🟡",    # 漲
        4: "🟢"   # 大漲
    }


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




# --- 訓練用函式 ---
def train_function(X_train, X_test, y_train, y_test, scaler, features_name, selector, pipeline_path, count):

    print(f"\n=== Training label column {count} ===")
    y_train = y_train[:, count]
    y_test  = y_test[:, count]

        
    all_results = []  # 儲存所有訓練結果
    best_test_acc = -1
    best_run_info = None

    # --- 分層隨機抽樣 50 萬 ---
    train_sample = get_random_samples_sparse_stratified(X_train, y_train)  # 裡面存[(X_sample), (y_sample), ...]

    # 定義模型
    log_reg = LogisticRegression(
        solver='saga', 
        max_iter=100000, 
        verbose=1, 
        penalty='l2', 
        C = C, 
        n_jobs=-1,
        multi_class="multinomial"   # 多類別 softmax
        )
    
    # model = OneVsRestClassifier(log_reg, n_jobs=-1)

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
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_run_info = {
                "run": run,
                "model": log_reg,
                "scaler": scaler,
                "selector": selector,
                "train_acc": train_acc,
                "test_acc": test_acc,
                # "params": random_search.best_params_
            }

        # 如果沒有用 random sampling 就只要跑一次迴圈就好
        if not ENABLE_SAMPLING:
            break
        
        # === 強制清理 ===
        # del random_search
        # del best_model
        # gc.collect()


    # --- 全部結果輸出 ---
    results_df = pd.DataFrame(all_results)
    print("\n=== 所有 Run 的結果 ===")
    print(results_df)
    results_df.to_csv(f"{OUTPUT_PATH}/logreg_sampling_results{count}_{N_SAMPLES}.csv", index=False)


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

    # print("\n被排除的日期（沒有推文或無法計算價格變化）:")
    # print(unprocessed_dates)

    # 最後一筆也無法計算（因為沒「明天」）
    # unprocessed_dates.append(df.loc[len(df)-1, "date"].strftime("%Y/%m/%d"))



# --- 預測用函式 ---
def predict_function(X_train, X_test, y_train, y_test, pipeline_path, count):

    y_train = y_train[:, count]
    y_test  = y_test[:, count]

     # === 載入最佳模型 ===
    pipeline = joblib.load(pipeline_path)
    model = pipeline["model"]
    
    # === 預測所有樣本 ===
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    print(y_pred_train.shape)
    print(y_pred_test.shape)

    # === 載入推文 ID 對應表 ===
    with open(f"{INPUT_PATH}/ids_train.pkl", "rb") as f:   # rb = read binary
        ids_train = pickle.load(f)
    with open(f"{INPUT_PATH}/ids_test.pkl", "rb") as f:   # rb = read binary
        ids_test = pickle.load(f)

    # 將 ids 轉成 np.array 方便接下來的處理
    ids_train = np.array(ids_train)
    ids_test = np.array(ids_test)

    
    print("\n分類報告 (Test set):")
    print(classification_report(y_test, model.predict(X_test)))


    # === 套用在 train / test ===
    train_daily, num_labels = evaluate_by_coin_date(ids_train, y_train, y_pred_train)
    test_daily, _           = evaluate_by_coin_date(ids_test,  y_test,  y_pred_test)

    # === 存成 JSON ===
    with open(f"{OUTPUT_PATH}/logreg_train_daily_results{count}_{N_SAMPLES}.json", "w", encoding="utf-8") as f:
        json.dump(train_daily, f, ensure_ascii=False, indent=4, default=int)

    with open(f"{OUTPUT_PATH}/logreg_test_daily_results{count}_{N_SAMPLES}.json", "w", encoding="utf-8") as f:
        json.dump(test_daily, f, ensure_ascii=False, indent=4, default=int)

    print("已輸出逐日預測結果：")
    print(f"- train: {OUTPUT_PATH}/logreg_train_daily_results{count}_{N_SAMPLES}.json")
    print(f"- test:  {OUTPUT_PATH}/logreg_test_daily_results{count}_{N_SAMPLES}.json")

    # === 合併 train + test ===
    combined_daily = {}
    for coin, daily in train_daily.items():
        combined_daily.setdefault(coin, {}).update(daily)
    for coin, daily in test_daily.items():
        combined_daily.setdefault(coin, {}).update(daily)

    # === 存成合併後的 TXT ===
    txt_path = f"{OUTPUT_PATH}/logreg_combined_results{count}_{N_SAMPLES}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        # === 初始化統計器 ===
        # label_correct = np.zeros(num_labels, dtype=int)
        # label_total   = np.zeros(num_labels, dtype=int)
        label_correct = np.zeros(1, dtype=int)
        label_total   = np.zeros(1, dtype=int)

        for coin, daily in combined_daily.items():
            f.write(f"\n=== {coin} ===\n")
            for date, stats in sorted(daily.items()):
                # --- 每日輸出 ---
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
                # for i, correct in enumerate(stats["majority_correct"]):
                #     label_total[i] += 1
                #     if correct:
                #         label_correct[i] += 1
                # --- 更新累積準確率 ---
                label_total[0] += 1
                if stats["majority_correct"]:
                    label_correct[0] += 1

            # --- 輸出整體準確率 (百分比) ---
            accuracy_summary = " ".join(
                f"{(c / t * 100):.2f}%" if t > 0 else "N/A"
                for c, t in zip(label_correct, label_total)
            )
            f.write(f"\n整體準確率: {accuracy_summary}\n")


    print(f"\n合併後的人類可讀版結果已輸出到：{txt_path}")

    # # === 同時印到 console 預覽 ===
    # with open(txt_path, "r", encoding="utf-8") as f:
    #     print("\n=== 輸出檔案內容預覽 ===\n")
    #     print(f.read())



def categorize_array_multi(Y, t1, t2):
    """
    Y: np.ndarray, shape = (N, num_labels), 價格變化率
    t1, t2: 五元分類閾值，百分比
    """
    # # 讀取價格 CSV
    # price_df = pd.read_csv(f"{PRICE_CSV_PATH}/{COIN_SHORT_NAME}_price.csv")
    # price_df['snapped_at'] = pd.to_datetime(price_df['snapped_at'], format="%Y-%m-%d %H:%M:%S %Z")
    # price_df.set_index('snapped_at', inplace=True)
    # price_df.index = price_df.index.tz_localize(None)  # 移除時區

    # # 建立每天價格的 dict，方便查詢
    # price_lookup = price_df['price'].to_dict()  # 假設 csv 有 price 欄
    # print("price_lookup:", list(price_lookup.items())[:5])



    # # 先建立空陣列
    # Y_pct = np.zeros_like(Y_diff, dtype=float)

    # for i, (coin, date, tweet_id) in tqdm(enumerate(ids), total=len(ids), desc="正在將價差轉成價錢變化率..."):
    #     # 確保 date 是 datetime.date
    #     if isinstance(date, str):
    #         date = pd.to_datetime(date)
    #     # 查當天價格
    #     if date in price_lookup:
    #         price_today = price_lookup[date]
    #     else:
    #         # 如果找不到日期，改用 1 避免除零
    #         print("找不到日期:", date)
    #         price_today = 1.0

    #     # 將整列的價格差轉百分比
    #     Y_pct[i, :] = Y_diff[i, :] / price_today

    # 五元分類
    labels = np.full_like(Y, 2, dtype=int)  # 預設持平
    labels[Y >= t1] = 4  # 大漲
    labels[(Y >= t2) & (Y < t1)] = 3  # 漲
    labels[(Y > -t1) & (Y <= -t2)] = 1  # 跌
    labels[Y <= -t1] = 0  # 大跌

    if np.any(Y == 0):  # 檢查是否有任何元素等於 0
        count = np.sum(Y == 0)
        print(f"共有 {count} 個 Y == 0")
        labels[Y == 0] = 4  # 為了校正TRUMP前兩天的價格相同 第一天設為大漲

    return labels



def load_and_preprocess():
    # 取得 ML 的 X
    X_train = sparse.load_npz(f"{INPUT_PATH}/X_train_filtered.npz")
    X_test = sparse.load_npz(f"{INPUT_PATH}/X_test.npz")

    print(X_train.shape)

    # 匯入已經分類好的 Y
    y_train_all = np.load(f"{INPUT_PATH}/Y_train_filtered_increase.npz")
    y_train_all = y_train_all['Y']
    y_test_all = np.load(f"{INPUT_PATH}/Y_test_increase.npz")
    y_test_all = y_test_all['Y']

    print(y_train_all.shape)

    # with open(f"{INPUT_PATH}/ids_train_filtered.pkl", 'rb') as file:
    #     ids_train_all = pickle.load(file)
    # with open(f"{INPUT_PATH}/ids_test.pkl", 'rb') as file:
    #     ids_test_all = pickle.load(file)
    
    # 建立 target label：五元分類
    y_train_categorized = categorize_array_multi(y_train_all, T1, T2)  # shape (N,5)
    y_test_categorized  = categorize_array_multi(y_test_all, T1, T2)   # shape (N,5)
    print("已成功分類別")

    # 統計每個類別數量
    print(f"大跌：-{T1 * 100}%以下, 跌：-{T2 * 100}% ~ -{T1 * 100}%, 持平：-{T2 * 100}% ~ {T2 * 100}%, 漲：{T2 * 100}% ~ {T1 * 100}%, 大漲：{T1 * 100}%以上")
    train_total_row = y_train_categorized.shape[0]
    test_total_row = y_test_categorized.shape[0]
    for col in range(y_train_categorized.shape[1]):
        counts = np.bincount(y_train_categorized[:, col], minlength=5)
        percentages = counts / train_total_row * 100
        percentages_str = " ".join([f"{p:.2f}%" for p in percentages])
        print(f"[TRAIN] column {col} 類別: {percentages_str}")

        counts = np.bincount(y_test_categorized[:, col], minlength=5)
        percentages = counts / test_total_row * 100
        percentages_str = " ".join([f"{p:.2f}%" for p in percentages])
        print(f"[TEST]  column {col} 類別: {percentages_str}\n")

    input("pasue...")



    scaler = StandardScaler(with_mean=False)  # 適合 sparse matrix
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)


    # 取得 all_keywords(features_name)
    with open(f"{INPUT_PATH}/keyword/filtered_keywords.json", "r", encoding="utf-8-sig") as jsonfile:
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--fs", type=str, default="none", help="Feature selection method")
    parser.add_argument("--k", type=int, default=600, help="Top k features")
    args = parser.parse_args()

    selector = make_selector(task="clf", method=args.fs, k=args.k)
    if selector is not None:
        X_train = selector.fit_transform(X_train, y_train_categorized)
        X_test = selector.transform(X_test)
        features_name = selector.get_feature_names_out(features_name)  # 更新 features_name
        print(f"[INFO] Feature selection ({args.fs}) done, X_train shape = {X_train.shape}")


    return X_train, X_test, y_train_categorized, y_test_categorized, scaler, features_name, selector



def main():

    # --- 載入資料 ---
    X_train, X_test, y_train, y_test, scaler, features_name, selector = load_and_preprocess()

    for count in range(LABELS):

        pipeline_path = f"{SAVE_MODEL_PATH}/logreg_best_pipeline{count}_{N_SAMPLES}.joblib"  # 儲存訓練模型的位置

        if IS_TRAIN:
            # --- 訓練模型 --- 
            train_function(X_train, X_test, y_train, y_test, scaler, features_name, selector, pipeline_path, count)

            # --- 預測模型 ---
            predict_function(X_train, X_test, y_train, y_test, pipeline_path, count)
        else:
            if not os.path.exists(pipeline_path):
                print("找不到已訓練好的模型，請先將 IS_TRAIN 設為 True")

            # --- 預測模型 ---
            predict_function(X_train, X_test, y_train, y_test, pipeline_path, count)



if __name__ == "__main__":
    main()