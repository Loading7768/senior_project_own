import json
import pandas as pd
import numpy as np
import random
from scipy import sparse
import os
import pickle
from tqdm import tqdm
from pulp import LpProblem, LpVariable, LpMinimize, LpBinary, lpSum, PULP_CBC_CMD


'''可修改參數'''
INPUT_PATH = "../data/ml/dataset"

MIN_COUNT = 10  # 設定刪掉出現次數 <= MIN_COUNT 的關鍵詞 (column)

COUNT_TOLERANCE = 1000  # 資料集中誤差推文數值 (1000 => +-1000)

LABEL_TOLERANCE = 0.02  # 資料集中誤差漲跌比例 (0.02 => +-2%)

LABEL_COUNT = 5  # Y 中有幾組 labels

random.seed(42)  # 42 可以換成你想要的數字
'''可修改參數'''








# --- 展開成每條推文一個單位 ---
def expand_by_tweet(df):
    expanded = []
    for _, row in df.iterrows():
        expanded.extend([row['date'].strftime("%Y/%m/%d")] * row['tweet_count'])
    return np.array(expanded)



# --- 列印漲跌比例 ---
def print_label_distribution(Y_train, Y_val, Y_test, COIN_SHORT_NAME, split_val=False):
    def get_stats(y):
        up = np.sum(y >= 0)
        down = np.sum(y < 0)
        total = len(y)
        return up, down, total, up / total, down / total

    train_up, train_down, train_total, train_up_ratio, train_down_ratio = get_stats(Y_train)
    if split_val:
        val_up, val_down, val_total, val_up_ratio, val_down_ratio = get_stats(Y_val)
    test_up, test_down, test_total, test_up_ratio, test_down_ratio = get_stats(Y_test)

    print(f"{COIN_SHORT_NAME} 標籤分佈：")
    print(f"  Train: 總數 {train_total}, 漲 {train_up} ({train_up_ratio:.2%}), 跌 {train_down} ({train_down_ratio:.2%})")
    if split_val:
        print(f"  Val  : 總數 {val_total}, 漲 {val_up} ({val_up_ratio:.2%}), 跌 {val_down} ({val_down_ratio:.2%})")
    print(f"  Test : 總數 {test_total}, 漲 {test_up} ({test_up_ratio:.2%}), 跌 {test_down} ({test_down_ratio:.2%})\n")



# --- 平衡先用日期切割的後的資料 ---
def balance_sets_by_swap(train_df, val_df, test_df, Y_train_df, Y_val_df, Y_test_df, Y_name, target_ratios=(0.8,0.1,0.1), target_label_ratio = 0.5):
    # 計算理論值
    total_tweets = train_df['tweet_count'].sum() + val_df['tweet_count'].sum() + test_df['tweet_count'].sum()
    target_train = int(total_tweets * target_ratios[0])
    target_val   = int(total_tweets * target_ratios[1])
    target_test  = total_tweets - target_train - target_val

    # 建立 DataFrame 複製以便交換
    train, val, test       = train_df.copy(), val_df.copy(), test_df.copy()
    Y_train, Y_val, Y_test = Y_train_df.copy(), Y_val_df.copy(), Y_test_df.copy()

    max_iter = 100000  # 避免無限迴圈
    for _ in tqdm(range(max_iter), desc="平衡資料中..."):
        # 計算目前三個 set 的推文總數
        train_sum = train['tweet_count'].sum()
        val_sum   = val['tweet_count'].sum()
        test_sum  = test['tweet_count'].sum()

        # 計算漲比例
        def get_up_ratio(Y_df):
            total_up = (Y_df[Y_name] > 0).sum()
            total_all = len(Y_df)
            return total_up / total_all
        
        train_up_ratio = get_up_ratio(Y_train)
        val_up_ratio   = get_up_ratio(Y_val)
        test_up_ratio  = get_up_ratio(Y_test)

        # 判斷是否所有 set 都在 tolerance 內
        # 判斷是否都在 tolerance 內
        number_ok = (abs(train_sum - target_train) <= COUNT_TOLERANCE and
                     abs(val_sum - target_val) <= COUNT_TOLERANCE and
                     abs(test_sum - target_test) <= COUNT_TOLERANCE)
        ratio_ok = (abs(train_up_ratio - target_label_ratio) <= LABEL_TOLERANCE and
                    abs(val_up_ratio   - target_label_ratio) <= LABEL_TOLERANCE and
                    abs(test_up_ratio  - target_label_ratio) <= LABEL_TOLERANCE)
        if number_ok and ratio_ok:
            break

        # 找一個超過理論值的 set 和小於理論值的 set    s[1] → DataFrame   s[3] → target count
        sets = [('train', train, Y_train, target_train, train_up_ratio),
                ('val',   val,   Y_val,   target_val,   val_up_ratio),
                ('test',  test,  Y_test,  target_test,  test_up_ratio)]
        over_sets  = [s for s in sets if s[1]['tweet_count'].sum() > s[3] or s[4] > target_label_ratio]
        under_sets = [s for s in sets if s[1]['tweet_count'].sum() < s[3] or s[4] < target_label_ratio]

        if not over_sets or not under_sets:
            break  # 無法交換時退出

        over_name, over_df, over_Y, _, _ = random.choice(over_sets)
        under_name, under_df, under_Y, _, _ = random.choice(under_sets)

        # 隨機選日期交換，但必須 over_row['tweet_count'] > under_row['tweet_count']
        attempt = 0
        max_attempt = 100
        while attempt < max_attempt:

            # 隨機選日期交換
            over_idx = random.choice(over_df.index)
            under_idx = random.choice(under_df.index)

            # 交換兩個 row
            over_row, under_row = over_df.loc[over_idx], under_df.loc[under_idx]
            over_y, under_y = over_Y.loc[over_idx], under_Y.loc[under_idx]
            
            # 交換條件：tweet_count 差距 or 漲跌符號不同
            if over_row['tweet_count'] > under_row['tweet_count'] or (over_y[Y_name] > 0) != (under_y[Y_name] > 0):
                break
            attempt += 1
        else:
            # 如果經過 max_attempt 次還找不到符合條件的，直接跳過這輪交換
            continue

        # 交換 tweet_count DataFrame
        over_df.loc[over_idx], under_df.loc[under_idx] = under_row, over_row
        # 交換 Y DataFrame
        over_Y.loc[over_idx], under_Y.loc[under_idx] = under_y, over_y

        # 更新集合
        if over_name == 'train': train, Y_train = over_df, over_Y
        elif over_name == 'val': val, Y_val = over_df, over_Y
        else: test, Y_test = over_df, over_Y

        if under_name == 'train': train, Y_train = under_df, under_Y
        elif under_name == 'val': val, Y_val = under_df, under_Y
        else: test, Y_test = under_df, under_Y

    return train, val, test


# ---------------- ILP 平衡函式 ----------------
# def balance_sets_by_ILP(df, df_Y_daily, Y_name, target_ratios=(0.8,0.1,0.1), target_label_ratio=0.5):
#     """
#     df_dates: 包含 'date' 與 'tweet_count'
#     df_Y: 每個日期對應的 Y 值 (sum 過每日)
#     Y_name: 欄位名稱，如 'Y_0'
#     target_ratios: train/val/test 比例
#     target_label_ratio: 漲推文比例目標
#     """
#     n = len(df)
#     total_tweets = df['tweet_count'].sum()
#     target_train = int(total_tweets * target_ratios[0])
#     target_val   = int(total_tweets * target_ratios[1])
#     target_test  = total_tweets - target_train - target_val

#     # 計算每日正推文數
#     df_Y_pos = df_Y_daily.copy()
#     df_Y_pos['pos'] = df_Y_pos[Y_name].apply(lambda x: max(0, x))

#     # --- 定義 ILP ---
#     prob = LpProblem("Balance_Dataset", LpMinimize)

#     # 每個日期分配到 train/val/test
#     x_train = [LpVariable(f"x_train_{i}", cat=LpBinary) for i in range(n)]
#     x_val   = [LpVariable(f"x_val_{i}",   cat=LpBinary) for i in range(n)]
#     x_test  = [LpVariable(f"x_test_{i}",  cat=LpBinary) for i in range(n)]

#     # 每個日期只能分配到一個集合
#     for i in range(n):
#         prob += x_train[i] + x_val[i] + x_test[i] == 1

#     # 計算總推文數
#     train_sum = lpSum([x_train[i]*df.loc[i,'tweet_count'] for i in range(n)])
#     val_sum   = lpSum([x_val[i]*df.loc[i,'tweet_count']   for i in range(n)])
#     test_sum  = lpSum([x_test[i]*df.loc[i,'tweet_count']  for i in range(n)])

#     # 計算正推文數
#     train_up = lpSum([x_train[i]*df_Y_pos.loc[i,'pos'] for i in range(n)])
#     val_up   = lpSum([x_val[i]*df_Y_pos.loc[i,'pos']   for i in range(n)])
#     test_up  = lpSum([x_test[i]*df_Y_pos.loc[i,'pos']  for i in range(n)])

#     # --- 線性化絕對值 ---
#     train_dev = LpVariable("train_dev", lowBound=0)
#     val_dev   = LpVariable("val_dev", lowBound=0)
#     test_dev  = LpVariable("test_dev", lowBound=0)

#     prob += train_sum - target_train <= train_dev
#     prob += target_train - train_sum <= train_dev

#     prob += val_sum - target_val <= val_dev
#     prob += target_val - val_sum <= val_dev

#     prob += test_sum - target_test <= test_dev
#     prob += target_test - test_sum <= test_dev

#     # 漲比例偏差
#     train_up_dev = LpVariable("train_up_dev", lowBound=0)
#     val_up_dev   = LpVariable("val_up_dev", lowBound=0)
#     test_up_dev  = LpVariable("test_up_dev", lowBound=0)

#     prob += train_up - target_label_ratio*train_sum <= train_up_dev
#     prob += target_label_ratio*train_sum - train_up <= train_up_dev

#     prob += val_up - target_label_ratio*val_sum <= val_up_dev
#     prob += target_label_ratio*val_sum - val_up <= val_up_dev

#     prob += test_up - target_label_ratio*test_sum <= test_up_dev
#     prob += target_label_ratio*test_sum - test_up <= test_up_dev

#     # --- 目標函數：最小化偏差 ---
#     prob += 1*(train_dev + val_dev + test_dev) + 10*(train_up_dev + val_up_dev + test_up_dev)

#     # --- 求解 ---
#     prob.solve(PULP_CBC_CMD(msg=0))

#     # --- 分配結果 ---
#     train_idx = [i for i in range(n) if x_train[i].varValue > 0.5]
#     val_idx   = [i for i in range(n) if x_val[i].varValue > 0.5]
#     test_idx  = [i for i in range(n) if x_test[i].varValue > 0.5]

#     train_df = df.iloc[train_idx].reset_index(drop=True)
#     val_df   = df.iloc[val_idx].reset_index(drop=True)
#     test_df  = df.iloc[test_idx].reset_index(drop=True)

#     return train_df, val_df, test_df


# --- ILP 平衡分配器（同時控制總數、漲數、跌數） ---
# def balance_sets_by_ILP(df_label, ratios=(0.8, 0.1, 0.1)):
#     """
#     df_label: DataFrame, columns = ["total", "up", "down"]
#     ratios: (train, val, test)
#     return: 字典 { "train": [idx...], "val": [...], "test": [...] }
#     """
#     n_dates = len(df_label)
#     sets = ["train", "val", "test"]

#     # --- 理論目標值 ---
#     total_all = df_label["total"].sum()
#     up_all    = df_label["up"].sum()
#     down_all  = df_label["down"].sum()

#     target = {
#         "train": {
#             "total": ratios[0] * total_all,
#             "up":    ratios[0] * up_all,
#             "down":  ratios[0] * down_all
#         },
#         "val": {
#             "total": ratios[1] * total_all,
#             "up":    ratios[1] * up_all,
#             "down":  ratios[1] * down_all
#         },
#         "test": {
#             "total": ratios[2] * total_all,
#             "up":    ratios[2] * up_all,
#             "down":  ratios[2] * down_all
#         }
#     }

#     # --- ILP 問題 ---
#     prob = LpProblem("BalanceSets", LpMinimize)

#     # x[i,s] = 1 if date i is assigned to set s
#     x = {(i, s): LpVariable(f"x_{i}_{s}", 0, 1, cat="Binary") 
#          for i in range(n_dates) for s in sets}

#     # --- 每個日期只能屬於一個集合 ---
#     for i in range(n_dates):
#         prob += lpSum(x[i, s] for s in sets) == 1

#     # --- 誤差變數 (線性化絕對值) ---
#     err = {}
#     for s in sets:
#         for k in ["total", "up", "down"]:
#             err[(s, k, "pos")] = LpVariable(f"err_{s}_{k}_pos", 0)
#             err[(s, k, "neg")] = LpVariable(f"err_{s}_{k}_neg", 0)

#     # --- 加入誤差限制 ---
#     for s in sets:
#         total_expr = lpSum(df_label.iloc[i]["total"] * x[i, s] for i in range(n_dates))
#         up_expr    = lpSum(df_label.iloc[i]["up"]    * x[i, s] for i in range(n_dates))
#         down_expr  = lpSum(df_label.iloc[i]["down"]  * x[i, s] for i in range(n_dates))

#         for k, expr in [("total", total_expr), ("up", up_expr), ("down", down_expr)]:
#             prob += expr - target[s][k] == err[(s, k, "pos")] - err[(s, k, "neg")]

#     # --- 目標函數：最小化總誤差 ---
#     prob += lpSum(err.values())

#     # --- 求解 ---
#     prob.solve(PULP_CBC_CMD(msg=False, timeLimit=180))

#     # --- 輸出結果 ---
#     assignment = {s: [] for s in sets}
#     for i in range(n_dates):
#         for s in sets:
#             if x[i, s].value() == 1:
#                 assignment[s].append(i)

#     return assignment



# --- 用日期為單位把資料切成 8:1:1 (train : validation : test) ---
def splitset_dates(COIN_SHORT_NAME):

    # --- 讀取每條推文的 ID .pkl ---
    with open(f"{INPUT_PATH}/keyword/{COIN_SHORT_NAME}_ids.pkl", "rb") as f:   # rb = read binary
        ids = pickle.load(f)  # array[('coin', 'date', 'no.'), (str, '%Y-%m-%d', int)]
    dates = np.array([row[1] for row in ids])  # 只把 'date' 取出來，並轉成 np.array
    if isinstance(dates[0], bytes):  # 如果是 bytes，要轉成 str
        dates = dates.astype(str)
    dates_dt = pd.to_datetime(dates, format="%Y-%m-%d")  # 轉成 datetime 方便排序

    # 統計每天出現次數
    unique_dates, counts = np.unique(dates_dt, return_counts=True)
    date_count_dict = {pd.Timestamp(d).strftime("%Y/%m/%d"): int(c) for d, c in zip(unique_dates, counts)}  # 建成 dict，key 用 "YYYY/MM/DD" 格式

    # 儲存 JSON
    json_output_path = f"{INPUT_PATH}/coin_price"
    os.makedirs(json_output_path, exist_ok=True)
    with open(f"{json_output_path}/{COIN_SHORT_NAME}_filtered_tweet_count.json", "w", encoding="utf-8") as f:
        json.dump(date_count_dict, f, ensure_ascii=False, indent=4)


    # --- 讀 JSON ---
    with open(f"{INPUT_PATH}/coin_price/{COIN_SHORT_NAME}_filtered_tweet_count.json", "r", encoding="utf-8") as f:
        tweet_count_dict = json.load(f)

    # --- 轉成 DataFrame ---
    df = pd.DataFrame(list(tweet_count_dict.items()), columns=['date', 'tweet_count'])
    df['date'] = pd.to_datetime(df['date'], format="%Y/%m/%d")

    # --- 讀取 Y 以取得 label ---
    Y_all = np.load(f"{INPUT_PATH}/coin_price/{COIN_SHORT_NAME}_price_diff.npy")  # shape = (總推文數, 5) -> 有五組 Y
    Y_all_labels = np.where(Y_all >= 0, 1, -1)  # 建立 target label：上漲與持平為 1，否則為 -1（二元分類） 

    # --- 建立 Y 的 DataFrame ---
    df_Y = pd.DataFrame(Y_all_labels, columns=[f"Y_{i}" for i in range(Y_all_labels.shape[1])])  # Y_all.shape = (總推文數, 5)
    df_Y['date'] = dates_dt  # 加上日期欄位
    df_Y_daily = df_Y.groupby('date', as_index=False).sum()    # --- 按日期聚合，相同日期的 row 相加 ---
    print(df_Y_daily.head())  # 查看前幾列

    # --- 按日期排序或隨機打亂（這裡先打亂） ---
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df_Y_daily = df_Y_daily.sample(frac=1, random_state=42).reset_index(drop=True)

    # --- 計算各資料集大小 ---
    n = len(df)
    train_size = int(n * 0.8)
    val_size = int(n * 0.1)
    test_size = n - train_size - val_size  # 剩下的給 test

    # --- 切分 ---
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:train_size + val_size]
    test_df = df.iloc[train_size + val_size:]


    for i in range(Y_all.shape[1]):  # 重複跑 Y_all.shape[1] 種不同組 Y
        df_i = df_Y_daily[['date', f'Y_{i}']].copy()  # 只保留 date 與該 Y
        # 每一輪 df_i = [date, Y_0], df_i = [date, Y_1], ...

        Y_name = df_i.columns[1]  # 取 Y_0, Y_1, Y_2, ...

        Y_train_df = df_i.iloc[:train_size]
        Y_val_df = df_i.iloc[train_size:train_size + val_size]
        Y_test_df = df_i.iloc[train_size + val_size:]
        print(f"目前正在平衡 {Y_train_df.columns[1]}...")

        # 隨機交換微調
        train_df, val_df, test_df = balance_sets_by_swap(train_df, val_df, test_df, Y_train_df, Y_val_df, Y_test_df, Y_name)

        # 先按照 tweet_count 由大到小排序
        train_df_sorted = train_df.sort_values(by='tweet_count', ascending=False)
        val_df_sorted   = val_df.sort_values(by='tweet_count', ascending=False)
        test_df_sorted  = test_df.sort_values(by='tweet_count', ascending=False)

        csv_output_path = f"{INPUT_PATH}/split_dates"
        os.makedirs(csv_output_path, exist_ok=True)
        train_df_sorted.to_csv(f"{csv_output_path}/{COIN_SHORT_NAME}_train_dates{i}.csv", index=False, encoding="utf-8-sig")
        val_df_sorted.to_csv(f"{csv_output_path}/{COIN_SHORT_NAME}_val_dates{i}.csv", index=False, encoding="utf-8-sig")
        test_df_sorted.to_csv(f"{csv_output_path}/{COIN_SHORT_NAME}_test_dates{i}.csv", index=False, encoding="utf-8-sig")

        # --- 展開成每條推文一個單位 ---
        dates_train_expanded = expand_by_tweet(train_df)
        dates_val_expanded = expand_by_tweet(val_df)
        dates_test_expanded = expand_by_tweet(test_df)

        print_split_number(dates_train_expanded, dates_val_expanded, dates_test_expanded, COIN_SHORT_NAME)



# --- 修改版 splitset_dates ---
# def splitset_dates(COIN_SHORT_NAME):
#     """
#     Y_all: shape (num_labels, num_samples)
#     dates: 每個樣本的日期 (len = num_samples)
#     coin_name: 幣種名稱
#     """

#     # --- 讀取每條推文的 ID .pkl ---
#     with open(f"{INPUT_PATH}/keyword/{COIN_SHORT_NAME}_ids.pkl", "rb") as f:   # rb = read binary
#         ids = pickle.load(f)  # array[('coin', 'date', 'no.'), (str, '%Y-%m-%d', int)]
#     dates = np.array([row[1] for row in ids])  # 只把 'date' 取出來，並轉成 np.array
#     if isinstance(dates[0], bytes):  # 如果是 bytes，要轉成 str
#         dates = dates.astype(str)
#     dates_dt = pd.to_datetime(dates, format="%Y-%m-%d")  # 轉成 datetime 方便排序

#     # --- 讀取 Y 以取得 label ---
#     Y_all = np.load(f"{INPUT_PATH}/coin_price/{COIN_SHORT_NAME}_price_diff.npy")  # shape = (總推文數, 5) -> 有五組 Y

#     results = {}

#     for y_idx in range(Y_all.shape[1]):
#         y = Y_all[:, y_idx]

#         # --- 先計算每天的 up/down ---
#         df = pd.DataFrame({"date": dates_dt, "y": y})
#         df_label = df.groupby("date").agg(
#             total=("y", "size"),
#             up=("y", lambda x: (x >= 0).sum()),
#             down=("y", lambda x: (x < 0).sum())
#         ).reset_index()

#         # --- 丟進 ILP 分配 ---
#         assignment = balance_sets_by_ILP(df_label)

#         # --- 紀錄結果 ---
#         results[f"Y_{y_idx}"] = assignment

#         # --- 檢查比例 ---
#         print(f"\n=== {COIN_SHORT_NAME} Y_{y_idx} ===")
#         for s in ["train", "val", "test"]:
#             subset = df_label.iloc[assignment[s]]
#             total, up, down = subset[["total", "up", "down"]].sum()
#             print(f"{s.capitalize()}: 總數 {int(total)}, 漲 {int(up)} ({up/total:.2%}), 跌 {int(down)} ({down/total:.2%})")




# ---------------- ILP 版本 splitset_dates ----------------
# def splitset_dates(COIN_SHORT_NAME):
#     # --- 讀取每條推文的 ID .pkl ---
#     with open(f"{INPUT_PATH}/keyword/{COIN_SHORT_NAME}_ids.pkl", "rb") as f:
#         ids = pickle.load(f)
#     dates = np.array([row[1] for row in ids])
#     if isinstance(dates[0], bytes):
#         dates = dates.astype(str)
#     dates_dt = pd.to_datetime(dates, format="%Y-%m-%d")

#     # 統計每天推文數
#     unique_dates, counts = np.unique(dates_dt, return_counts=True)
#     date_count_dict = {pd.Timestamp(d).strftime("%Y/%m/%d"): int(c) for d, c in zip(unique_dates, counts)}
#     json_output_path = f"{INPUT_PATH}/coin_price"
#     os.makedirs(json_output_path, exist_ok=True)
#     with open(f"{json_output_path}/{COIN_SHORT_NAME}_filtered_tweet_count.json", "w", encoding="utf-8") as f:
#         json.dump(date_count_dict, f, ensure_ascii=False, indent=4)

#     # 讀 JSON
#     with open(f"{INPUT_PATH}/coin_price/{COIN_SHORT_NAME}_filtered_tweet_count.json", "r", encoding="utf-8") as f:
#         tweet_count_dict = json.load(f)

#     # DataFrame
#     df = pd.DataFrame(list(tweet_count_dict.items()), columns=['date', 'tweet_count'])
#     df['date'] = pd.to_datetime(df['date'], format="%Y/%m/%d")

#     # 讀取 Y
#     Y_all = np.load(f"{INPUT_PATH}/coin_price/{COIN_SHORT_NAME}_price_diff.npy")  # (總推文數, 5)
#     Y_all_labels = np.where(Y_all>=0, 1, -1)
#     df_Y = pd.DataFrame(Y_all_labels, columns=[f"Y_{i}" for i in range(Y_all_labels.shape[1])])
#     df_Y['date'] = dates_dt
#     df_Y_daily = df_Y.groupby('date', as_index=False).sum()  # 每日聚合
#     print(df_Y_daily.head())

#     # 先打亂日期順序
#     df = df.sample(frac=1, random_state=42).reset_index(drop=True)
#     df_Y_daily = df_Y_daily.sample(frac=1, random_state=42).reset_index(drop=True)

#     # 對每個 Y 做平衡 ILP
#     for i in range(Y_all.shape[1]):
#         df_i = df_Y_daily[['date', f'Y_{i}']].copy()
#         Y_name = f'Y_{i}'
#         print(f"平衡 {Y_name}...")

#         train_df, val_df, test_df = balance_sets_by_ILP(df, df_i, Y_name)

#         # 排序 tweet_count
#         train_df_sorted = train_df.sort_values(by='tweet_count', ascending=False)
#         val_df_sorted   = val_df.sort_values(by='tweet_count', ascending=False)
#         test_df_sorted  = test_df.sort_values(by='tweet_count', ascending=False)

#         # 輸出 CSV
#         csv_output_path = f"{INPUT_PATH}/split_dates"
#         os.makedirs(csv_output_path, exist_ok=True)
#         train_df_sorted.to_csv(f"{csv_output_path}/{COIN_SHORT_NAME}_train_dates{i}.csv", index=False, encoding="utf-8-sig")
#         val_df_sorted.to_csv(f"{csv_output_path}/{COIN_SHORT_NAME}_val_dates{i}.csv", index=False, encoding="utf-8-sig")
#         test_df_sorted.to_csv(f"{csv_output_path}/{COIN_SHORT_NAME}_test_dates{i}.csv", index=False, encoding="utf-8-sig")

#         print(f"完成 {Y_name}")

#         # --- 展開成每條推文一個單位 ---
#         dates_train_expanded = expand_by_tweet(train_df)
#         dates_val_expanded = expand_by_tweet(val_df)
#         dates_test_expanded = expand_by_tweet(test_df)

#         print_split_number(dates_train_expanded, dates_val_expanded, dates_test_expanded, COIN_SHORT_NAME)



# --- 用平衡好的結果來按日期切割 X, Y，並可選擇是否要再分出 val (預設 False) ---
def splitset_XY(train_dates, val_dates, test_dates, Y, COIN_SHORT_NAME, count, split_val=False):

    # 讀取稀疏矩陣
    X = sparse.load_npz(f"{INPUT_PATH}/keyword/{COIN_SHORT_NAME}_X_sparse.npz")  # 二維陣列：colunm(關鍵詞) row(某天某推文) (但這裡是稀疏矩陣的格式)
    
    # 一維陣列：存放與 X row 對應的 ID
    with open(f"{INPUT_PATH}/keyword/{COIN_SHORT_NAME}_ids.pkl", "rb") as f:   # rb = read binary
        ids = pickle.load(f)  # array[('coin', 'date', 'no.'), (str, '%Y-%m-%d', int)
    ids = np.array(ids)  # 把 ids 轉成 numpy array
    dates = np.array([row[1] for row in ids])  # 只把 'date' 取出來，並轉成 np.array


    # 輸出長度 確保一致性
    print(f"{COIN_SHORT_NAME}：")
    print("X.shape[0] =", X.shape[0])
    print("ids.shape[0] =", len(ids))
    print("Y.shape[0] =", Y.shape[0],"\n")



    # 把 date 轉成 datetime 格式，方便比對
    dates_datetime = pd.to_datetime(dates)

    # 轉成 datetime
    train_dates = pd.to_datetime(train_dates)
    val_dates   = pd.to_datetime(val_dates)
    test_dates  = pd.to_datetime(test_dates)

    # 找出 index   逐筆檢查 date 中的每一個值，判斷它是否在 train_dates 裡
    train_mask = dates_datetime.isin(train_dates)
    val_mask   = dates_datetime.isin(val_dates)
    test_mask  = dates_datetime.isin(test_dates)

    # 切割 X
    X_train = X[train_mask, :]
    X_val   = X[val_mask, :]
    X_test  = X[test_mask, :]

    # 切割 Y
    Y_train = Y[train_mask]  # shape = (len(train_mask), 5)
    Y_val   = Y[val_mask]
    Y_test  = Y[test_mask]

    # 切割 ids
    ids_train = ids[train_mask]
    ids_val   = ids[val_mask]
    ids_test  = ids[test_mask]

    if not split_val:
        # 如果不需要 validation，就把 val + test 合併
        X_test = sparse.vstack([X_val, X_test], format="csr")
        Y_test = np.concatenate([Y_val, Y_test])
        ids_test = np.concatenate([ids_val, ids_test])
        X_val, Y_val, ids_val = None, None, None  # 不返回 validation

    # 列印各個幣種 split 後 每個資料集的漲跌比例
    # print(Y_train.shape)
    # print(Y_test.shape)
    print(ids_train.shape)
    print(ids_test.shape)

    val_y = Y_val if Y_val is not None else None
    print(f"第 {count} 組 Ｙ")
    print_label_distribution(Y_train, val_y, Y_test, COIN_SHORT_NAME)

    return X_train, X_val, X_test, Y_train, Y_val, Y_test, ids_train, ids_val, ids_test



# --- 將不必要的關鍵詞與推文刪除 ---
def filter_XY(X_train, X_val, X_test, Y_train, Y_val, Y_test, ids_train, ids_val, ids_test, all_vocab):
    '''重複 功能 1, 功能 2, 功能 3 直到沒有東西可以刪為止'''

    # 定義 功能 1
    def function_1(X, Y, ids):
        row_sums = np.array(X.sum(axis=1)).ravel()
        valid_rows = np.where(row_sums > 0)[0]
        invalid_rows = X.shape[0] - len(valid_rows)

        X = X[valid_rows, :]  # 也可寫 X = X[valid_rows]
        Y = Y[valid_rows]
        ids = ids[valid_rows]
        
        return X, Y, ids, invalid_rows, len(valid_rows)
            

    invalid_rows = -1
    delete_min_count = -1
    delete_only_test = -1

    total_delete_rows = 0
    total_delete_columns = 0

    # 若還有功能是可以刪資料的，就再繼續跑
    while invalid_rows != 0 or delete_min_count != 0 or delete_only_test != 0:

        # --- 功能 1: 刪掉沒有任何關鍵詞的推文 (刪 row) ---
        # train
        X_train, Y_train, ids_train, train_invalid_rows, train_valid_rows = function_1(X_train, Y_train, ids_train)
        
        # test
        X_test, Y_test, ids_test, test_invalid_rows, test_valid_rows = function_1(X_test, Y_test, ids_test)

        # val
        if X_val is not None:
            X_val, Y_val, ids_val, val_invalid_rows, val_valid_rows = function_1(X_val, Y_val, ids_val)

        # 計算保留、刪掉的筆數
        valid_rows = train_valid_rows + test_valid_rows
        invalid_rows = train_invalid_rows + test_invalid_rows
        if X_val is not None:
            valid_rows += val_valid_rows
            invalid_rows += val_invalid_rows
            
        # 只看 Train 的數量
        total_delete_rows += train_invalid_rows
        print("功能 1: 刪掉沒有任何關鍵詞的推文 (row):")
        print(f"\tTrain 保留 row 數量: {train_valid_rows}")
        print(f"\tTrain 刪掉 row 數量: {train_invalid_rows}\n")


        # --- 功能 2: 刪掉在 train 中出現次數 <= min_count 的關鍵詞 (刪 column) ---
        col_sums = np.array(X_train.sum(axis=0)).ravel()
        valid_cols = np.where(col_sums > MIN_COUNT)[0]

        # 每個關鍵詞的出現次數統計
        keyword_counts = {all_vocab[i]: int(col_sums[i]) for i in range(len(all_vocab))}
        stats_output_path = os.path.join(INPUT_PATH, "keyword", "keyword_counts.json")
        with open(stats_output_path, "w", encoding="utf-8") as f:
            json.dump(keyword_counts, f, ensure_ascii=False, indent=4)

        # 開始過濾
        X_train = X_train[:, valid_cols]
        X_test  = X_test[:, valid_cols]
        if X_val is not None:
            X_val = X_val[:, valid_cols]

        filtered_vocab = [all_vocab[i] for i in valid_cols]
        delete_min_count = len(all_vocab) - len(filtered_vocab)
        total_delete_columns += delete_min_count
        print("功能 2: 刪掉出現次數 <= min_count 的關鍵詞 (column):")
        print(f"\t原始 column 數量: {len(all_vocab)}")
        print(f"\t保留 column 數量: {len(filtered_vocab)}")
        print(f"\t刪掉 column 數量: {delete_min_count}\n")

        with open(os.path.join(f"{INPUT_PATH}/keyword", f"filtered_keywords.json"), "w", encoding="utf-8") as f:
            json.dump(filtered_vocab, f, ensure_ascii=False, indent=4)  


        # --- 功能 3: 只保留 train 出現過的關鍵詞 (刪 column) --- 
        '''
        X_train.nonzero() 會回傳一個 tuple (row_idx, col_idx)：
        row_idx → 非零元素所在的 row（推文 index）。
        col_idx → 非零元素所在的 column（關鍵詞 index）。

        X_train.nonzero()[1] 取的是所有非零值的 column index。
        → 這就是「有哪些關鍵詞至少在 train 裡出現過一次」。

        np.unique(...) 把它去重，得到一個 只出現在 train 的 column 清單。
        '''

        orig_cols = X_train.shape[1]
        keep_cols = np.unique(X_train.nonzero()[1])  # train 出現過的 column index
        new_cols = len(keep_cols)

        # column 過濾
        X_train = X_train[:, keep_cols]  # [:, keep_cols] 表示「保留所有 row，但只取出 keep_cols 這些 column」。
        X_test  = X_test[:, keep_cols]
        if X_val is not None:
            X_val = X_val[:, keep_cols]
        if all_vocab is not None:
            all_vocab = [all_vocab[i] for i in keep_cols]

        delete_only_test = orig_cols - new_cols
        total_delete_columns += delete_only_test
        print("功能 3: 只保留 train 出現過的關鍵詞 (column):")
        print(f"\t原始 column 數量: {orig_cols}")
        print(f"\t保留 column 數量: {new_cols}")
        print(f"\t刪掉 column 數量: {delete_only_test}\n")

    print(f"總共刪除 {total_delete_rows} 個推文 (row), {total_delete_columns} 個關鍵詞 (column)\n")
    print(f"已輸出所有關鍵詞出現次數統計到 {stats_output_path}")
    print(f"已輸出所有被過濾的關鍵詞到 {INPUT_PATH}/keyword\n")

    return X_train, X_val, X_test, Y_train, Y_val, Y_test, ids_train, ids_val, ids_test



# --- 打亂順序 (shuffle) ---
def shuffle_XY(X, Y, ids, seed=42):
    """
    Shuffle X and Y in unison.
    X: np.ndarray 或 scipy.sparse 矩陣
    Y: np.ndarray 一維標籤
    seed: 隨機種子
    """
    rng = np.random.default_rng(seed)
    indices = np.arange(Y.shape[0])  # 取得樣本數 indices = [0, 1, 2, ... , len(X)-1]
    rng.shuffle(indices)  # 把 indices 隨機重新排序

    X_shuffled = X[indices, :]  # 按照 indices 的順序重新排列
    Y_shuffled = Y[indices]
    ids_shuffled = ids[indices]

    return X_shuffled, Y_shuffled, ids_shuffled



# --- 計算每個日期總共的數量 (用以確認是否有切正確) ---
def count_per_day(ids, dataset_name):
    """
    dates: array-like, 每條推文的日期 (str 或 np.datetime64)
    dataset_name: 用於打印
    """
    dates = [row[1] for row in ids]
    # 如果是 bytes，先轉成 str
    if isinstance(dates[0], bytes):
        dates = dates.astype(str)

    # 轉成 datetime
    dates_dt = pd.to_datetime(dates)

    # 計算每天出現次數
    date_counts = dates_dt.value_counts().sort_index()  # 按日期排序
    df_counts = pd.DataFrame({"date": date_counts.index, "tweet_count": date_counts.values})

    df_counts.to_csv(f"{INPUT_PATH}/dates_{dataset_name}_counts.csv", index=False)

    return df_counts



# --- 將三種幣種的 X, Y 合併成完整的模型輸入值 (輸出 .npy 檔) ---
def merge(DOGE_X_train, DOGE_X_val, DOGE_X_test, DOGE_Y_train, DOGE_Y_val, DOGE_Y_test,
          PEPE_X_train, PEPE_X_val, PEPE_X_test, PEPE_Y_train, PEPE_Y_val, PEPE_Y_test,
          TRUMP_X_train, TRUMP_X_val, TRUMP_X_test, TRUMP_Y_train, TRUMP_Y_val, TRUMP_Y_test,
          DOGE_ids_train, DOGE_ids_val, DOGE_ids_test,
          PEPE_ids_train, PEPE_ids_val, PEPE_ids_test,
          TRUMP_ids_train, TRUMP_ids_val, TRUMP_ids_test,
          all_vocab, count):

    # 合併 X（稀疏矩陣用 sparse.vstack）
    X_train_list = [DOGE_X_train, PEPE_X_train, TRUMP_X_train]
    X_test_list  = [DOGE_X_test, PEPE_X_test, TRUMP_X_test]
    X_val_list   = [DOGE_X_val, PEPE_X_val, TRUMP_X_val] if DOGE_X_val is not None else []

    X_train = sparse.vstack(X_train_list, format="csr")  # np.vstack = vertical stack，把多個矩陣在「列方向」堆疊起來
    X_test  = sparse.vstack(X_test_list, format="csr")
    X_val   = sparse.vstack(X_val_list, format="csr") if X_val_list else None

    # 合併 Y
    Y_train = np.concatenate([DOGE_Y_train, PEPE_Y_train, TRUMP_Y_train])  # np.concatenate = 把多個一維陣列串接起來
    Y_test  = np.concatenate([DOGE_Y_test, PEPE_Y_test, TRUMP_Y_test])
    Y_val   = np.concatenate([DOGE_Y_val, PEPE_Y_val, TRUMP_Y_val]) if DOGE_Y_val is not None else None

    # 合併 ids
    ids_train = np.concatenate([DOGE_ids_train, PEPE_ids_train, TRUMP_ids_train])
    ids_test  = np.concatenate([DOGE_ids_test, PEPE_ids_test, TRUMP_ids_test])
    ids_val   = np.concatenate([DOGE_ids_val, PEPE_ids_val, TRUMP_ids_val]) if DOGE_ids_val is not None else None


    # 將不必要的關鍵詞與推文刪除
    X_train, X_val, X_test, Y_train, Y_val, Y_test, ids_train, ids_val, ids_test = filter_XY(X_train, X_val, X_test, Y_train, Y_val, Y_test, ids_train, ids_val, ids_test, all_vocab)
    print(ids_train.shape)


    # 打亂順序
    X_train, Y_train, ids_train = shuffle_XY(X_train, Y_train, ids_train)
    X_test,  Y_test,  ids_test  = shuffle_XY(X_test, Y_test, ids_test)
    if X_val is not None:
        X_val, Y_val, ids_val = shuffle_XY(X_val, Y_val, ids_val)

    # 儲存
    sparse.save_npz(f"{INPUT_PATH}/X_train{count}.npz", X_train)
    sparse.save_npz(f"{INPUT_PATH}/X_test{count}.npz", X_test)
    np.savez_compressed(f"{INPUT_PATH}/Y_train{count}.npz", Y=Y_train)
    np.savez_compressed(f"{INPUT_PATH}/Y_test{count}.npz",  Y=Y_test)

    if X_val is not None:
        sparse.save_npz(f"{INPUT_PATH}/X_val{count}.npz", X_val)
        np.savez_compressed(f"{INPUT_PATH}/Y_val{count}.npz", Y=Y_val)

    if ids_train is not None:
        print(ids_train.shape)
        print(ids_test.shape)
        with open(f"{INPUT_PATH}/ids_train{count}.pkl", 'wb') as file:
            pickle.dump(ids_train.tolist(), file)
        with open(f"{INPUT_PATH}/ids_test{count}.pkl", 'wb') as file:
            pickle.dump(ids_test.tolist(), file)
        if ids_val is not None:
            with open(f"{INPUT_PATH}/ids_val{count}.pkl", 'wb') as file:
                pickle.dump(ids_val.tolist(), file)

    # 檢查資料集維度
    assert X_train.shape[0] == Y_train.shape[0] == len(ids_train), "Train 維度不一致!"
    assert X_test.shape[0] == Y_test.shape[0] == len(ids_test), "Test 維度不一致!"
    if X_val is not None:
        assert X_val.shape[0] == Y_val.shape[0] == len(ids_val), "Val 維度不一致!"


    print("Merge 完成，資料已輸出到 ../data/ml/dataset\n")

    # 列印 merge, filter 後 每個資料集的漲跌比例
    # print(Y_train.shape)
    val_y = Y_val if Y_val is not None else None
    print(f"第 {count} 組 Ｙ")
    print_label_distribution(Y_train, val_y, Y_test, "ALL")

    # 計算每個資料集中每天的推文總數
    count_per_day(ids_train, "train")
    count_per_day(ids_test, "test")
    if ids_val is not None:
        count_per_day(ids_val, "val")

    print("已將不同資料集每天的推文總數輸出為 csv 到 ../data/ml/dataset\n")





# --- 列印平衡好的結果 ---
def print_split_number(train_expanded, val_expanded, test_expanded, COIN_SHORT_NAME):
    sum = len(train_expanded) + len(val_expanded) + len(test_expanded)
    print(COIN_SHORT_NAME + "：")
    print(f"理論值 Train: {int(sum * 0.8)},                Val: {int(sum * 0.1)},                Test: {int(sum * 0.1)}")
    print(f"實際值 Train: {len(train_expanded)} ({round((len(train_expanded) / sum), 10)}), Val: {len(val_expanded)} ({round((len(val_expanded) / sum), 10)}), Test: {len(test_expanded)} ({round((len(test_expanded) / sum), 10)})\n")



def split_XY_preprocess(COIN_SHORT_NAME, count):
    # 讀取三個集合的日期 (切割好的 CSV) 
    train_dates = pd.read_csv(f"{INPUT_PATH}/split_dates/{COIN_SHORT_NAME}_train_dates{count}.csv")['date']
    val_dates = pd.read_csv(f"{INPUT_PATH}/split_dates/{COIN_SHORT_NAME}_val_dates{count}.csv")['date']
    test_dates = pd.read_csv(f"{INPUT_PATH}/split_dates/{COIN_SHORT_NAME}_test_dates{count}.csv")['date']

    # 讀取 price_diff.npy
    Y_all = np.load(f"{INPUT_PATH}/coin_price/{COIN_SHORT_NAME}_price_diff.npy")  # shape = (5, 總推文數) -> 有五組 Y
    Y = Y_all[:, count]

    X_train, X_val, X_test, Y_train, Y_val, Y_test, ids_train, ids_val, ids_test = splitset_XY(train_dates, val_dates, test_dates, Y, COIN_SHORT_NAME, count)  # 若要分出 val => splitset_XY("DOGE", True)

    return X_train, X_val, X_test, Y_train, Y_val, Y_test, ids_train, ids_val, ids_test

def main():

    # 分別切資料集
    splitset_dates("DOGE")
    # print_split_number(DOGE_dates_train_expanded, DOGE_dates_val_expanded, DOGE_dates_test_expanded, "DOGE")
    splitset_dates("PEPE")
    # print_split_number(PEPE_dates_train_expanded, PEPE_dates_val_expanded, PEPE_dates_test_expanded, "PEPE")
    splitset_dates("TRUMP")
    # print_split_number(TRUMP_dates_train_expanded, TRUMP_dates_val_expanded, TRUMP_dates_test_expanded, "TRUMP")

    for count in range(LABEL_COUNT):
        
        DOGE_X_train, DOGE_X_val, DOGE_X_test, DOGE_Y_train, DOGE_Y_val, DOGE_Y_test, DOGE_ids_train, DOGE_ids_val, DOGE_ids_test = split_XY_preprocess("DOGE", count)  # 若要分出 val => splitset_XY("DOGE", True)
        PEPE_X_train, PEPE_X_val, PEPE_X_test, PEPE_Y_train, PEPE_Y_val, PEPE_Y_test, PEPE_ids_train, PEPE_ids_val, PEPE_ids_test = split_XY_preprocess("PEPE", count)
        TRUMP_X_train, TRUMP_X_val, TRUMP_X_test, TRUMP_Y_train, TRUMP_Y_val, TRUMP_Y_test, TRUMP_ids_train, TRUMP_ids_val, TRUMP_ids_test = split_XY_preprocess("TRUMP", count)

        # 讀取所有關鍵詞的名字
        json_path = os.path.join("../data/keyword/machine_learning", "all_keywords.json")
        with open(json_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)

        all_vocab = list(vocab)

        print(DOGE_ids_train.shape)
        print(DOGE_ids_test.shape)
        print(PEPE_ids_train.shape)
        print(PEPE_ids_test.shape)
        print(TRUMP_ids_train.shape)
        print(TRUMP_ids_test.shape)
        # 合併資料集
        merge(DOGE_X_train, DOGE_X_val, DOGE_X_test, DOGE_Y_train, DOGE_Y_val, DOGE_Y_test,
            PEPE_X_train, PEPE_X_val, PEPE_X_test, PEPE_Y_train, PEPE_Y_val, PEPE_Y_test,
            TRUMP_X_train, TRUMP_X_val, TRUMP_X_test, TRUMP_Y_train, TRUMP_Y_val, TRUMP_Y_test,
            DOGE_ids_train, DOGE_ids_val, DOGE_ids_test,
            PEPE_ids_train, PEPE_ids_val, PEPE_ids_test,
            TRUMP_ids_train, TRUMP_ids_val, TRUMP_ids_test,
            all_vocab, count)
    

if __name__ == "__main__":
    main()