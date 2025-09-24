from collections import defaultdict
import os
import json
import pandas as pd
from datetime import datetime, timedelta
from glob import glob
from tqdm import tqdm
import numpy as np

import sys
from pathlib import Path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
from config import COIN_SHORT_NAME, JSON_DICT_NAME

'''
要先把價錢和日期的 csv 檔放在 ../data/coin_price 中
檔名存為 {COIN_SHORT_NAME}_price.csv

DOGE_price.csv：
    snapped_at,price,market_cap,total_volume
    2014-01-28 00:00:00 UTC,0.00134193,50833265.0,2342040.0
    2014-01-29 00:00:00 UTC,0.00138856,53584322.0,1655470.0
    2014-01-30 00:00:00 UTC,0.00147485,58009736.0,2315200.0
    ...
'''


'''可修改參數'''
PRICE_CSV_PATH = f"../data/coin_price/{COIN_SHORT_NAME}_price.csv"

OUTPUT_TWEET_COUNT_PATH = "../data/ml/dataset/coin_price"

IS_FILTERED = False  # 看是否有分 normal 與 bot

IS_RUN_AUGUST = True  # 看現在是不是要跑 2025/08 的資料  START_DATE, END_DATE 會固定

START_DATE = "2013/12/15"

END_DATE   = "2025/07/31"

SHIFT = 5

FIRST_AND_SECOND_CLASSIFIER_Y = True

SECOND_CLASSIFIER_X = True
'''可修改參數'''

if IS_RUN_AUGUST:
    START_DATE = "2025/08/01"
    END_DATE   = "2025/08/31"

SUFFIX_FILTERED = "" if IS_FILTERED else "_non_filtered"
SUFFIX_AUGUST   = "" if IS_RUN_AUGUST else "_202508"

# === 修改為你的 CSV 檔與 JSON 資料夾路徑 ===
OUTPUT_CSV_PATH = f"../data/coin_price/{COIN_SHORT_NAME}_current_tweet_price_output{SUFFIX_FILTERED}{SUFFIX_AUGUST}.csv"
OUTPUT_TWEET_COUNT_PATH_FILE = f"{OUTPUT_TWEET_COUNT_PATH}/{COIN_SHORT_NAME}_current_tweet_count{SUFFIX_FILTERED}{SUFFIX_AUGUST}.json"
OUTPUT_SECOND_CLASSIFIER_Y = f"{OUTPUT_TWEET_COUNT_PATH}/{COIN_SHORT_NAME}_price_diff_original{SUFFIX_FILTERED}{SUFFIX_AUGUST}.npy"
OUTPUT_FIERT_CLASSIFIER_Y = f"{OUTPUT_TWEET_COUNT_PATH}/{COIN_SHORT_NAME}_price_diff{SUFFIX_FILTERED}{SUFFIX_AUGUST}.npy"
OUTPUT_SECOND_CLASSIFIER_X = f"{OUTPUT_TWEET_COUNT_PATH}/{COIN_SHORT_NAME}_price_diff_past{SHIFT}days{SUFFIX_FILTERED}{SUFFIX_AUGUST}.npy"

if IS_FILTERED:
    NORMAL_TWEETS_JSON_GLOB = f"../data/filtered_tweets/normal_tweets/{COIN_SHORT_NAME}/*/*/{COIN_SHORT_NAME}_*_normal.json"  # 是針對 normal_tweet 做運算
else:
    NORMAL_TWEETS_JSON_GLOB = f"../data/tweets/{COIN_SHORT_NAME}/*/*/{COIN_SHORT_NAME}_*.json"  # 是針對 原始 tweets 做運算


os.makedirs(OUTPUT_TWEET_COUNT_PATH, exist_ok=True)



# 轉成 datetime 方便比較
START_DATE_DT = pd.to_datetime(START_DATE, format="%Y/%m/%d")
END_DATE_DT   = pd.to_datetime(END_DATE, format="%Y/%m/%d")

# === 讀取價格 CSV ===
price_df = pd.read_csv(PRICE_CSV_PATH)
price_df['snapped_at'] = pd.to_datetime(price_df['snapped_at'], format="%Y-%m-%d %H:%M:%S %Z")
price_df.set_index('snapped_at', inplace=True)
price_df.index = price_df.index.tz_localize(None)  # 移除時區 只保留日期部分

# === 檢查是否有缺少日期 ===
all_days = pd.date_range(start=price_df.index.min(), end=price_df.index.max(), freq="D")
missing_days = all_days.difference(price_df.index)

if len(missing_days) == 0:
    print("✅ 價格資料完整，沒有缺少日期")
else:
    print(f"⚠️ 發現 {len(missing_days)} 天缺少價格資料")
    print(missing_days[:50])  # 只印出前 50 天，避免太多


# 🔹 過濾價格資料到時間範圍內
price_df = price_df.loc[(price_df.index >= START_DATE_DT) & (price_df.index <= END_DATE_DT + pd.Timedelta(days=1))]


# === 儲存推文資訊 若當天沒有推文則不會加進去 set 中 ===
tweet_dates = set()  # 收集 tweet 有出現的日期

tweet_count = defaultdict(int)  # 儲存每天的推文數量


json_files = glob(NORMAL_TWEETS_JSON_GLOB)
for json_path in tqdm(json_files, desc="正在找尋日期"):
    with open(json_path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    tweets = data[JSON_DICT_NAME]
    if not tweets:
        print("當天沒有推文：", json_path)
        continue

    try:
        # 取得日期
        date_str = datetime.strptime(
            tweets[0]['created_at'], "%a %b %d %H:%M:%S %z %Y"
        ).strftime("%Y/%m/%d")
        date_dt = pd.to_datetime(date_str)

        # 🔹 過濾掉不在範圍內的推文
        if not (START_DATE_DT <= date_dt <= END_DATE_DT):
            print("當天不在指定時間範圍內：", json_path)
            continue

        tweet_dates.add(date_dt)

        # 取得當天推文數量
        tweet_count[date_dt] = len(tweets)

    except Exception as e:
        print(f"[錯誤] {json_path}: {e}")

# === 依照 tweet 日期排序，決定整個時間範圍 ===
if not tweet_dates:
    print("沒有抓到任何推文日期")
    exit()

tweet_dates = sorted(tweet_dates)  # 因為抓進來的檔案順序可能會是亂的

# ----------- 將 tweet_count 輸出成 json 檔 -------------
# 將 datetime 轉成字串，defaultdict -> dict
tweet_count_dict = {
    date.strftime("%Y/%m/%d"): count
    for date, count in sorted(tweet_count.items())  # <- 這裡 sorted 會依 datetime 升序排序
}

# 儲存成 JSON
with open(OUTPUT_TWEET_COUNT_PATH_FILE, "w", encoding="utf-8") as f:
    json.dump(tweet_count_dict, f, ensure_ascii=False, indent=4)

print(f"✅ 已儲存 {COIN_SHORT_NAME}_tweet_count 到 {OUTPUT_TWEET_COUNT_PATH_FILE}")

total_tweets = sum(tweet_count.values())
if IS_FILTERED:
    print(f"\n全部 normal_tweet 的推文數量: {total_tweets}\n")
else:
    print(f"\n全部 原始 tweets 的推文數量: {total_tweets}\n")

# === 建立最終結果表 ===
output_rows = []

prev_date = None
for current_date in tqdm(tweet_dates, desc="正在儲存價錢"):
    if prev_date:

        # 若有缺少的日期 且 相鄰兩天間少於 31 天
        gap = (current_date - prev_date).days
        if 1 < gap < 31:
            for d in pd.date_range(prev_date + timedelta(days=1), current_date - timedelta(days=1)):
                
                row = price_df.loc[price_df.index == d]
                price = row['price'].values[0] if not row.empty else ""

                output_rows.append({
                    "date": d.strftime("%Y/%m/%d"),
                    "price": price,
                    "tweet_count": 0,
                    "has_tweet": False
                })
                print("當天沒有抓到推文：", d)
    # 當前 tweet 日期
    price = price_df.loc[current_date]['price'] if current_date in price_df.index else ""
    output_rows.append({
        "date": current_date.strftime("%Y/%m/%d"),
        "price": price,
        "tweet_count": tweet_count[current_date],
        "has_tweet": True
    })
    prev_date = current_date


# 將 output_rows 轉成 DataFrame
df_output = pd.DataFrame(output_rows)

# 轉換日期格式（方便後續計算）
df_output['date_dt'] = pd.to_datetime(df_output['date'], format='%Y/%m/%d')
df_output['price'] = pd.to_numeric(df_output['price'], errors='coerce')

# # ---------------- 計算 1~5 天的價差 ----------------
# day_shifts = [shift for shift in range(1, SHIFT + 1)]

# for shift in day_shifts:
#     col_name = f"price_diff_{shift}d"

#     def calc_price_diff_shift(row, shift=shift):
#         today = row['date_dt']
#         future = today + pd.Timedelta(days=shift)
#         try:
#             price_today = row['price']
#             price_future = price_df.loc[future]['price']
#             if pd.isna(price_today):
#                 return np.nan
#             return price_future - price_today
#         except KeyError:
#             return np.nan  # 缺少未來價格

#     df_output[col_name] = df_output.apply(calc_price_diff_shift, axis=1)

# ---------------- 計算相鄰日期的價差 ----------------
# 將 price_df 的價格對齊 df_output 的日期
price_map = price_df['price'].to_dict()

# 計算明天價格：直接用 price_map 查隔天
df_output['price_tomorrow'] = df_output['date_dt'].apply(
    lambda x: price_map.get(x + pd.Timedelta(days=1), np.nan)
)

# 計算明天 - 今天
df_output['price_diff_tomorrow'] = df_output['price_tomorrow'] - df_output['price']
df_output['price_diff_rate_tomorrow'] = df_output['price_diff_tomorrow'] / df_output['price']

# 動態生成「往回 SHIFT 天」的價差與變化率
for i in range(1, SHIFT + 1):
    col_price_prev = f"price_{i}daysbefore"
    df_output[col_price_prev] = df_output['price'].shift(i)

    col_diff = f"price_diff_{i}daysbefore"
    col_rate = f"price_diff_rate_{i}daysbefore"

    # 價差： (i-1) 天前價格 - i 天前價格
    df_output[col_diff] = df_output['price'].shift(i - 1) - df_output['price'].shift(i)

    # 價差變化率：差 ÷ i 天前價格
    df_output[col_rate] = df_output[col_diff] / df_output['price'].shift(i)

# 移除輔助欄位（所有 shift 出來的 price_*）
drop_cols = ['date_dt'] + ['price_tomorrow'] + [f"price_{i}daysbefore" for i in range(1, SHIFT + 1)]
df_output.drop(columns=drop_cols, inplace=True)

# 儲存 CSV
df_output.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")
print(f"✅ 已儲存到 {OUTPUT_CSV_PATH}")


# ---------------- 儲存 price_diff_rate_tomorrow 到 numpy ----------------
if FIRST_AND_SECOND_CLASSIFIER_Y:
    # 過濾出有推文且 price_diff_rate_tomorrow 不是 NaN
    filtered_df = df_output[(df_output['has_tweet'] == True) & (df_output['price_diff_rate_tomorrow'].notna())]

    # 先存原始的 price_diff_rate_tomorrow
    original_price_diff_array = filtered_df['price_diff_rate_tomorrow'].to_numpy(dtype=float)

    np.save(OUTPUT_SECOND_CLASSIFIER_Y, original_price_diff_array)
    print(original_price_diff_array[:20])  # 預覽前 20 筆
    print(f"✅ 已儲存原始 price_diff_rate_tomorrow 矩陣到 {OUTPUT_SECOND_CLASSIFIER_Y}，共: {len(original_price_diff_array)} 筆\n")
    

    # 依 tweet_count 重複價差
    expanded_price_diffs = []
    for _, row in filtered_df.iterrows():
        expanded_price_diffs.extend([row['price_diff_rate_tomorrow']] * row['tweet_count'])

    # 轉成 numpy 陣列並儲存
    price_diff_array = np.array(expanded_price_diffs, dtype=float)
    
    np.save(OUTPUT_FIERT_CLASSIFIER_Y, price_diff_array)

    print(expanded_price_diffs[:20])  # 預覽前 20 筆
    print(f"\n✅ 已儲存 price_diff_rate_tomorrow 矩陣到 {OUTPUT_FIERT_CLASSIFIER_Y}，共: {len(expanded_price_diffs)} 筆\n")



# ---------------- 儲存過去 SHIFT 天的價差及變化率 ----------------
if SECOND_CLASSIFIER_X:
    # 只保留沒有 NaN 的行
    df_output_clean = df_output.dropna()

    columns_to_save = []
    for i in range(1, SHIFT + 1):
        columns_to_save.append(f'price_diff_{i}daysbefore')
        columns_to_save.append(f'price_diff_rate_{i}daysbefore')

    # 過濾掉有 NaN 的行
    filtered_df = df_output.dropna(subset=columns_to_save)

    # 過濾有推文且對應欄位不是 NaN
    filtered_df = filtered_df[filtered_df['has_tweet'] == True].copy()

    # 直接轉成 numpy
    all_price_diffs_array = filtered_df[columns_to_save].to_numpy(dtype=float)

    # 儲存
    np.save(OUTPUT_SECOND_CLASSIFIER_X, all_price_diffs_array)

    print(all_price_diffs_array[:10])  # 預覽前 20 筆
    print(f"\n✅ 已儲存 {COIN_SHORT_NAME}_price_diff_past{SHIFT}days.npy，形狀: {all_price_diffs_array.shape}")



# # ---------------- 檢查 NaN ----------------
# for shift in day_shifts:
#     col_name = f"price_diff_{shift}d"
#     nan_rows = df_output[df_output[col_name].isna()]
#     if not nan_rows.empty:
#         print(f"\n以下日期 {col_name} 無法計算（可能缺少當天或未來 {shift} 天價格）:")
#         print(nan_rows[['date', 'price', 'tweet_count', 'has_tweet']])

# # ---------------- 儲存多組 price_diff.npy ----------------
# all_price_diffs = []  # 建立 price_diff 矩陣（每個 row 都是不同天數的價差）

# for shift in day_shifts:
#     col_name = f"price_diff_{shift}d"

#     # 過濾出有推文且價差不是 NaN
#     filtered_df = df_output[(df_output['has_tweet'] == True) & (df_output[col_name].notna())]

#     # 依 tweet_count 重複價差
#     expanded_price_diffs = []
#     for _, row in filtered_df.iterrows():
#         expanded_price_diffs.extend([row[col_name]] * row['tweet_count'])

#     all_price_diffs.append(expanded_price_diffs)
    
#     print(f"\n✅ 已加入 {COIN_SHORT_NAME}_price_diff_{shift}day（共 {len(expanded_price_diffs)} 筆）")
#     print(expanded_price_diffs[:20])  # 預覽前 20 筆


# # 轉成 numpy 陣列並儲存
# all_price_diffs = np.array(all_price_diffs, dtype=float)
# all_price_diffs_T = all_price_diffs.T  # 或者 np.transpose(all_price_diffs)
# # save_path = f"../data/ml/dataset/coin_price/{COIN_SHORT_NAME}_price_diff_{SHIFT}.npy"
# save_path = f"../data/ml/dataset/coin_price/{COIN_SHORT_NAME}_price_diff.npy"
# np.save(save_path, all_price_diffs_T)

# print(f"\n✅ 已儲存矩陣 {all_price_diffs_T}，形狀: {all_price_diffs_T.shape}")