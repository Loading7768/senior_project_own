import pandas as pd
import json
from glob import glob
from datetime import datetime
from pathlib import Path
import sys
from tqdm import tqdm
import os
import random
import matplotlib.pyplot as plt
import csv

import sys
from pathlib import Path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
from config import COIN_SHORT_NAME, JSON_DICT_NAME


OUTPUT_FILE = "../data/tweets/count/estimate/"
os.makedirs(OUTPUT_FILE, exist_ok=True)

hour_counter = []
partial_json_files = []
completed_json_files = []

# 所有原始檔案 (把所有結尾是 .json 的檔案抓出來)
json_files = glob(f'../data/tweets/{COIN_SHORT_NAME}/*/*/*.json')

def hour_distribution():

    for json_file in tqdm(json_files, desc="統計完整檔案小時比例"):
        with open(json_file, 'r', encoding="utf-8-sig") as file:
            data = json.load(file)

        tweets = data[JSON_DICT_NAME]

        # 先將最後一則推文的時間抓出來
        earliest_str = tweets[-1]["created_at"]
        earliest_dt = datetime.strptime(earliest_str, "%a %b %d %H:%M:%S %z %Y")
        earliest_time = earliest_dt.strftime("%H:%M:%S")

        # 判斷有沒有抓完 是否是 00:XX:XX
        if earliest_time.startswith("00:"):
            for tweet in tweets:
                created_at = tweet.get('created_at')
                created_dt = datetime.strptime(created_at, "%a %b %d %H:%M:%S %z %Y")
                hour_counter.append(created_dt.hour)  # 只取小時數
            
            completed_json_files.append(json_file)  # 把完整抓到的檔案名先存起來
        else:
            partial_json_files.append(json_file)  # 把沒有完整抓到的檔案名先存起來


    # 統計完整檔案的小時比例
    hour_df = pd.DataFrame({'hour': hour_counter})
    hour_distribution = hour_df['hour'].value_counts().sort_index()
    hour_distribution = hour_distribution / hour_distribution.sum()

    print("✅ 小時比例：")
    print(hour_distribution)

    output_path_completed = f"{OUTPUT_FILE}/{COIN_SHORT_NAME}_hour_distribution_and_errors.txt"
    with open(output_path_completed, 'w', encoding="utf-8-sig") as txtfile:
        txtfile.write(str(hour_distribution))
        txtfile.write("\n\n")

    return hour_distribution


def estimate(hour_distribution):
    results = []

    # 先清空 txt 內的資料
    output_path_partial_txt = f"{OUTPUT_FILE}/{COIN_SHORT_NAME}_estimate.txt"
    with open(output_path_partial_txt, 'w', encoding="utf-8-sig") as txtfile:
        txtfile.write("")

    output_path_partial_csv = f"{OUTPUT_FILE}/{COIN_SHORT_NAME}_estimate.csv"

    for json_file in tqdm(json_files, desc="估計數量中..."):
        with open(json_file, 'r', encoding="utf-8-sig") as file:
            data = json.load(file)

        tweets = data[JSON_DICT_NAME]
        if not tweets:
            continue

        # 估算每個 partial 檔案的總推文數
        if json_file in partial_json_files:
            # 統計該檔案目前已抓到的推文數量與小時分佈
            partial_hour_counter = [datetime.strptime(tweet['created_at'], "%a %b %d %H:%M:%S %z %Y").hour for tweet in tweets]
            
            # 轉成 DataFrame 再取唯一小時
            partial_hour_df = pd.DataFrame({'hour': partial_hour_counter})
            observed_hours = partial_hour_df['hour'].unique()  # .unique() 取得不重複的小時數

            observed_percentage = hour_distribution.loc[observed_hours].sum()  # .loc[observed_hours] → 只取出該檔案有抓到的小時範圍的比例
            estimated_total = len(tweets) / observed_percentage

            # 取得檔名
            file_name_with_json = os.path.basename(json_file)
            file_name = os.path.splitext(file_name_with_json)[0]

            # 取得日期
            date_str = (datetime.strptime(tweets[0]['created_at'], "%a %b %d %H:%M:%S %z %Y")).strftime("%Y-%m-%d")


            # 儲存結果 csv
            results.append({
                "filename": file_name,
                "date": date_str,
                "isCompleteData": False,
                "original_count": len(tweets),
                "predicted_count": int(round(estimated_total))
            })

            with open(output_path_partial_txt, 'a', encoding="utf-8-sig") as txtfile:
                txtfile.write(f"📄 檔案：{json_file}\n")
                txtfile.write(f"實際已抓數：{len(tweets)}，觀察小時範圍：{observed_hours}\n")
                txtfile.write(f"已抓佔比：{observed_percentage:.2%}，估算總數：約 {int(round(estimated_total))} 筆\n\n")
        
        # 若是有抓完的檔案 就直接把數量存進 csv 檔
        else:
            # 取得檔名
            file_name_with_json = os.path.basename(json_file)
            file_name = os.path.splitext(file_name_with_json)[0]

            # 取得日期
            date_str = (datetime.strptime(tweets[0]['created_at'], "%a %b %d %H:%M:%S %z %Y")).strftime("%Y-%m-%d")

            # 儲存結果 csv
            results.append({
                "filename": file_name,
                "date": date_str,
                "isCompleteData": True,
                "original_count": len(tweets),
                "predicted_count": len(tweets)
            })

        with open(output_path_partial_csv, 'w', newline='', encoding='utf-8-sig') as csvfile:
            fieldnames = ['filename', 'date', 'isCompleteData', 'original_count', 'predicted_count']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow(row)

    print(f"✅ 已全部執行完成 資料路徑: {OUTPUT_FILE}")


def accuracy(hour_distribution):
    errors = []
    results = []  # 儲存要輸出的結果

    for json_file in tqdm(completed_json_files, desc="驗證估算準確度"):
        with open(json_file, 'r', encoding="utf-8-sig") as file:
            data = json.load(file)

        tweets = data[JSON_DICT_NAME]
        if not tweets:
            continue

        # 取得日期（假設每個json檔都是一天資料）
        # 以第一筆推文時間為該日代表
        date_str = datetime.strptime(tweets[0]['created_at'], "%a %b %d %H:%M:%S %z %Y").strftime("%Y-%m-%d")

        # 模擬「只保留某些小時」： (都是連續的且從 23 開始往回)
        k = random.randint(1, 23)  # 你要選的連續小時數量
        end_hour = 23
        start_hour = max(0, end_hour - k + 1)  # 往前推 k 小時，但不能小於 0
        partial_hours = list(range(start_hour, end_hour + 1))
        partial_tweets = [tweet for tweet in tweets if datetime.strptime(
            tweet['created_at'], "%a %b %d %H:%M:%S %z %Y").hour in partial_hours]

        if not partial_tweets:
            continue

        # 用原本邏輯估算：
        observed_percentage = hour_distribution.loc[partial_hours].sum()
        estimated_total = len(partial_tweets) / observed_percentage

        # 計算誤差
        real_total = len(tweets)
        error_rate = abs(estimated_total - real_total) / real_total

        errors.append(error_rate)

        # 儲存結果
        results.append({
            "date": date_str,
            "actual_count": real_total,
            "predicted_count": int(round(estimated_total))
        })

    # 輸出 CSV
    os.makedirs(OUTPUT_FILE, exist_ok=True)
    output_csv_path = os.path.join(OUTPUT_FILE, f"{COIN_SHORT_NAME}_accuracy_predictions.csv")

    with open(output_csv_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
        fieldnames = ['date', 'actual_count', 'predicted_count']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)



def plot_errors():
    # 假設讀入一個csv，包含日期、真實值、預估值
    df = pd.read_csv(f"../data/tweets/count/estimate/{COIN_SHORT_NAME}_accuracy_predictions.csv")

    # 計算誤差
    df['abs_error'] = (df['predicted_count'] - df['actual_count']).abs()
    df['percentage_error'] = df['abs_error'] / df['actual_count'] * 100  # 百分比誤差

    # 基本統計指標
    mean_abs_error = df['abs_error'].mean()
    mean_percentage_error = df['percentage_error'].mean()
    median_percentage_error = df['percentage_error'].median()
    max_percentage_error = df['percentage_error'].max()
    min_percentage_error = df['percentage_error'].min()

    print(f"平均絕對誤差 (Mean Absolute Error): {mean_abs_error:.2f} (平均一天預估與實際相差推文數)")
    print(f"平均百分比誤差 (Mean Absolute Percentage Error, MAPE): {mean_percentage_error:.2f}% (預估數量平均偏離真實值)")
    print(f"中位數百分比誤差: {median_percentage_error:.2f}% (超過一半天的誤差率低於的趴數)")
    print(f"最大百分比誤差: {max_percentage_error:.2f}% (最嚴重一天預估偏差)")
    print(f"最小百分比誤差: {min_percentage_error:.2f}% (最好的一天預估偏差)")

    output_path_completed = f"{OUTPUT_FILE}/{COIN_SHORT_NAME}_hour_distribution_and_errors.txt"
    with open(output_path_completed, 'a', encoding="utf-8-sig") as txtfile:
        txtfile.write(f"平均絕對誤差 (Mean Absolute Error): {mean_abs_error:.2f} (平均一天預估與實際相差推文數)\n")
        txtfile.write(f"平均百分比誤差 (Mean Absolute Percentage Error, MAPE): {mean_percentage_error:.2f}% (預估數量平均偏離真實值)\n")
        txtfile.write(f"中位數百分比誤差: {median_percentage_error:.2f}% (超過一半天的誤差率低於的趴數)\n")
        txtfile.write(f"最大百分比誤差: {max_percentage_error:.2f}% (最嚴重一天預估偏差)\n")
        txtfile.write(f"最小百分比誤差: {min_percentage_error:.2f}% (最好的一天預估偏差)\n")

    output_figures = "../outputs/figures"

    # 繪製預估與實際數量比較圖
    plt.figure(figsize=(12,6))
    plt.plot(df['date'], df['actual_count'], label='actual_count')
    plt.plot(df['date'], df['predicted_count'], label='predicted_count', linestyle='--')
    plt.xticks(rotation=45)
    plt.xlabel('date')
    plt.ylabel('tweet_count')
    plt.title('actual_count vs. predicted_count')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_figures}/{COIN_SHORT_NAME}_actual_count_vs_predicted_count.png', dpi=300, bbox_inches='tight')
    plt.close()  # 關閉圖表，避免佔用記憶體或影響後續繪圖

    # 繪製百分比誤差分布圖
    plt.figure(figsize=(8,5))
    plt.hist(df['percentage_error'], bins=30, color='orange', alpha=0.7)
    plt.xlabel('Percentage Error (%)')
    plt.ylabel('Number of Days')
    plt.title('Distribution of Estimation Percentage Errors')
    plt.savefig(f'{output_figures}/{COIN_SHORT_NAME}_percentage_error_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()



def main():
    distribution = hour_distribution()
    estimate(distribution)
    accuracy(distribution)
    plot_errors()


if __name__ == '__main__':
    main()