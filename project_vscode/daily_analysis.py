import os
import json
import csv
from collections import Counter

# 🛠️ 原本的功能：分析推文，輸出每日統計成 CSV
def analyze_tweets(folder_path, coin_type, json_coin, output_csv):
    daily_author_stats = []  # 儲存每天的統計結果
    all_post_counts = set()  # 紀錄所有可能的發文次數（用來決定 CSV 欄位）

    # 🔍 取得資料夾裡所有 JSON 檔案，並排序（確保按日期順序）
    json_files = sorted([
        f for f in os.listdir(folder_path) if f.endswith(".json")
    ])

    # 📦 逐一處理每一個 JSON 檔
    for filename in json_files:
        # 解析出檔案名中的日期資訊，轉成 YYYY-MM-DD 格式
        date_str = filename.replace(f"{coin_type}_", "").replace(".json", "")
        date_formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
        
        file_path = os.path.join(folder_path, filename)

        try:
            # 讀取 JSON 檔案內容
            with open(file_path, "r", encoding="utf-8-sig") as f:
                data = json.load(f)

            # 如果 JSON 裡沒有指定的幣種資料，跳過
            if json_coin not in data:
                continue

            tweets = data[json_coin]  # 取得推文列表
            tweet_count = len(tweets)  # 當天推文總數
            author_counter = Counter(tweet["username"] for tweet in tweets)  # 每個作者發文數統計

            unique_authors = len(author_counter)  # 當天發過文的作者數量
            multiple_post_authors = sum(1 for c in author_counter.values() if c > 1)  # 發超過 1 篇的人數

            post_count_distribution = Counter(author_counter.values())  # 分析發文次數的分布
            all_post_counts.update(post_count_distribution.keys())  # 更新所有可能的發文次數

            # 整理成一行資料
            row = {
                "Date": date_formatted,
                "Unique Authors": unique_authors,
                "Tweet Count": tweet_count,
                "Authors With Multiple Posts": multiple_post_authors
            }

            # 加入各種「發 N 次」的作者數量
            for count, user_num in post_count_distribution.items():
                row[f"Authors Posting {count} Time{'s' if count > 1 else ''}"] = user_num

            daily_author_stats.append(row)  # 收集這天的統計資料

        except Exception as e:
            print(f"無法處理 {filename}：{e}")  # 讀取失敗時顯示錯誤訊息

    # 🏷️ 決定 CSV 的所有欄位名稱
    fixed_columns = ["Date", "Unique Authors", "Tweet Count", "Authors With Multiple Posts"]
    dynamic_columns = [f"Authors Posting {i} Time{'s' if i > 1 else ''}" for i in sorted(all_post_counts)]
    all_columns = fixed_columns + dynamic_columns

    # ✍️ 寫入 CSV 檔案
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_columns)
        writer.writeheader()
        for row in daily_author_stats:
            # 保證每一列都有齊所有欄位
            for col in dynamic_columns:
                row.setdefault(col, 0)
            writer.writerow(row)

    print(f"✅ 統計完成：{output_csv}")

# ✨ 新增的功能：提取發超過 144 篇的人，並儲存推文成 JSON
def extract_prolific_tweets(folder_path, coin_type, json_coin, output_json, threshold=144):
    prolific_tweets = []  # 儲存符合條件的推文
    prolific_authors = {}  # {日期: {作者: 發文數}}
    total_tweet_counts = {}  # {日期: 當天推文總數}

    # 🔍 取得資料夾裡所有 JSON 檔案，並排序
    json_files = sorted([
        f for f in os.listdir(folder_path) if f.endswith(".json")
    ])

    # 📦 逐一處理每一個 JSON 檔
    for filename in json_files:
        file_path = os.path.join(folder_path, filename)
        date_str = filename.replace(f"{coin_type}_", "").replace(".json", "")
        date_formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"

        try:
            with open(file_path, "r", encoding="utf-8-sig") as f:
                data = json.load(f)

            if json_coin not in data:
                continue

            tweets = data[json_coin]
            tweet_count = len(tweets)
            total_tweet_counts[date_formatted] = tweet_count  # 記錄當天總推文數

            author_counter = Counter(tweet["username"] for tweet in tweets)

            prolific = {author: count for author, count in author_counter.items() if count > threshold}
            if prolific:
                prolific_authors[date_formatted] = prolific

                for tweet in tweets:
                    if tweet["username"] in prolific:
                        prolific_tweets.append(tweet)

        except Exception as e:
            print(f"無法處理 {filename}：{e}")

    # （選擇性）可以排序推文
    # prolific_tweets.sort(key=lambda x: x.get("created_at", ""))

    # 🔥 寫入符合條件的推文成 JSON
    output_data = {json_coin: prolific_tweets}
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    print(f"✅ 發超過 {threshold} 篇的推文已存到：{output_json}")

    # 🔥 最後列出每天超過144篇的作者，含推文數與佔比
    if prolific_authors:
        print("\n🚀 發超過 144 篇推文的作者名單（含推文數與佔比）：")
        for date, authors in prolific_authors.items():
            total = total_tweet_counts.get(date, 0)
            print(f"\n【{date}】（當天總推文：{total} 篇）")
            for author, count in authors.items():
                if total > 0:
                    percentage = (count / total) * 100
                    print(f" - {author}：{count} 篇 (占比 {percentage:.2f}%)")
                else:
                    print(f" - {author}：{count} 篇 (占比 無法計算(總推文為 0))")
    else:
        print("\nℹ️ 沒有任何作者單日發超過 144 篇推文。")



# ✅ 使用範例（直接執行）
if __name__ == "__main__":
    folder = "data/DOGE/2025/3"             # 📂 資料夾路徑
    json_coin = "dogecoin"                  # 🪙 幣種名稱（對應 JSON 裡的 key）
    coin = "DOGE"                           # 檔案名中的幣種名稱
    
    # 1️⃣ 執行原本的統計功能
    output_csv = "doge_post_stats_202503.csv"
    analyze_tweets(folder, coin, json_coin, output_csv)
    
    # 2️⃣ 執行新的提取大量發文作者功能
    output_json = "doge_prolific_author_tweets_202503.json"
    extract_prolific_tweets(folder, coin, json_coin, output_json, threshold=144)
