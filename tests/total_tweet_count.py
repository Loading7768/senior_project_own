import pandas as pd
from glob import glob
import json
import os

# # 找出所有 summary.csv 檔案
# path = "../data/tweets/summary/*_tweet_summary.csv"
# csv_files = glob(path, recursive=True)

# total_tweets = 0
# print(csv_files)
# for file in csv_files:
#     df = pd.read_csv(file)
#     if "tweet_total" in df.columns:
#         total_tweets += df["tweet_total"].sum()

# print(f"所有檔案的 tweet_total 加總：{total_tweets}")



with open(f"../data/ml/dataset/coin_price/DOGE_current_tweet_count.json", "r", encoding="utf-8") as f:
    tweet_count_dict = json.load(f)

# 計算總和
total = sum(tweet_count_dict.values())

print("總和:", total)


