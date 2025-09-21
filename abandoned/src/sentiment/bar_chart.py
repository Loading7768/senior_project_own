import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# ======== 設定資料夾路徑（請修改為你的資料夾） ========
folder_path = "analysis/202502"  # ← 請自行修改資料夾路徑

# ======== 收集所有推文 ========
all_tweets = []

for filename in sorted(os.listdir(folder_path)):
    if filename.endswith(".json"):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            try:
                data = json.load(f)
                tweets = data.get("dogecoin", [])
                all_tweets.extend(tweets)
            except Exception as e:
                print(f"讀取錯誤：{filename}, 錯誤：{e}")

# ======== 檢查是否讀取到資料 ========
if not all_tweets:
    print("沒有找到任何 DOGE 推文")
    exit()

# ======== 建立 DataFrame 並處理日期 ========
df = pd.DataFrame(all_tweets)
df['created_at'] = pd.to_datetime(df['created_at'])
df['date'] = df['created_at'].dt.date

# ======== 計算每日情緒比例 ========
sentiment_counts = df.groupby(['date', 'sentiment']).size().unstack(fill_value=0)
sentiment_proportions = sentiment_counts.div(sentiment_counts.sum(axis=1), axis=0)

# ======== 整體情緒統計 ========
overall_sentiment_ratio = df['sentiment'].value_counts(normalize=True)

# ======== 畫圖：每日情緒比例長條圖 ========
plt.figure(figsize=(12, 6))
sentiment_proportions.plot(kind='bar', stacked=True, colormap='Set3')
plt.title("DOGE tweet ratio")
plt.ylabel("Ratio")
plt.xlabel("Date")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("daily_sentiment_ratio.png")
plt.show()

# ======== 印出整體情緒比例 ========
print("\n整體情緒占比：")
for sentiment, ratio in overall_sentiment_ratio.items():
    print(f"{sentiment}: {ratio:.2%}")
