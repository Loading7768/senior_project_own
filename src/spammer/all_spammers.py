import os
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm
import csv
import json

from pathlib import Path
import sys
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
from config import JSON_DICT_NAME, COIN_SHORT_NAME

# === 自訂參數 ===
LIMIT_DAY = 50      # 每天總推文數超過這個就是高發文天
LIMIT_HOUR = 6      # 小時內推文超過這個就是高小時發文
TOTAL_COUNT = 5     # 相加數字 >= 5 的話就算 spammer
root_folder = f"../data/author_all/{COIN_SHORT_NAME}"
output_folder = f"../data/spammer/{COIN_SHORT_NAME}/"
os.makedirs(output_folder, exist_ok=True)
csv_file = os.path.join(output_folder, f"{COIN_SHORT_NAME}_high_post_summary.csv")
spammer_list_file = os.path.join(output_folder, f"{COIN_SHORT_NAME}_spammers.txt")

# 解析時間
# def parse_time(s):
#     try:
#         return datetime.strptime(s, "%a %b %d %H:%M:%S %z %Y")
#     except:
#         return None

# author -> date -> count
author_day_counts = defaultdict(lambda: defaultdict(int))
author_hour_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

# 收集所有 JSON 檔
all_files = []
for dirpath, _, filenames in os.walk(root_folder):
    for filename in sorted(filenames):
        if filename.endswith(".json"):
            all_files.append(os.path.join(dirpath, filename))

print(f"🔍 準備處理 {len(all_files)} 個檔案...")

# 流式解析 JSON
for file_path in tqdm(all_files, desc="讀取 JSON 檔案"):
    try:
        with open(file_path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)

        tweets = data.get(JSON_DICT_NAME, [])
        if not isinstance(tweets, list):
            continue

        for tw in tweets:
            user = tw.get("user_account") or tw.get("username")
            t = datetime.strptime(tw.get("created_at", ""), "%a %b %d %H:%M:%S %z %Y")
            if user and t:
                date = t.date()
                hour = t.hour
                author_day_counts[user][date] += 1
                author_hour_counts[user][date][hour] += 1
    except Exception as e:
        print(f"❌ 讀取失敗：{file_path}，錯誤：{e}")
        continue

# 計算天數
summary = {}
spammers = set()
for user in set(list(author_day_counts.keys()) + list(author_hour_counts.keys())):
    # 每天總推文 >= LIMIT_DAY
    high_day_count = sum(1 for count in author_day_counts[user].values() if count >= LIMIT_DAY)
    # 每天任一小時 >= LIMIT_HOUR
    high_hour_count = sum(1 for hour_dict in author_hour_counts[user].values() if any(c >= LIMIT_HOUR for c in hour_dict.values()))
    summary[user] = (high_day_count, high_hour_count)

    # 判斷 spammer：兩個統計相加 >= TOTAL_COUNT
    if high_day_count + high_hour_count >= TOTAL_COUNT:
        spammers.add(user)

# 輸出 CSV，只列出任一統計大於 0 的帳號
with open(csv_file, "w", encoding="utf-8-sig", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "user_account",
        f"days_over_{LIMIT_DAY}_tweets",
        f"days_hour_ge_{LIMIT_HOUR}_tweets",
        "total_days_over_limit"
    ])
    for user, (day_count, hour_count) in sorted(summary.items()):
        if day_count > 0 or hour_count > 0:  # 只列出任一大於 0
            total = day_count + hour_count
            writer.writerow([user, day_count, hour_count, total])

print(f"✅ 高發文天數與高小時發文天數 CSV 已存入（只列出任一大於 0）: {csv_file}")

# 輸出 spammer 名單到文字檔
with open(spammer_list_file, "w", encoding="utf-8-sig") as f:
    for user in sorted(spammers):
        f.write(user + "\n")

print(f"✅ Spammer 名單已存入：{spammer_list_file} （days_over + hours_over >= 5）")

