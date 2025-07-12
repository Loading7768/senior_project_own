import os
import json
from datetime import datetime
from collections import defaultdict

# === 自訂參數 ===
json_coin = "dogecoin"
coin_type = "DOGE"
year = "2022"
month = "05"

# === 資料夾設定 ===
folder_path = f"../data/tweets/{coin_type}/{year}/{month}"
# folder_path = f"data/tweets"
file_prefix = f"{coin_type}_{year}{month}"

# === 儲存結果
qualified_authors_per_day = defaultdict(list)  # {date: [(author, count), ...]}
daily_stats = {}  # {date: {"authors": N, "tweets": M}}
spammers = []

# === 掃描所有檔案 ===
for filename in sorted(os.listdir(folder_path)):
    if filename.startswith(file_prefix) and filename.endswith(".json"):
        date_str = filename.replace(f"{coin_type}_", "").replace(".json", "")
        formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
        file_path = os.path.join(folder_path, filename)

        try:
            with open(file_path, "r", encoding="utf-8-sig") as f:
                data = json.load(f)
        except Exception as e:
            # print(f"讀取 {filename} 時失敗：{e}")
            continue

        if json_coin not in data:
            continue

        # 計算當天總作者與貼文數
        user_tweets = defaultdict(list)
        for tweet in data[json_coin]:
            username = tweet["username"]
            tweet_time = datetime.strptime(tweet["created_at"], "%a %b %d %H:%M:%S %z %Y")
            user_tweets[username].append(tweet_time)

        total_authors = len(user_tweets)
        total_tweets = sum(len(t) for t in user_tweets.values())
        daily_stats[formatted_date] = {
            "authors": total_authors,
            "tweets": total_tweets
        }

        # 分析每位作者
        for user, times in user_tweets.items():
            if len(times) <= 1:
                continue  # 不計算發一篇的人

            times.sort()

            # 是否有 1 小時內發 6 篇以上
            for i in range(len(times)):
                count = 1
                for j in range(i + 1, len(times)):
                    if (times[j] - times[i]).total_seconds() <= 3600:
                        count += 1
                        if count >= 6:
                            qualified_authors_per_day[formatted_date].append((user, len(times)))
                            break
                    else:
                        break
                if count >= 6:
                    break

# === 輸出結果 ===
for date in sorted(daily_stats.keys()):
    # print(f"\n📅 {date}")
    # print(f"總作者數：{daily_stats[date]['authors']} 位")
    # print(f"總貼文數：{daily_stats[date]['tweets']} 篇")

    # 讀檔抓貼文資料
    all_authors_counter = defaultdict(int)
    file_name = f"{coin_type}_{date.replace('-', '')}.json"
    file_path = os.path.join(folder_path, file_name)

    try:
        with open(file_path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        if json_coin in data:
            for tweet in data[json_coin]:
                try:
                    user = tweet["user_account"]
                except Exception:
                    user = tweet["username"]
                all_authors_counter[user] += 1
    except Exception as e:
        # print(f"無法讀取或解析 {file_name}：{e}")
        continue

    # 當天所有作者按發文數排序
    sorted_authors = sorted(all_authors_counter.items(), key=lambda x: -x[1])

    # 這天有符合發文快速的作者嗎？
    fast_authors = {user for user, _ in qualified_authors_per_day.get(date, [])}
    fast_authors_total_posts = sum(all_authors_counter[user] for user in fast_authors if user in all_authors_counter)

    if fast_authors:
        # print("⚡ 發文太快的作者（1小時內6篇以上）：")
        for user, count in sorted(qualified_authors_per_day[date], key=lambda x: -x[1]):
            spammers.append(user)
            # print(f"    {user}: {count} 篇")
        # print(f"🔢 這些快速作者合計貼文數：{fast_authors_total_posts} 篇")
    #else:
        # print("⚠ 無符合『1小時內至少6篇且不只1篇』的作者")

    # 前 N 名發文最多的人，N=這天有符合條件的數量 or 10，取最大
    rank_limit = max(len(qualified_authors_per_day.get(date, [])), 10)

    top_n_authors = sorted_authors[:rank_limit]
    top_n_total_posts = sum(count for _, count in top_n_authors)

    # print(f"📊 當天發文數最多的前 {rank_limit} 名：")
    #for i, (user, count) in enumerate(top_n_authors, start=1)
        # print(f"    {i}. {user}: {count} 篇")
    # print(f"🔢 這些前{rank_limit}名合計貼文數：{top_n_total_posts} 篇")

    # === 🔥 最後加上你要的比例統計 ===
    total_authors = daily_stats[date]["authors"]
    total_tweets = daily_stats[date]["tweets"]
    
    fast_authors_count = len(fast_authors)

    ratio_fast_authors = fast_authors_count / total_authors if total_authors else 0
    ratio_fast_posts = fast_authors_total_posts / total_tweets if total_tweets else 0
    ratio_topn_posts = top_n_total_posts / total_tweets if total_tweets else 0

    # print(f"📈 發文太快的作者比例：{fast_authors_count} / {total_authors} = {ratio_fast_authors:.2%}")
    # print(f"📈 發文太快作者的貼文比例：{fast_authors_total_posts} / {total_tweets} = {ratio_fast_posts:.2%}")
    # print(f"📈 當天前{rank_limit}名作者的貼文比例：{top_n_total_posts} / {total_tweets} = {ratio_topn_posts:.2%}")

spammers = list(set(spammers))
# for spammer in spammers: 
#     print(spammer)

output_path = f"../data/spammer/{year}"
os.makedirs(output_path, exist_ok=True)
output_txt_path = f"../data/spammer/{year}/spammer_{year}{month}.txt"
with open(output_txt_path, "w", encoding="utf-8-sig") as txtfile:
    for spammer in sorted(spammers): 
        txtfile.write(spammer)
        txtfile.write("\n")

print(f"已儲存在 spammer_{year}{month}.txt")