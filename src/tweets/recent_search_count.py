import requests
from datetime import datetime, timedelta, timezone
import time

# ==== 請替換為你的 Bearer Token ====
BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAADKN2wEAAAAAQzHFCG%2F8j%2BBcUFctjuNQ4RZuDXA%3DevK4FR2FO8A1wcBR24Z9rH2W5I07KRb2huZGLQGZoK4eYySkVe"

# ==== 你要查的 keyword ====
QUERY = '(dogecoin OR \"doge coin\" OR \"doge meme coin\" OR dogecoin OR \"$DOGE\" OR \"dollar doge\") lang:en'
# '(dogecoin OR \"doge coin\" OR \"doge meme coin\" OR dogecoin OR \"dollar doge\") lang:en'

# ==== 查詢日期範圍 ====
start_date = datetime(2025, 7, 7, 0, 0, 0, tzinfo=timezone.utc)
end_date = datetime(2025, 7, 10, 0, 0, 0, tzinfo=timezone.utc)

# ==== Twitter API endpoint ====
URL = "https://api.twitter.com/2/tweets/counts/recent"

# ==== Header ====
HEADERS = {
    "Authorization": f"Bearer {BEARER_TOKEN}"
}

# ==== 開始查詢 ====

# 查詢參數
params = {
    "query": QUERY,
    "granularity": "day",  # 可以選 "minute", "hour", "day"
    "start_time": start_date.isoformat(),
    "end_time": end_date.isoformat()
}

print(f"🔍 查詢 {start_date.isoformat()} ~ {end_date.isoformat()}")
response = requests.get(URL, headers=HEADERS, params=params)
print("Status Code:", response.status_code)

if response.status_code == 200:
    data = response.json()
    print(data)
    # 可以在這裡把 data 存到檔案或資料庫
else:
    print("⚠️ 查詢失敗:", response.text)
