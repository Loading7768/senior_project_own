import os
import json
import codecs
from datetime import datetime
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import download

# === 設定路徑參數 ===
import sys
from pathlib import Path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
from config import COIN_SHORT_NAME, JSON_DICT_NAME

# === 下載 VADER 情緒字典 ===
download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# === 設定年份與月份（可自由更改）===
YEAR = "2019"
MONTH = "03"

# === 路徑設定 ===
INPUT_DIR = f"../data/tweets/{COIN_SHORT_NAME}/{YEAR}/{MONTH}"
BASE_OUTPUT_DIR = f"../data/sentiment/{COIN_SHORT_NAME}/{YEAR}/{MONTH}"
POS_DIR = os.path.join(BASE_OUTPUT_DIR, "positive")
NEU_DIR = os.path.join(BASE_OUTPUT_DIR, "neutral")
NEG_DIR = os.path.join(BASE_OUTPUT_DIR, "negative")

# 建立輸出子資料夾
os.makedirs(POS_DIR, exist_ok=True)
os.makedirs(NEU_DIR, exist_ok=True)
os.makedirs(NEG_DIR, exist_ok=True)

# === 開始處理所有 JSON 檔案 ===
for file in sorted(os.listdir(INPUT_DIR)):
    if not file.endswith(".json"):
        continue

    date_str = file.replace(f"{COIN_SHORT_NAME}_", "").replace(".json", "")
    try:
        datetime.strptime(date_str, "%Y%m%d")  # 驗證格式是否為 yyyymmdd
    except ValueError:
        print(f"❌ 跳過不合法日期檔案：{file}")
        continue

    filepath = os.path.join(INPUT_DIR, file)
    try:
        with codecs.open(filepath, "r", encoding="utf-8-sig") as f:
            raw = json.load(f)

        if isinstance(raw, dict) and JSON_DICT_NAME in raw:
            tweets = raw[JSON_DICT_NAME]
        else:
            print(f"❌ 格式錯誤或缺少 {JSON_DICT_NAME}：{file}")
            continue
    except Exception as e:
        print(f"❌ 無法解析 {file}：{e}")
        continue

    # === 情緒分類 ===
    pos, neu, neg = [], [], []
    for tweet in tweets:
        text = tweet.get("text", "")
        if not text.strip():
            continue
        score = sia.polarity_scores(text)
        c = score["compound"]
        if c >= 0.05:
            pos.append(tweet)
        elif c <= -0.05:
            neg.append(tweet)
        else:
            neu.append(tweet)

    # === 儲存結果 JSON 檔案 ===
    def save_json(subfolder, name, content):
        path = os.path.join(subfolder, f"{name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(content, f, ensure_ascii=False, indent=2)

    save_json(POS_DIR, f"{COIN_SHORT_NAME}_{date_str}_positive", pos)
    save_json(NEU_DIR, f"{COIN_SHORT_NAME}_{date_str}_neutral", neu)
    save_json(NEG_DIR, f"{COIN_SHORT_NAME}_{date_str}_negative", neg)

    print(f"✅ {file} → 完成分類：👍 {len(pos)} | 😐 {len(neu)} | 👎 {len(neg)}")
