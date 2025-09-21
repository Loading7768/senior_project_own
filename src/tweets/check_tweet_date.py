import os
import json
from glob import glob
from datetime import datetime
from tqdm import tqdm

# === 幣別設定 ===
COIN_SHORT_NAME = "DOGE"
JSON_DICT_NAME = "dogecoin"  # 外層 key 和幣別一樣

# === JSON 檔案搜尋路徑 ===
JSON_GLOB_PATH = f"../data/filtered_tweets/normal_tweets/{COIN_SHORT_NAME}/*/*/{COIN_SHORT_NAME}_*_normal.json"

def extract_date_from_filename(path):
    """從檔名中擷取日期（格式如 20240201 → datetime.date）"""
    filename = os.path.basename(path)
    try:
        date_str = filename.split("_")[1]
        return datetime.strptime(date_str, "%Y%m%d").date()
    except Exception as e:
        print(f"[❌ 錯誤] 無法從檔名解析日期: {path} -> {e}")
        return None

def main():
    json_files = glob(JSON_GLOB_PATH)
    print(f"🔍 共找到 {len(json_files)} 個 JSON 檔案")

    error_files = []

    for path in tqdm(json_files, desc="檢查推文日期"):
        expected_date = extract_date_from_filename(path)
        if not expected_date:
            continue

        try:
            with open(path, "r", encoding="utf-8-sig") as f:
                data = json.load(f)

            tweets = data.get(COIN_SHORT_NAME, [])
            for tweet in tweets:
                created_at = tweet.get("created_at", "")
                try:
                    actual_date = datetime.strptime(created_at, "%a %b %d %H:%M:%S %z %Y").date()
                    if actual_date != expected_date:
                        error_files.append(path)
                        break  # 只列一次這個檔案
                except Exception:
                    continue  # 有些推文可能格式錯，就跳過這筆
        except Exception as e:
            print(f"[⚠️ 錯誤] 無法讀取 {path}: {e}")

    # === 輸出所有錯誤檔名 ===
    print(f"\n🚨 共 {len(error_files)} 個檔案中含有錯誤推文日期：\n")
    for f in error_files:
        print(f)

if __name__ == "__main__":
    main()
