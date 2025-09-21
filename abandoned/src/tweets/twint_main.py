import twint

# 建立設定
c = twint.Config()

# === 設定抓取條件 ===
# c.Username = "elonmusk"      # 指定帳號
c.Search = "AI"              # 關鍵字
c.Since = "2025-08-01"       # 起始日期
c.Until = "2025-09-01"       # 結束日期
c.Limit = 100                # 最多抓多少推文（可刪掉 = 不限）
c.Lang = "en"                # 限定語言 (例如 "en", "zh")

# === 輸出設定 ===
c.Store_csv = True           # 存成 CSV
c.Output = "elon_tweets.csv" # 輸出檔名

# === 執行抓取 ===
twint.run.Search(c)
