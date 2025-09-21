import pandas as pd

def check_missing_dates(csv_path):
    # 讀取 CSV
    df = pd.read_csv(csv_path)

    # 把 snapped_at 轉成 datetime
    df['snapped_at'] = pd.to_datetime(df['snapped_at'])

    # 只保留日期（去掉時間部分）
    df['date'] = df['snapped_at'].dt.date

    # 建立完整日期範圍
    full_range = pd.date_range(start=df['date'].min(), end=df['date'].max(), freq="D")

    # 找出缺少的日期
    missing_dates = full_range.difference(df['date'])

    if len(missing_dates) == 0:
        print("✅ 沒有缺少的日期")
    else:
        print("⚠️ 缺少的日期如下：")
        for d in missing_dates:
            print(d.date())

# 使用方式
COIN_SHORT_NAME = "TRUMP"
check_missing_dates(f"../data/coin_price/{COIN_SHORT_NAME}_price.csv")
