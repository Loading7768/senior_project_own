import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle




'''可修改參數'''
COIN_SHORT_NAME = ["DOGE", "PEPE", "TRUMP"]

PRICE_CSV_PATH = "../data/coin_price"

INPUT_PATH = "../data/ml/dataset"

OUTPUT_PATH = "../data/ml/dataset"
'''可修改參數'''



def categorize_array_multi(Y_diff, ids):
    """
    Y_diff: np.ndarray, shape = (N, num_labels), 原始價格差價
    ids: list of (coin, date, tweet_id) 對應每個樣本
    price_csv_path: coin 的價格 csv 檔
    t1, t2: 五元分類閾值，百分比
    """

    print("Y_diff 前 5 筆：", Y_diff[:5])
    print("Y_diff 非零數量：", np.count_nonzero(Y_diff))


    # for coin_short_name in COIN_SHORT_NAME:
    #     print(f"目前正在轉換 {coin_short_name}...")
    #     # 讀取價格 CSV
    #     price_df = pd.read_csv(f"{PRICE_CSV_PATH}/{coin_short_name}_price.csv")
    #     price_df['snapped_at'] = pd.to_datetime(price_df['snapped_at'], format="%Y-%m-%d %H:%M:%S %Z")
    #     price_df.set_index('snapped_at', inplace=True)
    #     price_df.index = price_df.index.tz_localize(None)  # 移除時區

    #     # 建立每天價格的 dict，方便查詢
    #     price_lookup = price_df['price'].to_dict()  # 假設 csv 有 price 欄
    #     print("price_lookup:", list(price_lookup.items())[:5])



    #     # 先建立空陣列
    #     Y_pct = np.zeros_like(Y_diff, dtype=float)

    #     for i, (coin, date, tweet_id) in tqdm(enumerate(ids), total=len(ids), desc="正在將價差轉成價錢變化率..."):
    #         if coin == coin_short_name:
    #             # 確保 date 是 datetime.date
    #             if isinstance(date, str):
    #                 date = pd.to_datetime(date)
    #             # 查當天價格
    #             if date in price_lookup:
    #                 price_today = price_lookup[date]
    #                 # print("\ncoin:", coin)
    #                 # print("date:", date)
    #                 # print("price_today:", price_today)
    #                 # input()
    #             else:
    #                 # 如果找不到日期，改用 1 避免除零
    #                 print("找不到日期:", date)
    #                 price_today = 1.0

    #             # 將整列的價格差轉百分比
    #             Y_pct[i, :] = Y_diff[i, :] / price_today
    #             print("Y_pct[i, :]:", Y_pct[i, :])


    Y_pct = np.zeros_like(Y_diff, dtype=float)

    # 建立每個幣種的 price_lookup
    price_lookup_dict = {}
    for coin_short_name in COIN_SHORT_NAME:
        price_df = pd.read_csv(f"{PRICE_CSV_PATH}/{coin_short_name}_price.csv")
        price_df['snapped_at'] = pd.to_datetime(price_df['snapped_at'], format="%Y-%m-%d %H:%M:%S %Z")
        price_df.set_index('snapped_at', inplace=True)
        price_df.index = price_df.index.tz_localize(None)

        price_lookup_dict[coin_short_name] = {d.date(): p for d, p in price_df['price'].items()}

    # 更新每一行
    for i, (coin, date, tweet_id) in tqdm(enumerate(ids), total=len(ids), desc="計算價格變化率..."):
        if isinstance(date, str):
            date = pd.to_datetime(date).date()
        else:
            date = date.date()

        price_lookup = price_lookup_dict.get(coin)
        if price_lookup is None:
            print(f"警告: 找不到幣種 {coin} 的價格資料")
            price_today = 1.0
        else:
            price_today = price_lookup.get(date)
            if price_today is None:
                print(f"警告: 找不到日期 {date} 對應幣種 {coin}")
                price_today = 1.0

        # 計算百分比變化率
        Y_pct[i, :] = Y_diff[i, :] / price_today

        # 可選 debug: 前 5 筆
        if i < 5:
            print(f"\nrow {i} | coin: {coin}, date: {date}, price_today: {price_today}, Y_diff: {Y_diff[i]}, Y_pct: {Y_pct[i]}")

    print("轉換完成！")



    print("已成功轉換")

    return Y_pct


def load_and_preprocess():

    y_train_all = np.load(f"{INPUT_PATH}/Y_train_filtered.npz")
    y_train_all = y_train_all['Y']
    y_test_all = np.load(f"{INPUT_PATH}/Y_test.npz")
    y_test_all = y_test_all['Y']

    print(y_train_all.shape)

    with open(f"{INPUT_PATH}/ids_train_filtered.pkl", 'rb') as file:
        ids_train_all = pickle.load(file)
    with open(f"{INPUT_PATH}/ids_test.pkl", 'rb') as file:
        ids_test_all = pickle.load(file)
    
    # 轉換為價格變化率並儲存成新的 Y
    y_train = categorize_array_multi(y_train_all, ids_train_all)  # shape (N,5)
    np.savez_compressed(f'{OUTPUT_PATH}/Y_train_filtered_increase.npz', Y=y_train)
    print("y_train.shape", y_train.shape)
    print(y_train[:5])

    y_test  = categorize_array_multi(y_test_all, ids_test_all)   # shape (N,5)
    np.savez_compressed(f'{OUTPUT_PATH}/Y_test_increase.npz', Y=y_test)
    print("y_test.shape", y_test.shape)
    print(y_test[:5])

    return y_train, y_test



def main():
    # --- 載入資料 ---
    y_train_categorized, y_test_categorized= load_and_preprocess()



if __name__ == "__main__":
    main()