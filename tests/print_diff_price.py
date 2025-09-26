import numpy as np
import pickle

COIN_SHORT_NAME = "DOGE"

IS_FILTERED = True  # 看是否有分 normal 與 bot

IS_RUN_AUGUST = False  # 看現在是不是要跑 2025/08 的資料

SUFFIX_FILTERED = "" if IS_FILTERED else "_non_filtered"
SUFFIX_AUGUST   = "_202508" if IS_RUN_AUGUST else ""


diff_price = np.load(f"../data/ml/dataset/coin_price/{COIN_SHORT_NAME}_price_diff_original{SUFFIX_FILTERED}{SUFFIX_AUGUST}.npy")

print(f"diff_price 有 {np.sum(diff_price == 0)} 個 0")

with open(f"../data/ml/dataset/keyword/{COIN_SHORT_NAME}_ids{SUFFIX_FILTERED}{SUFFIX_AUGUST}.pkl", "rb") as f:   # rb = read binary
    ids = pickle.load(f)  # array[('coin', 'date', 'no.'), (str, '%Y-%m-%d', int)
ids = np.array(ids)  # 把 ids 轉成 numpy array
dates = np.array([row[1] for row in ids])  # 只把 'date' 取出來，並轉成 np.array

# 設定輸出不省略
# np.set_printoptions(threshold=np.inf)

# print(dates)

np.savetxt(f"{COIN_SHORT_NAME}_ids{SUFFIX_FILTERED}{SUFFIX_AUGUST}.txt", dates, fmt="%s")

print(diff_price[:180])