import numpy as np

'''可修改參數'''
COIN_SHORT_NAME = ["DOGE", "PEPE", "TRUMP"]

INPUT_PATH = "../data/ml/dataset/coin_price"

INPUT_FIRST_CLASSIFIER_PATH = "../data/ml/classification/logistic_regression"
'''可修改參數'''

def merge():
    X = []
    Y = []

    # 將不同幣種的 X, Y 分別讀取進來
    for coin_short_name in COIN_SHORT_NAME:
        # --- 讀取 X ---
        X_diff_past = np.load(f"{INPUT_PATH}/{coin_short_name}_price_diff_past5days.npy")  # 讀取 前面幾天 的 價差、價錢變化率
        X_XGBoost = np.load(f"{INPUT_PATH}/{coin_short_name}_XGBoost_features.npy")  # 讀取 XBGoost 所使用的 features
        # XGBoost_dates = np.load(f"{INPUT_PATH}/{coin_short_name}_XGBoost_dates.npy")  # 讀取 XBGoost 所使用的 dates
        # X_first_classifier = np.load(f"{INPUT_FIRST_CLASSIFIER_PATH}/{coin_short_name}_combined_daily_predictions.npy")  # 讀取 第一個分類器 預測的結果
        print("X_diff_past.shape:", X_diff_past.shape)
        print("X_XGBoost.shape:", X_XGBoost.shape)
        # print("X_first_classifier.shape:", X_first_classifier.shape)

        # --- 讀取 Y ---
        Y_single_coin = np.load(f"{INPUT_PATH}/{coin_short_name}_price_diff_original.npy")  # 讀取 明天 的價錢變化率 (price_diff_rate_tomorrow)
        print("Y_single_coin.shape:", Y_single_coin.shape)

        # --- 對齊時間軸 ---
        start_idx = 13   # 因為XGBoost特徵要跳過前13天
        X_diff_past = X_diff_past[(start_idx - 5):]  # 原本少5天 → 再切掉到8
        X_XGBoost = X_XGBoost[start_idx:]            # 原本就少13天
        X_first_classifier = X_first_classifier[start_idx:]
        Y_single_coin = Y_single_coin[start_idx:]

        # --- 合併特徵 ---
        X_single_coin = np.hstack([X_diff_past, X_XGBoost, X_first_classifier.reshape(-1, 1)])

        # --- 存進總集合 ---
        X.append(X_single_coin)
        Y.append(Y_single_coin)

    # --- 把三個幣種合併成一個大陣列 ---
    X = np.vstack(X)
    Y = np.concatenate(Y)

    return X, Y

def main():
    X, Y = merge()

if __name__ == "__main__":
    main()
