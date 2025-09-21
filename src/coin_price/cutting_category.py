# import pandas as pd
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import norm, shapiro

# coin_short_names = ["DOGE", "PEPE", "TRUMP"]
# data_dir = "../data/coin_price"

# dfs = []

# for coin in coin_short_names:
#     file_path = os.path.join(data_dir, f"{coin}_price.csv")
#     df = pd.read_csv(file_path)
#     df['Coin'] = coin
#     df = df[['snapped_at', 'Coin', 'price']]
#     dfs.append(df)

# df_long = pd.concat(dfs, ignore_index=True)
# df_long.to_csv("../data/coin_price/all_coins_price_long.csv", index=False)
# print("✅ 合併完成")




# # 1. 固定百分比分界（可以依需求調整）
# fixed_thresholds = {
#     "大跌": -0.10,  # < -10%
#     "跌": -0.03,    # -10% ~ -3%
#     "持平_low": -0.03,
#     "持平_high": 0.03,
#     "漲": 0.03,     # 3% ~ 10%
#     "大漲": 0.10    # > 10%
# }

# def compute_thresholds(df):
#     # 計算日變化率
#     df["pct_change"] = df["price"].pct_change().dropna()
#     changes = df["pct_change"].dropna()

#     results = {}

#     # 1. 固定百分比法
#     results["固定百分比法"] = fixed_thresholds

#     # 2. 分位數法
#     q10 = changes.quantile(0.10)
#     q40 = changes.quantile(0.40)
#     q60 = changes.quantile(0.60)
#     q90 = changes.quantile(0.90)
#     results["分位數法"] = {
#         "大跌": f"< {q10:.4f}",
#         "跌": f"{q10:.4f} ~ {q40:.4f}",
#         "持平": f"{q40:.4f} ~ {q60:.4f}",
#         "漲": f"{q60:.4f} ~ {q90:.4f}",
#         "大漲": f"> {q90:.4f}"
#     }

#     # 3. 標準差法
#     mu = changes.mean()
#     sigma = changes.std()
#     results["標準差法"] = {
#         "大跌": f"< {mu - 2*sigma:.4f}",
#         "跌": f"{mu - 2*sigma:.4f} ~ {mu - sigma:.4f}",
#         "持平": f"{mu - sigma:.4f} ~ {mu + sigma:.4f}",
#         "漲": f"{mu + sigma:.4f} ~ {mu + 2*sigma:.4f}",
#         "大漲": f"> {mu + 2*sigma:.4f}"
#     }

#     return results

# # 主程式
# # for coin in coin_short_names:
# file_path = os.path.join(data_dir, f"all_coins_price_long.csv")
# df = pd.read_csv(file_path)

# # thresholds = compute_thresholds(df)

# # print(f"\n=== {coin} ===")
# # for method, values in thresholds.items():
# #     print(f"\n【{method}】")
# #     for k, v in values.items():
# #         print(f"{k}: {v}")


# # 假設有 "Close" 欄位作為收盤價
# df["pct_change"] = df["price"].pct_change().dropna()
# returns = df["pct_change"].dropna()

# # 計算平均值和標準差
# mu, sigma = returns.mean(), returns.std()

# # 畫直方圖
# plt.figure(figsize=(8, 5))
# count, bins, ignored = plt.hist(returns, bins=50, density=True, alpha=0.6, color='blue', edgecolor='black')

# # 疊加常態分布曲線
# x = np.linspace(returns.min(), returns.max(), 1000)
# plt.plot(x, norm.pdf(x, mu, sigma), 'r', linewidth=2, label=f"N({mu:.4f}, {sigma:.4f}²)")

# plt.title(f"Daily Return Distribution vs Normal Fit")
# plt.xlabel("Daily Return")
# plt.ylabel("Density")
# plt.legend()
# plt.grid(True)
# plt.show()

# # Shapiro-Wilk 正態性檢驗
# stat, p_value = shapiro(returns)
# print(f"Shapiro-Wilk 檢驗統計量: {stat:.4f}, p-value: {p_value:.4f}")

# if p_value > 0.05:
#     print("📊 結果：數據與常態分布沒有顯著差異（可視為近似常態）")
# else:
#     print("📊 結果：數據顯著偏離常態分布（不是常態）")



import pandas as pd
import os

coin_short_names = ["DOGE", "PEPE", "TRUMP"]
data_dir = "../data/coin_price"

all_returns = []

for coin in coin_short_names:
    df = pd.read_csv(os.path.join(data_dir, f"{coin}_price.csv"))
    df['pct_change'] = df['price'].pct_change()
    all_returns.append(df['pct_change'].dropna())

# 合併三個幣種的日變化率
combined_returns = pd.concat(all_returns, ignore_index=True)

# 計算分位數
q10 = combined_returns.quantile(0.10)
q40 = combined_returns.quantile(0.40)
q60 = combined_returns.quantile(0.60)
q90 = combined_returns.quantile(0.90)

print("三幣種共用分位數法切界線：")
print(f"大跌 < {q10:.4f}")
print(f"跌: {q10:.4f} ~ {q40:.4f}")
print(f"持平: {q40:.4f} ~ {q60:.4f}")
print(f"漲: {q60:.4f} ~ {q90:.4f}")
print(f"大漲 > {q90:.4f}")
