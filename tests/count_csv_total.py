import pandas as pd

# 讀取 CSV
df = pd.read_csv("../data/tweets/count/estimate/PEPE_estimate.csv")

# 計算 predicted_count 總和
total_predicted = df["predicted_count"].sum()

print(f"✅ predicted_count 總和 = {total_predicted}")
