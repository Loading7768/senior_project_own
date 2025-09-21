import json

path = "../data/keyword/machine_learning"

# 讀取兩個 JSON 檔
with open(f"{path}/feature_name.json", "r", encoding="utf-8") as f1:
    data1 = json.load(f1)

with open(f"{path}/DOGE_keywords.json", "r", encoding="utf-8") as f2:
    data2 = json.load(f2)

# 轉成集合，方便比對
set1 = set(data1)
set2 = set(data2)

output_path = "../tests"
# 各種比對結果
results = {
    "common.json": list(set1 & set2),
    "only_in_feature_name.json": list(set1 - set2),
    "only_in_DOGE_keywords.json": list(set2 - set1),
    "different.json": list(set1 ^ set2),
}

# 存成 json 檔
for filename, result in results.items():
    with open(f"{output_path}/{filename}", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

print("✅ 已輸出 common.json, only_in_file1.json, only_in_file2.json, different.json")
