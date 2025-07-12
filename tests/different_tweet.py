import json
from deepdiff import DeepDiff  # pip install deepdiff

# 讀取兩個 JSON 檔案
with open('tests/TRUMP_20250119_sean.json', 'r', encoding='utf-8-sig') as f1:
    data1 = json.load(f1)

with open('tests/TRUMP_20250119_drifter.json', 'r', encoding='utf-8-sig') as f2:
    data2 = json.load(f2)

# 使用 DeepDiff 來比較
diff = DeepDiff(data1, data2, ignore_order=True)

# 輸出差異
if diff:
    print("兩個 JSON 檔案的差異如下：")
    with open('tests/different.json', 'w', encoding='utf-8-sig') as file:
        json.dump(diff, file, indent=4, ensure_ascii=False)
else:
    print("兩個 JSON 檔案完全相同")
