import os

# 設定檔案與資料夾路徑
spammer_file = "../data/spammer/2021/spammer_202105.txt"
data_folder = "../data/dice/analysis/202105"

# 讀取 spammer 名單
with open(spammer_file, "r", encoding="utf-8") as f:
    spammers = [line.strip() for line in f if line.strip()]

# 過濾掉已經有 json 和 txt 檔案的使用者
filtered_spammers = []
for name in spammers:
    base_filename = name.replace(" ", "_")  # 若檔名是用底線取代空白
    json_path = os.path.join(data_folder, f"{base_filename}_202105.json")
    txt_path = os.path.join(data_folder, f"{base_filename}_202105.txt")
    
    if not (os.path.exists(json_path) and os.path.exists(txt_path)):
        filtered_spammers.append(name)

# 將結果寫回 spammer_list.txt 或另存新檔
with open("spammer_202105_filtered.txt", "w", encoding="utf-8") as f:
    for name in filtered_spammers:
        f.write(name + "\n")

print("✅ 已完成過濾，結果儲存在 spammer_202105_filtered.txt")
