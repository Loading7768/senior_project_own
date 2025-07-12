import json

with open("raw.json", "r", encoding="utf-8") as f:
    data = json.load(f)

with open("formatted.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)