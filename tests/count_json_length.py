import json
import pickle

with open("../data/keyword/machine_learning/all_keywords.json", 'r', encoding="utf-8") as file:
    data = json.load(file)

print(len(data))

with open(f"../abandoned/data/ml_old/dataset/ids_train.pkl", "rb") as f:   # rb = read binary
    ids = pickle.load(f)  # array[('coin', 'date', 'no.'), (str, '%Y-%m-%d', int)

print(ids)