from datetime import datetime
import gc
import os
import json
import pickle
import numpy as np
from glob import glob
import random
from scipy import sparse
from sklearn.discriminant_analysis import StandardScaler
from tqdm import tqdm
import sys
import math
import pandas as pd

from collections import Counter, defaultdict

from sklearn.metrics import accuracy_score, classification_report

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
import transformers

from sklearn.model_selection import train_test_split

import joblib




'''可修改變數'''
N_SAMPLES = 100  # random sampling 取的數量

N_RUNS = 1

NUM_CATEGORIES = 5  # 類別數量

EPOCHS = 5

T1 = 0.0590 # 0.1

T2 = 0.0102 # 0.00125
 
T3 = 0.0060

T4 = 0.0657

START_DATE = {"DOGE": "2013/12/15", "PEPE": "2024/02/01", "TRUMP": "2025/01/18"}

END_DATE   = {"DOGE": "2025/07/31", "PEPE": "2025/07/31", "TRUMP": "2025/07/31"}

COIN_SHORT_NAME = ["DOGE", "PEPE", "TRUMP"]

JSON_DICT_NAME = ["dogecoin", "PEPE", "(officialtrump OR \"official trump\" OR \"trump meme coin\" OR \"trump coin\" OR trumpcoin OR $TRUMP OR \"dollar trump\")"]

# PRICE_CSV_PATH = "../data/coin_price"

INPUT_PATH = "../data/ml/dataset"

OUTPUT_PATH = "../data/ml/classification/BERT"

SAVE_MODEL_PATH = "../data/ml/models/BERT"

MODEL_NAME = ["logistic_regression", "logreg"]  # 第二個分類器目前輸入的模型名字(未完成)

RUN_FIRST_CLASSIFIER = True  # 是否要跑第一個分類器

RUN_SECOND_CLASSIFIER = False  # 是否要跑第二個分類器(未完成)

IS_GROUPED_CV = False  # 是否要跑第二個分類器的交叉驗證(未完成)

IS_TRAIN = True  # 看是否要訓練

IS_FILTERED = False  # 看是否有分 normal 與 bot

IS_RUN_AUGUST = False  # 看現在是不是要跑 2025/08 的資料(未完成)
'''可修改變數'''

os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(SAVE_MODEL_PATH, exist_ok=True)

SUFFIX_FILTERED = "" if IS_FILTERED else "_non_filtered"
SUFFIX_AUGUST   = "_202508" if IS_RUN_AUGUST else ""

# 轉成 datetime 方便比較
START_DATE_DT = {k: pd.to_datetime(v, format="%Y/%m/%d") for k, v in START_DATE.items()}
END_DATE_DT   = {k: pd.to_datetime(v, format="%Y/%m/%d") for k, v in END_DATE.items()}



# --- 讀取檔案 (只處理 normal, non_filtered) ---
def load_and_preprocess():
    if RUN_FIRST_CLASSIFIER:
        X_train = []
        X_test = []
        y_train = []
        y_test = []
        ids_train = []
        ids_test = []

        for coin_short_name, json_dict_name in zip(COIN_SHORT_NAME, JSON_DICT_NAME):
            print(f"====== 目前在處理 {coin_short_name} ======")

            # 讀取 price_diff_original 作為 y
            y_single_coin = np.load(f"{INPUT_PATH}/y_input/{coin_short_name}/{coin_short_name}_price_diff{SUFFIX_FILTERED}{SUFFIX_AUGUST}.npy")
            print("y_single_coin.shape[0]:", y_single_coin.shape[0])

            with open(f"{INPUT_PATH}/ids_input/{coin_short_name}/{coin_short_name}_ids{SUFFIX_FILTERED}{SUFFIX_AUGUST}.pkl", "rb") as f:   # 讀取一開始訓練用的 ids
                ids_single_coin = pickle.load(f)
                print("len(ids_single_coin):", len(ids_single_coin))
            
            # dates_single_coin = [(c, d) for (c, d, _) in ids_single_coin]  # 只取 date 加入集合
            # dates_single_coin = pd.to_datetime(dates_single_coin, format="%Y-%m-%d")

            # 先把 dates_single_coint 只保留當前幣種的日期
            # dates_single_coin = set([d for (c, d) in dates_single_coin if c == coin_short_name])
            # dates_single_coin = sorted(dates_single_coin)


            print("y_single_coin[:10]\n", y_single_coin[:10])
            print("dates_single_coin[:10]\n", ids_single_coin[:10])
            print()


            # 讀取 原始推文 text
            origianl_single_coin_tweet_text = []  # (N, 2) = (樣本數, (text, date))
            if IS_FILTERED:
                tweets_path = f"../data/filtered_tweets/normal_tweets/{coin_short_name}/*/*/{coin_short_name}_*_normal.json"
            else:
                tweets_path = f"../data/tweets/{coin_short_name}/*/*/{coin_short_name}_*.json"

            original_tweets_file = glob(tweets_path)
            for file in tqdm(original_tweets_file, desc=f"讀取 {coin_short_name} 的原始推文與日期..."):
                with open(file, "r", encoding="utf-8-sig") as fp:
                    data = json.load(fp)
                
                tweets_single_coin = data[json_dict_name]
                if not tweets_single_coin:
                    print("當天沒有推文：", file)
                    continue

                # 取得日期
                date_str = datetime.strptime(
                    tweets_single_coin[0]['created_at'], "%a %b %d %H:%M:%S %z %Y"
                ).strftime("%Y/%m/%d")
                date_dt = pd.to_datetime(date_str)

                # 🔹 過濾掉不在範圍內的推文
                if not (START_DATE_DT[coin_short_name] <= date_dt <= END_DATE_DT[coin_short_name]):
                    print("當天不在指定時間範圍內：", file)
                    continue

                # 儲存 原始推文, 日期(datetime)
                for tweet in tweets_single_coin:
                    origianl_single_coin_tweet_text.append([tweet["text"], date_dt])
            
            print("len(origianl_single_coin_tweet_text):", len(origianl_single_coin_tweet_text))
                
            # --- 讀取 merge_and_splitset 中已經切好資料集的 日期 ---
            # 讀取 Train
            df_split_train = pd.read_csv(f"{INPUT_PATH}/split_dates/{coin_short_name}_train_dates{SUFFIX_FILTERED}.csv")
            df_split_train['date'] = pd.to_datetime(df_split_train['date'], format="%Y-%m-%d")  # 把 date 欄位轉成日期格式

            # 讀取 Test, Val 並把兩個合併
            df_split_test = pd.read_csv(f"{INPUT_PATH}/split_dates/{coin_short_name}_test_dates{SUFFIX_FILTERED}.csv")
            df_split_test['date'] = pd.to_datetime(df_split_test['date'], format="%Y-%m-%d")  # 把 date 欄位轉成日期格式

            # df_split_val = pd.read_csv(f"{INPUT_PATH}/split_dates/{coin_short_name}_val_dates{SUFFIX_FILTERED}.csv")
            # df_split_val['date'] = pd.to_datetime(df_split_val['date'], format="%Y-%m-%d")  # 把 date 欄位轉成日期格式

            # df_split_test = pd.concat([df_split_only_test, df_split_val], ignore_index=True)


            # 把 train/test/val 的日期集合化，加速查詢  切割資料集
            train_dates = set(df_split_train["date"])
            test_dates = set(df_split_test["date"])

            # 切割資料集
            for (text, tweet_date), price_diff, (coin, ids_date, ids_idx) in zip(origianl_single_coin_tweet_text, y_single_coin, ids_single_coin):
                if tweet_date in train_dates:
                    X_train.append(text)
                    y_train.append(price_diff)  # 這裡要對應 y_single_coin
                    ids_train.append([coin, ids_date, ids_idx])

                elif tweet_date in test_dates:
                    X_test.append(text)
                    y_test.append(price_diff)
                    ids_test.append([coin, ids_date, ids_idx])

            # mask_train = [date in df_split_train["date"] for date in origianl_single_coin_tweet_text[0]]
            # mask_test = [date in df_split_test["date"] for date in origianl_single_coin_tweet_text[0]]

            # X_train += origianl_single_coin_tweet_text[mask_train]
            # X_test += origianl_single_coin_tweet_text[mask_test]



            print("len(X_train):", len(X_train))
            print("len(X_test):", len(X_test))
            print("len(y_train):", len(y_train))
            print("len(y_test):", len(y_test))
            print("len(ids_train):", len(ids_train))
            print("len(ids_test):", len(ids_test))

            print(f"\n已成功切割 {coin_short_name} 的資料集\n")


        X_train = np.array(X_train)
        X_test = np.array(X_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)

        print("\n合併完成後的形狀:")
        print("X_train.shape:", X_train.shape)
        print("X_test.shape:", X_test.shape)
        print("y_train.shape:", y_train.shape)
        print("y_test.shape:", y_test.shape)
        print("len(ids_train):", len(ids_train))
        print("len(ids_test):", len(ids_test))

        input("\n按 Enter 以繼續...")
        
    
    elif RUN_SECOND_CLASSIFIER:
        # 取得資料
        X = np.load(f"{INPUT_PATH}/final_input/price_classifier/{MODEL_NAME[0]}/{MODEL_NAME[1]}_X_classifier_2{SUFFIX_FILTERED}{SUFFIX_AUGUST}.npy")
        y = np.load(f"{INPUT_PATH}/final_input/price_classifier/{MODEL_NAME[0]}/{MODEL_NAME[1]}_Y_classifier_2{SUFFIX_FILTERED}{SUFFIX_AUGUST}.npy")
        with open(f"{INPUT_PATH}/final_input/price_classifier/{MODEL_NAME[0]}/{MODEL_NAME[1]}_ids_classifier_2{SUFFIX_FILTERED}{SUFFIX_AUGUST}.pkl", 'rb') as file:
            ids = pickle.load(file)

        X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
            X, y, ids, test_size=0.2, random_state=42, shuffle=True
        )

        print("X_test shape:", X_test.shape)
        print("y_test shape:", y_test.shape)

        print("X_train shape:", X_train.shape)
        print("X_test shape:", X_test.shape)
        print("y_train shape:", y_train.shape)
        print("y_test shape:", y_test.shape)
        print("Train IDs count:", len(ids_train))
        print("Test IDs count:", len(ids_test))

    else:
        raise ValueError("必須指定 run_first_classifier 或 run_second_classifier")
    
    # 建立 target label：五元分類
    y_train_categorized = categorize_array_multi(y_train, T1, T2, T3, T4, ids_train)  # shape (N,)
    y_test_categorized  = categorize_array_multi(y_test, T1, T2, T3, T4, ids_test)   # shape (N,)
    print("已成功分類別")

    # 統計每個類別數量
    print(f"大跌：-{T1 * 100:.2f}%以下, 跌：-{T1 * 100:.2f}% ~ -{T2 * 100}%, 持平：-{T2 * 100}% ~ {T3 * 100}%, 漲：{T3 * 100}% ~ {T4 * 100:.2f}%, 大漲：{T4 * 100:.2f}%以上")
    train_total_row = y_train_categorized.shape[0]
    test_total_row = y_test_categorized.shape[0]
    # for col in range(y_train_categorized.shape[1]):
    counts = np.bincount(y_train_categorized, minlength=5)
    percentages = counts / train_total_row * 100
    percentages_str = " ".join([f"{p:.2f}%" for p in percentages])
    print(f"[TRAIN] column 類別: {percentages_str}")

    counts = np.bincount(y_test_categorized, minlength=5)
    percentages = counts / test_total_row * 100
    percentages_str = " ".join([f"{p:.2f}%" for p in percentages])
    print(f"[TEST]  column 類別: {percentages_str}\n")

    input("pasue...")

    return X_train, X_test, y_train_categorized, y_test_categorized, ids_train, ids_test



# --- 五元分類 ---
def categorize_array_multi(Y, t1, t2, t3, t4, ids=None):
    """
    Y: np.ndarray, shape = (num_labels,), 價格變化率
    """

    print("Y.shape:", Y.shape)
    # print(len(ids))

    # 五元分類
    labels = np.full_like(Y, 2, dtype=int)  # 預設持平
    labels[Y <= -t1] = 0  # 大跌
    labels[(Y > -t1) & (Y <= -t2)] = 1  # 跌
    labels[(Y >= t3) & (Y < t4)] = 3  # 漲
    labels[Y >= t4] = 4  # 大漲

    if ids is not None:
        # 找出 Y==0 的索引
        zero_idx = np.where(Y == 0)[0]
        # 只取對應的 ids
        dates_is_0 = set((ids[i][0], ids[i][1]) for i in zero_idx)
        if len(dates_is_0) > 0:
            print(f"共有 {len(dates_is_0)} 天 Y==0")
            for id in sorted(dates_is_0):
                print(id)

    if np.any(Y == 0):  # 檢查是否有任何元素等於 0
        count = np.sum(Y == 0)
        print(f"共有 {count} 個 Y == 0")
        labels[Y == 0] = 4  # 為了校正 TRUMP 前兩天的價格相同 第一天設為大漲

    return labels



# def get_random_samples_sparse_stratified(X, y, seed: int = 42):
    """
    X: 原始推文 text
    y: shape=(N,)  多類別標籤
    """
    X = np.array(X)  # 強制轉換成 np.array
    y = np.array(y)

    global N_SAMPLES
    # global ENABLE_SAMPLING
    # n_total = X.shape[0]

    print(X)
    input()

    n_total = len(X['input_ids'])


    if N_SAMPLES == 0:
        print(f"[INFO] 不做 random sampling，使用所有樣本數: {n_total} 筆")
        # ENABLE_SAMPLING = False
        return [(X, y)]  # 回傳一個原始數量的 (X, y) tuple

    classes = np.unique(y)
    n_classes = len(classes)
    if N_SAMPLES < n_classes:
        raise ValueError(f"樣本數 {N_SAMPLES} 太少，無法平均分配到每個類別 ({n_classes})")
    
    samples_per_class = N_SAMPLES // n_classes

    # 建立索引字典
    class_indices = defaultdict(list)
    for idx, label in enumerate(y):
        class_indices[label].append(idx)

    samples = []
    for run in range(N_RUNS):
        np.random.seed(seed + run)
        selected_indices = []

        for c in classes:
            idx_list = class_indices[c]
            if len(idx_list) <= samples_per_class:
                # 如果該類別數量不夠，就全部拿
                selected_indices.extend(idx_list)
            else:
                selected_indices.extend(np.random.choice(idx_list, samples_per_class, replace=False))

        # 如果總數少於 N_SAMPLES，從剩餘樣本補足
        if len(selected_indices) < N_SAMPLES:
            # set(range(n_total)) 是所有樣本索引（0 ~ n_total-1）   set(selected_indices) 是已被選過的索引集合
            remaining_idx = list(set(range(n_total)) - set(selected_indices))
            remaining_needed = N_SAMPLES - len(selected_indices)
            selected_indices.extend(np.random.choice(remaining_idx, remaining_needed, replace=False))

        np.random.shuffle(selected_indices)  # 打亂順序
        X_sample = X[selected_indices]
        y_sample = y[selected_indices]
        samples.append((X_sample, y_sample))

        # === 新增：統計類別數量與比例 ===
        unique, counts = np.unique(y_sample, return_counts=True)
        total = len(y_sample)
        print(f"\n[INFO] Run {run}: Stratified sample X_train={X_sample.shape}, y_train={y_sample.shape}")
        for cls, cnt in zip(unique, counts):
            pct = cnt / total * 100
            print(f"   Class {cls}: {cnt} samples ({pct:.2f}%)")

    return samples



def get_random_samples_sparse_stratified(X, y, seed: int = 42):
    """
    X: dict, {'input_ids': np.array, 'attention_mask': np.array}
    y: shape=(N,)  多類別標籤
    """
    # 不要轉成 np.array，保持 dict
    y = np.array(y)

    global N_SAMPLES, N_RUNS

    n_total = X['input_ids'].shape[0]

    if N_SAMPLES == 0:
        print(f"[INFO] 不做 random sampling，使用所有樣本數: {n_total} 筆")
        return [(X, y)]  # 回傳原始數量的 (X, y) tuple

    classes = np.unique(y)
    n_classes = len(classes)
    if N_SAMPLES < n_classes:
        raise ValueError(f"樣本數 {N_SAMPLES} 太少，無法平均分配到每個類別 ({n_classes})")
    
    samples_per_class = N_SAMPLES // n_classes

    # 建立索引字典
    class_indices = defaultdict(list)
    for idx, label in enumerate(y):
        class_indices[label].append(idx)

    samples = []
    for run in range(N_RUNS):
        np.random.seed(seed + run)
        selected_indices = []

        for c in classes:
            idx_list = class_indices[c]
            if len(idx_list) <= samples_per_class:
                selected_indices.extend(idx_list)
            else:
                selected_indices.extend(np.random.choice(idx_list, samples_per_class, replace=False))

        # 如果總數少於 N_SAMPLES，從剩餘樣本補足
        if len(selected_indices) < N_SAMPLES:
            remaining_idx = list(set(range(n_total)) - set(selected_indices))
            remaining_needed = N_SAMPLES - len(selected_indices)
            selected_indices.extend(np.random.choice(remaining_idx, remaining_needed, replace=False))

        np.random.shuffle(selected_indices)  # 打亂順序

        # ⚡ 對每個 key 分別索引
        X_sample = {k: v[selected_indices] for k, v in X.items()}
        y_sample = y[selected_indices]
        samples.append((X_sample, y_sample))

        # === 統計類別數量與比例 ===
        unique, counts = np.unique(y_sample, return_counts=True)
        total = len(y_sample)
        print(f"\n[INFO] Run {run}: Stratified sample X_train keys={list(X_sample.keys())}, y_train={y_sample.shape}")
        for cls, cnt in zip(unique, counts):
            pct = cnt / total * 100
            print(f"   Class {cls}: {cnt} samples ({pct:.2f}%)")

    return samples




# 自訂 Dataset 來適配 Hugging Face
# class TweetDataset(Dataset):
#     def __init__(self, texts, labels, tokenizer, max_length=128):
#         self.texts = texts
#         self.labels = labels
#         self.tokenizer = tokenizer
#         self.max_length = max_length

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         text = str(self.texts[idx])
#         label = int(self.labels[idx])

#         encoding = self.tokenizer(
#             text,
#             truncation=True,
#             padding="max_length",   # 可以改成 "longest" 或 "max_length"
#             max_length=self.max_length,
#             return_tensors="pt"
#         )

#         # squeeze 0 維，變成單筆 tensor
#         item = {key: val.squeeze(0) for key, val in encoding.items()}
#         item["labels"] = torch.tensor(label, dtype=torch.long)
#         return item

class TweetDataset(Dataset):
    def __init__(self, encodings, labels):
        """
        encodings: dict, 包含 'input_ids', 'attention_mask', (optional: 'token_type_ids')
        labels: shape=(N,)
        """
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if idx >= len(self.encodings['input_ids']):
            print(f"[ERROR] idx={idx} 超出範圍，dataset 長度={len(self.encodings['input_ids'])}")
            raise IndexError
        # 每個 sample 已經是 dict
        item = {key: torch.tensor(self.encodings[key][idx]) for key in self.encodings}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    


# ==========================================================
# 分批 Tokenize + 存檔
# ==========================================================
def tokenize_and_save_in_batches(X, y, tokenizer, save_path, prefix, batch_size=5000, max_len=128):
    os.makedirs(save_path, exist_ok=True)

    total_batches = math.ceil(len(X) / batch_size)
    print(f"📦 {prefix}: 共需 {total_batches} 個 batch，每個大小 {batch_size}（最後一批可能較少）")

    file_paths = []
    for batch_idx in tqdm(range(total_batches), desc=f"Tokenizing {prefix}"):
        start = batch_idx * batch_size
        end = start + batch_size

        batch_texts = X[start:end].astype(str).tolist() if isinstance(X, np.ndarray) else [str(x) for x in X[start:end]]
        batch_labels = y[start:end]

        # print(f"🔍 batch {batch_idx} 型態：", type(batch_texts))
        # print(f"🔍 第一個元素型態：", type(batch_texts[0]))
        # print(f"🔍 第一個元素內容：", batch_texts[0])

        encodings = tokenizer(
            batch_texts,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="np"
        )

        file_path = os.path.join(save_path, f"{prefix}_batch{batch_idx}{SUFFIX_FILTERED}.pkl")
        with open(file_path, "wb") as f:
            pickle.dump((encodings, batch_labels), f)
        file_paths.append(file_path)

    print(f"✅ {prefix} 全部 {total_batches} 個 batch 已存檔完成")

    return file_paths


# ==========================================================
# 載入分批資料 → 合併成單一 Dataset
# ==========================================================
# def load_tokenized_batches(save_path, prefix):
#     all_encodings = []
#     all_labels = []

#     files = sorted([f for f in os.listdir(save_path) if f.startswith(prefix)])
#     for f_name in tqdm(files, desc=f"正在讀取分批的 {prefix} tokenize..."):
#         with open(os.path.join(save_path, f_name), "rb") as f:
#             encodings, labels = pickle.load(f)
#             all_encodings.append(encodings)
#             all_labels.extend(labels)

#     # 合併成單一 dict (numpy)
#     merged_encodings = {
#         "input_ids": np.concatenate([e["input_ids"] for e in all_encodings]),
#         "attention_mask": np.concatenate([e["attention_mask"] for e in all_encodings]),
#     }

#     print(f"✅ {prefix} 成功合併成單一 tokenize")

#     return merged_encodings, all_labels

def load_tokenized_batches(save_path, prefix):
    merged_path = f"../data/ml/classification/BERT/tokenize/{prefix}_token_merged{SUFFIX_FILTERED}.pkl"

    # 🔹 若合併檔已存在，直接載入
    if os.path.exists(merged_path):
        print(f"📂 偵測到已存在的合併檔：{merged_path}")
        input("❓ 是否要使用這份 合併的 Token? (按 Enter 以繼續 或 Ctrl + C ...)")
        with open(merged_path, "rb") as f:
            merged_encodings, labels_all = pickle.load(f)
        print(f"✅ 已直接載入 {prefix}_merged.pkl")
        return merged_encodings, labels_all

    # 🔹 否則就進行合併
    input_ids_list = []
    attention_mask_list = []
    labels_all = []

    files = sorted([f for f in os.listdir(save_path) if f.startswith(prefix)])
    for f_name in tqdm(files, desc=f"正在讀取分批的 {prefix} tokenize..."):
        file_path = os.path.join(save_path, f_name)

        # 讀取單一檔案
        with open(file_path, "rb") as f:
            encodings, labels = pickle.load(f)

        # 合併
        input_ids_list.append(encodings["input_ids"])
        attention_mask_list.append(encodings["attention_mask"])
        labels_all.extend(labels)

        # 清理暫存
        del encodings, labels
        gc.collect()

    labels_all = np.array(labels_all)  #------------------------------------------------------


    # 🔹 合併為單一陣列
    merged_encodings = {
        "input_ids": np.concatenate(input_ids_list, axis=0),
        "attention_mask": np.concatenate(attention_mask_list, axis=0),
    }

    # 清理暫存
    del input_ids_list, attention_mask_list
    gc.collect()

    print(f"✅ {prefix} 成功合併成單一 tokenize")

    # 🔹 將結果快取下來，下次就能直接載入
    with open(merged_path, "wb") as f:
        pickle.dump((merged_encodings, labels_all), f)
    print(f"💾 已將合併結果存成 {merged_path}")

    return merged_encodings, labels_all


# ==========================================================
# 主要 Tokenize & Save function（改用分批）
# ==========================================================
def tokenize_and_save(X_train, X_test, y_train, y_test, save_path, model_name="bert-base-uncased", batch_size=5000):
    os.makedirs(save_path, exist_ok=True)
    tokenizer = BertTokenizerFast.from_pretrained(model_name)

    print("🛠️ Tokenizing Train Data...")
    tokenize_and_save_in_batches(X_train, y_train, tokenizer, save_path, prefix="train", batch_size=batch_size)

    print("🛠️ Tokenizing Test Data...")
    tokenize_and_save_in_batches(X_test, y_test, tokenizer, save_path, prefix="test", batch_size=batch_size)

    print(f"✅ All tokenized data saved to {save_path}")


def load_tokenized_data(save_path):
    X_train_enc, y_train = load_tokenized_batches(save_path, prefix="train")
    X_test_enc, y_test = load_tokenized_batches(save_path, prefix="test")
    return X_train_enc, X_test_enc, y_train, y_test




def train_function(X_train, X_test, y_train, y_test, pipeline_path, model_name="bert-base-uncased"):

    print("transformers.__version__:", transformers.__version__)

    all_results = []
    best_test_acc = -1
    best_run_info = None

    tokenizer = BertTokenizerFast.from_pretrained(model_name)

    if RUN_FIRST_CLASSIFIER:
        # 確保是 list，方便 Trainer   #------------------------------------------------------
        # X_train = list(X_train)
        # X_test  = list(X_test)
        # y_train = list(y_train)
        # y_test  = list(y_test)

        if IS_FILTERED:
            tokenize_path = "filtered"
        else:
            tokenize_path = "non_filtered"

        # 檢查是否已經有 tokenized data
        if os.path.exists(f"{OUTPUT_PATH}/tokenize/{tokenize_path}/train_batch0{SUFFIX_FILTERED}.pkl"):
            print("📂 載入已存的 Tokenized Data")
            input("❓ 是否要使用這份 Tokenized Data? (按 Enter 以繼續 或 Ctrl + C ...)")
            X_train_enc, X_test_enc, y_train, y_test = load_tokenized_data(f"{OUTPUT_PATH}/tokenize/{tokenize_path}")
        else:
            print("🛠️ 第一次執行，開始 Tokenize 並存檔...")
            tokenize_and_save(X_train, X_test, y_train, y_test, save_path=f"{OUTPUT_PATH}/tokenize/{tokenize_path}", model_name="bert-base-uncased")
            X_train_enc, X_test_enc, y_train, y_test = load_tokenized_data(f"{OUTPUT_PATH}/tokenize/{tokenize_path}")

        
        # --- 取得分層隨機取樣 ---
        train_sample = get_random_samples_sparse_stratified(X_train_enc, y_train)  # [(X_sample, y_sample), ...]
        run_count = len(train_sample)

        # X_test_enc = list(X_test_enc)  #-----------------------------
        # y_test = list(y_test)

        # test 包裝成 Dataset
        test_dataset = TweetDataset(X_test_enc, y_test)

    elif RUN_SECOND_CLASSIFIER:
        train_sample = [(X_train, y_train)]
        run_count = 1

        # X_test = list(X_test)  #---------------------------------
        # y_test = list(y_test)

        # test 包裝成 Dataset
        test_dataset = TweetDataset(X_test, y_test)

    else:
        raise ValueError("請設定 RUN_FIRST_CLASSIFIER 或 RUN_SECOND_CLASSIFIER")

    # --- 執行 N_RUNS 次 ---
    for run in range(run_count):
        print(f"\n===== RUN {run} =====")

        X_train_sample, y_train_sample = train_sample[run]
        # X_train_sample = list(X_train_sample)    # -----------------------------------------------
        # y_train_sample = list(y_train_sample)
        train_dataset = TweetDataset(X_train_sample, y_train_sample)

        # 初始化模型
        num_labels = len(set(y_train_sample))
        model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

        # 訓練參數（這裡你可以隨機抽 hyperparams，模擬 RandomizedSearchCV）
        training_args = TrainingArguments(
            output_dir=f"./results_run_{run}",
            # evaluation_strategy="epoch",
            save_strategy="no",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            num_train_epochs=3,
            weight_decay=0.01,
            logging_dir=f"./logs_run_{run}",
            load_best_model_at_end=False,
            report_to="none"
        )

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            acc = accuracy_score(labels, preds)
            return {"accuracy": acc}

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics,
        )

        trainer.train()

        # 評估
        train_metrics = trainer.evaluate(train_dataset)
        test_metrics = trainer.evaluate(test_dataset)

        train_acc = train_metrics["eval_accuracy"]
        test_acc = test_metrics["eval_accuracy"]

        print(f"[RUN {run}] Train acc={train_acc:.4f}, Test acc={test_acc:.4f}")

        all_results.append({
            "run": run,
            "train_acc": train_acc,
            "test_acc": test_acc,
        })

        if (RUN_FIRST_CLASSIFIER and test_acc > best_test_acc) or RUN_SECOND_CLASSIFIER:
            best_test_acc = test_acc
            best_run_info = {
                "run": run,
                "model": model,
                "tokenizer": tokenizer,
                "train_acc": train_acc,
                "test_acc": test_acc,
            }

    # --- 儲存所有結果 ---
    results_df = pd.DataFrame(all_results)
    results_df.to_csv("bert_results.csv", index=False)

    # --- 儲存最佳模型 ---
    best_model = best_run_info["model"]
    best_model.save_pretrained(pipeline_path)
    tokenizer.save_pretrained(pipeline_path)

    print("\n=== 最佳模型 ===")
    print(f"Run {best_run_info['run']} | Train acc={best_run_info['train_acc']:.4f}, Test acc={best_run_info['test_acc']:.4f}")
    
    best_model = best_run_info["model"]
    best_tokenizer = best_run_info["tokenizer"]

    trainer = Trainer(
        model=best_model,
        args=training_args,  # 可以重用最後一個 run 的 training_args
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    preds = trainer.predict(test_dataset).predictions
    preds = np.argmax(preds, axis=-1)
    print(classification_report(y_test, preds))



def evaluate_by_coin_date(ids, y_true, y_pred):
    LABEL_SYMBOLS = {
        0: "🔴",  # 大跌
        1: "🟠",    # 跌
        2: "⚪",    # 持平
        3: "🟡",    # 漲
        4: "🟢"   # 大漲
    }

    if RUN_FIRST_CLASSIFIER:
        results = defaultdict(list)

        # 聚合
        for (coin, date, _), t, p in zip(ids, y_true, y_pred):
            results[(coin, date)].append((t, p))

        daily_summary = {}
        for (coin, date), samples in results.items():
            truths, preds = zip(*samples)
            truths = np.array(truths)
            preds  = np.array(preds)

            # 多數決
            values, counts = np.unique(preds, return_counts=True)
            majority_pred = values[np.argmax(counts)]

            true_label = truths[0]  # 假設同一天真實標籤一致
            correct = (majority_pred == true_label)


            daily_summary.setdefault(coin, {})

            # 將各類別出現次數轉成 list（保持原本 up_counts/down_counts 的感覺）
            class_counts = [np.sum(preds == i) for i in range(5)]  # 0~4 五類
            pred_symbols = [LABEL_SYMBOLS[majority_pred]]           # 單一預測符號

            true_symbols   = [LABEL_SYMBOLS[int(true_label)]]   # 真實符號
            result_symbols = ["✅" if correct else "❌"]         # 對錯符號


            daily_summary[coin][date] = {
                "true_label": int(true_label),
                "majority_pred": int(majority_pred),
                "majority_correct": bool(correct),
                "class_counts": class_counts,    # 替代 up_counts/down_counts
                "total_counts": len(preds),      # 原本 total_counts
                "pred_symbols": pred_symbols,
                "true_symbols": true_symbols,     # 真實符號
                "result_symbols": result_symbols  # 對錯符號
            }

        return daily_summary, len(np.unique(y_true))
    
    # --- 未完成 ---
    elif RUN_SECOND_CLASSIFIER:
        daily_summary = {}

        for (coin, date), t, p in zip(ids, y_true, y_pred):
            correct = (p == t)

            # 各類別計數 (這裡因為只有一筆，只有一個類別會是 1，其餘都是 0)
            class_counts = [1 if p == i else 0 for i in range(5)]

            daily_summary.setdefault(coin, {})
            daily_summary[coin][date] = {
                "true_label": int(t),
                "majority_pred": int(p),
                "majority_correct": bool(correct),
                "class_counts": class_counts,
                "total_counts": 1,
                "pred_symbols": [LABEL_SYMBOLS[int(p)]],
                "true_symbols": [LABEL_SYMBOLS[int(t)]],
                "result_symbols": ["✅" if correct else "❌"]
            }

        return daily_summary, len(np.unique(y_true))



def predict_function(X_train, X_test, y_train, y_test, ids_train, ids_test, model_path, model_name="bert-base-uncased"):
    
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_path)

    # 建立 Dataset
    train_dataset = TweetDataset(X_train, y_train)
    test_dataset  = TweetDataset(X_test, y_test)

    trainer = Trainer(model=model)  # 只用來做 predict，不需要 training args

    # 預測
    train_preds = trainer.predict(train_dataset).predictions
    test_preds  = trainer.predict(test_dataset).predictions

    # 取 argmax
    train_preds = np.argmax(train_preds, axis=-1)
    test_preds  = np.argmax(test_preds, axis=-1)

    # 評估分類報告
    print("\nTrain Classification Report:")
    print(classification_report(y_train, train_preds, zero_division=0))
    print("\nTest Classification Report:")
    print(classification_report(y_test, test_preds, zero_division=0))

    # 套用你原本的 daily aggregation
    train_daily, _ = evaluate_by_coin_date(ids_train, y_train, train_preds)
    test_daily, _  = evaluate_by_coin_date(ids_test, y_test, test_preds)

    if RUN_FIRST_CLASSIFIER:

        # === 存成 JSON ===
        with open(f"{OUTPUT_PATH}/BERT_train_daily_results_{N_SAMPLES}{SUFFIX_FILTERED}{SUFFIX_AUGUST}.json", "w", encoding="utf-8") as f:
            json.dump(train_daily, f, ensure_ascii=False, indent=4, default=int)

        with open(f"{OUTPUT_PATH}/BERT_test_daily_results_{N_SAMPLES}{SUFFIX_FILTERED}{SUFFIX_AUGUST}.json", "w", encoding="utf-8") as f:
            json.dump(test_daily, f, ensure_ascii=False, indent=4, default=int)

        print("已輸出逐日預測結果：")
        print(f"- train: {OUTPUT_PATH}/BERT_train_daily_results_{N_SAMPLES}{SUFFIX_FILTERED}{SUFFIX_AUGUST}.json")
        print(f"- test:  {OUTPUT_PATH}/BERT_test_daily_results_{N_SAMPLES}{SUFFIX_FILTERED}{SUFFIX_AUGUST}.json")

        # === 合併 train + test ===
        combined_daily = {}
        for coin, daily in train_daily.items():
            combined_daily.setdefault(coin, {}).update(daily)
        for coin, daily in test_daily.items():
            combined_daily.setdefault(coin, {}).update(daily)

        # === 存成合併後的 TXT ===
        txt_path = f"{OUTPUT_PATH}/BERT_combined_results_{N_SAMPLES}{SUFFIX_FILTERED}{SUFFIX_AUGUST}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            # === 初始化統計器 ===
            label_correct = np.zeros(1, dtype=int)
            label_total   = np.zeros(1, dtype=int)

            for coin, daily in combined_daily.items():
                f.write(f"\n=== {coin} ===\n")

                # 用來存放每天的 (date, pred_class)
                records = []

                for date, stats in sorted(daily.items()):
                    # --- 每日輸出到 TXT ---
                    class_str = " ".join(f"{x:5d}" for x in stats['class_counts'])
                    line = (
                        f"{date} → 📊 {class_str}  "
                        f"總數: {stats['total_counts']:5d}  "
                        f"預測: {''.join(stats['pred_symbols'])}  "
                        f"真實: {''.join(stats['true_symbols'])}  "
                        f"結果: {''.join(stats['result_symbols'])}\n"
                    )
                    f.write(line)

                    # --- 更新累積準確率 ---
                    label_total[0] += 1
                    if stats["majority_correct"]:
                        label_correct[0] += 1

                    # --- 取當天預測類別 (class_counts 最大的 index) ---
                    pred_class = int(np.argmax(stats["class_counts"]))
                    records.append((date, pred_class))

                # --- 輸出整體準確率 (百分比) ---
                accuracy_summary = " ".join(
                    f"{(c / t * 100):.2f}%" if t > 0 else "N/A"
                    for c, t in zip(label_correct, label_total)
                )
                f.write(f"\n整體準確率: {accuracy_summary}\n")

                # === 存成 .npy (每日預測結果，依日期排序) ===
                if records:
                    records.sort(key=lambda x: x[0])
                    _, preds = zip(*records)
                    preds = np.array(preds, dtype=np.int32)

                    npy_path = f"{OUTPUT_PATH}/{coin}_BERT_classifier_1_result{SUFFIX_FILTERED}{SUFFIX_AUGUST}.npy"
                    np.save(npy_path, preds)
                    print(preds[:50])
                    print(f"{coin} → {npy_path} 已完成, shape={preds.shape}")


        print(f"\n合併後的人類可讀版結果已輸出到：{txt_path}")

    elif RUN_SECOND_CLASSIFIER:
        # === 存成 JSON ===
        with open(f"{OUTPUT_PATH}/BERT_train_daily_classifier_2_results{SUFFIX_FILTERED}{SUFFIX_AUGUST}.json", "w", encoding="utf-8") as f:
            json.dump(train_daily, f, ensure_ascii=False, indent=4, default=int)

        with open(f"{OUTPUT_PATH}/BERT_test_daily_classifier_2_results{SUFFIX_FILTERED}{SUFFIX_AUGUST}.json", "w", encoding="utf-8") as f:
            json.dump(test_daily, f, ensure_ascii=False, indent=4, default=int)

        print("已輸出逐日預測結果：")
        print(f"- train: {OUTPUT_PATH}/BERT_train_daily_classifier_2_results{SUFFIX_FILTERED}{SUFFIX_AUGUST}.json")
        print(f"- test:  {OUTPUT_PATH}/BERT_test_daily_classifier_2_results{SUFFIX_FILTERED}{SUFFIX_AUGUST}.json")

        # === 合併 train + test ===
        combined_daily = {}
        for coin, daily in train_daily.items():
            combined_daily.setdefault(coin, {}).update(daily)
        for coin, daily in test_daily.items():
            combined_daily.setdefault(coin, {}).update(daily)

        # === 存成合併後的 TXT ===
        txt_path = f"{OUTPUT_PATH}/BERT_combined_classifier_2_results{SUFFIX_FILTERED}{SUFFIX_AUGUST}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            label_correct = 0
            label_total = 0

            for coin, daily in combined_daily.items():
                f.write(f"\n=== {coin} ===\n")

                records = []
                for date, stats in sorted(daily.items()):
                    # --- 每日輸出到 TXT ---
                    line = (
                        f"{date} → "
                        f"預測: {''.join(stats['pred_symbols'])}  "
                        f"真實: {''.join(stats['true_symbols'])}  "
                        f"結果: {''.join(stats['result_symbols'])}\n"
                    )
                    f.write(line)

                    # --- 更新累積準確率 ---
                    label_total += 1
                    if stats["majority_correct"]:
                        label_correct += 1

                    # --- 保存每日預測類別 ---
                    records.append((date, stats["majority_pred"]))

                # --- 輸出整體準確率 ---
                acc = (label_correct / label_total * 100) if label_total > 0 else 0
                f.write(f"\n整體準確率: {acc:.2f}%\n")

        print(f"\n合併後的人類可讀版結果已輸出到：{txt_path}")



# --- 未完成 ---
def predict_august_function(pipeline_path):
    combined_daily = {}  # 用來放 合併 三種幣種 的資料 ===

    # --- 載入資料 ---
    for coin_short_name in ['DOGE', 'PEPE', 'TRUMP']:
        if RUN_FIRST_CLASSIFIER:
            X_august = sparse.load_npz(f'{INPUT_PATH}/X_input/keyword_classifier/{coin_short_name}/{coin_short_name}_X_sparse{SUFFIX_FILTERED}{SUFFIX_AUGUST}.npz')
            y_august = np.load(f'{INPUT_PATH}/y_input/{coin_short_name}/{coin_short_name}_price_diff{SUFFIX_FILTERED}{SUFFIX_AUGUST}.npy')
            with open(f'{INPUT_PATH}/ids_input/{coin_short_name}/{coin_short_name}_ids{SUFFIX_FILTERED}{SUFFIX_AUGUST}.pkl', 'rb') as file:
                ids_august = pickle.load(file)

        elif RUN_SECOND_CLASSIFIER:
            X_august = np.load(f"{INPUT_PATH}/X_input/keyword_classifier/{coin_short_name}/{coin_short_name}_{MODEL_NAME}_X_classifier_2{SUFFIX_FILTERED}{SUFFIX_AUGUST}.npy")
            y_august = np.load(f"{INPUT_PATH}/y_input/{coin_short_name}/{coin_short_name}_price_diff_original{SUFFIX_FILTERED}{SUFFIX_AUGUST}.npy")
            with open(f"{INPUT_PATH}/ids_input/{coin_short_name}/{coin_short_name}_{MODEL_NAME}_ids_classifier_2{SUFFIX_FILTERED}{SUFFIX_AUGUST}.pkl", 'rb') as file:
                ids_august = pickle.load(file)

        y_august_categorized = categorize_array_multi(y_august, T1, T2, T3, T4)

        # === 載入最佳模型 ===
        pipeline = joblib.load(pipeline_path)
        model = pipeline["model"]
        
        # === 預測所有樣本 ===
        y_pred_august = model.predict(X_august)
        print(y_pred_august.shape)

        # 將 ids 轉成 np.array 方便接下來的處理
        ids_august = np.array(ids_august)

        
        print(f"\n分類報告 ({coin_short_name} August set):")
        print(classification_report(y_august_categorized, y_pred_august, zero_division=0))

        # august_score = knn.score(X_august, Y_august)
        print(f'{coin_short_name} August accuracy')  

        print("ids_august[:5]", ids_august[:5])

        august_daily, _ = evaluate_by_coin_date(ids_august, y_august_categorized, y_pred_august)

        if RUN_FIRST_CLASSIFIER:
            # === 存成 JSON ===
            with open(f"{OUTPUT_PATH}/{coin_short_name}_logreg_august_daily_results_{N_SAMPLES}{SUFFIX_FILTERED}{SUFFIX_AUGUST}.json", "w", encoding="utf-8") as f:
                json.dump(august_daily, f, ensure_ascii=False, indent=4, default=int)

            print("已輸出逐日預測結果：")
            print(f"- august: {OUTPUT_PATH}/{coin_short_name}_logreg_august_daily_results_{N_SAMPLES}{SUFFIX_FILTERED}{SUFFIX_AUGUST}.json")

            # === 合併 三種幣種 ===
            for coin, daily in august_daily.items():
                combined_daily.setdefault(coin, {}).update(daily)

            # === 存成合併後的 TXT ===
            txt_path = f"{OUTPUT_PATH}/logreg_combined_results_{N_SAMPLES}{SUFFIX_FILTERED}{SUFFIX_AUGUST}.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                # === 初始化統計器 ===
                label_correct = np.zeros(1, dtype=int)
                label_total   = np.zeros(1, dtype=int)

                for coin, daily in combined_daily.items():
                    f.write(f"\n=== {coin} ===\n")

                    # 用來存放每天的 (date, pred_class)
                    records = []

                    for date, stats in sorted(daily.items()):
                        # --- 每日輸出到 TXT ---
                        class_str = " ".join(f"{x:5d}" for x in stats['class_counts'])
                        line = (
                            f"{date} → 📊 {class_str}  "
                            f"總數: {stats['total_counts']:5d}  "
                            f"預測: {''.join(stats['pred_symbols'])}  "
                            f"真實: {''.join(stats['true_symbols'])}  "
                            f"結果: {''.join(stats['result_symbols'])}\n"
                        )
                        f.write(line)

                        # --- 更新累積準確率 ---
                        label_total[0] += 1
                        if stats["majority_correct"]:
                            label_correct[0] += 1

                        # --- 取當天預測類別 (class_counts 最大的 index) ---
                        pred_class = int(np.argmax(stats["class_counts"]))
                        records.append((date, pred_class))

                    # --- 輸出整體準確率 (百分比) ---
                    accuracy_summary = " ".join(
                        f"{(c / t * 100):.2f}%" if t > 0 else "N/A"
                        for c, t in zip(label_correct, label_total)
                    )
                    f.write(f"\n整體準確率: {accuracy_summary}\n")

                    # === 存成 .npy (每日預測結果，依日期排序) ===
                    if records:
                        records.sort(key=lambda x: x[0])
                        _, preds = zip(*records)
                        preds = np.array(preds, dtype=np.int32)

                        npy_path = f"{OUTPUT_PATH}/{coin}_logreg_classifier_1_result{SUFFIX_FILTERED}{SUFFIX_AUGUST}.npy"
                        np.save(npy_path, preds)
                        print(preds[:50])
                        print(f"{coin} → {npy_path} 已完成, shape={preds.shape}")


            print(f"\n合併後的人類可讀版結果已輸出到：{txt_path}")

        elif RUN_SECOND_CLASSIFIER:

            # === 存成 JSON ===
            with open(f"{OUTPUT_PATH}/{coin_short_name}_logreg_train_daily_classifier_2_results{SUFFIX_FILTERED}{SUFFIX_AUGUST}.json", "w", encoding="utf-8") as f:
                json.dump(august_daily, f, ensure_ascii=False, indent=4, default=int)

            print("已輸出逐日預測結果：")
            print(f"- august: {OUTPUT_PATH}/{coin_short_name}_logreg_train_daily_classifier_2_results{SUFFIX_FILTERED}{SUFFIX_AUGUST}.json")

            # === 合併 三種幣種 ===
            for coin, daily in august_daily.items():
                combined_daily.setdefault(coin, {}).update(daily)

            # === 存成合併後的 TXT ===
            txt_path = f"{OUTPUT_PATH}/logreg_combined_classifier_2_results{SUFFIX_FILTERED}{SUFFIX_AUGUST}.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                label_correct = 0
                label_total = 0

                for coin, daily in combined_daily.items():
                    f.write(f"\n=== {coin} ===\n")

                    records = []
                    for date, stats in sorted(daily.items()):
                        # --- 每日輸出到 TXT ---
                        line = (
                            f"{date} → "
                            f"預測: {''.join(stats['pred_symbols'])}  "
                            f"真實: {''.join(stats['true_symbols'])}  "
                            f"結果: {''.join(stats['result_symbols'])}\n"
                        )
                        f.write(line)

                        # --- 更新累積準確率 ---
                        label_total += 1
                        if stats["majority_correct"]:
                            label_correct += 1

                        # --- 保存每日預測類別 ---
                        records.append((date, stats["majority_pred"]))

                    # --- 輸出整體準確率 ---
                    acc = (label_correct / label_total * 100) if label_total > 0 else 0
                    f.write(f"\n整體準確率: {acc:.2f}%\n")

            print(f"\n合併後的人類可讀版結果已輸出到：{txt_path}")



def main():

    if RUN_FIRST_CLASSIFIER:

        pipeline_path = f"{SAVE_MODEL_PATH}/BERT_best_pipeline_{N_SAMPLES}{SUFFIX_FILTERED}.joblib"  # 儲存訓練模型的位置

        if not IS_RUN_AUGUST:
            # --- 載入資料 ---
            X_train, X_test, y_train, y_test, ids_train, ids_test = load_and_preprocess()

            # for count in range(LABELS):

            if IS_TRAIN:
                # --- 訓練模型 --- 
                train_function(X_train, X_test, y_train, y_test, pipeline_path)

                # --- 預測模型 ---
                predict_function(X_train, X_test, y_train, y_test, ids_train, ids_test, pipeline_path)
            else:
                if not os.path.exists(pipeline_path):
                    print("找不到已訓練好的 第一個分類器 模型，請先將 IS_TRAIN 設為 True")

                # --- 預測模型 ---
                predict_function(X_train, X_test, y_train, y_test, ids_train, ids_test, pipeline_path)

        else:
            # --- 預測 2025-08 ---
            predict_august_function(pipeline_path)

    elif RUN_SECOND_CLASSIFIER:

        pipeline_path = f"{SAVE_MODEL_PATH}/BERT_classifier_2{SUFFIX_FILTERED}.joblib"  # 儲存訓練模型的位置

        if not IS_RUN_AUGUST:
            if IS_GROUPED_CV == False:
                # --- 載入資料 ---
                X_train, X_test, y_train, y_test, ids_train, ids_test= load_and_preprocess()

                if IS_TRAIN:
                    # --- 訓練模型 --- 
                    train_function(X_train, X_test, y_train, y_test, pipeline_path)

                    # --- 預測模型 ---
                    predict_function(X_train, X_test, y_train, y_test, ids_train, ids_test, pipeline_path)
                else:
                    if not os.path.exists(pipeline_path):
                        print("找不到已訓練好的 第二個分類器 模型，請先將 IS_TRAIN 設為 True")

                    # --- 預測模型 ---
                    predict_function(X_train, X_test, y_train, y_test, ids_train, ids_test, pipeline_path)

            else:
                # 取得資料
                X = np.load(f"{INPUT_PATH}/final_input/price_classifier/{MODEL_NAME[0]}/{MODEL_NAME[1]}_X_classifier_2{SUFFIX_FILTERED}{SUFFIX_AUGUST}.npy")
                y = np.load(f"{INPUT_PATH}/final_input/price_classifier/{MODEL_NAME[0]}/{MODEL_NAME[1]}_Y_classifier_2{SUFFIX_FILTERED}{SUFFIX_AUGUST}.npy")
                with open(f"{INPUT_PATH}/final_input/price_classifier/{MODEL_NAME[0]}/{MODEL_NAME[1]}_ids_classifier_2{SUFFIX_FILTERED}{SUFFIX_AUGUST}.pkl", 'rb') as file:
                    ids = pickle.load(file)

                y_categorized = categorize_array_multi(y, ids, T1, T2, T3, T4)  # shape (N,)

                # results_all = coin_month_cv(X, y_categorized, ids, C=C)

        else:
            # --- 預測 2025-08 ---
            predict_august_function(pipeline_path)  




    # texts = load_tweets()
    # Y = load_price_diff(price_dir, coin_short_name)  # (N_coin, )

    # # print(len(texts))
    # # print(Y.shape[0])

    # assert len(texts) == Y.shape[0], f"{coin_short_name} texts and Y length mismatch!"

    # all_texts.extend(texts)
    # all_Y.append(Y)

    # all_Y = np.concatenate(all_Y)  # shape = (N_total, )

    # if IS_TRAIN:
    #     print(f"=== Processing Y (all coins combined) ===")
    #     labels = categorize_array_multi(all_Y)
    #     model_dir = f"{SAVE_PATH}/allcoins_y"

    #     # 訓練 + 預測全部推文
    #     trainer = train_single_model(
    #         all_texts,
    #         labels,
    #         num_categories=NUM_CATEGORIES,
    #         model_dir=model_dir,
    #         epochs=EPOCHS,
    #         n_samples=N_SAMPLES,
    #         balanced=True
    #     )
        
    # print("\n開始預測全部推文...")
    # # 預測全部推文 + 輸出 CSV/JSON
    # fast_predict_all_models(all_texts, all_Y, tokenized_path=f"{SAVE_PATH}/tokenized_tweets.pt")





if __name__ == "__main__":
    main()