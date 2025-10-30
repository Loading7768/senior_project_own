from datetime import datetime
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

from collections import Counter

from sklearn.metrics import accuracy_score

import torch
from torch.utils.data import Dataset

from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
import transformers

from sklearn.model_selection import train_test_split





'''可修改變數'''
N_SAMPLES = 250_000  # random sampling 取的數量

NUM_CATEGORIES = 5  # 類別數量

EPOCHS = 5

START_DATE = {"DOGE": "2013/12/15", "PEPE": "2024/02/01", "TRUMP": "2025/01/18"}

END_DATE   = {"DOGE": "2025/07/31", "PEPE": "2025/07/31", "TRUMP": "2025/07/31"}

SAVE_PATH = "../data/ml/classification/BERT"


T1 = 0.0590 # 0.1

T2 = 0.0102 # 0.00125
 
T3 = 0.0060

T4 = 0.0657

PRICE_CSV_PATH = "../data/coin_price"

INPUT_PATH = "../data/ml/dataset"

OUTPUT_PATH = "../data/ml/classification/logistic_regression"

SAVE_MODEL_PATH = "../data/ml/models/classification"

MODEL_NAME = "logreg"  # 第二個分類器目前輸入的模型名字

RUN_FIRST_CLASSIFIER = True  # 是否要跑第一個分類器

RUN_SECOND_CLASSIFIER = False  # 是否要跑第二個分類器

IS_GROUPED_CV = False  # 是否要跑第二個分類器的交叉驗證

IS_TRAIN = True  # 看是否要訓練

IS_FILTERED = True  # 看是否有分 normal 與 bot

IS_RUN_AUGUST = False  # 看現在是不是要跑 2025/08 的資料
'''可修改變數'''

os.makedirs(SAVE_PATH, exist_ok=True)

SUFFIX_FILTERED = "" if IS_FILTERED else "_non_filtered"
SUFFIX_AUGUST   = "_202508" if IS_RUN_AUGUST else ""

# 轉成 datetime 方便比較
START_DATE_DT = {k: pd.to_datetime(v, format="%Y/%m/%d") for k, v in START_DATE.items()}
END_DATE_DT   = {k: pd.to_datetime(v, format="%Y/%m/%d") for k, v in END_DATE.items()}



def load_tweets(data_dir, coin_short_name, json_dict_name):
    if RUN_FIRST_CLASSIFIER:
        # 取得 ML 的 X
        X_train = sparse.load_npz(f"{INPUT_PATH}/X_train{SUFFIX_FILTERED}.npz")
        X_test = sparse.load_npz(f"{INPUT_PATH}/X_test{SUFFIX_FILTERED}.npz")

        print(X_train.shape)

        # 匯入 Y
        y_train = np.load(f"{INPUT_PATH}/Y_train{SUFFIX_FILTERED}.npz")
        y_train = y_train['Y']
        y_test = np.load(f"{INPUT_PATH}/Y_test{SUFFIX_FILTERED}.npz")
        y_test = y_test['Y']

        print(y_train.shape)

        with open(f"{INPUT_PATH}/ids_train{SUFFIX_FILTERED}.pkl", 'rb') as file:
            ids_train = pickle.load(file)
        with open(f"{INPUT_PATH}/ids_test{SUFFIX_FILTERED}.pkl", 'rb') as file:
            ids_test = pickle.load(file)

        scaler = StandardScaler(with_mean=False)  # 適合 sparse matrix
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # 取得 all_keywords(features_name)
        with open(f"{INPUT_PATH}/keyword/filtered_keywords{SUFFIX_FILTERED}.json", "r", encoding="utf-8-sig") as jsonfile:
            features_name = json.load(jsonfile)
    
    elif RUN_SECOND_CLASSIFIER:
        # 取得資料
        X = np.load(f"{INPUT_PATH}/{MODEL_NAME}_X_classifier_2{SUFFIX_FILTERED}{SUFFIX_AUGUST}.npy")
        y = np.load(f"{INPUT_PATH}/{MODEL_NAME}_Y_classifier_2{SUFFIX_FILTERED}{SUFFIX_AUGUST}.npy")
        with open(f"{INPUT_PATH}/{MODEL_NAME}_ids_classifier_2{SUFFIX_FILTERED}{SUFFIX_AUGUST}.pkl", 'rb') as file:
            ids = pickle.load(file)

        X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
            X, y, ids, test_size=0.2, random_state=42, shuffle=True
        )

        print("X_test shape:", X_test.shape)
        print("y_test shape:", y_test.shape)


        scaler = None
        features_name = None

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

    return X_train, X_test, y_train_categorized, y_test_categorized, ids_train, ids_test, scaler, features_name


    """讀取某幣種所有推文，回傳 [list of texts]"""
    files = glob(os.path.join(data_dir, coin_short_name, "*", "*", f"{coin_short_name}_*_normal.json"))
    texts = []
    for f in tqdm(files, desc=f"Loading tweets for {coin_short_name}"):
        with open(f, "r", encoding="utf-8-sig") as fp:
            data = json.load(fp)

        tweets = data[json_dict_name]
        if not tweets:
            print("當天沒有推文：", f)
            continue

        # 取得日期
        date_str = datetime.strptime(
            tweets[0]['created_at'], "%a %b %d %H:%M:%S %z %Y"
        ).strftime("%Y/%m/%d")
        date_dt = pd.to_datetime(date_str)

        # 🔹 過濾掉不在範圍內的推文
        if not (START_DATE_DT[coin_short_name] <= date_dt <= END_DATE_DT[coin_short_name]):
            print("當天不在指定時間範圍內：", f)
            continue

        for tw in tweets:
            texts.append(tw["text"])

    return texts



def load_price_diff(price_path, coin_short_name):
    """讀取某幣種的價差 (N, )"""
    return np.load(os.path.join(price_path, f"{coin_short_name}_price_diff.npy"))



# --- 五元分類 ---
def categorize_array_multi(Y, t1=0.0590, t2=0.0102, t3=0.0060, t4=0.0657):
    # 五元分類
    labels = np.full_like(Y, 2, dtype=int)  # 預設持平
    labels[Y <= -t1] = 0  # 大跌
    labels[(Y > -t1) & (Y <= -t2)] = 1  # 跌
    labels[(Y >= t3) & (Y < t4)] = 3  # 漲
    labels[Y >= t4] = 4  # 大漲

    if np.any(Y == 0):  # 檢查是否有任何元素等於 0
        count = np.sum(Y == 0)
        print(f"共有 {count} 個 Y == 0")
        labels[Y == 0] = 4  # 為了校正 TRUMP 前兩天的價格相同 第一天設為大漲

    return labels



class TweetDataset(Dataset):
    def __init__(self, texts=None, labels=None, tokenizer=None, max_len=64, pre_tokenized=None):
        if pre_tokenized is not None:
            # 已經處理好的 encoding
            self.encodings = pre_tokenized
            self.labels = labels
        else:
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_len = max_len
            self.encodings = []
            print("Tokenizing texts...")
            for txt in tqdm(self.texts, desc="Tokenizing"):
                encoding = self.tokenizer(
                    txt,
                    truncation=True,
                    padding="max_length",
                    max_length=self.max_len,
                    return_tensors="pt"
                )
                self.encodings.append({
                    "input_ids": encoding["input_ids"].flatten(),
                    "attention_mask": encoding["attention_mask"].flatten()
                })

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        item = self.encodings[idx]
        if hasattr(self, "labels") and self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item





def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}



def balanced_sampling(texts, labels, n_samples, num_categories, random_state=42):
    """
    Balanced sampling, with a target total sample count n_samples.
    Attempts to have roughly equal number of samples per class.
    
    texts: list of text samples
    labels: np.array of shape (N,)
    n_samples: total number of samples to draw
    num_categories: number of classes
    """

    # np.random.seed(random_state)
    # labels = np.array(labels)
    # sampled_texts = []
    # sampled_labels = []

    # # 計算每類目標樣本數
    # target_per_class = n_samples // num_categories
    # all_indices = set(range(len(labels)))
    # sampled_indices_set = set()

    # # 每類抽樣
    # for cls in range(num_categories):
    #     cls_indices = np.where(labels == cls)[0]
    #     n_cls_sample = min(len(cls_indices), target_per_class)
    #     selected = np.random.choice(cls_indices, size=n_cls_sample, replace=False)
    #     sampled_indices_set.update(selected)

    # # 剩餘數量
    # remaining = n_samples - len(sampled_indices_set)
    # if remaining > 0:
    #     available_indices = np.array(list(all_indices - sampled_indices_set))
    #     extra_selected = np.random.choice(available_indices, size=min(len(available_indices), remaining), replace=False)
    #     sampled_indices_set.update(extra_selected)

    # # 最終抽樣
    # sampled_indices = np.array(list(sampled_indices_set))
    # sampled_texts = [texts[i] for i in sampled_indices]
    # sampled_labels = labels[sampled_indices]

    # # 類別統計
    # counter_sampled = Counter(sampled_labels)
    # total = len(sampled_labels)
    # print("Balanced sampled class distribution (approx):")
    # for k in range(num_categories):
    #     print(f"  Class {k}: {counter_sampled[k]} samples, {counter_sampled[k]/total*100:.2f}%")
    # print(f"  Total sampled: {total} samples\n")

    # return sampled_texts, sampled_labels


    np.random.seed(random_state)
    labels = np.array(labels)
    sampled_texts = []
    sampled_labels = []

    # 計算每類目標樣本數
    target_per_class = n_samples // num_categories

    # 用索引操作，避免重複
    all_indices = np.arange(len(labels))
    used_indices = set()

    for cls in range(num_categories):
        cls_indices = np.where(labels == cls)[0]
        if len(cls_indices) == 0:
            continue
        n_cls_sample = min(len(cls_indices), target_per_class)
        selected = np.random.choice(cls_indices, size=n_cls_sample, replace=False)
        sampled_texts.extend([texts[i] for i in selected])
        sampled_labels.extend(labels[selected])
        used_indices.update(selected)

    # 剩餘樣本數，從未用過的樣本隨機分配
    remaining = n_samples - len(sampled_labels)
    if remaining > 0:
        available_indices = np.array([i for i in all_indices if i not in used_indices])
        if len(available_indices) > 0:
            extra_selected = np.random.choice(
                available_indices, 
                size=min(len(available_indices), remaining), 
                replace=False
            )
            sampled_texts.extend([texts[i] for i in extra_selected])
            sampled_labels.extend(labels[extra_selected])

    sampled_labels = np.array(sampled_labels)

    # 統計抽樣結果
    counter = Counter(sampled_labels)
    print("Balanced sampled class distribution (approx):")
    total = len(sampled_labels)
    for cls in range(num_categories):
        print(f"  Class {cls}: {counter[cls]} samples, {counter[cls]/total*100:.2f}%")
    print(f"  Total sampled: {total} samples")

    return sampled_texts, sampled_labels




def train_single_model(texts, labels, num_categories, model_dir=None,
                       epochs=3, n_samples=None, balanced=True):
    """
    texts: 訓練用的推文
    labels: 對應標籤
    all_texts_for_pred: 要全部丟去預測的推文 (包含訓練用的)
    """
    labels = np.array(labels)
    sampled_texts = []
    sampled_labels = []

    if n_samples is None:
        n_samples = len(texts)

    if balanced:
        sampled_texts, sampled_labels = balanced_sampling(
            texts, labels, n_samples, num_categories
        )
    else:
        n = len(texts)
        sampled_indices = np.random.choice(range(n), size=min(n_samples, n), replace=False)
        sampled_texts = [texts[i] for i in sampled_indices]
        sampled_labels = labels[sampled_indices]

    # if balanced:
    #     # 計算每類目標樣本數
    #     target_per_class = n_samples // num_categories
    #     all_indices = set(range(len(labels)))
    #     sampled_indices_set = set()

    #     # 每類抽樣
    #     for cls in range(num_categories):
    #         cls_indices = np.where(labels == cls)[0]
    #         n_cls_sample = min(len(cls_indices), target_per_class)
    #         selected = np.random.choice(cls_indices, size=n_cls_sample, replace=False)
    #         sampled_indices_set.update(selected)

    #     # 剩餘數量
    #     remaining = n_samples - len(sampled_indices_set)
    #     if remaining > 0:
    #         available_indices = np.array(list(all_indices - sampled_indices_set))
    #         extra_selected = np.random.choice(available_indices, size=min(len(available_indices), remaining), replace=False)
    #         sampled_indices_set.update(extra_selected)

    #     # 最終抽樣
    #     sampled_indices = np.array(list(sampled_indices_set))
    #     sampled_texts = [texts[i] for i in sampled_indices]
    #     sampled_labels = labels[sampled_indices]

    #     # 類別統計
    #     counter_sampled = Counter(sampled_labels)
    #     total = len(sampled_labels)
    #     print("Balanced sampled class distribution (approx):")
    #     for k in range(num_categories):
    #         print(f"  Class {k}: {counter_sampled[k]} samples, {counter_sampled[k]/total*100:.2f}%")
    #     print(f"  Total sampled: {total} samples\n")
    # else:
    #     # 普通隨機抽樣
    #     n = len(texts)
    #     sampled_indices = np.random.choice(range(n), size=min(n_samples, n), replace=False)
    #     sampled_texts = [texts[i] for i in sampled_indices]
    #     sampled_labels = labels[sampled_indices]

    # Tokenizer + Dataset
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    dataset = TweetDataset(sampled_texts, sampled_labels, tokenizer)

    # train/val split
    split = int(0.8 * len(dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [split, len(dataset)-split])

    # 模型
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_categories)

    training_args = TrainingArguments(
        output_dir=model_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        load_best_model_at_end=True,
        logging_steps=10,
        report_to="none",
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    print(f"=== Training {model_dir} ===")
    train_result = trainer.train()

    # 儲存模型
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    # 訓練結果
    metrics = train_result.metrics
    print(f"Training metrics for {model_dir}:")
    print(json.dumps(metrics, indent=4))
    with open(os.path.join(model_dir, "train_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # 驗證集結果
    eval_metrics = trainer.evaluate()
    print(f"Validation metrics for {model_dir}:")
    print(json.dumps(eval_metrics, indent=4))

    return trainer



def tokenize_and_save(all_texts, tokenizer, max_len=64, save_path=None): 
    if os.path.exists(save_path): 
        print(f"Loading pre-tokenized tweets from {save_path}") 
        return torch.load(save_path) 
    
    encodings = [] 
    print("Tokenizing texts...") 
    for txt in tqdm(all_texts, desc="Tokenizing"): 
        encoding = tokenizer( 
            txt, 
            truncation=True, 
            padding="max_length", 
            max_length=max_len, 
            return_tensors="pt" 
        ) 
        encodings.append({ 
            "input_ids": encoding["input_ids"].flatten(), 
            "attention_mask": encoding["attention_mask"].flatten() 
        }) 
        
    torch.save(encodings, save_path) 
    print(f"Saved tokenized tweets to {save_path}") 
    
    return encodings



def fast_predict_all_models(all_texts, all_Y, tokenized_path=None,
                            save_path=SAVE_PATH, batch_size=512, device=None):
    """
    分批預測所有模型，完成一個 label 就單獨存成一個檔案並釋放記憶體
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 載入 pre-tokenized
    tokenizer = BertTokenizerFast.from_pretrained(f"{save_path}/allcoins_y")  # 用 model0 的 tokenizer
    encodings = tokenize_and_save(all_texts, tokenizer, max_len=64, save_path=tokenized_path)
    input_ids = torch.stack([e["input_ids"] for e in encodings]).to(device)
    attention_mask = torch.stack([e["attention_mask"] for e in encodings]).to(device)
    n_samples = input_ids.size(0)

    os.makedirs(save_path, exist_ok=True)

    model_dir = f"{save_path}/allcoins_y"
    print(f"=== Predicting Y with model {model_dir} ===")

    model = BertForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    preds = []

    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_input_ids = input_ids[start:end]
            batch_attention_mask = attention_mask[start:end]

            outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
            logits = outputs.logits
            batch_pred = torch.argmax(logits, dim=-1).cpu().numpy()
            preds.append(batch_pred)

            if (start // batch_size) % 50 == 0:
                print(f"  Processed {end}/{n_samples} samples")

    preds = np.concatenate(preds)

    # 建立 DataFrame，只包含當前 label
    df = pd.DataFrame({
        "text": all_texts,
        f"true_y": all_Y,
        f"pred_y": preds,
    })
    df[f"correct_y"] = df[f"true_y"] == df[f"pred_y"]

    # 存檔（每個 label 獨立檔案）
    csv_path = os.path.join(save_path, f"predictions_y.csv")
    json_path = os.path.join(save_path, f"predictions_y.json")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    df.to_json(json_path, orient="records", force_ascii=False, indent=4)

    print(f"Saved predictions for label to {csv_path} and {json_path}")

    # # 清理記憶體
    # del model
    # torch.cuda.empty_cache()

    print("✅ 全部 label 預測完成！")







def main():
    data_dir = "../data/filtered_tweets/normal_tweets"
    price_dir = "../data/ml/dataset/coin_price"

    COIN_SHORT_NAME = ["DOGE", "PEPE", "TRUMP"]
    JSON_DICT_NAME = ["dogecoin", "PEPE", "(officialtrump OR \"official trump\" OR \"trump meme coin\" OR \"trump coin\" OR trumpcoin OR $TRUMP OR \"dollar trump\")"]

    all_texts = []
    all_Y = []

    # 先把三種幣的資料合併
    for coin_short_name, json_dict_name in zip(COIN_SHORT_NAME, JSON_DICT_NAME):
        print(f"=== Loading data for {coin_short_name} ===")
        texts = load_tweets(data_dir, coin_short_name, json_dict_name)
        Y = load_price_diff(price_dir, coin_short_name)  # (N_coin, )

        # print(len(texts))
        # print(Y.shape[0])

        assert len(texts) == Y.shape[0], f"{coin_short_name} texts and Y length mismatch!"

        all_texts.extend(texts)
        all_Y.append(Y)

    all_Y = np.concatenate(all_Y)  # shape = (N_total, )

    if IS_TRAIN:
        print(f"=== Processing Y (all coins combined) ===")
        labels = categorize_array_multi(all_Y)
        model_dir = f"{SAVE_PATH}/allcoins_y"

        # 訓練 + 預測全部推文
        trainer = train_single_model(
            all_texts,
            labels,
            num_categories=NUM_CATEGORIES,
            model_dir=model_dir,
            epochs=EPOCHS,
            n_samples=N_SAMPLES,
            balanced=True
        )
        
    print("\n開始預測全部推文...")
    # 預測全部推文 + 輸出 CSV/JSON
    fast_predict_all_models(all_texts, all_Y, tokenized_path=f"{SAVE_PATH}/tokenized_tweets.pt")





if __name__ == "__main__":
    main()