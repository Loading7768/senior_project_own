import json
import os
from multiprocessing import Pool, cpu_count
from itertools import combinations
from tqdm import tqdm
from collections import defaultdict
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import time
import psutil
import gc

from pathlib import Path
import sys
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
import config

'''可修改參數'''
YEAR = "2025"

MONTH = "07"

FOLDER_PATH = f"../data/spammer/{YEAR}/{MONTH}"  # 選擇要對哪個資料夾執行
# "../Kmeans/data/clustered/"
# "../project_vscode/data/spammer/04"

OUTPUT_FOLDER_NAME = f"{YEAR}{MONTH}"  # 設定要儲存到的資料夾名稱   ex. "../LCS/analysis/{OUTPUT_FOLDER_NAME}/"

JSON_DICT_NAME = config.JSON_DICT_NAME  # 設定推文所存的 json 檔中字典的名稱

DICE_COEFFICIENT = 70  # 設定 Dice 算出來的結果門檻值（也就是相似度）  60 => 60%

LENGTH_RATIO = 80  # 設定 Y(被比對的推文) 的長度相對於 X(當基準的推文) 的百分比   80 => 80%
# 這是要確認兩篇推文的長度落在合理範圍內，推文 Y 的長度至少有 X 的 80% 長（不能太短）

MAX_PAIRS_THRESHOLD = 1_000_000

IS_CLUSTERED = False  # 設定是否要用有分群的檔案來比對
'''可修改參數'''

# create folders if not existed
os.makedirs("../data/dice/analysis", exist_ok=True)
os.makedirs("../data/dice/robot_account", exist_ok=True)
os.makedirs("../data/dice/robot_list", exist_ok=True)

# 取得英文停用詞集合
stop_words = set(stopwords.words('english'))

def preprocess_tweets(tweets):
    cache = {}
    for tweet in tweets:
        text = tweet["text"]
        if text not in cache:
            tokens = word_tokenize(text.lower(), language='english')
            tokens = [word for word in tokens if word.isalnum() and (word not in stop_words)]
            cache[text] = tokens
        tweet["tokens"] = cache[text]
    return tweets

def dice(X, Y, x_tokens, y_tokens):
    x_set = set(x_tokens)
    y_set = set(y_tokens)
    overlap = len(x_set & y_set)
    total = len(x_set) + len(y_set)
    if total == 0:
        return 0.0, x_tokens, y_tokens
    dice_score = (2 * overlap) / total
    return dice_score, x_tokens, y_tokens

def compare_pair(args):
    i, j, tweets = args
    X = tweets[i]
    Y = tweets[j]

    length_ratio_X = (len(Y["text"]) / len(X["text"]))
    length_ratio_Y = (len(X["text"]) / len(Y["text"]))
    if length_ratio_X * 100 < LENGTH_RATIO or length_ratio_Y * 100 < LENGTH_RATIO:
        return None

    dice_coefficient, x_tokens, y_tokens = dice(X["text"], Y["text"], X["tokens"], Y["tokens"])
    if dice_coefficient * 100 < DICE_COEFFICIENT:
        return None

    return {
        "X": X,
        "Y": Y,
        "X_token": ", ".join(x_tokens),
        "Y_token": ", ".join(y_tokens),
        "length_ratio_X": length_ratio_X,
        "length_ratio_Y": length_ratio_Y,
        "dice_coefficient": dice_coefficient
    }



# 為了不要 MemoryError
# 邊用 imap_unordered 邊產生組合，不需要一次生成所有組合
def generate_pairs(tweets, batch_size=100000):  # Increased batch_size
    batch = []
    for i, j in combinations(range(len(tweets)), 2):
        batch.append((i, j, tweets))
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


# def write_txt_result(filetxt, res):
#     X, Y = res["X"], res["Y"]

#     filetxt.write(f"X = [{repr(X['text'])[1:-1]}]\n")  # repr(): 讓 \n 保持為 \n 輸出
#     filetxt.write(f"X_token = [{res['X_token']}]\n")
#     filetxt.write(f"\tX tweet_count = [{X['tweet_count']}]\n")
#     filetxt.write(f"\tX username = [{X['username']}]\n")

#     filetxt.write(f"Y = [{repr(Y['text'])[1:-1]}]\n")
#     filetxt.write(f"Y_token = [{res['Y_token']}]\n")
#     filetxt.write(f"\tY tweet_count = [{Y['tweet_count']}]\n")
#     filetxt.write(f"\tY username = [{Y['username']}]\n")

#     filetxt.write(f"Total Length: X = {len(X['text'])}, Y = {len(Y['text'])} "
#                   f"(Y / X = {res['length_ratio_X'] * 100:.2f}  X / Y = {res['length_ratio_Y'] * 100:.2f})\n")
#     filetxt.write(f"Dice Coefficient: {res['dice_coefficient'] * 100:.2f}% \n\n")



def write_json_result(res, cluster_id=None):
    X, Y = res["X"], res["Y"]
    data = {
        "X_text": X["text"],
        "X_token": res["X_token"],
        "X_tweet_count": X["tweet_count"],
        "X_username": X["username"],
        "Y_text": Y["text"],
        "Y_token": res["Y_token"],
        "Y_tweet_count": Y["tweet_count"],
        "Y_username": Y["username"],
        "X_length": len(X["text"]),
        "Y_length": len(Y["text"]),
        "Y_length_percent_of_X": round(res['length_ratio_X'] * 100, 2),
        "X_length_percent_of_Y": round(res['length_ratio_Y'] * 100, 2),
        "dice_coefficient": round(res['dice_coefficient'] * 100, 2)  # round( , 2) 輸出到小數點後第二位
    }
    if cluster_id is not None:
        data["cluster_id"] = cluster_id
    return data



def process_tweet_group(tweets_group, json_output, json_output_path, cluster_id=None):
    start_time = time.time()
    writed_compare = 0
    fail_count = 0
    repetitive_counts = defaultdict(int)  # Track repetitive counts during processing

    os.makedirs(os.path.dirname(json_output_path), exist_ok=True)

    tweets_group = preprocess_tweets(tweets_group)

    num_processes = min(cpu_count() // 2, 6)
    pool = Pool(processes=num_processes)
    
    temp_file_path = f"./temp_results_{cluster_id if cluster_id is not None else 'all'}.jsonl"
    total_pairs = (len(tweets_group) * (len(tweets_group) - 1)) // 2
    batch_size = 100000
    chunksize = 10000
    print(f"Processing {len(tweets_group)} tweets, {total_pairs} pairs, batch_size={batch_size}, chunksize={chunksize}, num_processes={num_processes}")
    print(f"Initial memory: {psutil.Process().memory_info().rss / 1024**2:.2f} MB")
    
    write_json = total_pairs <= MAX_PAIRS_THRESHOLD
    if not write_json:
        print(f"⚠️ Skipping JSON output for {total_pairs} pairs (exceeds threshold of {MAX_PAIRS_THRESHOLD})")

    try:
        if write_json:
            with open(temp_file_path, 'w', encoding='utf-8') as temp_file:
                pair_count = 0
                for batch in generate_pairs(tweets_group, batch_size=batch_size):
                    batch_start = time.time()
                    for res in pool.imap_unordered(compare_pair, batch, chunksize=max(1, batch_size // num_processes)):
                        if res is not None:
                            json.dump(res, temp_file, ensure_ascii=False)
                            temp_file.write('\n')
                            writed_compare += 1

                            # Update repetitive_counts
                            repetitive_counts[res["X"]["username"]] += 1
                            repetitive_counts[res["Y"]["username"]] += 1
                    pair_count += len(batch)
                    print(f"Processed {pair_count}/{total_pairs} pairs, {writed_compare} results, "
                            f"batch time: {time.time() - batch_start:.2f}s, "
                            f"memory: {psutil.Process().memory_info().rss / 1024**2:.2f} MB")
                    # Free memory used by the batch
                    del batch
                    gc.collect()
        else:
            # Still process pairs but don't write to temp file
            pair_count = 0
            for batch in generate_pairs(tweets_group, batch_size=batch_size):
                batch_start = time.time()
                for res in pool.imap_unordered(compare_pair, batch, chunksize=max(1, batch_size // num_processes)):
                    if res is not None:
                        writed_compare += 1

                        # Update repetitive_counts
                        repetitive_counts[res["X"]["username"]] += 1
                        repetitive_counts[res["Y"]["username"]] += 1
                pair_count += len(batch)
                print(f"Processed {pair_count}/{total_pairs} pairs, {writed_compare} results, "
                      f"batch time: {time.time() - batch_start:.2f}s, "
                      f"memory: {psutil.Process().memory_info().rss / 1024**2:.2f} MB")
                del batch
                gc.collect()
    finally:
        pool.close()
        pool.join()

    if write_json and os.path.exists(temp_file_path):
        with open(temp_file_path, 'r', encoding='utf-8') as temp_file:
            for line in temp_file:
                res = json.loads(line.strip())
                json_output.append(write_json_result(res, cluster_id=cluster_id))
                # write_txt_result(filetxt, res)

        os.unlink(temp_file_path)

        try:
            with open(json_output_path, "w", encoding="utf-8") as f:
                json.dump(json_output, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"[ERROR] Failed to write after fail_count = {fail_count}: {e}")
            fail_count += 1

    print(f"Total time: {time.time() - start_time:.2f}s, {writed_compare} results")
    return writed_compare, repetitive_counts



# 🧠 主程式入口：處理整個資料夾
if __name__ == "__main__":
    all_files = [f for f in os.listdir(FOLDER_PATH) if f.endswith(".json")]
    print(f"📂 總共找到 {len(all_files)} 個檔案要處理")

    # 先清空 robottxt.txt
    robottxt = f"../data/dice/robot_account/{OUTPUT_FOLDER_NAME}.txt"
    with open(robottxt, "w", encoding="utf-8-sig") as robotfile:
        robotfile.write("")

    for file in all_files:
        # run_for_file(os.path.join(FOLDER_PATH, file))
        filepath = os.path.join(FOLDER_PATH, file)
        filename = os.path.basename(filepath)  # ex: DOGE_20210428.json
        analysis_name = os.path.splitext(filename)[0]  # ex: DOGE_20210428

        # 設定 txtname, json_output_path 的名稱
        # txtname = f"../data/dice/analysis/{OUTPUT_FOLDER_NAME}/{analysis_name}.txt"
        json_output_path = f"../data/dice/analysis/{OUTPUT_FOLDER_NAME}/{analysis_name}.json"

        # 確認是否有輸出時需使用的資料夾
        output_folder_path = f"../data/dice/analysis/{OUTPUT_FOLDER_NAME}/"
        os.makedirs(output_folder_path, exist_ok=True)

        # 讀入 json 檔
        with open(filepath, 'r', encoding="utf-8-sig") as file:
            data_json = json.load(file)

        tweets = data_json[JSON_DICT_NAME]
        print(f"\n📄 正在處理檔案：{filename}，共 {len(tweets)} 筆推文")

        # # 先把 txt 檔裡清空
        # with open(txtname, 'w', encoding="utf-8-sig") as filetxt:
        #     filetxt.write("")

        json_output = []  # 用來儲存所有比對結果

        # 先把輸出的 json 檔裡清空
        with open(json_output_path, 'w', encoding='utf-8-sig') as f_json:
            json.dump(json_output, f_json, indent=4, ensure_ascii=False)



        total_compare = 0  # 計算總共寫入的結果數

        if IS_CLUSTERED:
            # defaultdict: Python 的一種特殊字典
            # 當你存取一個不存在的 key 時，它會自動建立對應的預設值  ex. clusters["0"] 若原本不存在，會自動被建立成 []（空 list）
            clusters = defaultdict(list)

            for tweet in tweets:
                clusters[tweet["cluster"]].append(tweet)  # 把推文加入對應的群集  ex. clusters[0].append(推文)

            # .items() 來一次取得 key（cluster_id）與對應的 value（cluster_tweets，一個 list）
            for cluster_id, cluster_tweets in clusters.items():
                if len(cluster_tweets) < 2:
                    continue  # 不需要比對

                # filetxt.write(f"cluster {cluster_id}, 共 {len(cluster_tweets)} 筆\n")

                # 呼叫 process_tweet_group 來執行比對，並回傳當前 Cluster 的實際寫入數量
                # total_compare, repetitive_counts += process_tweet_group(cluster_tweets, json_output, json_output_path, cluster_id=cluster_id)
        else:
            # 如果是沒有分類過的檔案 直接呼叫 process_tweet_group 來執行比對
            total_compare, repetitive_counts = process_tweet_group(tweets, json_output, json_output_path)

        print()
        print(f"✅ 已儲存 JSON 結果到 {json_output_path}")
        print(f"實際寫入的全部結果數：{total_compare}")



        # 建立一個字典記錄每個帳號有多少重複推文
        # repetitive_counts = defaultdict(int)

        # 當你從 Dice 對比結果中抓出重複推文時
        # 你可以記錄帳號出現的次數
        # for tweet in json_output:
        #     X_user = tweet["X_username"]
        #     Y_user = tweet["Y_username"]
        #     repetitive_counts[X_user] += 1
        #     repetitive_counts[Y_user] += 1

        tweet_len = len(tweets)
        total_pairs = (tweet_len * (tweet_len - 1)) // 2
        # Free memory used by tweets and json_output
        del tweets
        del json_output
        gc.collect()

        robottxt = f"../data/dice/robot_account/{OUTPUT_FOLDER_NAME}.txt"
        # 印出出現次數大於 10 的帳號，符合的話就輸出到 txt 檔中
        robotlist = [] # list of user that has ressemblence over threshold
        print()
        with open(robottxt, "a", encoding="utf-8-sig") as robotfile:
            robotfile.write(f"{filename}\n")
            robotfile.write(f"共 {tweet_len} 筆推文\n")

            # with open(json_output_path, "r", encoding="utf-8-sig") as jsonfile:
            #     output_json = json.load(jsonfile)
            if total_compare == 0:
                robotfile.write(f"(沒有符合條件的推文)\n")

            for user, count in sorted(repetitive_counts.items(), key=lambda x: x[1], reverse=True):
                resemblance = ((count / 2) / total_pairs) * 100 if total_pairs > 0 else 0
                robotfile.write(f"整體推文相似度：{resemblance:.2f}%\n")

                if resemblance > 80.0 :
                    robotlist.append(user)

                if int(count / 2) > 10:
                    robotfile.write(f"🤖 疑似洗版帳號：{user}，重複出現次數：{int(count / 2)}\n")
            robotfile.write("\n")

        robotlisttxt = f"../data/dice/robot_list/{OUTPUT_FOLDER_NAME}_list.txt"
        with open(robotlisttxt, 'a', encoding="utf-8-sig") as robotlistfile:
            for robot in robotlist:
                robotlistfile.write(f"{robot}\n")