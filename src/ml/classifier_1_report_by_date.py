import json
import pandas as pd
from sklearn.metrics import classification_report
import numpy as np
from pathlib import Path
import os
import pickle

from sklearn.model_selection import train_test_split



'''可修改參數'''
COIN_SHORT_NAME = ['DOGE', 'PEPE', 'TRUMP']

COIN_DELETE_DATE = [13, 0, 12]  # 每個幣種要刪除的天數

MODEL_SHORT_NAME = "sgd"  # "logreg" "rf" "sgd"

MODEL_PATH_NAME = "SGD"  # "logistic_regression" "random_forest" "SGD"

IS_FILTERED = False  # 看是否有分 normal 與 bot

IS_BASEON_CLASSIFIER_1 = True  # 看是否要根據原先第一個分類器的 Train、Test 來按日期輸出分類報告
'''可修改參數'''

SUFFIX_FILTERED = "" if IS_FILTERED else "_non_filtered"
LATEX_SUFFIX_FILTERED = "_filtered" if IS_FILTERED else "_non_filtered"

save_json_path = f"../outputs/classification_report/{MODEL_PATH_NAME}"
os.makedirs(save_json_path, exist_ok=True)




def categorize_array_multi(Y, t1=-0.0590, t2=-0.0102, t3=0.0060, t4=0.0657, ids=None):
    """
    Y: np.ndarray, shape = (num_labels,), 價格變化率
    t1, t2: 五元分類閾值，百分比
    """

    # 五元分類
    labels = np.full_like(Y, 2, dtype=int)  # 預設持平
    labels[Y <= t1] = 0  # 大跌
    labels[(Y > t1) & (Y <= t2)] = 1  # 跌
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



def Latex(report_train, report_test):
    classes = ['大跌', '小跌', '持平', '小漲', '大漲']

    latex_str = r"""\begin{table}[H]
\centering
{\fontsize{12.5}{16}\selectfont
\begin{tabular}{c|ccc|ccc}
& \multicolumn{3}{c|}{Train set} & \multicolumn{3}{c}{Test set} \\
\hline
Class & Precision & Recall & F1-score & Precision & Recall & F1-score \\
\hline
"""

    for cls in classes:
        train_prec = report_train[cls]['precision']
        train_rec = report_train[cls]['recall']
        train_f1 = report_train[cls]['f1-score']
        
        test_prec = report_test[cls]['precision']
        test_rec = report_test[cls]['recall']
        test_f1 = report_test[cls]['f1-score']
        
        latex_str += f"{cls} & {train_prec:.3f} & {train_rec:.3f} & {train_f1:.3f} & {test_prec:.3f} & {test_rec:.3f} & {test_f1:.3f} \\\\\n"

    # 加上 Macro avg
    train_macro = report_train['macro avg']
    test_macro = report_test['macro avg']
    latex_str += r"\hline" + "\n"
    latex_str += f"Macro avg & {train_macro['precision']:.3f} & {train_macro['recall']:.3f} & {train_macro['f1-score']:.3f} & {test_macro['precision']:.3f} & {test_macro['recall']:.3f} & {test_macro['f1-score']:.3f} \\\\\n"

    latex_str += r"""\end{tabular}
}
\caption{"""
    latex_str += MODEL_SHORT_NAME.capitalize()
    
    latex_str += r""" 以日期為單位的訓練與測試準確度}
\label{tab:classifier_1_report_"""
    latex_str += f"{MODEL_SHORT_NAME}{LATEX_SUFFIX_FILTERED}"
    latex_str += r"""}
\end{table}"""

    # 印出 LaTeX
    print(latex_str)

    # 可選：存成檔案
    with open(f"{save_json_path}/{MODEL_SHORT_NAME}_report{SUFFIX_FILTERED}.tex", "w", encoding="utf-8") as f:
        f.write(latex_str)




y_true_final, y_pred_final = [], []
y_true_train, y_pred_train, y_dates_train, y_true_test, y_pred_test, y_dates_test = [], [], [], [], [], []
for csn, delete in zip(COIN_SHORT_NAME, COIN_DELETE_DATE):
    print(f"\n目前正在執行 {csn} ...\n")
    Y_TRUE_PATH = Path(f'../data/ml/dataset/y_input/{csn}/{csn}_price_diff_original{SUFFIX_FILTERED}.npy')
    Y_PRED_PATH = Path(f'../data/ml/classification/{MODEL_PATH_NAME}/keyword_classifier/single_coin_result/{csn}/{csn}_{MODEL_SHORT_NAME}_classifier_1_result{SUFFIX_FILTERED}.npy')
    # Y_DATE_PATH = Path(f'../data/coin_price/{csn}_current_tweet_price_output{SUFFIX_FILTERED}.csv')
 
    with open(f"../data/ml/dataset/ids_input/{csn}/{csn}_ids{SUFFIX_FILTERED}.pkl", "rb") as f:   # 讀取最初的 ids 
        ids_single_coin_original = pickle.load(f)
        ids_single_coin_original = sorted({(c, d) for (c, d, no) in ids_single_coin_original})
        print("len(ids_single_coin_original):", len(ids_single_coin_original))
        print("ids_single_coin_original[:10]:\n", ids_single_coin_original[:10])

    with open(f"../data/ml/dataset/final_input/keyword_classifier/ids_train{SUFFIX_FILTERED}.pkl", "rb") as f:   # 讀取一開始訓練用的 ids
        ids_train_classifier_1 = pickle.load(f)
        print("len(ids_train_classifier_1):", len(ids_train_classifier_1))
    with open(f"../data/ml/dataset/final_input/keyword_classifier/ids_test{SUFFIX_FILTERED}.pkl", "rb") as f:   # 讀取一開始訓練用的 ids
        ids_test_classifier_1 = pickle.load(f)
        print("len(ids_test_classifier_1):", len(ids_test_classifier_1))

    ids_all = ids_train_classifier_1 + ids_test_classifier_1
    print("全部幣種的 len(ids_all):", len(ids_all))
    ids_single_coin = {d for c, d, no in ids_all if c == csn}  # 轉成集合 (set)
    print(f"{csn} 的 len(ids_single_coin):", len(ids_single_coin))
    print("list(ids_single_coin)[:10]:\n", list(ids_single_coin)[:10])


    y_true = categorize_array_multi(np.load(Y_TRUE_PATH)).tolist()
    y_pred = np.load(Y_PRED_PATH).tolist()

    # 將 merge_and_splitset 中被過濾掉的資料 這裡也過濾掉
    ids_mask = np.array([d in ids_single_coin for (_, d) in ids_single_coin_original])
    ids_single_coin_original = (np.array(ids_single_coin_original))[ids_mask]
    y_true = (np.array(y_true))[ids_mask]
    print("過濾完後的 len(ids_single_coin_original):", len(ids_single_coin_original))
    print("過濾完後的 len(y_true):", len(y_true))

    # df = pd.read_csv(Y_DATE_PATH)
    # df_filtered = df[df["has_tweet"] == True]  # 篩選 has_tweet 為 True 的資料
    # y_dates = pd.to_datetime(df_filtered["date"], format="%Y/%m/%d").dt.strftime("%Y-%m-%d").tolist()  # 先轉成 datetime，再轉成 YYYY-MM-DD 字串

    y_dates = [str(d) for (_, d) in ids_single_coin_original]
    print("y_dates[:10]:\n", y_dates[:10])
    input("按 Enter 以繼續 ...")


    if IS_BASEON_CLASSIFIER_1:
        # 讀取每個幣種第一個分類的資料集日期
        single_coin_train_date = pd.read_csv(f"../data/ml/dataset/split_dates/{csn}_train_dates{SUFFIX_FILTERED}.csv")
        single_coin_test_date= pd.read_csv(f"../data/ml/dataset/split_dates/{csn}_test_dates{SUFFIX_FILTERED}.csv")

        single_coin_train_date = set(single_coin_train_date["date"])
        # if "2013-12-16" in single_coin_train_date:
        #     print("Train 有 2013-12-16")
        # print(single_coin_train_date)
        # input("按 Enter 以繼續 ...")
        single_coin_test_date = set(single_coin_test_date["date"])
        # if "2013-12-16" in single_coin_test_date:
        #     print("Test 有 2013-12-16")
        # print(single_coin_train_date)

        # 建立對應 train/test 的 mask（布林列表）
        train_mask = [d in single_coin_train_date for d in y_dates]
        test_mask = [d in single_coin_test_date for d in y_dates]

        # 使用 mask 對 y_true, y_pred, y_dates 分割
        y_true_train += [yt for yt, m in zip(y_true, train_mask) if m]
        print("len(y_true_train):", len(y_true_train))
        y_pred_train += [yp for yp, m in zip(y_pred, train_mask) if m]
        print("len(y_pred_train):", len(y_pred_train))
        y_dates_train += [d for d, m in zip(y_dates, train_mask) if m]
        print("len(y_dates_train):", len([d for d, m in zip(y_dates, train_mask) if m]))
        print("y_dates_train[:10]:\n", [d for d, m in zip(y_dates, train_mask) if m][:10])

        y_true_test += [yt for yt, m in zip(y_true, test_mask) if m]
        print("len(y_true_test):", len(y_true_test))
        y_pred_test += [yp for yp, m in zip(y_pred, test_mask) if m]
        print("len(y_pred_test):", len(y_pred_test))
        y_dates_test += [d for d, m in zip(y_dates, test_mask) if m]
        print("len(y_dates_test):", len([d for d, m in zip(y_dates, test_mask) if m]))
        print("y_dates_test[:10]:\n", [d for d, m in zip(y_dates, test_mask) if m][:10])
        input("按 Enter 以繼續 ...")
    
    else:
        print("🚩 刪除資料前")
        print("len(y_true):", len(y_true))
        print("len(y_pred):", len(y_pred))
        print("y_true[:10]:", y_true[:30])
        print("y_pred[:10]:", y_pred[:30])

        y_true = y_true[delete:]
        y_pred = y_pred[delete:]
        y_true_final += y_true
        y_pred_final += y_pred

        print("🚩 刪除資料後")
        print("len(y_true):", len(y_true))
        print("len(y_pred):", len(y_pred))
        print("y_true[:10]:", y_true[:30])
        print("y_pred[:10]:", y_pred[:30])


if not IS_BASEON_CLASSIFIER_1:  # 要跟第二個分類器一樣才使用
    # --- 打亂 X, Y, ids ---
    rng = np.random.default_rng(42)  # 可自訂種子
    indices = np.arange(len(y_pred_final))
    rng.shuffle(indices)


    y_true_final = [y_true_final[i] for i in indices]
    y_pred_final = [y_pred_final[i] for i in indices]


    print("🚩 打亂後")
    print("len(y_true_final):", len(y_true_final))
    print("len(y_pred_final):", len(y_pred_final))
    print("y_true_final[:10]:", y_true_final[:30])
    print("y_pred_final[:10]:", y_pred_final[:30])


    y_true_train, y_true_test, y_pred_train, y_pred_test = train_test_split(
        y_true_final, y_pred_final, test_size=0.2, random_state=42, shuffle=True
    )


print("\n🚨 Debug Info")
print("len(y_true_train):", len(y_true_train))
print("len(y_pred_train):", len(y_pred_train))
print("len(y_true_test):", len(y_true_test))
print("len(y_pred_test):", len(y_pred_test))

print(classification_report(
    y_true_train, y_pred_train,
    digits=3,
    target_names=['大跌', '小跌', '持平', '小漲', '大漲']
))

print(classification_report(
    y_true_test, y_pred_test,
    digits=3,
    target_names=['大跌', '小跌', '持平', '小漲', '大漲']
))


# --- 生成分類報告 dict ---
report_train = classification_report(
    y_true_train, y_pred_train,
    digits=3,
    target_names=['大跌', '小跌', '持平', '小漲', '大漲'],
    output_dict=True   # <-- 這裡把報告轉成 dict
)

report_test = classification_report(
    y_true_test, y_pred_test,
    digits=3,
    target_names=['大跌', '小跌', '持平', '小漲', '大漲'],
    output_dict=True
)

Latex(report_train, report_test)


# --- 存成 JSON ---
save_json_path = f"../outputs/classification_report/{MODEL_PATH_NAME}"
os.makedirs(save_json_path, exist_ok=True)

with open(f"{save_json_path}/{MODEL_SHORT_NAME}_train_report{SUFFIX_FILTERED}.json", "w", encoding="utf-8") as f:
    json.dump(report_train, f, ensure_ascii=False, indent=4)

with open(f"{save_json_path}/{MODEL_SHORT_NAME}_test_report{SUFFIX_FILTERED}.json", "w", encoding="utf-8") as f:
    json.dump(report_test, f, ensure_ascii=False, indent=4)

print("\n✅ JSON 檔案已存好")