from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, learning_curve, LearningCurveDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import loguniform, uniform
from scipy.sparse import csr_matrix

import json
import numpy as np
from scipy import sparse
import os
import matplotlib.pyplot as plt
import argparse
import joblib
from tqdm import tqdm
import math
from collections import defaultdict

import sys
from pathlib import Path
parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))
from config import COIN_SHORT_NAME

# ----------------------------------------paths----------------------------------------
FEATURE_NAME_PATH = Path('../data/keyword/machine_learning/all_keywords.json')
X_TRAIN_PATH = Path('../data/ml/dataset/X_train.npz')
X_TEST_PATH = Path('../data/ml/dataset/X_test.npz')
Y_TRAIN_PATH = Path('../data/ml/dataset/Y_train.npz')
Y_TEST_PATH = Path('../data/ml/dataset/Y_test.npz')
OUTPUT_PATH = Path('../data/ml/regression/bayesian_importance.json')
FIGURE_PATH = Path('../outputs/figures/ml/regression/bayesing_learning_curve.png')
MODEL_PATH = Path('../data/ml/models/classification/knn.pkl')
# ----------------------------------------paths----------------------------------------

# ----------------------------------------parameters----------------------------------------
N_SAMPLES = 1000000
N_RUNS = 1
# ----------------------------------------parameters----------------------------------------

def categorize_array_multi(Y, t1 = 0.0590, t2 = 0.0102, t3 = 0.0060, t4 = 0.0657):
    """
    Y: np.ndarray, shape = (num_labels,), 價格變化率
    t1, t2: 五元分類閾值，百分比
    """

    # 五元分類
    labels = np.full_like(Y, 2, dtype=int)  # 預設持平
    labels[Y <= -t1] = 0  # 大跌
    labels[(Y > -t1) & (Y <= -t2)] = 1  # 跌
    labels[(Y >= t3) & (Y < t4)] = 3  # 漲
    labels[Y >= t4] = 4  # 大漲

    if np.any(Y == 0):  # 檢查是否有任何元素等於 0
        count = np.sum(Y == 0)
        print(f"共有 {count} 個 Y == 0")
        labels[Y == 0] = 4  # 為了校正TRUMP前兩天的價格相同 第一天設為大漲

    return labels

def prepare_data():
    '''
    args: feature selection arguments
    '''
    # load dataset
    with open(FEATURE_NAME_PATH, 'r', encoding='utf-8') as file:
        feature_names = json.load(file)
    X_train = sparse.load_npz(X_TRAIN_PATH)
    X_test = sparse.load_npz(X_TEST_PATH)
    Y_train = np.load(Y_TRAIN_PATH)['Y']
    Y_test = np.load(Y_TEST_PATH)['Y']

    # scaler = StandardScaler(with_mean=False)
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)

    print(Y_train[:50])

    Y_train = categorize_array_multi(Y_train)
    Y_test  = categorize_array_multi(Y_test)

    # 統計每個類別數量
    print(f"大跌：-{0.0590 * 100:.2f}%以下, 跌：-{0.0590 * 100:.2f}% ~ -{0.0102 * 100}%, 持平：-{0.0102 * 100}% ~ {0.0060 * 100}%, 漲：{0.0060 * 100}% ~ {0.0657 * 100:.2f}%, 大漲：{0.0657 * 100:.2f}%以上")
    train_total_row = Y_train.shape[0]
    test_total_row = Y_test.shape[0]
    # for col in range(y_train_categorized.shape[1]):
    counts = np.bincount(Y_train, minlength=5)
    percentages = counts / train_total_row * 100
    percentages_str = " ".join([f"{p:.2f}%" for p in percentages])
    print(f"[TRAIN] column 類別: {percentages_str}")

    counts = np.bincount(Y_test, minlength=5)
    percentages = counts / test_total_row * 100
    percentages_str = " ".join([f"{p:.2f}%" for p in percentages])
    print(f"[TEST]  column 類別: {percentages_str}\n")

    input("pasue...")
    
    return X_train, X_test, Y_train, Y_test, feature_names

def get_random_samples_sparse_stratified(X: csr_matrix, y: np.ndarray, seed: int = 42):
    """
    X: csr_matrix
    y: np.ndarray, shape=(N,)  多類別標籤
    """
    global N_SAMPLES, ENABLE_SAMPLING
    n_total = X.shape[0]

    if N_SAMPLES == 0:
        print(f"[INFO] 不做 random sampling，使用所有樣本數: {n_total} 筆")
        ENABLE_SAMPLING = False
        return [(X, y)] * N_RUNS

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

def train_and_evaluate(X_train, X_test, Y_train, Y_test):
    knn = KNeighborsClassifier(n_jobs=-1)
    max_neighbors = int(math.sqrt(N_SAMPLES))
    param_grid = {
        'n_neighbors': [100, 1000, 5000],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=True)
    grid_search.fit(X_train, Y_train)

    print('Best params:', grid_search.best_params_)
    print('Best CV score:', grid_search.best_score_)

    best_model = grid_search.best_estimator_
    test_score = best_model.score(X_test, Y_test)
    print('Test accuracy:', test_score)

    # 測試集詳細報告
    print("\n分類報告 (Test set):")
    print(classification_report(Y_test, best_model.predict(X_test)))

# def validation(best_model, X_train, X_test, Y_train, Y_test):
#     display = LearningCurveDisplay.from_estimator(
#         best_model, X_train, Y_train, cv=5, scoring='neg_mean_squared_error',
#         train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
#     )
#     display.plot()
#     os.makedirs(FIGURE_PATH.parent, exist_ok=True)
#     plt.savefig(FIGURE_PATH)
#     plt.show()

def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--fs", type=str, default="none", help="Feature selection method")
    # parser.add_argument("--k", type=int, default=600, help="Top k features")
    # args = parser.parse_args()

    print('Loading data...')
    X_train, X_test, Y_train, Y_test, feature_names = prepare_data()
    # train_and_evaluate(X_train, X_test, Y_train, Y_test)
    samples = get_random_samples_sparse_stratified(X_train, Y_train) 
    for s in samples:
        X_train, Y_train = s[0], s[1]
        # print('\nTunning hyperparameters and Training model...')
        train_and_evaluate(X_train, X_test, Y_train, Y_test)

        # print('\nPlotting learning curve...')
        # validation(best_model, X_train, X_test, Y_train, Y_test)

if __name__ == '__main__':
    main()          
