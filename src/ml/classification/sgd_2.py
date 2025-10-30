from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
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
import pickle

import sys
from pathlib import Path
parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))
from config import COIN_SHORT_NAME

# ----------------------------------------parameters----------------------------------------
FIRST_MODEL_NAME = ["random_forest", "rf"]  # ["logistic_regression", "logreg"] ["random_forest", "rf"] ["SGD", "sgd"] 
USE_NON_FILTERED = True
NO_KEYWORD_CLASSIFIER = False
# ----------------------------------------parameters----------------------------------------
FILTERED_SUFFIX = '_non_filtered' if USE_NON_FILTERED else ''
KEYWORD_CLASSIFIER_SUFFIX = '_non_classifier_1' if NO_KEYWORD_CLASSIFIER else ''

def prepare_data():
    '''
    args: feature selection arguments
    '''
    # X_PATH = Path(f'../data/ml/dataset/{FIRST_MODEL_NAME}_X_classifier_2{FILTERED_SUFFIX}{KEYWORD_CLASSIFIER_SUFFIX}.npy')
    # Y_PATH = Path(f'../data/ml/dataset/{FIRST_MODEL_NAME}_Y_classifier_2{FILTERED_SUFFIX}.npy')

    # X = np.load(X_PATH)
    # Y = np.load(Y_PATH)
    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    X_TRAIN_PATH = Path(f'../data/ml/dataset/final_input/price_classifier/{FIRST_MODEL_NAME[0]}/{FIRST_MODEL_NAME[1]}_X_train_classifier_2{FILTERED_SUFFIX}{KEYWORD_CLASSIFIER_SUFFIX}.npy')
    X_TEST_PATH = Path(f'../data/ml/dataset/final_input/price_classifier/{FIRST_MODEL_NAME[0]}/{FIRST_MODEL_NAME[1]}_X_test_classifier_2{FILTERED_SUFFIX}{KEYWORD_CLASSIFIER_SUFFIX}.npy')
    Y_TRAIN_PATH = Path(f'../data/ml/dataset/final_input/price_classifier/{FIRST_MODEL_NAME[0]}/{FIRST_MODEL_NAME[1]}_Y_train_classifier_2{FILTERED_SUFFIX}.npy')
    Y_TEST_PATH = Path(f'../data/ml/dataset/final_input/price_classifier/{FIRST_MODEL_NAME[0]}/{FIRST_MODEL_NAME[1]}_Y_test_classifier_2{FILTERED_SUFFIX}.npy')

    X_train = np.load(X_TRAIN_PATH)
    X_test  = np.load(X_TEST_PATH)
    Y_train = np.load(Y_TRAIN_PATH)
    Y_test  = np.load(Y_TEST_PATH)

    # Y_train = Y_train
    # Y_test  = Y_test

    scaler = StandardScaler(with_mean=False)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, Y_train, Y_test


def train_and_evaluate(X_train, X_test, Y_train, Y_test):
    if USE_NON_FILTERED:
        sgd = SGDClassifier(max_iter=5000, tol=1e-3, random_state=42, n_jobs=-1)
    else: 
        sgd = SGDClassifier(max_iter=5000, tol=1e-3, random_state=42, n_jobs=-1)

    param_grid = {
        'loss': ['log_loss', 'hinge'],  # logistic regression vs linear SVM
        'penalty': ['l2', 'l1', 'elasticnet'],
        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
        'alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        'learning_rate': ['optimal', 'adaptive'],
        'eta0': [1e-3, 1e-2, 1e-1],
    }
    grid_search = GridSearchCV(sgd, param_grid, cv=5, scoring='balanced_accuracy', n_jobs=-1, verbose=True)
    grid_search.fit(X_train, Y_train)

    print('Best params:', grid_search.best_params_)
    print('Best CV score:', grid_search.best_score_)

    sgd = grid_search.best_estimator_
    sgd.fit(X_train, Y_train)

    MODEL_PATH = Path(f'../data/ml/models/SGD/{FIRST_MODEL_NAME[1]}_SGD_2{FILTERED_SUFFIX}{KEYWORD_CLASSIFIER_SUFFIX}.pkl')
    os.makedirs(MODEL_PATH.parent, exist_ok=True)
    joblib.dump(sgd, MODEL_PATH)

    train_score = sgd.score(X_train, Y_train)
    test_score = sgd.score(X_test, Y_test)
    clf_rpt_train = classification_report(Y_train, sgd.predict(X_train), digits=3, target_names=['大跌', '小跌', '持平', '小漲', '大漲'])
    clf_rpt_test = classification_report(Y_test, sgd.predict(X_test), digits=3, target_names=['大跌', '小跌', '持平', '小漲', '大漲'])
    model_report = f'Train accuracy: {train_score}\nTest accuracy: {test_score}\n\nClassification Report (Train set):\n{clf_rpt_train}\n\nClassification Report (Test set):\n{clf_rpt_test}\n'
    print(model_report)

    pre_trained = []
    pre_trained += sgd.predict(X_train).tolist()
    pre_trained += sgd.predict(X_test).tolist()
    Y = np.concatenate((Y_train, Y_test), axis=0)
    pre_trained = [[pt, y] for pt, y in zip(pre_trained, Y)]

    aug_clf_rpt_all = []
    pre_august_all = []
    # for csn in ['DOGE', 'PEPE', 'TRUMP']:
    #     X_AUGUST_PATH = f'../data/ml/dataset/keyword/{csn}_{FIRST_MODEL_NAME}_X_classifier_2{FILTERED_SUFFIX}_202508{KEYWORD_CLASSIFIER_SUFFIX}.npy'
    #     Y_AUGUST_PATH = f'../data/ml/dataset/coin_price/{csn}_price_diff_original{FILTERED_SUFFIX}_202508.npy'

    #     X_august = np.load(X_AUGUST_PATH)
    #     Y_august = np.load(Y_AUGUST_PATH)

    #     pre_august = sgd.predict(X_august).tolist()

    #     august_score = sgd.score(X_august, Y_august)
    #     aug_clf_rpt = classification_report(Y_august, sgd.predict(X_august), digits=3, labels=[0, 1, 2, 3, 4], target_names=['大跌', '小跌', '持平', '小漲', '大漲'])
    #     aug_clf_rpt_all.append(f'Classification Report ({csn} 202508):\n{aug_clf_rpt}\n{august_score}')
        
    #     pre_august = [[pt, y] for pt, y in zip(pre_august, Y_august)]

    #     pre_august_all.append(pre_august)

    return pre_trained, model_report, pre_august_all, aug_clf_rpt_all

def sort_and_save(pre_trained, model_report, pre_august_all, aug_clf_rpt_all):
    cat = ["🔴","🟠","⚪","🟡","🟢"]
    OUTPUT_PATH = f'../data/ml/classification/SGD/price_classifier/{FIRST_MODEL_NAME[0]}/train_result'
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    IDS_TRAIN_PATH = f'../data/ml/dataset/final_input/price_classifier/{FIRST_MODEL_NAME[0]}/{FIRST_MODEL_NAME[1]}_ids_train_classifier_2{FILTERED_SUFFIX}.pkl'
    IDS_TEST_PATH = f'../data/ml/dataset/final_input/price_classifier/{FIRST_MODEL_NAME[0]}/{FIRST_MODEL_NAME[1]}_ids_test_classifier_2{FILTERED_SUFFIX}.pkl'

    with open(IDS_TRAIN_PATH, 'rb') as file:
        ids_train = pickle.load(file).tolist()
    with open(IDS_TEST_PATH, 'rb') as file:
        ids_test = pickle.load(file).tolist()

    ids = ids_train + ids_test

    ids = [i + pt for i, pt in zip(ids, pre_trained)]
    ids = sorted(ids, key=lambda x: (x[0], x[1]))

    csn = ['DOGE', 'PEPE', 'TRUMP']
    model_report += f'=={csn.pop(0)}==\n'
    for i in ids:
        model_report += f'{i[1]} → 預測: {cat[i[2]]}  真實: {cat[i[3]]}  結果: {'✅' if i[3] == i[2] else '❌'}\n'
        if i[1] == '2025-07-31' and csn:
            model_report += f'\n=={csn.pop(0)}==\n'

    with open(f'{OUTPUT_PATH}/{FIRST_MODEL_NAME[1]}_sgd_combined_classifier_2_results{FILTERED_SUFFIX}.txt', 'w', encoding='utf-8-sig') as file:
        file.write(model_report)

    # august_report = ''
    # for csn, pre_august in zip(['DOGE', 'PEPE', 'TRUMP'], pre_august_all):
    #     IDS_AUGUST_PATH = f'../data/ml/dataset/keyword/{csn}_{FIRST_MODEL_NAME}_ids_classifier_2{FILTERED_SUFFIX}_202508.pkl'

    #     with open(IDS_AUGUST_PATH, 'rb') as file:
    #         august_ids = pickle.load(file)
    #     august_ids = [list(i) for i in august_ids]
    #     august_ids = [i + pa for i, pa in zip(august_ids, pre_august)]

    #     august_report += f'=={csn}==\n'
    #     for i in august_ids:
    #         august_report += f'{i[1]} → 預測: {cat[i[2]]}  真實: {cat[i[3]]}  結果: {'✅' if i[3] == i[2] else '❌'}\n'
                
    #     august_report += f'\n{aug_clf_rpt_all.pop(0)}\n\n'

    # with open(f'{OUTPUT_PATH}/{FIRST_MODEL_NAME}_sgd_combined_classifier_2_results{FILTERED_SUFFIX}_202508.txt', 'w', encoding='utf-8-sig') as file:
    #     file.write(august_report)

def main():
    print(f'Using results from {'all' if USE_NON_FILTERED else 'normal'} tweets.')

    print('Loading data...')
    X_train, X_test, Y_train, Y_test = prepare_data()
    pre_trained, model_report, pre_august_all, aug_clf_rpt_all = train_and_evaluate(X_train, X_test, Y_train, Y_test)
    sort_and_save(pre_trained, model_report, pre_august_all, aug_clf_rpt_all)

if __name__ == '__main__':
    main()          
