#!/usr/bin/env python
# -*- coding: utf-8 -*-


from tqdm import tqdm
import json
import numpy as np
from sklearn.model_selection import train_test_split


def load_from_json(json_filename, train_size):
    print('Start to load json file')
    with open(json_filename) as f:
        json_datas = json.load(f)
    print('Finish to load json file')

    print('Start to compute data')
    xs = []
    ys = np.array([], dtype=np.int64)
    for json_data in json_datas[:10]:
        xs.append(np.array(json_data['function_coeffs']))
        ys = np.append(ys, json_data['function_type'])
    xs = np.asarray(xs)
    ys = np.eye(10)[ys.astype(int)]  # 1-of-K 表現に変換
    return train_test_split(xs, ys, train_size=train_size)


def main():
    json_filename = 'assets/input_function_datas.json'
    train_size=0.8
    X_train, X_test, Y_train, Y_test =\
        load_from_json(json_filename=json_filename, train_size=0.8)
    print(X_train[:10])
