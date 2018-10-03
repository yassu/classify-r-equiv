#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import json
from tqdm import tqdm
from sympy import *
import random
from itertools import product

x, y = symbols("x y")
MIN_VAR, MAX_VAR, STEP_VAR = 0, 3, 1
JSON_FILENAME = 'assets/input_function_datas.json'

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def near_eq(f1, f2):
    return f1 - f2 < 0.1 ** 3


def get_diffeo(t):
        return t[0] * x + t[1] * y

def get_function_infos(seed_functions, diffeosWithTs):
    number_of_samples = 2000000
    yielded_keys = list()
    for _ in range(number_of_samples):
        seed_function = random.choice(seed_functions)
        phi1, t1 = random.choice(diffeosWithTs)
        phi2, t2 = random.choice(diffeosWithTs)
        if (t1[0] * t2[1] - t1[1] * t2[0] == 0):
            continue
        key = str(seed_function) + str(t1) + str(t2)
        if key in yielded_keys:
            continue
        yielded_keys.append(key)
        yield (seed_function, (phi1, t1), (phi2, t2))


def update_data():
    seed_functions = (
        # (function, r-codimension, function-id)
        (lambda x_, y_: x_ * x_ - y_ * y_, 0, 0),
        (lambda x_, y_: x_ * x_ + y_ * y_, 0, 1),
        (lambda x_, y_: x_ * x_ * x_ - x_ * y_ * y_, 3, 2),
        (lambda x_, y_: x_ * x_ * x_ + y_ * y_ * y_, 3, 3),
        (lambda x_, y_: x_ * x_ * y_ + y_ * y_ * y_ * y_, 4, 4),
        (lambda x_, y_: - x_ * x_ * y_ - y_ * y_ * y_ * y_, 4, 5),
    )
    print('Compute ts')
    ts = list(product(np.arange(MIN_VAR, MAX_VAR, STEP_VAR), repeat=2))
    print('Finish to compute ts')
    print('Compute diffeos')
    diffeos = list()
    with tqdm(total=len(ts)) as pbar:
        for t in ts:
            diffeos.append(get_diffeo(t))
            pbar.update(1)
    print('Finish to compute diffeos')

    print('Compute datas')
    datas = []
    function_infos = list(get_function_infos(seed_functions,
        list(zip(diffeos, ts))))
    with tqdm(total=len(function_infos)) as pbar:
        for function, (phi1, t1), (phi2, t2) in function_infos:
            func = function[0]
            updated_func = expand(func(phi1, phi2))
            datas.append({
                "seed_function": str(func(x, y)),
                "function_type": function[2],
                "t1": t1,
                "t2": t2,
                "function": str(updated_func)
            })
            pbar.update(1)
    print('Finish to compute datas')

    with open(JSON_FILENAME, 'w') as f:
        json.dump(datas, f, indent=4, cls=MyEncoder)


def main():
    update_data()


if __name__ == '__main__':
    update_data()
