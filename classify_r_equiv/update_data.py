#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import json
from classify_r_equiv.const import get_seed_functions
from tqdm import tqdm
from sympy import *
import random
from itertools import product

x, y = symbols("x y")


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

def get_ts(numbers, number_of_samples):
    tmat = list()
    for i in range(14):
        ts = np.array(numbers)
        np.random.shuffle(ts)
        tmat.append(ts)
    return zip(*tmat)

def get_diffeo(t):
        return t[0] * x + t[1] * y + (
            t[2] * x * x + t[3] * x * y + t[4] * y * y +
            t[5] * x ** 3 + t[6] * x ** 2 * y + t[7] * x * y ** 2
                + t[8] * y ** 3 +
            t[9] * x ** 4 + t[10] * x ** 3 * y + t[11] * x ** 2 * y ** 2 +
                t[12] * x * y ** 3 + t[13] * y ** 4)


def get_function_infos(diffeosWithTs, difficulty, number_of_samples):
    yielded_keys = list()
    cnt = 0
    seed_functions = get_seed_functions(difficulty)
    while True:
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

        cnt += 1
        if (cnt == number_of_samples):
            return


def update_data(
        max_deg,
        min_var,
        max_var,
        step_var,
        difficulty,
        json_filename,
        number_of_samples,):
    coeff_keys = [
        x ** (k - i) * y ** i
        for k in range(1, max_deg + 1) for i in range(k + 1)]
    print('Compute diffeos')
    numbers = np.arange(min_var, max_var, step_var)
    diffeos = list()
    ts = list(get_ts(numbers, number_of_samples))
    with tqdm(total=len(ts)) as pbar:
        for t in ts:
            diffeos.append(get_diffeo(t))
            pbar.update(1)
    print('Finish to compute diffeos')

    print('Compute datas')
    datas = []
    function_infos = get_function_infos(
            list(zip(diffeos, ts)),
            difficulty,
            number_of_samples)
    with tqdm(total=number_of_samples) as pbar:
        for function, (phi1, t1), (phi2, t2) in function_infos:
            func = function[0]
            updated_func = expand(func(phi1, phi2))
            datas.append({
                "seed_function": str(func(x, y)),
                "function_type": function[2],
                "t1": t1,
                "t2": t2,
                "function": str(updated_func),
                "function_coeffs": [
                    float(updated_func.coeff(coeff_key).subs([(x, 0), (y, 0)]))
                    for coeff_key in coeff_keys],
            })
            pbar.update(1)
    print('Finish to compute datas')

    with open(json_filename, 'w') as f:
        json.dump(datas, f, indent=4, cls=MyEncoder)


def main(
    max_deg,
    min_var,
    max_var,
    step_var,
    json_filename,
    difficulty,
    number_of_samples,):
    update_data(
        max_deg=max_deg,
        min_var=min_var,
        max_var=max_var,
        step_var=step_var,
        difficulty=difficulty,
        number_of_samples=number_of_samples,
        json_filename=json_filename,)
