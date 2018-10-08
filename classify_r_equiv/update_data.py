#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import json
from classify_r_equiv.const import SEED_FUNCTIONS
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
    t1s = np.array(numbers)
    np.random.shuffle(t1s)
    t2s = np.array(numbers)
    np.random.shuffle(t2s)
    return zip(
            t1s[:number_of_samples],
            t2s[:number_of_samples]
        )

def get_diffeo(t):
        return t[0] * x + t[1] * y


def get_function_infos(diffeosWithTs, number_of_samples=None):
    yielded_keys = list()
    for i in range(number_of_samples):
        seed_function = random.choice(SEED_FUNCTIONS)
        phi1, t1 = random.choice(diffeosWithTs)
        phi2, t2 = random.choice(diffeosWithTs)
        if (t1[0] * t2[1] - t1[1] * t2[0] == 0):
            continue
        key = str(seed_function) + str(t1) + str(t2)
        if key in yielded_keys:
            continue
        yielded_keys.append(key)
        yield (seed_function, (phi1, t1), (phi2, t2))


def update_data(
        max_deg=None,
        min_var=None,
        max_var=None,
        step_var=None,
        json_filename=None,
        number_of_samples=None,):
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
    max_deg=None,
    min_var=None,
    max_var=None,
    step_var=None,
    json_filename=None,
    number_of_samples=None,):
    update_data(
        max_deg=max_deg,
        min_var=min_var,
        max_var=max_var,
        step_var=step_var,
        number_of_samples=number_of_samples,
        json_filename=json_filename,)
