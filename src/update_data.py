#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import json
from sympy import *
from itertools import product

x, y = symbols("x y")
MIN_VAR, MAX_VAR, STEP_VAR = -100, 100, 0.1
JSON_FILENAME = 'assets/function_datas.json'


def near_eq(f1, f2):
    return f1 - f2 < 0.1 ** 3


def to(from_, to_, step):
    now = from_
    while (now < to_ or near_eq(now, to_)):
        yield now
        now += step
    yield now


def get_diffeo(t):
        return t[0] * x + t[1] * y


def update_data():
    seed_functions = (
        (lambda x_, y_: x_ * x_ - y_ * y_, 0),
        (lambda x_, y_: x_ * x_ + y_ * y_, 0),
    )
    ts = list(product(np.arange(MIN_VAR, MAX_VAR, STEP_VAR), repeat=2))
    diffeos = list()
    for t in ts:
        diffeos.append(get_diffeo(t))

    datas = []
    for function in seed_functions:
        func = function[0]
        for phi1, t1 in zip(diffeos, ts):
            for phi2, t2 in zip(diffeos, ts):
                if (t1[0] * t2[1] - t1[1] * t2[0] == 0):
                    continue
                updated_func = expand(func(phi1, phi2))

                datas.append({
                    "seed_function": str(func(x, y)),
                    "phi1": str(phi1),
                    "phi2": str(phi2),
                    "updated_function": str(updated_func)
                })

    with open(JSON_FILENAME, 'w') as f:
        json.dump(datas, f, indent=4)


def main():
    update_data()


if __name__ == '__main__':
    update_data()
