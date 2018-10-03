#!/usr/bin/env python
# -*- coding: utf-8 -*-
from sympy import *
from itertools import product

x, y = symbols("x y")
MIN_X, MAX_X, STEP_X = -100, 100, 0.1
MIN_Y, MAX_Y, STEP_Y = -100, 100, 0.1

def near_eq(f1, f2):
    return f1 - f2 < 0.1 ** 3

def to(from_, to_, step):
    now = from_
    while (now < to_ or near_eq(now, to_)):
        yield now
        now += step
    yield now

def update_data():
    seed_functions = (
        (lambda x_, y_: x_ * x_ - y_ * y_, 0),
        (lambda x_, y_: x_ * x_ + y_ * y_, 0),
    )
    functions = []
    ran = list(range(0, 3 + 1))
    l = list(product(ran, repeat=2))

    for a, b in l:
        phi1 = lambda x_, y_: a * x_ + b * y_
        for c, d in l:
            if (a * d - b * c == 0):
                continue
            phi2 = lambda x_, y_: c * x_ + d * y_
            for function in seed_functions:
                func = function[0]
                updated_func = expand(func(phi1(x, y), phi2(x, y)))
                functions.append((updated_func, function[1]))

    for j in range(10):
        print(functions[j])


def main():
    update_data()


if __name__ == '__main__':
    update_data()
