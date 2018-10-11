#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ilinq.ilinq import Linq
from itertools import chain


_SEED_FUNCTIONS = {
    0: (
        # (function, r-codimension, function-id)
        (lambda x, y: x * x - y * y, 0, 0),
        (lambda x, y: x * x + y * y, 0, 1),
        (lambda x, y: x * x * x - x * y * y, 3, 2),
        (lambda x, y: x * x * x + y * y * y, 3, 3),
        (lambda x, y: x * x * y + y * y * y * y, 4, 4),
        (lambda x, y: - x * x * y - y * y * y * y, 4, 5),
    ),
    1: (
        # unstable functions
        (lambda x, y: x * y * (x - y) * (x - 0.5 * y), None, 6),
        (lambda x, y: x * y * (x - y) * (x - 0.51 * y), None, 7),
    ),
    2: (
        (lambda x, y: x * x * x, None, 8),
        (lambda x, y: x * x * x + 0.01 * y * y * y * y * y, None, 9)
    )
}

def get_seed_functions(difficulty):
    return list(chain.from_iterable(
        Linq(_SEED_FUNCTIONS.items())
            .where(lambda t: t[0] <= difficulty)
            .select(lambda t: t[1])
    ))

if __name__ == '__main__':
    print(get_seed_functions(1))
    print(len(get_seed_functions(1)))
    print(len(get_seed_functions(2)))
