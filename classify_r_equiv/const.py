#!/usr/bin/env python
# -*- coding: utf-8 -*-


SEED_FUNCTIONS = (
    # (function, r-codimension, function-id)
    (lambda x, y: x * x - y * y, 0, 0),
    (lambda x, y: x * x + y * y, 0, 1),
    (lambda x, y: x * x * x - x * y * y, 3, 2),
    (lambda x, y: x * x * x + y * y * y, 3, 3),
    (lambda x, y: x * x * y + y * y * y * y, 4, 4),
    (lambda x, y: - x * x * y - y * y * y * y, 4, 5),
)

