#!/usr/bin/env python
# -*- coding: utf-8 -*-


SEED_FUNCTIONS = (
    # (function, r-codimension, function-id)
    (lambda x_, y_: x_ * x_ - y_ * y_, 0, 0),
    (lambda x_, y_: x_ * x_ + y_ * y_, 0, 1),
    (lambda x_, y_: x_ * x_ * x_ - x_ * y_ * y_, 3, 2),
    (lambda x_, y_: x_ * x_ * x_ + y_ * y_ * y_, 3, 3),
    (lambda x_, y_: x_ * x_ * y_ + y_ * y_ * y_ * y_, 4, 4),
    (lambda x_, y_: - x_ * x_ * y_ - y_ * y_ * y_ * y_, 4, 5),
)

