#!/usr/bin/env python
# -*- coding: utf-8 -*-
from invoke import task
from classify_r_equiv import update_data


DEFAULT_MAX_DEG = 10


@task
def update(ctx, max_deg=DEFAULT_MAX_DEG):
    """ 入力のデータを作成/更新する """
    update_data.main(max_deg=int(max_deg))


@task
def learning(ctx):
    """ 学習する """


@task
def test(ctx):
    """ テストする """


@task
def run(ctx):
    """ 学習後, テストする """
