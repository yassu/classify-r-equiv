#!/usr/bin/env python
# -*- coding: utf-8 -*-
from invoke import task
from classify_r_equiv import update_data


@task
def update(ctx):
    """ 入力のデータを作成/更新する """
    update_data.main()


@task
def learning(ctx):
    """ 学習する """


@task
def test(ctx):
    """ テストする """


@task
def run(ctx):
    """ 学習後, テストする """
