#!/usr/bin/env python
# -*- coding: utf-8 -*-
from invoke import task


@task
def update(ctx):
    """ 入力のデータを作成/更新する """
    ctx.run('python {}'.format('src/update_data.py'))


@task
def learning(ctx):
    """ 学習する """


@task
def test(ctx):
    """ テストする """


@task
def run(ctx):
    """ 学習後, テストする """
