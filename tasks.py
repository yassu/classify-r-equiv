#!/usr/bin/env python
# -*- coding: utf-8 -*-
from invoke import task
from classify_r_equiv import (
    update_data as _update_data,
    learn_and_test as _learn_and_test)


MAX_DEG = 10
MIN_VAR, MAX_VAR, STEP_VAR = -100.0, 100.0, 0.01
TRAIN_RATE = 0.8
EPOCHS = 700
N_HIDDEN = 200
JSON_FILENAME = 'assets/input_function_datas.json'
NUMBER_OF_SAMPLES = 10 ** 6


@task(
    help={
        'json-filename': 'output data to json-filename',
        'number-of-samples': 'number of made samples',
        'min-var': 'consider coefficient of function >= min-var',
        'max-var': 'consider coefficient of function <= max-var',
        'max-deg': 'consider degree of max <= max-deg',
        'step-var': 'consider list from min-var to max-var by step-var step',
        }
)
def update(
        ctx,
        max_deg=MAX_DEG,
        min_var=MIN_VAR,
        max_var=MAX_VAR,
        step_var=STEP_VAR,
        number_of_samples=NUMBER_OF_SAMPLES,
        difficulty=0,
        json_filename=JSON_FILENAME,):
    """ 入力のデータを作成/更新する """
    _update_data.main(
        max_deg=int(max_deg),
        min_var=float(min_var),
        max_var=float(max_var),
        step_var=float(step_var),
        difficulty=int(difficulty),
        number_of_samples=number_of_samples,
        json_filename=json_filename,)


@task
def learn_and_test(
        ctx,
        train_rate=TRAIN_RATE,
        epochs=EPOCHS,
        n_hidden=N_HIDDEN,
        difficulty=0,
        json_filename=JSON_FILENAME,):
    """ 学習する """
    _learn_and_test.main(
        train_rate=float(train_rate),
        epochs=int(epochs),
        n_hidden=int(n_hidden),
        difficulty=int(difficulty),
        json_filename=str(json_filename)
    )


@task
def test(ctx):
    """ テストする """


@task
def run(ctx):
    """ 学習後, テストする """
