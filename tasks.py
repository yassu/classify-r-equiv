#!/usr/bin/env python
# -*- coding: utf-8 -*-
from invoke import task
from classify_r_equiv import update_data


MAX_DEG = 10
MIN_VAR, MAX_VAR, STEP_VAR = -100.0, 100.0, 0.01
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
        json_filename=JSON_FILENAME,):
    """ 入力のデータを作成/更新する """
    update_data.main(
        max_deg=int(max_deg),
        min_var=float(min_var),
        max_var=float(max_var),
        step_var=float(step_var),
        number_of_samples=number_of_samples,
        json_filename=json_filename,)


@task
def learning(ctx):
    """ 学習する """


@task
def test(ctx):
    """ テストする """


@task
def run(ctx):
    """ 学習後, テストする """
