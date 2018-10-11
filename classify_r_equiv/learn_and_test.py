#!/usr/bin/env python
# -*- coding: utf-8 -*-


from tqdm import tqdm
import json
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from classify_r_equiv.const import get_seed_functions
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def load_from_json(json_filename, train_size):
    print('Start to load json file')
    with open(json_filename) as f:
        json_datas = json.load(f)
    print('Finish to load json file')

    print('Start to compute data')
    xs = []
    ys = np.array([], dtype=np.int64)
    number_of_datas = len(json_datas)
    with tqdm(total=number_of_datas) as pbar:
        for json_data in json_datas:
            xs.append(np.array(json_data['function_coeffs']))
            ys = np.append(ys, json_data['function_type'])
            pbar.update(1)
    xs = np.asarray(xs)
    seed_functions = get_seed_functions(0)
    ys = np.eye(len(seed_functions))[ys.astype(int)]
    print('Finish to compute data')
    return train_test_split(xs, ys, train_size=train_size)


def prelu(x, alpha):
    return tf.maximum(tf.zeros(tf.shape(x)), x) \
        + alpha * tf.minimum(tf.zeros(tf.shape(x)), x)


def execute(X_train, X_test, Y_train, Y_test, epochs, n_hidden):
    n_in = len(X_train[0])
    n_out = len(Y_train[0])

    # make model
    x = tf.placeholder(tf.float32, shape=[None, n_in])
    t = tf.placeholder(tf.float32, shape=[None, n_out])
    # 入力層 - 隠れ層
    W0 = tf.Variable(tf.truncated_normal([n_in, n_hidden], stddev=0.01))
    b0 = tf.Variable(tf.zeros([n_hidden]))
    h0 = tf.nn.tanh(tf.matmul(x, W0) + b0)

    # 隠れ層 - 隠れ層
    W1 = tf.Variable(tf.truncated_normal([n_hidden, n_hidden], stddev=0.01))
    b1 = tf.Variable(tf.zeros([n_hidden]))
    h1 = tf.nn.tanh(tf.matmul(h0, W1) + b1)

    W2 = tf.Variable(tf.truncated_normal([n_hidden, n_hidden], stddev=0.01))
    b2 = tf.Variable(tf.zeros([n_hidden]))
    h2 = tf.nn.tanh(tf.matmul(h1, W2) + b2)

    W3 = tf.Variable(tf.truncated_normal([n_hidden, n_hidden], stddev=0.01))
    b3 = tf.Variable(tf.zeros([n_hidden]))
    alpha3 = tf.Variable(tf.zeros([n_hidden]))
    h3 = tf.nn.tanh(tf.matmul(h2, W3) + b3)

    # 隠れ層 - 出力層
    W4 = tf.Variable(tf.truncated_normal([n_hidden, n_out], stddev=0.01))
    b4 = tf.Variable(tf.zeros([n_out]))
    y = tf.nn.softmax(tf.matmul(h3, W4) + b4)

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(t * tf.log(y), axis=1))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(t, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    '''
    モデル学習
    '''

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # TODO: n_batchesの定義を消す. 合わせて X_ = X_train, Y_=Y_train
    # とする
    n_batches = len(X_train)
    for epoch in range(epochs):
        X_, Y_ = shuffle(X_train, Y_train)

        sess.run(train_step, feed_dict={
            x: X_[:n_batches],
            t: Y_[:n_batches]
        })

        # 訓練データに対する学習の進み具合を出力
        loss = cross_entropy.eval(session=sess, feed_dict={
            x: X_,
            t: Y_
        })
        acc = accuracy.eval(session=sess, feed_dict={
            x: X_,
            t: Y_
        })
        print('epoch:', f'{epoch}/{epochs}', ' loss:', loss, ' accuracy:', acc)

    '''
    予測精度の評価
    '''
    accuracy_rate = accuracy.eval(session=sess, feed_dict={
        x: X_test,
        t: Y_test
    })
    print('accuracy: ', accuracy_rate)


def main(
    train_rate,
    epochs,
    n_hidden,
    json_filename,
    ):
    X_train, X_test, Y_train, Y_test =\
        load_from_json(json_filename=json_filename, train_size=train_rate)
    execute(X_train, X_test, Y_train, Y_test,
        epochs=epochs,
        n_hidden=n_hidden)
