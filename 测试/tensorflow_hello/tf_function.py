# -*- coding: utf-8 -*-
# @Time : 2020/2/16 9:40 下午
# @Author : 徐缘
# @FileName: tf_function.py
# @Software: PyCharm


from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np

import tensorflow as tf


def hello():
    # The tf.function decorator     annotate a function with tf.function
    @tf.function
    def simple_nn_layer(x, y):
        return tf.nn.relu(tf.matmul(x, y))

    x = tf.random.uniform((3, 3))
    y = tf.random.uniform((3, 3))
    print(x)                        #
    print(simple_nn_layer(x, y))    # 它的输出是一个张量
    print(simple_nn_layer)          # <tensorflow.python.eager.def_function.Function object at 0x107de42e8>


def conv_timeit():
    """
    Eager conv:    1.4411156220012344
    Function conv: 1.101218842988601
    Note how there's not much difference in performance for convolutions

    :return:
    """
    import timeit
    conv_layer = tf.keras.layers.Conv2D(100, 3)

    @tf.function
    def conv_fn(image):
        return conv_layer(image)

    image = tf.zeros([1, 200, 200, 100])
    # warm up
    conv_layer(image);
    conv_fn(image)
    print("Eager conv:", timeit.timeit(lambda: conv_layer(image), number=10))
    print("Function conv:", timeit.timeit(lambda: conv_fn(image), number=10))
    print("Note how there's not much difference in performance for convolutions")


def lstm_timeit():
    """
    eager lstm:    0.00813881799695082
    function lstm: 0.0066553520009620115
    :return:
    """
    import timeit
    lstm_cell = tf.keras.layers.LSTMCell(10)

    @tf.function
    def lstm_fn(input, state):
        return lstm_cell(input, state)

    input = tf.zeros([10, 10])
    state = [tf.zeros([10, 10])] * 2
    # warm up
    lstm_cell(input, state)
    lstm_fn(input, state)
    print("eager lstm:", timeit.timeit(lambda: lstm_cell(input, state), number=10))
    print("function lstm:", timeit.timeit(lambda: lstm_fn(input, state), number=10))


if __name__ == '__main__':
    # hello()
    # conv_timeit()
    lstm_timeit()


