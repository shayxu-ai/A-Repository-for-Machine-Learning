# -*- coding: utf-8 -*-
# @Time : 2020/2/16 10:50 下午
# @Author : 徐缘
# @FileName: tf_distribute_strategy.py
# @Software: PyCharm


"""
    https://tensorflow.google.cn/api_docs/python/tf/distribute/experimental/ParameterServerStrategy
"""

from __future__ import absolute_import, division, print_function, unicode_literals      # 导入一些熟悉的陌生人
# 绝对引入，精确除法，print，unicode类型字符串。都是为了适配python2，不加也罢

import sys
import os
import json

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model


with np.load('mnist.npz', allow_pickle=True) as f:
    x_train, y_train = f['x_train'], f['y_train']  # ndarray 就numpy 的数据类型
    x_test, y_test = f['x_test'], f['y_test']

x_train, x_test = x_train / 255.0, x_test / 255.0


def make_dataset(images, labels, epochs, batch_size, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if shuffle:
        dataset = dataset.shuffle(10000)
    dataset = dataset.repeat(epochs).batch(batch_size).prefetch(50)
    return dataset


# 输入 + 全连接 + Dropout + Softmax
os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ["localhost:12345", "localhost:23456"]
    },
    'tasks': {'type': 'worker', 'index': 0}
})
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

NUM_WORKERS = 2
GLOBAL_BATCH_SIZE = 64 * NUM_WORKERS

with strategy.scope():
    # Creation of dataset, and model building/compiling need to be within `strategy.scope()`.
    train_dataset = make_dataset(x_train, x_test, 5, GLOBAL_BATCH_SIZE)

    model = tf.keras.Sequential([  # alias tf.keras.models.Sequential Linear stack of layers.
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Configures the model for training
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])


# Keras' `model.fit()` trains the model with specified number of epochs and
# number of steps per epoch. Note that the numbers here are for demonstration
# purposes only and may not sufficiently produce a model with good quality.
model.fit(train_dataset, epochs=1, steps_per_epoch=5)
