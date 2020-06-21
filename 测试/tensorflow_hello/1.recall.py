# -*- coding: utf-8 -*-
# @Time : 2020/2/4 12:05 下午
# @Author : 徐缘
# @FileName: 1.recall.py
# @Software: PyCharm


"""
    https://tensorflow.google.cn/api_docs/python/tf     api文档

    1、将数据以numpy.ndarray格式导入python
    2、用keras.Sequential构建序贯的模型。（不同的任务，设置不同的模型）
    3、设置优化函数，损失函数。
    4、设置batch size和epoch并训练
    5、save and load模型

    可能有的问题
    1、过拟合与欠拟合


    .npz文件用np.load读
"""

from __future__ import absolute_import, division, print_function, unicode_literals      # 导入一些熟悉的陌生人
# 绝对引入，精确除法，print，unicode类型字符串。都是为了适配python2，不加也罢

import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model


def version():
    print(tf.version.VERSION)
    print(tf.version.GIT_VERSION)
    return


def mnist_recognize():
    """
    train-images-idx3-ubyte.gz:  training set images (9912422 bytes) 训练样本：共60000个
    train-labels-idx1-ubyte.gz:  training set labels (28881 bytes)
    t10k-images-idx3-ubyte.gz:   test set images (1648877 bytes) 测试样本：共10000个
    t10k-labels-idx1-ubyte.gz:   test set labels (4542 bytes)

    28 x 28 x 60000

    """

    # 我选择放弃，这个傻吊数据集。为啥是gz还要解压多麻烦。
    with np.load('mnist.npz', allow_pickle=True) as f:
        x_train, y_train = f['x_train'], f['y_train']   # ndarray 就numpy 的数据类型
        x_test, y_test = f['x_test'], f['y_test']
    print(type(x_train))
    # print(x_train[0, :])    # 查看第一条记录
    x_train, x_test = x_train / 255.0, x_test / 255.0
    # print(x_train[0, :])

    # 步骤就是生成一个model类的对象。选上损失、优化函数。训练。
    # 输入 + 全连接 + Dropout + Softmax
    model = tf.keras.Sequential([   # alias tf.keras.models.Sequential Linear stack of layers.
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Configures the model for training
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Trains the model for a fixed number of epochs (iterations on a dataset).
    model.fit(x_train, y_train, epochs=5)

    # Returns the loss value & metrics values for the model in test mode.
    # Computation is done in batches.
    model.evaluate(x_test, y_test, verbose=2)

    return


def mnist_recognize_advanced():
    """
    CNN
    废物电脑 CNN 就跑成这样了
    """
    batch_size = 32
    # tf.keras.backend.set_floatx('float64')    # 将模型dtype改成float64

    class MyModel(Model):
        """
        子类化使用。更加高级
        """
        def __init__(self):
            super(MyModel, self).__init__()
            self.conv1 = Conv2D(32, 3, activation='relu')   # 二维卷积
            self.flatten = Flatten()
            self.d1 = Dense(128, activation='relu')
            self.d2 = Dense(10, activation='softmax')

        def call(self, inputs, training=None, mask=None):
            inputs = self.conv1(inputs)
            inputs = self.flatten(inputs)
            inputs = self.d1(inputs)
            return self.d2(inputs)

    with np.load('mnist.npz', allow_pickle=True) as f:
        x_train, y_train = f['x_train'], f['y_train']   # ndarray 就numpy 的数据类型
        x_test, y_test = f['x_test'], f['y_test']
    x_train, x_test = x_train / 255.0, x_test / 255.0
    # print(x_train.dtype)      # float64
    x_train = x_train.astype(np.float32)
    y_train = y_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    y_test = y_test.astype(np.float32)

    # Add a channels dimension
    x_train = x_train[..., tf.newaxis]  # np 还是蛮神奇的
    x_test = x_test[..., tf.newaxis]
    # print(x_train.shape, y_train.shape )    # (60000, 28, 28, 1)

    # Data: API for input pipelines.
    # 从前一万个开始随机抽，每抽一个往里面补一个
    # 每个batch 32个记录
    # Allocation 报的内存是Dataset的大小。网上的人和我讲你马的batch size 呢
    # 把样本和标签放一起大小大了4倍。（32，28, 28, 1)， （32，）
    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(10000).batch(32)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

    # print(len(list(train_ds.as_numpy_iterator())))  # 1875： 1875 * 32 = 60000
    # for images, labels in train_ds:     # 生成一个Iterator。这么看比较合适。或者转换成list
    #     print(images.shape, labels.shape)


    # 生成实例
    model = MyModel()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    # 将python的函数编译成tensorflow的图结构
    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:  # 梯度带Record operations for automatic differentiation.
            predictions = model(images)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)

    @tf.function
    def test_step(images, labels):
        predictions = model(images)
        t_loss = loss_object(labels, predictions)

        test_loss(t_loss)
        test_accuracy(labels, predictions)

    EPOCHS = 5

    for epoch in range(EPOCHS):
        # 在下一个epoch开始时，重置评估指标
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        for images, labels in train_ds:
            train_step(images, labels)

        for test_images, test_labels in test_ds:
            test_step(test_images, test_labels)

        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(epoch + 1,
                              train_loss.result(),
                              train_accuracy.result() * 100,
                              test_loss.result(),
                              test_accuracy.result() * 100))
    return


if __name__ == '__main__':
    version()
    # mnist_recognize_advanced()




