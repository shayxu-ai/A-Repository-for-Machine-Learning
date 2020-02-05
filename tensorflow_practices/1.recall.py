# -*- coding: utf-8 -*-
# @Time : 2020/2/4 12:05 下午
# @Author : 徐缘
# @FileName: 1.recall.py
# @Software: PyCharm


"""
    第一个最好是看一眼就能全部想起来那种
    https://tensorflow.google.cn/api_docs/python/tf

    其实也没几个模块
"""

import numpy as np
import tensorflow as tf

print(tf.version.VERSION)
print(tf.version.GIT_VERSION)

x_data = np.random.rand(100).astype(np.float32)     # cast to a specified type
y_data = x_data * 0.1 + 0.3

# dtype=tf.dtypes.float32
Weights = tf.Variable(tf.random.uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

y = Weights * x_data + biases

loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.keras.optimizers.SGD(0.5)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
# 创建session，进行参数初始化
sess = tf.Session()
sess.run(init)
# 开始训练200步，每隔20步输出一下两个参数
for step in range(201):
    sess.run(train)
if step % 20 == 0:
    print(step,sess.run(Weights),sess.run(biases))










