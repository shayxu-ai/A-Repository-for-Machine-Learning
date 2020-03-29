# -*- coding: utf-8 -*-
# @Time: 2020/3/25,025 14:38
# @Last Update: 2020/3/25,025 14:38
# @Author: 徐缘
# @FileName: NNLM-Tensor.py.py
# @Software: PyCharm


"""
    https://github.com/graykode/nlp-tutorial
    Neural Network Language Model
    酷诶
    很奇怪的问题是，我公司里的tf是v1.0吗？我记得我看过是v2.0的啊

"""

import tensorflow as tf
import numpy as np

# 清除默认图形堆栈并重置全局默认图形
# tf.compat.v1.reset_default_graph()

sentences = ["i like dog", "i love coffee", "i hate milk"]

word_list = " ".join(sentences).split()
word_list = list(set(word_list))
# 以词为键值
word_dict = {w: i for i, w in enumerate(word_list)}
# 以序号为键值
number_dict = {i: w for i, w in enumerate(word_list)}

n_class = len(word_dict)    # number of Vocabulary

# NNLM Parameter
n_step = 2  # number of steps ['i like', 'i love', 'i hate']
n_hidden = 2    # number of hidden units

# 准备训练数据
input_batch = list()
target_batch = list()
for sen in sentences:
    word = sen.split()  #   [i, like, dog]
    input_ = [word_dict[n] for n in word[:-1]]
    target = word_dict[word[-1]]

    input_batch.append(np.eye(n_class)[input_])
    target_batch.append(np.eye(n_class)[target])

# Model
X = tf.placeholder(tf.float32, [None, n_step, n_class]) # [batch_size, number of steps, number of Vocabulary]
Y = tf.placeholder(tf.float32, [None, n_class])

input = tf.reshape(X, shape=[-1, n_step * n_class]) # [batch_size, n_step * n_class]
H = tf.Variable(tf.random_normal([n_step * n_class, n_hidden]))
d = tf.Variable(tf.random_normal([n_hidden]))
U = tf.Variable(tf.random_normal([n_hidden, n_class]))
b = tf.Variable(tf.random_normal([n_class]))

tanh = tf.nn.tanh(d + tf.matmul(input, H))  # [batch_size, n_hidden]
model = tf.matmul(tanh, U) + b  # [batch_size, n_class]

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
prediction = tf.argmax(model, 1)

# Training
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


for epoch in range(5000):
    _, loss = sess.run([optimizer, cost], feed_dict={X: input_batch, Y: target_batch})
    if (epoch + 1)%1000 == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

# Predict
predict = sess.run([prediction], feed_dict={X: input_batch})

# Test
input = [sen.split()[:2] for sen in sentences]
print([sen.split()[:2] for sen in sentences], '->', [number_dict[n] for n in predict[0]])
