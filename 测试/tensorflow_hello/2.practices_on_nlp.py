# -*- coding: utf-8 -*-
# @Time: 2020/2/5,005 22:02
# @Last Update: 2020/2/5,005 22:02
# @Author: 徐缘
# @FileName: 2.practices_on_nlp.py
# @Software: PyCharm


from __future__ import absolute_import, division, print_function, unicode_literals      # 导入一些熟悉的陌生人
# 绝对引入，精确除法，print，unicode类型字符串。都是为了适配python2，不加也罢

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
from tensorflow import keras


import tensorflow_hub as hub    # 模型库
import tensorflow_datasets as tfds  # 数据|库 https://tensorflow.google.cn/datasets/api_docs/python/tfds?hl=en
tfds.disable_progress_bar()


def version():
    """
    国际惯例，先看下版本
    """
    print("Eager mode: ", tf.executing_eagerly())
    print("Hub version: ", hub.__version__)
    print("tfds version", tfds.__version__)
    print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")


def tf_hub_hello():
    """
    预训练word2vector(迁移学习) + 全连接层
    loss: 0.329
    accuracy: 0.858 我记得 cnn 文本分类可以有95%呢

    """
    train_data, validation_data, test_data = tfds.load(
        name="imdb_reviews", split=('train[:60%]', 'train[60%:]', 'test'),
        as_supervised=True)
    train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))
    print(train_examples_batch)
    print(train_labels_batch)

    embedding = "https://hub.tensorflow.google.cn/google/tf2-preview/gnews-swivel-20dim/1"
    hub_layer = hub.KerasLayer(embedding, input_shape=[],
                               dtype=tf.string, trainable=True)
    print(hub_layer(train_examples_batch[:3]))

    model = tf.keras.Sequential()
    model.add(hub_layer)
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    # model.summary()

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(train_data.shuffle(10000).batch(512),
                        epochs=20,
                        validation_data=validation_data.batch(512),
                        verbose=1)

    results = model.evaluate(test_data.batch(512), verbose=2)

    for name, value in zip(model.metrics_names, results):
        print("%s: %.3f" % (name, value))


def preprocess_text():
    """


    """
    (train_data, test_data), info = tfds.load(
        # Use the version pre-encoded with an ~8k vocabulary.
        'imdb_reviews/subwords8k',
        # Return the train/test datasets as a tuple.
        split=(tfds.Split.TRAIN, tfds.Split.TEST),
        # Return (example, label) pairs from the dataset (instead of a dictionary).
        as_supervised=True,
        # Also return the `info` structure.
        with_info=True)

    encoder = info.features['text'].encoder
    print('Vocabulary size: {}'.format(encoder.vocab_size))

    sample_string = 'Hello TensorFlow.'

    encoded_string = encoder.encode(sample_string)
    print('Encoded string is {}'.format(encoded_string))

    original_string = encoder.decode(encoded_string)
    print('The original string: "{}"'.format(original_string))

    assert original_string == sample_string

    for ts in encoded_string:
        print('{} ----> {}'.format(ts, encoder.decode([ts])))

    for train_example, train_label in train_data.take(1):
        print('Encoded text:', train_example[:10].numpy())
        print('Label:', train_label.numpy())

    encoder.decode(train_example)

    BUFFER_SIZE = 1000

    train_batches = (
        train_data
            .shuffle(BUFFER_SIZE)
            .padded_batch(32, train_data.output_shapes))

    test_batches = (
        test_data
            .padded_batch(32, train_data.output_shapes))

    for example_batch, label_batch in train_batches.take(2):
        print("Batch shape:", example_batch.shape)
        print("label shape:", label_batch.shape)

    model = keras.Sequential([
        keras.layers.Embedding(encoder.vocab_size, 16),
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dense(1, activation='sigmoid')])

    model.summary()

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(train_batches,
                        epochs=10,
                        validation_data=test_batches,
                        validation_steps=30)

    loss, accuracy = model.evaluate(test_batches)

    print("Loss: ", loss)
    print("Accuracy: ", accuracy)

    history_dict = history.history
    history_dict.keys()

    import matplotlib.pyplot as plt

    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(acc) + 1)

    # "bo" is for "blue dot"
    plt.plot(epochs, loss, 'bo', label='Training loss')
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

    plt.clf()  # clear figure

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

    plt.show()
    return


if __name__ == '__main__':
    # version()
    preprocess_text()


