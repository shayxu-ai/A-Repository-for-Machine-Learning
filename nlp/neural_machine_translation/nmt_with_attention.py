# -*- coding: utf-8 -*-
# @Time : 2020/6/15 12:13 上午
# @Author : 徐缘
# @FileName: nmt_with_attention.py
# @Software: PyCharm


"""
    https://tensorflow.google.cn/tutorials/text/nmt_with_attention
"""

import tensorflow as tf

import matplotlib.pyplot as plt  # 作图
import matplotlib.ticker as ticker  # 坐标轴上的刻度
from sklearn.model_selection import train_test_split  # 把数据集分成训练集和测试集

import unicodedata
import re
import numpy as np
import os
import io
import time
from matplotlib.font_manager import FontProperties


# 将 unicode 文件转换为 ascii
def unicode_to_ascii(s):
    # https://zhuanlan.zhihu.com/p/93029007
    # 通过unicodedata把字符分门别类。去除不要的字符
    # [Mn] Mark, Nonspacing。换行符这种吧 \n \t \r\n
    # normalize 将形状相同，但编码不同的字符，进行归一化
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


# 给每个句子添加一个<start>和一个<end>
def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())

    # 在单词与跟在其后的标点符号之间插入一个空格
    # 例如： "he is a boy." => "he is a boy ."
    # 参考：https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,¿\u4e00-\u9fa5])", r" \1 ", w)
    w = re.sub(r'[" ]+', " ", w)

    # 除了 (a-z, A-Z, ".", "?", "!", ",")，将所有字符替换为空格
    # w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

    w = w.strip()

    # 给句子加上开始和结束标记
    # 以便模型知道何时开始和结束预测
    w = '<start> ' + w + ' <end>'
    return w


# 1. 去除重音符号
# 2. 清理句子
# 3. 返回这样格式的单词对：[ENGLISH, SPANISH]
def create_dataset(path, num_examples):
    lines = open(path, encoding='UTF-8').read().strip().split('\n')

    # 新的语料每行后面加了贡献者的信息。所以只取前两行
    word_pairs = [[preprocess_sentence(w) for w in line.split('\t')[0:2]] for line in lines[:num_examples]]

    return list(zip(*word_pairs))


def max_length(tensor):
    return max(len(t) for t in tensor)


def tokenize(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(  # Text tokenization utility class.
        filters='')
    lang_tokenizer.fit_on_texts(lang)

    tensor = lang_tokenizer.texts_to_sequences(lang)  # 给每个字标序号，独热码
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post', maxlen=None)  # 补到和最长的一样长

    return tensor, lang_tokenizer


def load_dataset(path, num_examples=None):
    # 创建清理过的输入输出对
    targ_lang, inp_lang = create_dataset(path, num_examples)

    input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
    target_tensor, targ_lang_tokenizer = tokenize(targ_lang)

    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer


def convert(lang, tensor):
    for t in tensor:
        if t != 0:
            print("%d ----> %s" % (t, lang.index_word[t]))


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    # 直接添加参数重写父类方法, 虽然也能正常使用，但是在pycharm里有警告提示：
    #  def call(self, inputs, training=None, mask=None)
    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    # ???
    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))


# https://tensorflow.google.cn/tutorials/text/nmt_with_attention#write_the_encoder_and_decoder_model
class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # query hidden state shape == (batch_size, hidden size)
        # query_with_time_axis shape == (batch_size, 1, hidden size)
        # values shape == (batch_size, max_len, hidden size)
        # we are doing this to broadcast addition along the time axis to calculate the score
        query_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)

        # 用于注意力
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        # 编码器输出 （enc_output） 的形状 == （批大小，最大长度，隐藏层大小）
        context_vector, attention_weights = self.attention(hidden, enc_output)

        # x 在通过嵌入层后的形状 == （批大小，1，嵌入维度）
        x = self.embedding(x)

        # x 在拼接 （concatenation） 后的形状 == （批大小，1，嵌入维度 + 隐藏层大小）
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # 将合并后的向量传送到 GRU
        output, state = self.gru(x)

        # 输出的形状 == （批大小 * 1，隐藏层大小）
        output = tf.reshape(output, (-1, output.shape[2]))

        # 输出的形状 == （批大小，vocab）
        x = self.fc(output)

        return x, state, attention_weights


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


@tf.function
def train_step(inp, targ, enc_hidden):
    loss = 0

    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)

        dec_hidden = enc_hidden

        dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * BATCH_SIZE, 1)

        # 教师强制 - 将目标词作为下一个输入
        for t in range(1, targ.shape[1]):
            # 将编码器输出 （enc_output） 传送至解码器
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

            loss += loss_function(targ[:, t], predictions)

            # 使用教师强制
            dec_input = tf.expand_dims(targ[:, t], 1)

    batch_loss = (loss / int(targ.shape[1]))

    variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, variables)

    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss


def train(EPOCHS):
    for epoch in range(EPOCHS):
        start = time.time()

        enc_hidden = encoder.initialize_hidden_state()
        total_loss = 0

        for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
            # print(inp, enc_hidden)

            batch_loss = train_step(inp, targ, enc_hidden)
            # print(encoder.summary())

            total_loss += batch_loss

            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, batch_loss.numpy()))
        # 每 2 个周期（epoch），保存（检查点）一次模型
        if (epoch + 1) % 2 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / steps_per_epoch))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


def evaluate(sentence):
    attention_plot = np.zeros((max_length_targ, max_length_inp))

    sentence = preprocess_sentence(sentence)

    inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                           maxlen=max_length_inp,
                                                           padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''

    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)

    for t in range(max_length_targ):
        predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)

        # storing the attention weights to plot later on
        attention_weights = tf.reshape(attention_weights, (-1,))
        attention_plot[t] = attention_weights.numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()

        result += targ_lang.index_word[predicted_id] + ' '

        if targ_lang.index_word[predicted_id] == '<end>':
            return result, sentence, attention_plot

        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence, attention_plot


# function for plotting the attention weights
def plot_attention(attention, sentence, predicted_sentence):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')

    fontdict = {'fontsize': 14}

    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def translate(sentence):
    result, sentence, attention_plot = evaluate(sentence)

    print('Input: %s' % (sentence))
    print('Predicted translation: {}'.format(result))

    attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
    plot_attention(attention_plot, sentence.split(' '), result.split(' '))


if __name__ == '__main__':
    # May I borrow this book? ¿Puedo tomar prestado este libro?
    path_to_file = "./spa-eng/spa.txt"  # http://www.manythings.org/anki/
    path_to_file = "./spa-eng/cmn.txt"

    # 测试预处理
    """
    1、给每个句子添加一个<start>和一个<end>标记（token）。
    2、删除特殊字符以清理句子。
    3、创建一个单词索引和一个反向单词索引（即一个从单词映射至 id 的词典和一个从 id 映射至单词的词典）。
    4、将每个句子填充（pad）到最大长度。
    """

    # 尝试实验不同大小的数据集
    num_examples = 10000
    # inp_lang_tokenizer
    input_tensor, target_tensor, inp_lang, targ_lang = load_dataset(path_to_file, num_examples)

    print("input:", type(input_tensor), input_tensor.shape, input_tensor[0])
    print("target:", type(target_tensor), target_tensor.shape, target_tensor[0])

    # 计算目标张量的最大长度 （max_length）
    max_length_targ, max_length_inp = max_length(target_tensor), max_length(input_tensor)
    print("max length", max_length_targ, max_length_inp )

    # 采用 80 - 20 的比例切分训练集和验证集
    input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor,
                                                                                                    target_tensor,
                                                                                                    test_size=0.2)

    # 显示长度
    print(type(input_tensor_train), len(input_tensor_train), len(target_tensor_train), len(input_tensor_val),
          len(target_tensor_val))

    print("Input Language; index to word mapping")
    convert(inp_lang, input_tensor_train[0])
    print("Target Language; index to word mapping")
    convert(targ_lang, target_tensor_train[0])
    print()

    print("开始训练")  # 模型部分
    # Create a tf.data dataset
    BUFFER_SIZE = len(input_tensor_train)  # 缓存和样本一样。从中随机抽取
    BATCH_SIZE = 32
    steps_per_epoch = len(input_tensor_train) // BATCH_SIZE
    embedding_dim = 64  # 维度有点高
    units = 128  #
    vocab_inp_size = len(inp_lang.word_index) + 1
    vocab_tar_size = len(targ_lang.word_index) + 1
    print("vocab_inp_size:", vocab_inp_size, "vocab_tar_size:", vocab_tar_size)
    dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)

    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)  # 除不尽的部分不要了。
    # example_input_batch, example_target_batch = next(iter(dataset))

    encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
    decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)

    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)

    EPOCHS = 10
    train(EPOCHS)
    font = FontProperties(fname=r"../../../src/simsun.ttc", size=10)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    translate(u'跑。')
    translate(u'我赢了。')
    translate(u'好可爱啊。')
    translate(u'请跟着我。')
    translate(u'请跟着我们。')

