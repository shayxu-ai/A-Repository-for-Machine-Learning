# -*- coding: utf-8 -*-
# @Time : 2020/6/11 1:31 上午
# @Author : 徐缘
# @FileName: normalization.py
# @Software: PyCharm


"""

@author: liushuchun
"""
import re
import string
import jieba

# 加载停用词
with open("dict/stop_words.utf8", encoding="utf8") as f:
    stopword_list = f.readlines()


def tokenize_text(text):
    tokens = jieba.lcut(text)
    tokens = [token.strip() for token in tokens]
    return tokens


def remove_special_characters(text):
    tokens = tokenize_text(text)
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    filtered_tokens = filter(None, [pattern.sub('', token) for token in tokens])
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text


def remove_stopwords(text):
    tokens = tokenize_text(text)
    filtered_tokens = [token for token in tokens if token not in stopword_list]
    filtered_text = ''.join(filtered_tokens)
    return filtered_text


def normalize_corpus(corpus):
    normalized_corpus = []
    for text in corpus:

        text = " ".join(jieba.lcut(text))   # jieba.lcut 直接生成的就是一个list。 cut: 生成器
        normalized_corpus.append(text)

    return normalized_corpus


if __name__ == '__main__':

    import pandas as pd

    book_data = pd.read_csv('data/data.csv')  # 读取文件
    # pandas 真方便啊
    # print(book_data.head())
    book_titles = book_data['title'].tolist()
    book_content = book_data['content'].tolist()
    # print('书名:', book_titles[0])
    # print('内容:', book_content[0][:10])
    print(book_content[0])
    # normalize corpus
    norm_book_content = normalize_corpus(book_content)
    print(norm_book_content[0])
