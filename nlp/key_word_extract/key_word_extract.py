# -*- coding: utf-8 -*-

import math

import jieba
import jieba.posseg as psg
from gensim import corpora, models
from jieba import analyse
import functools
import re


# 停用词表加载方法
def get_stopword_list():
    # 停用词表存储路径，每一行为一个词，按行读取进行加载
    # 进行编码转换确保匹配准确率
    stop_word_path = './stopword.txt'
    stopword_list = [sw.replace('\n', '') for sw in open(stop_word_path, encoding='utf-8-sig').readlines()]
    stop_word_path = './stopword_custom.txt'
    stopword_custom_list = [sw.replace('\n', '') for sw in open(stop_word_path, encoding='utf-8-sig').readlines()]
    stopword_list.extend(stopword_custom_list)
    # print(stopword_list)
    return stopword_list


# 分词方法，调用结巴接口
def seg_to_list(sentence, pos_flag=False):
    if not pos_flag:
        # 不进行词性标注的分词方法
        seg_list_tmp = jieba.cut(sentence)
    else:
        # 进行词性标注的分词方法
        seg_list_tmp = psg.cut(sentence)

    return seg_list_tmp


# 去除干扰词
def word_filter(seg_list, pos_flag=False):
    stopword_list = get_stopword_list()
    filtered_list_tmp = []
    # 根据POS参数选择是否词性过滤
    # 不进行词性过滤，则将词性都标记为n，表示全部保留
    for seg in seg_list:
        if not pos_flag:
            word = seg
            flag = 'n'
        else:
            word = seg.word
            flag = seg.flag
        if not (flag.startswith('n') or flag.startswith('v')):
            continue
        # 过滤停用词表中的词，以及长度为<2的词
        if word not in stopword_list and len(word) > 1:
            filtered_list_tmp.append(word)

    return filtered_list_tmp


# 数据加载，pos为是否词性标注的参数，corpus_path为数据集路径
def load_data(pos_tmp=False, corpus_path='./complaints.txt'):
    # 调用上面方式对数据集进行处理，处理后的每条数据仅保留非干扰词
    doc_list = []
    for line in open(corpus_path, 'r', encoding='utf-8'):
        content = line.strip()
        seg_list = seg_to_list(content, pos_tmp)
        filter_list = word_filter(seg_list, pos_tmp)
        doc_list.append(filter_list)

    return doc_list


# idf值统计方法
def train_idf(doc_list):
    idf_dic = {}
    # 总文档数
    tt_count = len(doc_list)

    # 每个词出现的文档数
    for doc in doc_list:
        for word in set(doc):
            idf_dic[word] = idf_dic.get(word, 0.0) + 1.0

    # 按公式转换为idf值，分母加1进行平滑处理
    for k, v in idf_dic.items():
        idf_dic[k] = math.log(tt_count / (1.0 + v))

    # 对于没有在字典中的词，默认其仅在一个文档出现，得到默认idf值。 这明明是没出现吧
    default_idf = math.log(tt_count / 1.0)
    return idf_dic, default_idf


#  排序函数，用于topK关键词的按值排序
def cmp(e1, e2):
    import numpy as np
    res = np.sign(e1[1] - e2[1])
    if res != 0:
        return res
    else:
        a = e1[0] + e2[0]
        b = e2[0] + e1[0]
        if a > b:
            return 1
        elif a == b:
            return 0
        else:
            return -1


# TF-IDF类
class TfIdf(object):
    # 四个参数分别是：训练好的idf字典，默认idf值，处理后的待提取文本，关键词数量
    def __init__(self, idf_dic, default_idf, word_list, keyword_num):
        self.word_list = word_list
        self.idf_dic, self.default_idf = idf_dic, default_idf
        self.tf_dic = self.get_tf_dic()
        self.keyword_num = keyword_num

    # 统计tf值
    def get_tf_dic(self):
        tf_dic = {}
        for word in self.word_list:
            tf_dic[word] = tf_dic.get(word, 0.0) + 1.0

        tt_count = len(self.word_list)
        for k, v in tf_dic.items():
            tf_dic[k] = float(v) / tt_count

        return tf_dic

    # 按公式计算tf-idf
    def get_tfidf(self):
        tfidf_dic = {}
        for word in self.word_list:
            idf = self.idf_dic.get(word, self.default_idf)
            tf = self.tf_dic.get(word, 0)

            tfidf = tf * idf
            tfidf_dic[word] = tfidf

        tfidf_dic.items()
        # 根据tf-idf排序，去排名前keyword_num的词作为关键词
        for k, v in sorted(tfidf_dic.items(), key=functools.cmp_to_key(cmp), reverse=True)[:self.keyword_num]:
            print(k + "/ ", end='')
        print()


# 主题模型
class TopicModel(object):
    # 三个传入参数：处理后的数据集，关键词数量，具体模型（LSI、LDA），主题数量
    def __init__(self, doc_list, keyword_num, model='LSI', num_topics=4):
        # 使用gensim的接口，将文本转为向量化表示
        # 先构建词空间
        self.dictionary = corpora.Dictionary(doc_list)
        # 使用BOW模型向量化
        corpus = [self.dictionary.doc2bow(doc) for doc in doc_list]
        # 对每个词，根据tf-idf进行加权，得到加权后的向量表示
        self.tfidf_model = models.TfidfModel(corpus)
        self.corpus_tfidf = self.tfidf_model[corpus]

        self.keyword_num = keyword_num
        self.num_topics = num_topics
        # 选择加载的模型
        if model == 'LSI':
            self.model = self.train_lsi()
        else:
            self.model = self.train_lda()

        # 得到数据集的主题-词分布
        word_dic = self.word_dictionary(doc_list)
        self.wordtopic_dic = self.get_wordtopic(word_dic)
        # print(self.wordtopic_dic)
        # print(self.model.show_topics())
        # [(0, '0.002*"孩子" + 0.002*"运动鞋" + 0.002*"孤儿" + 0.002*"搜

    def train_lsi(self):
        lsi = models.LsiModel(self.corpus_tfidf, id2word=self.dictionary, num_topics=self.num_topics)
        return lsi

    def train_lda(self):
        lda = models.LdaModel(self.corpus_tfidf, id2word=self.dictionary, num_topics=self.num_topics)
        return lda

    def get_wordtopic(self, word_dic):
        wordtopic_dic = {}

        for word in word_dic:
            single_list = [word]
            wordcorpus = self.tfidf_model[self.dictionary.doc2bow(single_list)]
            wordtopic = self.model[wordcorpus]
            wordtopic_dic[word] = wordtopic
        return wordtopic_dic

    # 计算词的分布和文档的分布的相似度，取相似度最高的keyword_num个词作为关键词
    def get_simword(self, word_list):
        sentcorpus = self.tfidf_model[self.dictionary.doc2bow(word_list)]
        senttopic = self.model[sentcorpus]
        # print(senttopic)
        # [(0, 0.0353493), (1, 0.03489105), (2, 0.034628212), (3, 0.89513147)]
        # 余弦相似度计算
        def calsim(l1, l2):
            a, b, c = 0.0, 0.0, 0.0
            for t1, t2 in zip(l1, l2):
                x1 = t1[1]
                x2 = t2[1]
                a += x1 * x1
                b += x1 * x1
                c += x2 * x2
            sim = a / math.sqrt(b * c) if not (b * c) == 0.0 else 0.0
            return sim

        # 计算输入文本和每个词的主题分布相似度
        sim_dic = {}
        for k, v in self.wordtopic_dic.items():
            if k not in word_list:
                continue
            sim = calsim(v, senttopic)
            sim_dic[k] = sim

        for k, v in sorted(sim_dic.items(), key=functools.cmp_to_key(cmp), reverse=True)[:self.keyword_num]:
            print(k + "/ ", end='')
        print()

    # 词空间构建方法和向量化方法，在没有gensim接口时的一般处理方法
    def word_dictionary(self, doc_list):
        dictionary = []
        for doc in doc_list:
            dictionary.extend(doc)

        dictionary = list(set(dictionary))

        return dictionary

    def doc2bowvec(self, word_list):
        vec_list = [1 if word in word_list else 0 for word in self.dictionary]
        return vec_list


def tfidf_extract(word_list, pos=False, keyword_num=10):
    doc_list = load_data(pos)
    idf_dic, default_idf = train_idf(doc_list)
    tfidf_model = TfIdf(idf_dic, default_idf, word_list, keyword_num)
    tfidf_model.get_tfidf()


def textrank_extract(text, pos=False, keyword_num=10):
    textrank = analyse.textrank
    keywords = textrank(text, keyword_num, withWeight=True)
    # 输出抽取出的关键词
    for keyword in keywords:
        print(keyword[0], end="")
        print(str(keyword[1]))
    print()


def topic_extract(word_list, model, pos=False, keyword_num=10):
    doc_list = load_data(pos)
    topic_model = TopicModel(doc_list, keyword_num, model=model)
    topic_model.get_simword(word_list)


if __name__ == '__main__':
    text = '我方在核心TXP上ping测198.13.42.222丢包严重，请集团协查。21:50联系客户，告知最新查证结果是上海移动出口访问国外VPN服务器IP地址丢包严重，解决方案和时间是将此问题上报集团优化网络，预计一周时间内解决，客户认可，并希望尽快收到VPN网络能否正常使用的明确答复'

    jieba.load_userdict('./user_dict.utf8')

    # 对待提取关键字的文章 1、进行分词 2、去除停用词
    pos = True      # 词性过滤

    seg_list = seg_to_list(text, pos)   # 分词
    # 生成器生成了就没了
    # print(list(seg_list))
    filtered_list = word_filter(seg_list, pos)      # 列表
    # print(filtered_list)

    print('TF-IDF模型结果：')
    tfidf_extract(filtered_list)

    # # 长文本其实好用的，反复提及的词权重高。
    print('TextRank模型结果：')
    textrank_extract(text)

    print('LSI模型结果：')
    topic_extract(filtered_list, 'LSI', pos)
    print('LDA模型结果：')
    topic_extract(filtered_list, 'LDA', pos)
