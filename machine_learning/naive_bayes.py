#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer  # 从sklearn.feature_extraction.text里导入文本特征向量化模块
from sklearn.naive_bayes import MultinomialNB  # 从sklean.naive_bayes里导入朴素贝叶斯模型
# from sklearn.naive_bayes import GaussianNB
# from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import classification_report

X = np.array([
    [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
    [4, 5, 5, 4, 4, 4, 5, 5, 6, 6, 6, 5, 5, 6, 6]
])
X = X.T
# y = np.array([-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1])
y = np.array([1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1])

# In[8]:


# 2.数据预处理：训练集和测试集分割，文本特征向量化
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=33)  # 随机采样25%的数据样本作为测试集
print(X_train)  # 查看训练样本
print(y_train)  # 查看标签

# 文本特征向量化
# vec = CountVectorizer()
# X_train = vec.fit_transform(X_train)
# X_test = vec.transform(X_test)

# 3.使用朴素贝叶斯进行训练
mnb = MultinomialNB()  # 使用默认配置初始化朴素贝叶斯
mnb.fit(X_train, y_train)  # 利用训练数据对模型参数进行估计
y_predict = mnb.predict(X_test)  # 对参数进行预测

# 4.获取结果报告
print('The Accuracy of Naive Bayes Classifier is:', mnb.score(X_test, y_test))
print(classification_report(y_test, y_predict))

# In[ ]:
