# -*- coding: utf-8 -*-
# @Time: 2020/6/29,029 13:51
# @Last Update: 2020/6/29,029 13:51
# @Author: 徐缘
# @FileName: f1_score_hello.py
# @Software: PyCharm


"""
    F1 = 2 * (precision * recall) / (precision + recall)

"""

from sklearn import metrics


y_true = [0, 1, 2, 0, 1, 2]
y_pred = [0, 2, 1, 0, 0, 1]

f1 = metrics.f1_score(y_true, y_pred, average='weighted')
print(f1)
