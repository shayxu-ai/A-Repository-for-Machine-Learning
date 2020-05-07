# -*- coding: utf-8 -*-
# @Time : 2020/5/2 6:10 下午
# @Author : 徐缘
# @FileName: linear_model_hello.py
# @Software: PyCharm


"""
    学习一下线性回归的两种解法。并使用scikit中的线性回归模型
    https://scikit-learn.org/stable/modules/classes.html?highlight=linear_model#module-sklearn.linear_model
"""


def linear_model_linear_regression():
    from sklearn import linear_model
    import numpy as np
    model = linear_model.LinearRegression()
    model.fit(X, y)
    a = model.predict([[12]])
    # a[0][0]
    print("预测一张12英寸匹萨价格：{:.2f}".format(model.predict([[12]])[0][0]))


if __name__ == '__main__':
    X = [[6], [8], [10], [14], [18]]
    y = [[7], [9], [13], [17.5], [18]]
    linear_model_linear_regression()

