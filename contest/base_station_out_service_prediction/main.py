# -*- coding: utf-8 -*-
# @Time: 2020/6/29,029 14:04
# @Last Update: 2020/6/29,029 14:04
# @Author: 徐缘
# @FileName: main.py
# @Software: PyCharm


import csv
import datetime
import pandas as pd


if __name__ == '__main__':
    # 1:10 还是 1:3
    train = pd.read_csv("train_data.csv")
    print(train["label"].sum())


