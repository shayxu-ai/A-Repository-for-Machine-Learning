# -*- coding: utf-8 -*-
# @Time: 2020/6/29,029 14:04
# @Last Update: 2020/6/29,029 14:04
# @Author: 徐缘
# @FileName: main.py
# @Software: PyCharm


import csv
import datetime


def read_data():
    with open("data/AlertHW_DF_1.csv", encoding="utf") as f:
        reader = csv.reader(f, header=1)
        for line in reader:
            print(line)

    return


if __name__ == '__main__':

    inp = read_data()

