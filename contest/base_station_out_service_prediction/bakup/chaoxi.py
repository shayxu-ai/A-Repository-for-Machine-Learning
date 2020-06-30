# -*- coding: utf-8 -*-
# @Time: 2020/6/29,029 15:14
# @Last Update: 2020/6/29,029 15:14
# @Author: 徐缘
# @FileName: chaoxi.py
# @Software: PyCharm


import os
import pandas as pd
from tqdm import tqdm
# import numpy as np
# import datetime
# import requests
# import json
# from sklearn.model_selection import StratifiedKFold,GridSearchCV,train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.decomposition import TruncatedSVD
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import f1_score,roc_auc_score,roc_curve,auc,accuracy_score,precision_score
# from sklearn.preprocessing import OneHotEncoder,LabelEncoder
# import lightgbm as lgb
# import gc
#
# import time
# import datetime

"""

"""

test_0322_path = '/mnt/5/Alert_BTS_HW_0316-0322'    # 测试集1
test_0330_path = '/mnt/5/Alert_BTS_HW_0324-0330'    # 测试集2

all_test_data = pd.DataFrame()
all_test_data2 = pd.DataFrame()

# 将测试集1目录下的
for now_csv in tqdm(os.listdir(test_0322_path)):
    data = pd.read_csv(os.path.join(test_0322_path, now_csv))
    all_test_data = all_test_data.append(data)
    tmp1 = all_test_data[(all_test_data['告警名称'] == '小区不可用告警') |
                         (all_test_data['告警名称'] == '网元连接中断')]
    tmp1_label1_IDs = tmp1['基站名称'].unique()     # 出过退服告警的基站

for now_csv in tqdm(os.listdir(test_0330_path)):
    data = pd.read_csv(os.path.join(test_0330_path, now_csv))
    all_test_data2 = all_test_data2.append(data)
    tmp2 = all_test_data2[(all_test_data2['告警名称'] == '小区不可用告警') |
                          (all_test_data2['告警名称'] == '网元连接中断')]
    tmp2_label1_IDs = tmp2['基站名称'].unique()

sub1 = pd.read_csv('/mnt/5/提交文件样例/Sample23日.csv', encoding='gbk')
sub2 = pd.read_csv('/mnt/5/提交文件样例/Sample31日.csv', encoding='gbk')
sub1['未来24小时发生退服类告警的概率'] = sub1['基站名称'].apply(lambda x: 1 if x in tmp1_label1_IDs else 0)
sub2['未来24小时发生退服类告警的概率'] = sub2['基站名称'].apply(lambda x: 1 if x in tmp2_label1_IDs else 0)
sub1.to_csv('/root/models/results/Sample23日.csv', index=False)
sub2.to_csv('/root/models/results/Sample31日.csv', index=False)
