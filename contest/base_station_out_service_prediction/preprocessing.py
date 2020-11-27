# -*- coding: utf-8 -*-
# @Time: 2020/7/3,003 14:57
# @Last Update: 2020/7/3,003 14:57
# @Author: 徐缘
# @FileName: preprocessing.py
# @Software: PyCharm

import os
import random

import numpy as np
import pandas as pd
from tqdm import tqdm


class Processor:
    def __init__(self):
        self.train_path = '/mnt/5/Alert_BTS_HW_1001-0309'
        self.test_path_1 = '/mnt/5/Alert_BTS_HW_0316-0322'
        self.test_path_2 = '/mnt/5/Alert_BTS_HW_0324-0330'
        self.endtime = {
            "test_1": "2020-03-23",
            "test_2": "2020-03-31",
            "train": "2020-03-10"
        }
        self.sample_rate = 0.1

    # flag决定加的endtime是不一样的
    def read_data(self, path, flag):
        samples = pd.DataFrame(columns=['基站名称', '告警数量', '曾经退服', 'label'])
        delta_time = np.timedelta64(1, 'D')

        for csvs in tqdm(os.listdir(path)):

            # 先可以每个基站分开做。之后再考虑是否可以合起来利用
            data = pd.read_csv(os.path.join(path, csvs))
            n_rows = len(data)
            if n_rows <= 1:
                continue

            data = pd.to_datetime(data['告警开始时间'], format='%Y-%m-%d %H:%M:%S')

            # 有几条告警，按比例出样本
            n_samples = int(n_rows * self.sample_rate)
            for n in range(n_samples):
                rand = random.randint(1, len(data) - 1)
                pre = rand - 1

                n_warnings = 0
                was_out_service = 0

                while pre >= 0 and (data['告警开始时间'][rand] - data['告警开始时间'][pre] <= delta_time):
                    if data['告警名称'][pre] in ['网元连接中断', '小区不可用告警']:
                        was_out_service = 1
                    n_warnings += 1
                    pre -= 1

                if data['告警名称'][rand] in ['网元连接中断', '小区不可用告警']:
                    label = 1
                else:
                    label = 0

                samples.append(pd.DateFrame([data['告警开始时间'][rand]['基站名称'], n_warnings, was_out_service, label],
                                            columns=['基站名称', '告警数量', '曾经退服', 'label']), ignore_index=True)

        return samples

    def run(self):
        test_1 = self.read_data(self.test_path_1, 'test_1')
        print(test_1)


#         test_2 = self.read_data(self.test_path_2, 'test_2')
#         train = self.read_data(self.train_path, 'train')


process = Processor()
process.run()

