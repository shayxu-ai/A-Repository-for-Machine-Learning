# -*- coding: utf-8 -*-
# @Time : 2020/2/18 10:50 下午
# @Author : 徐缘
# @FileName: search.py
# @Software: PyCharm


import csv
import time

import requests

"""
    查询api(GET)   https://api.bilibili.com/x/web-interface/search/all/v2        这个可能优化过查询结果吧。毕竟页面上什么都不点的话，就会查这个api。
    实际上用这个api才能分类查询 https://api.bilibili.com/x/web-interface/search/type    而且有参数检查
    
    很可惜，搜索API只能查到50 * 20 = 1000 个视频。
    那我就很好奇，用同一个关键字和不同的排序方式，查出来的会是同样的1000个视频吗？
    那么就有一个问题，如果要得到答案，我至少要访问页面 50 * 2 = 100次。 短时间内频繁访问，IP可能会被B站封掉。
    那我就看不了小破站了。这可怎么权衡呢。
    50 * 2.5s = 125s ~= 2m   问题不大

    参数:
        keyword=tensorflow
        page=1      ~50
        order=[totalrank, click,   pubdate, dm,      stow] 
              [综合排序,   最多点击, 最新发布  最多弹幕, 最多收藏]

        tids_1=0    ~17    '全部分区'和17个分区
        tids_2=0            小分区
        duration=[0, 1, 2, 3, 4]    [全部时长, 10分钟以下, 10-30分钟, 30-60分钟, 60分钟以上]

        # 一些没搞懂的参数
        context
        __refresh__=true
        __reload__=false
        highlight=1
        single_column=0
        jsonp=jsonp
        callback=__jp0        

    Conclusion:
        使用search/type api不同排序方式查下来会有不一样的结果。
        去除重复项留有1825。
        删除最新发布后，去重有1650

        点击、弹幕、收藏以及和发布时间都和综合排序相关。
        所以知道这些有什么用。搜索的时候可以更高效？

        不如直接 echo "127.0.0.1 www.bilibili.com" >> /etc/hosts
"""

if __name__ == '__main__':
    keyword = 'tensorflow'
    url = 'https://api.bilibili.com/x/web-interface/search/all/v2?keyword=' + keyword  # 这个api改order没用
    url_2 = 'https://api.bilibili.com/x/web-interface/search/type?context=&page=1&order=pubdate&keyword=tensorflow&duration=0&tids_2=&__refresh__=true&search_type=video&tids=0&highlight=1&single_column=0'

    rows = list()
    for order in ['totalrank', 'click', 'pubdate']:

        for page in range(1, 51):

            print("order:", order, "page:", page)
            url_tmp = 'https://api.bilibili.com/x/web-interface/search/type?' \
                      'context=&page=' + str(page) + '&order=' + order + '&keyword=' + keyword + \
                      '&duration=0&tids_2=&__refresh__=true&search_type=video&tids=0&highlight=1&single_column=0'
            # print(url_tmp)
            try:
                f = requests.get(url_tmp).json()

                # tmp_dict = json.loads(f)['data']['result'][-1]['data']    # all/v2 json格式
                tmp_dict = f['data']['result']
                time.sleep(2)
            except 'TypeError':
                print(url_tmp)
                continue

            for item in tmp_dict:
                av = item['id']
                # print('id:', av)
                title = item['title'].replace(r'<em class="keyword">', '').replace(r'</em>', '')
                # print('title:', title)
                pubdate = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(item['pubdate'])))
                # print('pubdate:', pubdate)
                row_tmp = [av, title, pubdate, order, page]
                print(row_tmp)
                print()
                rows.append(row_tmp)

    with open('search_result.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(rows)





