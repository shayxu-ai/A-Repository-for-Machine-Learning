import jieba.posseg as psg

"""
    词性标注（jieba词性标注实战:jieba.posseg） https://blog.csdn.net/qq_35164554/article/details/90205175
"""

sent = "还有什么是比jieba更好的中文分词工具呢？"
seg_list = psg.cut(sent)
# print(list(seg_list))
result = " ".join(["{0}/{1}".format(w, t) for w, t in seg_list])

print(result)

