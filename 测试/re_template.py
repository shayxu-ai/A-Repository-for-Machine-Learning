import jieba
import re


"""
    圆括号()是组
"""

# 正常匹配
re_han_internal = re.compile(r"([\u4E00-\u9FD5a-zA-Z0-9+#&._]+)")
