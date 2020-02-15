# A-Repository-for-Machine-Learning

在github上clone了一大堆项目，没有一个变成自己的东西。所以还是自己整理一下。

markdown教程 https://www.runoob.com/markdown/md-code.html
```
Python 3.7.5    (然而python版本其实没有什么影响)
Tensorflow-cpu 2.1.0rc2    (虽然安装包大小和原版一样，单既然有就装CPU版吧，区别未知。GPU版没有分MACOS。p.s. Github上能找到支持AVX的版本)
pytorch 它不香吗

# 切换版本 
pyenv versions
pyenv which pip
pyenv local 3.7.5   (pyenv 用不了matplotlib.pyplot, 可以试试virtualenv用来创建同版本的虚拟环境) https://www.v2ex.com/t/583958

python版本镜像
https://www.python.org/ftp/python/3.5.2/Python-3.5.2.tar.xz
http://mirrors.sohu.com/python/3.7.6/Python-3.7.6.tar.xz

Tensorflow最新只支持到CP37  
Python 3.7.6 - Dec. 18, 2019    https://www.python.org/downloads/mac-osx/
Python 3.7.6                    https://github.com/pyenv/pyenv#homebrew-on-macos
Tensorflow 2.1.0 Jan 9, 2020 https://pypi.org/project/tensorflow/#files
Tensorflow 2.1.0rc2 https://pypi.tuna.tsinghua.edu.cn/simple/tensorflow/

# 关键是requirements.txt

```
tensorflow_datasets 2.0.0 (被旧版本教程坑了，subsplit在新版本报错。)
发布候选版 (RC=Release Candidate)  
最终版 (RTM=Release To Manufacture)

