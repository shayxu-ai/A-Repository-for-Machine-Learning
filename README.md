# A-Repository-for-Machine-Learning

在github上clone了一大堆项目，没有一个变成自己的东西。所以还是自己整理一下。

markdown教程 https://www.runoob.com/markdown/md-code.html
```
Python 3.7.5
Tensorflow-cpu 2.1.0rc2    (虽然安装包大小和原版一样，单既然有就装CPU版吧，区别未知。GPU版没有分MACOS。p.s. Github上能找到支持AVX的版本)
pytorch 它不香吗

# 切换版本 
pyenv versions
pyenv which pip
pyenv local 3.7.5   (pyenv 用不了matplotlib.pyplot, 可以试试virtualenv用来创建同版本的虚拟环境) https://www.v2ex.com/t/583958

python版本镜像
https://www.python.org/ftp/python/3.5.2/Python-3.5.2.tar.xz
(推荐)http://mirrors.sohu.com/python/3.7.6/Python-3.7.6.tar.xz  浏览器点不进python的目录，但是可以下载

Tensorflow最新只支持到CP37  
Python 3.7.6 - Dec. 18, 2019    https://www.python.org/downloads/mac-osx/
Python 3.7.6                    https://github.com/pyenv/pyenv#homebrew-on-macos
Tensorflow 2.1.0 Jan 9, 2020 https://pypi.org/project/tensorflow/#files
Tensorflow 2.1.0rc2 https://pypi.tuna.tsinghua.edu.cn/simple/tensorflow/

# 关键是requirements.txt

# 整理几个镜像源
http://mirrors.sohu.com/
https://developer.aliyun.com/mirror/  阿里云
http://mirrors.ustc.edu.cn/  中国科学技术大学
https://mirrors.tuna.tsinghua.edu.cn/  清华大学
```
tensorflow_datasets 2.0.0 (被旧版本教程坑了，subsplit在新版本报错。)
发布候选版 (RC=Release Candidate)  
最终版 (RTM=Release To Manufacture)

https://www.bilibili.com/video/av59503393
听了个很有意思的公开课。
比起学理论，更重要的是增加自己的能力。
会用什么工具，能起到什么作用，可以给你的manager提供什么帮助。

LR SVM DT BOOST K-MEANS CRF-BiLSTM
CNN RNN LSTM Attention transform gpt bert
one-hot bow w2v fasttest glove elmo
分词 词性标注 命名实体识别 依存句法分析

LR，贝叶斯分类，单层感知机、线性回归，SVM（线性核）
决策树、RF、GBDT、多层感知机、SVM（高斯核）等。

GradientTree Boosting
GBDT泛指所有梯度提升树算法，包括XGBoost，它也是GBDT的一种变种，为了区分它们，GBDT一般特指“Greedy Function Approximation：A Gradient Boosting Machine”里提出的算法，只用了一阶导数信息。

https://www.paddlepaddle.org.cn/ 飞浆开源平台


