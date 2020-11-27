"""

    https://blog.csdn.net/shenxiaoming77/article/details/72627882

分类模型评估：
Precision	精准度	from sklearn.metrics import precision_score
Recall	召回率	from sklearn.metrics import recall_score
F1	F1值	from sklearn.metrics import f1_score
Confusion Matrix	混淆矩阵	from sklearn.metrics import confusion_matrix
ROC	ROC曲线(Receiver Operating Characteristic)	from sklearn.metrics import roc
AUC	ROC曲线下的面积(Area Under the Curve)	from sklearn.metrics import auc

回归模型评估：
Mean Square Error (MSE, RMSE)	平均方差	from sklearn.metrics import mean_squared_error
Absolute Error (MAE, RAE)	绝对误差	from sklearn.metrics import mean_absolute_error, median_absolute_error
R-Squared	R平方值	from sklearn.metrics import r2_score

"""


import numpy as np
from sklearn import metrics

y = np.array([1, 1, 2, 2])
scores = np.array([0.1, 0.4, 0.35, 0.8])
fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)
print(fpr, tpr, thresholds)

import numpy as np
from sklearn.metrics import roc_auc_score
y_true = np.array([0, 0, 1, 1])
y_scores = np.array([0.1, 0.4, 0.35, 0.8])
print(roc_auc_score(y_true, y_scores))
