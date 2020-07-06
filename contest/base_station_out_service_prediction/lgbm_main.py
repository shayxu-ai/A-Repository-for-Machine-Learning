import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn import metrics


trian_path = 'train_data.csv'
train_data = pd.read_csv(trian_path)

x_train = train_data[['n_warnings', 'was_out_service']]
y_train = train_data['label']

# print(x_train)

test_path = 'test_data.csv'
test_data = pd.read_csv(test_path)

x_test = test_data[['n_warnings', 'was_out_service']]
y_test = test_data['label']
# print(x_test)

param = {
    'boosting_type': 'gbdt',
    'colsample_bytree': 1.0,
    'learning_rate': 0.075286,
    'max_depth': 20,
    'n_estimators': 2000,
    'n_jobs': 4,
    'num_leaves': 100,
    'reg_alpha': 0.0,
    'reg_lambda': 0.0,
    'subsample': 0.7,
    'objective': 'binary',
    'metric': 'auc',
    'num_threads': 20,
    'feature_fraction': 0.5,
    'bagging_fraction': 0.6792676,
    'verbose': 1,
    'max_bin': 255,
    'min_sum_hessian_in_leaf': 1
}

model = lgb.LGBMClassifier(**param)
model.fit(x_train, y_train, eval_set=(x_test, y_test), eval_metric='auc', early_stopping_rounds=50, verbose=10,
          feature_name='auto', categorical_feature='auto', callbacks=None)


y_pred = model.predict(x_test, num_iteration=model.best_iteration_)
f1 = metrics.f1_score(y_test, y_pred, average='weighted')
print(f1)


# 预测结果

test_01 = 'test01.csv'
test_02 = 'test02.csv'

output = {
    test_01: "Sample23日.csv",
    test_02: "Sample31日.csv"
}

sample = {
    test_01: r"bakup\Sample23日.csv",
    test_02: r"bakup\Sample31日.csv"
}

for path in [test_01, test_02]:
    data = pd.read_csv(path)

    x = data[['n_warnings', 'was_out_service']]
    index = data['基站名称']

    # y = model.predict_proba(x, num_iteration=model.best_iteration_)
    # print(list(zip(*y))[1])
    # tmp = pd.DataFrame({"基站名称": index, "未来24小时发生退服类告警的概率": [round(x, 3) for x in list(zip(*y))[1]]})

    y = model.predict(x, num_iteration=model.best_iteration_)
    tmp = pd.DataFrame({"基站名称": index, "未来24小时发生退服类告警的概率": y})

    sample_index = pd.read_csv(sample[path])
    sample_index = sample_index.drop(["未来24小时发生退服类告警的概率"], axis=1)
    ans = sample_index.merge(tmp, on=["基站名称"], how='outer')

    ans.to_csv(output[path], index=False, encoding="gbk")





