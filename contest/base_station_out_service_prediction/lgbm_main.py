import lightgbm as lgb
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pandas as pd
pd.set_option('display.max_rows', None)

data_path = 'train_data.csv'

train_data = pd.read_csv(data_path).dropna(axis=0)
train_data.reset_index(drop=True, inplace=True)
# print(len(train_data))
# print(len(train_data[(train_data['1']>0) | (train_data['37']>0)][['1', '37']]))

train_data.info()
train_data.tail(10)

train_data[['113', '113_6d', '113_6d_bool', '152', '152_6d', '152_6d_bool', 'label']]

# train_data['label'] = train_data.apply(lambda row: 1 if (row['1'] or row['37']) else 0, axis=1).shift(axis=0, periods=-1)

# train_data = train_data[train_data['starttime'] != "2020-03-09"]
train_data_1 = train_data[train_data['label'] == 1]
train_data_0 = train_data[train_data['label'] == 0]
# .sample(n=len(train_data_1)*3)
train_data_sampled = train_data_1.append(train_data_0).sample(frac=1).reindex()

train_data_sampled[['113_6d_bool','152_6d_bool', 'label']]


def node_id_encode():
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    le.fit(train_data['node_id'])
    train_data['node_id'] = le.transform(train_data['node_id'])

# node_id_encode()

def select_k_best():
    from sklearn.feature_selection import SelectKBest, f_classif

    feature_cols = train_data_sampled.columns.drop('label')

    # Keep 5 features
    selector = SelectKBest(f_classif, k=5)
    X_new = selector.fit_transform(train_data_sampled[feature_cols], train_data_sampled['label'])

# select_k_best()


def baseline(X, Y):
    X_predict = X.apply(lambda row: 1 if row['113_2d_bool'] + row['152_2d_bool'] > 0 else 0, axis=1)
    #     print(X_predict)

    f1 = metrics.f1_score(X_predict, Y, average='weighted')
    print(f1)
    f1 = metrics.f1_score(X_predict, Y)
    print(f1)

    print(metrics.classification_report(X_predict, Y, labels=None, target_names=None, sample_weight=None, digits=2))
    return


baseline(train_data_sampled, train_data_sampled['label'])

cols = [i for i in train_data.columns if i not in ['label']]
# cols = ['113_6d_bool', '152_6d_bool']
x_train, x_test, y_train, y_test = train_test_split(train_data_sampled[cols], train_data_sampled['label'], test_size=0.25, random_state=0)

param = {
    'boosting_type': 'gbdt',
    'colsample_bytree': 1.0,
    'learning_rate': 0.075286,
    'max_depth': 20,
    'n_estimators': 2000,
    'n_jobs': 4,
    'num_leaves': 100,
    'reg_alpha': 1,
    'reg_lambda': 1,
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
model.fit(x_train, y_train, eval_set=(x_test, y_test), eval_metric='binary_logloss', early_stopping_rounds=50, verbose=10,
          feature_name='auto', categorical_feature='auto', callbacks=None)

model.booster_.feature_importance(importance_type='gain')

imp = pd.DataFrame(model.booster_.feature_importance(importance_type='gain').tolist(),index=cols)
imp.sort_values(by=[0], ascending=False)

y_pred = model.predict(x_test, num_iteration=model.best_iteration_)
f1 = metrics.f1_score(y_test, y_pred, average='weighted')
print(f1)
f1 = metrics.f1_score(y_test, y_pred)
print(f1)

print(metrics.classification_report(y_test, y_pred, labels=None, target_names=None, sample_weight=None, digits=2))