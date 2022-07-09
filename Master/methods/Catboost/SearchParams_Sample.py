# !/usr/bin/env python
# coding: utf-8
'''
@File    :   SearchParams.py
@Time    :   2020/05/01 22:28:51
@Author  :   Wang Kai 
@Version :   1.0
@Contact :   wk15@mail.ustc.edu.cn
'''

# Search Parameters

# First
# conda install hyperopt

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from hyperopt import fmin, hp, tpe, Trials, STATUS_OK, atpe, rand

train_data = lgb.Dataset(data=X_train, label=y_train, free_raw_data=False)
validate_data = lgb.Dataset(
    data=X_validate, label=y_validate, reference=train_data, free_raw_data=False)


def objective(params):
    def src_lightgbm(preds, dtrain):
        labels = dtrain.get_label()
        correlation, pvalue = spearmanr(preds, labels)
        return "SRC", correlation, True

    params = {
        "max_depth": int(params["max_depth"]),
        "num_leaves": int(params["num_leaves"]),
        "learning_rate": params["learning_rate"],
        "min_data_in_leaf": int(params["min_data_in_leaf"]),
        "reg_lambda": params["reg_lambda"],

        "random_seed": 2020,
        "nthread": 8,
        "objective": "regression",
        "metric": ["mae", "mse"],
        "boosting": "gbdt",
        "tree_learner": "serial",
        "device_type": "gpu",
        "gpu_platform_id": 0,
        "gpu_device_id": 6
    }
    n_estimators = 40000
    early_stop = 1000
    result = {}

    bst = lgb.train(params, train_data, num_boost_round=n_estimators, feval=src_lightgbm,
                    valid_sets=[train_data, validate_data], valid_names=[
                        "train", "validate"],
                    early_stopping_rounds=early_stop, evals_result=result, verbose_eval=False, keep_training_booster=False)

    preds = bst.predict(validate_data.get_data(),
                        num_iteration=bst.best_iteration)
    mae = mean_absolute_error(validate_data.get_label(), preds)
    mse = mean_squared_error(validate_data.get_label(), preds)
    rho, pval = spearmanr(validate_data.get_label(), preds)
    print("-"*10)
    print(params, bst.best_iteration)
    print("mae:{}\tmse:{}\tsrc:{}".format(mae, mse, rho))
    return {
        "loss": mae,
        "mse": mse,
        "src": rho,
        "status": STATUS_OK
    }


space = {
    "max_depth": hp.quniform("max_depth", 6, 16, 2),
    "num_leaves": hp.choice("num_leaves", [31, 59, 99]),
    "learning_rate": hp.choice("learning_rate", [0.01, 0.03, 0.15, 0.3]),
    "min_data_in_leaf": hp.choice("min_data_in_leaf", [5, 10, 20, 50]),
    "reg_lambda": hp.choice("reg_lambda", [1.0, 2.0, 5.0])
}
trials = Trials()
bst = fmin(
    fn=objective,
    space=space,
    algo=atpe.suggest,
    # algo=tpe.suggest,
    # algo=rand.suggest,
    max_evals=100,
    trials=trials
)
print(bst)
# bst 返回的是choice的索引... 别的uniform是值.
# 比如{'learning_rate': 0, 'max_depth': 14.0, 'min_data_in_leaf': 3, 'num_leaves': 2, 'reg_lambda': 2}
print(trials.trials)
# 参考https://github.com/hyperopt/hyperopt/wiki/FMin
# 参考https://github.com/hyperopt/hyperopt

# hp.choice(label, options)
# hp.randint(label, upper)
# hp.uniform(label, low, high)
# hp.quniform(label, low, high, q)
# hp.loguniform(label, low, high)
# hp.qloguniform(label, low, high, q)
# hp.normal(label, mu, sigma)
# hp.qnormal(label, mu, sigma, q)
# hp.lognormal(label, mu, sigma)
# hp.qlognormal(label, mu, sigma, q)


# GridSearch

param_dist = {
    'max_depth': list(range(8, 17, 2)),
    "num_leaves": [31, 65, 127]
    #     'learning_rate':[0.01,0.03,0.15,0.3]
}
lgb_model = lgb.LGBMRegressor(boosting_type="gbdt", n_estimators=5000, objective="regression", metric=[
                              "mae", "mse"], min_child_samples=20, random_seed=random_seed, silent=False, n_jobs=-1, device_type="gpu", gpu_platform_id=0, gpu_device_id=5)
lgbsearch = GridSearchCV(lgb_model, param_dist, scoring="neg_mean_absolute_error", cv=KFold(
    n_splits=5, shuffle=False), n_jobs=-1, verbose=1)
lgbsearch.fit(X_train, y_train)

print("best estimator", lgbsearch.best_estimator_)

print("best params:", lgbsearch.best_params_)
print("best best score", lgbsearch.best_score_)
print("cv results", lgbsearch.cv_results_)
df_cv = pd.DataFrame(lgbsearch.cv_results_)
df_cv

preds = lgbsearch.best_estimator_.predict(X_validate)
mae = mean_absolute_error(y_validate, preds)
mse = mean_squared_error(y_validate, preds)
rho, pval = spearmanr(y_validate, preds)
print("mae:{}\tmse:{}\tsrc:{}".format(mae, mse, rho))
