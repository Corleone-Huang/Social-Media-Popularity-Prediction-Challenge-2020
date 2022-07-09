# !/usr/bin/env python
# coding: utf-8
'''
@File    :   utils.py
@Time    :   2020/04/19 00:33:25
@Author  :   Wang Kai 
@Version :   1.0
@Contact :   wk15@mail.ustc.edu.cn
'''
# This is used to realize SRC obj and feval

from scipy.stats import spearmanr


def src_loss(preds, dtrain):
    # I don't know how to calculate the gred and hess
    # labels=dtrain.get_label()
    pass
    # gred=
    # hess=
    # return gred, hess


def src_lightgbm(preds, dtrain):
    labels = dtrain.get_label()
    correlation, pvalue = spearmanr(preds, labels)
    return "SRC", correlation, True

# gbm = lgb.train(fobj=src_loss, feval=src_lightgbm))


# user defined evaluation function, return a pair metric_name, result

# NOTE: when you do customized loss function, the default prediction value is
# margin. this may make builtin evaluation metric not function properly for
# example, we are doing logistic loss, the prediction is score before logistic
# transformation the builtin evaluation error assumes input is after logistic
# transformation Take this in mind when you use the customization, and maybe
# you need write customized evaluation function
def src_xgboost(preds, dtrain):
    labels = dtrain.get_label()
    correlation, pvalue = spearmanr(preds, labels)
    return "SRC", correlation


# training with customized objective, we can also do step by step training
# simply look at xgboost.py's implementation of train

# bst = xgb.train(param, dtrain, num_round, watchlist, obj=src_loss, feval=src_xgboost)
