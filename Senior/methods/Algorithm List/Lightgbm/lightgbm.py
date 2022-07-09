# -*- coding: utf-8 -*-
# @Date    : 2020-04-16
# @Time    : 15:52
# @Author  : Zhang Jingjing
# @FileName: lightgbm.py
# @Software: PyCharm
# 中文文档: https://lightgbm.apachecn.org/#/
# 官方文档: https://lightgbm.readthedocs.io/en/latest/Python-Intro.html
# 示例代码: https://github.com/Microsoft/LightGBM/tree/master/examples/python-guide
# github: https://github.com/Microsoft/LightGBM/blob/master/docs/Parameters.rst
# https://zhuanlan.zhihu.com/p/55259112
# install: pip install lightgbm --install-option=--gpu
import lightgbm as lgb
import pandas as pd
import numpy as np
import json
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
import datetime
from utils import src_lightgbm
# from xgb_reg import LoadData
<<<<<<< HEAD

visible_gpu_device = '0,1,2,3,4,5,6,7'
default_feature_path = '../../results/merge/wordchar_merge.csv'
default_train_path = '../../data/label/train_label_postdate.csv'
default_val_path = '../../data/label/validate_label_postdate.csv'
default_save_dir = ''


os.environ['CUDA_VISIBLE_DEVICES'] = visible_gpu_device
def parse_args():
    parser = argparse.ArgumentParser(description='------LightGBM-------')
    parser.add_argument('--feature_path', default=default_feature_path, \
                        type=str, help='feature.csv的路径')
    parser.add_argument('--train_path', default=default_train_path, \
        type=str, help='划分好的train：pid|label的路径')
    parser.add_argument('--val_path', default=default_val_path, \
        type=str, help='划分好的val：pid|label的路径')
    parser.add_argument('--save_dir', default=default_save_dir, type=str, help='保存model的文件夹路径')

=======
visible_cuda_id = '0,1,2,3,4,5,6,7'
train_feature_path = '../../results/merge/wordchar_merge.csv'
train_label_post = "../../data/label/train_label_postdate.csv"
test_label_post = "../../data/label/validate_label_postdate.csv"
save_model_dir = '../'

os.environ['CUDA_VISIBLE_DEVICES'] = visible_cuda_id
def parse_args():
    parser = argparse.ArgumentParser(description='------LightGBM-------')
    parser.add_argument('--feature_path', default= train_feature_path, \
                        type=str, help='feature.csv的路径')
    parser.add_argument('--train_path', default= train_label_post, \
        type=str, help='划分好的train：pid|label的路径')
    parser.add_argument('--val_path', default= test_label_post, \
        type=str, help='划分好的val：pid|label的路径')
    parser.add_argument('--save_dir', default= save_model_dir, type=str, help='保存model的文件夹路径')
>>>>>>> 123ace228f8d4e7adae313daa6cfb971f1df5708
    args = parser.parse_args()
    return args


# 从路径加载划分好的数据集和特征
def LoadData(feature_path, train_path, val_path):
    # fature_path 为特征路径
    # train_path,val_path 为划分好的训练集验证集路径
    # 返回合并划分好的训练验证集
    print('Loading data...')
    feature_df = pd.read_csv(feature_path)
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    train = pd.merge(train_df, feature_df, on='pid')
    val = pd.merge(val_df, feature_df, on='pid')
    # 格式：pid label uid feature____
    # 提取特征列/标签列
    x_train = np.array(train.iloc[:, 3:])
    y_train = np.array(train.iloc[:, 1])

    x_val = np.array(val.iloc[:, 3:])
    y_val = np.array(val.iloc[:, 1])
    return x_train, y_train, x_val, y_val


def plot_result(model_path, model, result):
    fig = plt.figure(num='LGBM'+feature_name + split, figsize=(19.2,4.8), dpi=80, clear=False)
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)
    # ax4 = fig.add_subplot(2, 2, 4)
    # 显示重要特征
    # lgb.plot_importance(model,ax=ax1, max_num_features=10,title='importance')
    lgb.plot_metric(result, metric="SRC", ax=ax1, title='SRC')
    lgb.plot_metric(result, metric="l1", ax=ax2, title='l1')
    lgb.plot_metric(result, metric="l2",ax=ax3, title='l2')
    fig.savefig(os.path.join(model_path, 'LGBM_'+feature_name + '_' + split + '.png'))

def lgb_trainer(x_train, x_test, y_train, y_test):
    params = {
        'boosting_type': 'gbdt',# 设置提升类型，默认gbdt,传统梯度提升决策树
        'objective': 'regression', # 目标函数L2(MSE)
        'metric': {'l2', 'l1'}, # 评估函数
        'first_metric_only':True,#以l2为早停指标
        "min_data_in_leaf": 20,
        'max_depth' : 16, # 树的深度 按层，默认-1
        'num_leaves': 99, # 在一棵树中叶子节点数,默认31
        'learning_rate': 0.03, # 学习速率
        'feature_fraction': 0.9, # 建树的特征选择比例
        'bagging_fraction': 0.8, # 建树的样本采样比例
        'bagging_freq': 4, # k 意味着每 k 次迭代执行bagging
        'verbose': 1, # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
        'device': 'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id':6,
        # 'num_iterations': 100 # 默认100,别名=num_iteration,
        # num_tree, num_trees, num_round, num_rounds, num_boost_round
    }
    print('Starting training...')
    starttime = datetime.datetime.now()
    # create dataset for lightgbm
    lgb_train = lgb.Dataset(x_train, y_train)
    lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)
    result={}
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=10000, # 默认100
                    early_stopping_rounds=1000, # 若有验证集,
                    # 根据所有的metric
                    # 使用提前停止找到最佳数量的boosting rounds梯度次数
                    feval=src_lightgbm, # 自定义评价函数
                    evals_result=result,
                    valid_sets =[lgb_train, lgb_eval],
                    valid_names=["train", "validate"],
                    verbose_eval=10,
                    keep_training_booster=False,
                    )
    # 模型将开始训练, 直到验证得分停止提高为止.
    # 验证错误需要至少每个 early_stopping_rounds 减少以继续训练.
    # 如果提前停止, 模型将有 1 个额外的字段: bst.best_iteration.
    # train() 将从最后一次迭代中返回一个模型, 而不是最好的一个
    endtime = datetime.datetime.now()
    print("total time: {}s".format((endtime - starttime).seconds))
    feature_name = feature_path.split('/')[-1].split('.csv')[0]
    model_path = os.path.join(save_dir, 'LGBM_'+ feature_name  + '_' +split)
    os.mkdir(model_path)

    # print('Saving model...')
    # # 保存模型
    # gbm.save_model(os.path.join(model_path, 'lgbm_model'+'.txt'))
    # 绘制结果图
    plot_result(model_path, gbm, result)

    print('Starting predicting...')
    # predict
    y_pred = gbm.predict(x_test, num_iteration=gbm.best_iteration)
    # eval
    print('mean_squared_error(MSE) : ', mean_squared_error(y_test, y_pred))
    print('mean_absolute_error(MAE) : ', mean_absolute_error(y_test, y_pred))

    correlation, _ = spearmanr(y_pred, y_test)
    print("SRC correlation : %.4g" % correlation)

if __name__ == "__main__":
    args = parse_args()
    feature_path = args.feature_path
    train_path = args.train_path
    val_path = args.val_path
    save_dir = args.save_dir
    feature_name = feature_path.split('/')[-1].split('_')[0]
    # 划分规则是random/postdate
    split = train_path.split('.')[-2].split('_')[-1]
    # 划分训练集和测试集
    x_train, y_train, x_val, y_val = LoadData(feature_path, train_path, val_path)
    # 训练
    lgb_trainer(x_train, x_val, y_train, y_val)
