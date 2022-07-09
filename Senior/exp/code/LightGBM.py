import os
import pandas as pd
import json
import numpy as np
from tqdm import tqdm
from tqdm import trange
from pandas import DataFrame
import argparse
import time
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import metrics
from scipy.stats import spearmanr
from utils import src_xgboost, src_lightgbm
from hyperopt import fmin, hp, tpe, Trials, STATUS_OK, atpe, rand
import matplotlib.pyplot as plt

<<<<<<< HEAD
visible_gpu_device = '2'
deafult_train_pkl = "./lgbm_train_feature.pkl"
default_val_pkl = "./lgbm_test_feature.pkl"
default_train_path = "./all_label_postdate.csv"
default_val_path = "./validate_label_postdate_05.csv"
deafault_save_path = './lgbm_add.csv'
=======
visible_cuda_id = '2'
train_feature__pkl_path = "./lgbm_train_feature.pkl"
test_feature__pkl_path = "./lgbm_test_feature.pkl"
use_userinfo = 'add'
train_lgbm_model = 2
train_label_post = "./all_label_postdate.csv"
test_label_post = "./validate_label_postdate_05.csv"
save_train_picture_dir = '../picture'
model_save_path = '../model'
save_predict_path = '../prediction/add.csv'

os.environ['CUDA_VISIBLE_DEVICES']= visible_cuda_id
>>>>>>> 123ace228f8d4e7adae313daa6cfb971f1df5708


os.environ['CUDA_VISIBLE_DEVICES']=visible_gpu_device
def parse_args():
    parser = argparse.ArgumentParser(description='------LightGBM-------')
<<<<<<< HEAD
    parser.add_argument('--train_pkl', default=deafult_train_pkl, dest='test_json',
                        type=str, help='the path of test_feature_pkl')
    parser.add_argument('--test_pkl', default=default_val_pkl, dest='test_json',
=======
    parser.add_argument('--train_pkl', default=train_feature__pkl_path, dest='test_json',
                        type=str, help='the path of test_feature_pkl')
    parser.add_argument('--test_pkl', default=test_feature__pkl_path, dest='test_json',
>>>>>>> 123ace228f8d4e7adae313daa6cfb971f1df5708
                        type=str, help='the path of test_feature_pkl')
    parser.add_argument('--userinfo', default=use_userinfo, type=str, dest='userinfo',
                        help='use:add or noadd')
    parser.add_argument('--train_model', default=train_lgbm_model, dest='train_model',
                        type=int, help='')
<<<<<<< HEAD
    parser.add_argument('--train_path', default=default_train_path,
                        type=str, help='')
    parser.add_argument('--val_path', default=default_val_path,
=======
    parser.add_argument('--train_path', default=train_label_post,
                        type=str, help='')
    parser.add_argument('--val_path', default=test_label_post,
>>>>>>> 123ace228f8d4e7adae313daa6cfb971f1df5708
                        type=str, help='')
    parser.add_argument('--picture_path', default=save_train_picture_dir,
                        type=str, help='picture_folder')
    parser.add_argument('--model_save_path', default=model_save_path,
                        type=str, help='model_folder')
<<<<<<< HEAD
    parser.add_argument('--save_path', default=deafault_save_path,
=======
    parser.add_argument('--save_path', default=save_predict_path,
>>>>>>> 123ace228f8d4e7adae313daa6cfb971f1df5708
                        type=str, help='prediction_csv_path')

    args = parser.parse_args()
    return args


def feature_concat(type, feature_path):
    print('Loading data from' + feature_path)
    df = pd.read_pickle(feature_path)
    if args.userinfo == 'add':
        if type == 'train':
            user_df = pd.read_csv("./train_userdata_9d_305613.csv")
        if type == 'test':
            if args.train_model == 1:
                user_df = pd.read_csv("./test_userdata_9d_180581.csv")
            if args.train_model == 2:
                user_df = pd.read_csv("./test_userdata_9d_64243.csv")
        df = pd.merge(df, user_df, on=['uid', 'pid'])

    if args.train_model ==2 and type == "test":
        if args.userinfo == 'add':
            uid = pd.read_csv("./test_uid_9611.csv")#116338pid
        elif args.userinfo == 'noadd':
            uid = pd.read_csv("./test_uid_21781.csv")#64243pid
        df = pd.merge(uid, df, on='uid')
    print(df)
    return df


def load_data():

    feature_df = feature_concat("train", args.train_pkl)
    train_df = pd.read_csv(filepath_or_buffer=train_path)
    val_df = pd.read_csv(filepath_or_buffer=val_path)

    train = pd.merge(train_df, feature_df, on='pid')
    val = pd.merge(val_df, feature_df, on='pid')

    print(train)
    print(val)

    x_train = train.iloc[:, 3:].astype('float')
    y_train = train.iloc[:, 1].astype('float')
    x_val = val.iloc[:, 3:].astype('float')
    y_val = val.iloc[:, 1].astype('float')

    return x_train, y_train, x_val, y_val

def plot_result(picture_path, lgbm_result):
    fig = plt.figure(num='LGBM'+feature_name+split, figsize=(19.2, 4.8), dpi=80, clear=False)

    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)
    lgb.plot_metric(lgbm_result, metric="SRC", ax=ax1, title='SRC')
    lgb.plot_metric(lgbm_result, metric="l1", ax=ax2, title='MAE')
    lgb.plot_metric(lgbm_result, metric="l2", ax=ax3, title='MSE')
    fig.savefig(os.path.join(picture_path, 'LGBM_'+feature_name + '_' + split + '.png'))

def lgb_regresssion(x_train, y_train, x_val, y_val):
    print("training lightgbm...")
    train_data = lgb.Dataset(data=x_train, label=y_train)
    validate_data = lgb.Dataset(data=x_val, label=y_val, reference=train_data)
    param = {
        "max_depth": 48,
        "num_leaves": 60,
        "learning_rate": 0.007,
        "nthread": 16,
        "reg_alpha": 0.5,
        "reg_lambda": 0.01,
        "random_seed": 2020,
        "objective": "regression",
        "metric": ["mse", "mae"],
        'first_metric_only': True,
        "boosting": "gbdt",
        "tree_learner": "serial",
        "device_type": "gpu",
        "gpu_platform_id": 0,
        "min_child_samples": 12,
        "min_child_weight": 0.002,
        "feature_fraction": 0.5804531627406809,
        'bagging_fraction': 0.6013832939955783,
    }
    n_estimators = 20000
    lgbm_result = {}
    start = time.time()
    model = lgb.train(param, train_data, num_boost_round=n_estimators, feval=src_lightgbm,
                    valid_sets=[train_data, validate_data], valid_names=["train", "validate"],
                    evals_result=lgbm_result, verbose_eval=10, init_model=None, keep_training_booster=True)
    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # save model
    print('save model to ' + model_save_path + 'LGBM_' + feature_name + '_' + split + '.txt')
    model.save_model(os.path.join(model_save_path, 'LGBM_'+feature_name + '_' + split + '.txt'))
    # predict
    y_pred = model.predict(x_val, num_iteration=model.best_iteration)
    # plot loss
    plot_result(picture_path, lgbm_result)
    y_true = y_val
    correlation, pvalue = spearmanr(y_pred, y_true)
    print("performance on val")
    print("mean_squared_error : %.4g" % mean_squared_error(y_true, y_pred))
    print("mean_absolute_error : %.4g" % mean_absolute_error(y_true, y_pred))
    print("SRC correlation : %.4g" % correlation)


    names = model.feature_name()
    importances = model.feature_importance().tolist()
    importances_df = pd.DataFrame({"feature": names, "importance": importances})
    importances_df.sort_values(by='importance', inplace=True, ascending=False)
    importances_df.to_csv(os.path.join(picture_path, "importance_" + 'LGBM_' + feature_name + '_' + split + '.csv'), index=None)


def lgb_test(model_path, test_pkl, save_path):
    test_df = feature_concat("test", test_pkl)
    pid = pd.DataFrame(test_df['pid'])
    x_test = test_df.iloc[:, 2:].astype('float')

    print("load model from " + model_path)
    trained_model = lgb.Booster(model_file=model_path)
    y_pred = trained_model.predict(x_test, num_iteration=trained_model.best_iteration)
    y_pred = pd.DataFrame(y_pred)
    y_pred.columns = ['label']

    data = pid.join(y_pred)
    print(data)
    data.to_csv(save_path, sep=',', header=True, index=True)


if __name__ == '__main__':
    args = parse_args()
    train_path = args.train_path
    val_path = args.val_path
    picture_path = args.picture_path
    model_save_path = args.model_save_path
    save_path = args.save_path
    feature_name = "feature_"+args.userinfo+"_model"+str(args.train_model)
    split = train_path.split('.')[-2].split('_')[-1]
    # # train
    x_train, y_train, x_val, y_val = load_data()
    lgb_regresssion(x_train, y_train, x_val, y_val)
    # # predict
    model_path = os.path.join(model_save_path, 'LGBM_'+feature_name + '_' + split + '.txt')
    lgb_test(model_path, args.test_pkl, save_path)
