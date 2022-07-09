# !/usr/bin/env python
# coding: utf-8
'''
@File    :   pipline.py
@Time    :   2020/05/17 16:42:35
@Author  :   Wang Kai 
@Version :   1.0
@Contact :   wk15@mail.ustc.edu.cn
'''


import argparse
import gc
import json
import os
import time

import catboost
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

random_seed = 2020
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

train_feature_pklpath = "../../features/extracted_features/train+label_with_user_result_305613.pkl"
test_feature_pklpath = "../../features/extracted_features/test+label_with_user_no_filled_result_180581.pkl"
to_submit_path="../to_submit"

log={}

train_feature_filepath = {
    "uid":"/home/wkai/SMP/feature/train_feature/train_uid_305613.csv",
    "Fasttext_tag": "/home/wkai/SMP/feature/train_feature/FastText_tags_305613.csv",
    "Fasttext_title": "/home/wkai/SMP/feature/train_feature/FastText_title_305613.csv",
    "Fasttext_ave3": "/home/wkai/SMP/feature/train_feature/FastText_tags+title_305613_average_3.csv",
    "Bert_tag": "/home/wkai/SMP/feature/train_feature/bert_base_alltags_average_3.csv",
    "Bert_title": "/home/wkai/SMP/feature/train_feature/bert_base_titles_average_3.csv",
    "Bert_ave3": "/home/wkai/SMP/feature/train_feature/Bert_tags+title_305613_average_3.csv",
    "Bert_tsvd512_ave3": "/home/wkai/SMP/feature/train_feature/Bert_tags+title_305613_tsvd512_average_3.csv",
    "glove": "/home/wkai/SMP/feature/train_feature/glove_tags_305613.csv",
    "glove_ave3":"/home/wkai/SMP/feature/train_feature/Glove_tags_305613_average_3.csv",
    "lsa_ave3": "/home/wkai/SMP/feature/train_feature/LSA_tags+title_305613_average_3.csv",
    "tfidf_ave3": "/home/wkai/SMP/feature/train_feature/pca_tfidf_train_512_windows3.csv",
    "wordchar": "/home/wkai/SMP/feature/train_feature/wordchar_merge.csv",
    "category": "/home/wkai/SMP/feature/train_feature/train_category.csv",
    "other": "/home/wkai/SMP/feature/train_feature/data_305613.csv",
    "pathalias": "/home/wkai/SMP/feature/train_feature/Pathalias_Category_305613.csv",
    "pathalias_tsvd100":"/home/wkai/SMP/feature/train_feature/Pathalias_Category_tsvd100_305613.csv",
    # "userdata": "/home/wkai/SMP/feature/train_feature/train_userdata_305613.csv",
    "userdata": "/home/wkai/SMP/feature/train_feature/train_userdata_true_305613.csv",
    # 降维
    "TSVD512_category":"/home/wkai/SMP/feature/train_feature/TSVD512_Category_1epoch3_ResNext101_image_305613.csv", 
    "TSVD512_subcategory":"/home/wkai/SMP/feature/train_feature/TSVD512_Subcategory_1epoch3_ResNext101_image_305613.csv",
    # 先降维后滑动窗口
    "TSVD512_ave5_category":"/home/wkai/SMP/feature/train_feature/TSVD512_ave5_Category_1epoch3_ResNext101_image_305613.csv",
    "TSVD512_ave5_subcategory":"/home/wkai/SMP/feature/train_feature/TSVD512_ave5_Subcategory_1epoch3_ResNext101_image_305613.csv",
    # 滑动窗口
    "image_resnext_category_ave5":"/home/wkai/SMP/feature/train_feature/Category_ResNext101_image_305613_average_5.csv",
    "image_resnext_subcategory_ave5":"/home/wkai/SMP/feature/train_feature/Subcategory_ResNext101_image_305613_average_5.csv",
    # 先滑动窗口后降维
    "image_resnext_category_tsvd512_ave5": 
                        "/home/wkai/SMP/feature/train_feature/Category_ResNext101_image_305613_tsvd512_average_5.csv",
    "image_resnext_subcategry_tsvd512_ave5":
                        "/home/wkai/SMP/feature/train_feature/ Subcategory_ResNext101_image_305613_tsvd512_average_5.csv",
    "image_resnext_pretrain": "/home/wkai/SMP/feature/train_feature/Pretrained_ResNext101_image_305613.csv",
    "image_resnest_pretrain": "/home/wkai/SMP/feature/train_feature/Pretrained_ResNest269_image_305613.csv",
    "image_resnext_category": "/home/wkai/SMP/feature/train_feature/Category_1epoch3_ResNext101_image_305613.csv",
    "image_resnest_category": "/home/wkai/SMP/feature/train_feature/Category_1epoch3_ResNest269_image_305613.csv",
    "image_resnext_subcategory": 
            "/home/wkai/SMP/feature/train_feature/Subcategory_1epoch3_ResNext101_image_305613.csv",
    "image_resnest_subcategory": 
            "/home/wkai/SMP/feature/train_feature/Subcategory_1epoch3_ResNest269_image_305613.csv"
}

test_feature_filepath = {
    "uid": "/home/wkai/SMP/feature/test_feature/test_uid_180581.csv",
    "Fasttext_tag": "/home/wkai/SMP/feature/test_feature/FastText_tags.csv",
    "Fasttext_title": "/home/wkai/SMP/feature/test_feature/FastText_title.csv",
    "Fasttext_ave3":"/home/wkai/SMP/feature/test_feature/FastText_tags+title_180581_average_3.csv", 
    "Bert_tag": "/home/wkai/SMP/feature/test_feature/tags_bert_180581_average3.csv",
    "Bert_title": "/home/wkai/SMP/feature/test_feature/title_bert_180581_average3.csv",
    "Bert_ave3": "/home/wkai/SMP/feature/test_feature/Bert_tags+title_180581_average_3.csv",
    "Bert_tsvd512_ave3": "/home/wkai/SMP/feature/test_feature/Bert_tags+title_180581_tsvd512_average_3.csv",
    "glove": "/home/wkai/SMP/feature/test_feature/glove_tags_180581.csv",
    "glove_ave3": "/home/wkai/SMP/feature/test_feature/Glove_tags_180581_average_3.csv",
    "lsa_ave3": "/home/wkai/SMP/feature/test_feature/LSA_tags+title_180581_average_3.csv",
    "tfidf_ave3": "/home/wkai/SMP/feature/test_feature/pca_tfidf_test_512_windows3.csv",
    "wordchar": "/home/wkai/SMP/feature/test_feature/wordchar_merge_180581.csv",
    "category": "/home/wkai/SMP/feature/test_feature/test_category.csv",
    "other": "/home/wkai/SMP/feature/test_feature/data_180581.csv",
    "pathalias": "/home/wkai/SMP/feature/test_feature/Pathalias_Category_180581.csv",
    "pathalias_tsvd100":"/home/wkai/SMP/feature/test_feature/Pathalias_Category_tsvd100_180581.csv",
    "userdata": "/home/wkai/SMP/feature/test_feature/test_userdata_180581.csv",
    # 降维
    "TSVD512_category":"/home/wkai/SMP/feature/test_feature/TSVD512_Category_1epoch3_ResNext101_image_180581.csv", 
    "TSVD512_subcategory":"/home/wkai/SMP/feature/test_feature/TSVD512_Subcategory_1epoch3_ResNext101_image_180581.csv",
    # 先降维后滑动窗口
    "TSVD512_ave5_category":"/home/wkai/SMP/feature/test_feature/TSVD512_ave5_Category_1epoch3_ResNext101_image_180581.csv",
    "TSVD512_ave5_subcategory":"/home/wkai/SMP/feature/test_feature/TSVD512_ave5_Subcategory_1epoch3_ResNext101_image_180581.csv",
    # 滑动窗口
    "image_resnext_category_ave5":"/home/wkai/SMP/feature/test_feature/Category_ResNext101_image_180581_average_5.csv",
    "image_resnext_subcategory_ave5":"/home/wkai/SMP/feature/test_feature/Subcategory_ResNext101_image_180581_average_5.csv",
    # 先滑动窗口后降维
    "image_resnext_category_tsvd512_ave5":
                    "/home/wkai/SMP/feature/test_feature/Category_ResNext101_image_180581_tsvd512_average_5.csv",
    "image_resnext_subcategry_tsvd512_ave5":
                    "/home/wkai/SMP/feature/test_feature/Subcategory_ResNext101_image_180581_tsvd512_average_5.csv",
    "image_resnext_pretrain": "/home/wkai/SMP/feature/test_feature/Pretrained_ResNext101_image.csv",
    "image_resnest_pretrain": "/home/wkai/SMP/feature/test_feature/Pretrained_ResNest269_image.csv",
    "image_resnext_category": "/home/wkai/SMP/feature/test_feature/Category_1epoch3_ResNext101_image.csv",
    "image_resnest_category": "/home/wkai/SMP/feature/test_feature/Category_1epoch3_ResNest269_image.csv",
    "image_resnext_subcategory": "/home/wkai/SMP/feature/test_feature/Subcategory_1epoch3_ResNext101_image.csv",
    "image_resnest_subcategory": "/home/wkai/SMP/feature/test_feature/Subcategory_1epoch3_ResNest269_image.csv"
}

# # postdate label
# train_label_filepath = "../../features/splited_label/train_label.csv"
# validate_label_filepath = "../../features/splited_label/train_label.csv"

all_label_filepath="../../data/data_source/train/train_label.json"

def load_dataset(feature_list,flag="train"):
    if flag == "train":
        load_path=train_feature_filepath
    else:
        load_path=test_feature_filepath
    for i, feature_name in enumerate(feature_list):
        print("Loading {} ...".format(feature_name))
        feature = pd.read_csv(load_path[feature_name])
        print("feature: {}, len:{}".format(feature_name,len(feature.columns)))
        if i == 0:
            all_feature = feature
        else:
            all_feature = pd.merge(all_feature,feature)

    print(all_feature)
    print("all feature's len: {}".format(all_feature.columns))
    return all_feature


def train_catboost(X_train,y_train):
    class SrcMetric(object):
        def is_max_optimal(self):
            # Returns whether great values of metric are better
            return True

        def evaluate(self, approxes, target, weight):
            # approxes is a list of indexed containers
            # (containers with only __len__ and __getitem__ defined),
            # one container per approx dimension.
            # Each container contains floats.
            # weight is a one dimensional indexed container.
            # target is a one dimensional indexed container.
            assert len(approxes) == 1
            assert len(target) == len(approxes[0])
            weight_sum = 0.0
            for i in range(len(approxes[0])):
                w = 1.0 if weight is None else weight[i]
                weight_sum += w
            correlation, pvalue = spearmanr(approxes[0], target)
            return correlation, weight_sum

        def get_final_error(self, error, weight):
            # Returns final value of metric based on error and weight
            return error

    params = {
        "objective": "RMSE",
        "eval_metric": SrcMetric(),
        "custom_metric": ["MAE","RMSE","MAPE"],
        "learning_rate": 0.03,
        "max_depth": 6,
        "max_leaves": 31,
        "boost_from_average": True,
        "reg_lambda": 3.0,
        # "use_best_model": True,
        "thread_count": -1,
        # "one_hot_max_size":255,
        "task_type": "CPU",
        "boosting_type": "Plain",
        "train_dir": "RESULT_ALL",
        "random_seed": random_seed
    }


    cat_features = ["uid", "Category", "Subcategory", "Concept", "Mediatype", "Ispublic",
                    "ispro", "canbuypro"]

    print("params: {}\ncat_features: {}".format(params,cat_features))
    train_data = catboost.Pool(X_train,y_train,cat_features=cat_features)
    
    del X_train
    gc.collect()

    iterations = 200000
    bst = catboost.train(train_data, params, iterations=iterations,
                         verbose=30, plot=False,save_snapshot=True,snapshot_file="RESULT_ALL",snapshot_interval=600)

    print(bst.get_params())


    preds = bst.predict(train_data)
    correlation, pvalue = spearmanr(train_data.get_label(),preds)
    mae = mean_absolute_error(train_data.get_label(), preds)
    mse = mean_squared_error(train_data.get_label(), preds)
    print("\nIn Best_iteration: SRC:{},MAE:{},MSE:{},iterations:{}".format(correlation, mae, mse, iterations))
    
    log["all_params"]=bst.get_all_params()
    log["used_params"]=bst.get_params()
    log["best_iteration"]=iterations
    log["SRC"]=correlation
    log["MAE"]=mae
    log["MSE"]=mse

    # bst.save_model(fname="/home/wkai/SMP/model/RESULT_ALL_model",pool=train_data)
    return bst


def write_log(path="/home/wkai/SMP/log"):
    filename="log_RESULT_ALL"+"_".join([str(i) for i in time.localtime(time.time())[:5]])+".txt"
    with open (os.path.join(path,filename),"a") as f:
        f.write("Model: Catboost\n")
        f.write("used feature: {}\n".format(log["used_feature"]))
        f.write("all params: {}\n".format(log["all_params"]))
        f.write("used params: {}\n".format(log["used_params"]))
        f.write("best_iteration: {}\n".format(log["best_iteration"]))
        f.write("SRC: {}\n".format(log["SRC"]))
        f.write("MAE: {}\n".format(log["MAE"]))
        f.write("MSE: {}\n".format(log["MSE"]))


def write_submission(path="./submission", pids=None, preds=None):
    result = [
        {
            "post_id": "post"+str(pids[i]),
            "popularity_score":round(float(preds[i]), 4)
        } for i in range(len(pids))
    ]
    submission = {
        "version": "VERSION 0.1",
        "result": result,
        "external_data": {
            "used": "true",
            "details": "ResNext pre-trained on ImageNet training set"
        }
    }
    tail = [str(i) for i in time.localtime(time.time())[:5]]
    tail = "_".join(tail)
    filepath = os.path.join(path, "submission_with_user"+tail+".json")
    with open(filepath, "w") as file:
        json.dump(submission, file)


def main(args):
    log["used_feature"]=args.feature+["boost_fill","mlp"]

    print("Loading all feature ...")
    train=pd.read_pickle(train_feature_pklpath)
    print(train)

    X_train,y_train = train.iloc[:, 1:-1], train["label"]
    del train

    print("Train catboost model...")
    if args.model == "catboost":
        bst = train_catboost(X_train,y_train)
    del X_train,y_train
    
    print("Loading test feature...")
    # test=pd.read_pickle("/home1/wangph/kai/fill/test+label_with_user_result_180581.pkl")
    test=pd.read_pickle(test_feature_pklpath)
    pids, X_test= test["pid"], test.iloc[:,1:]
    # predict
    preds = bst.predict(X_test)

    print("Start store log...")
    write_log(path="/home/wkai/SMP/log")

    print("Start store result...")
    write_submission(path=args.submission_path, pids=pids, preds=preds)

    # feature importance 
    df_important=pd.DataFrame({"feature_name":bst.feature_names_,"importance":bst.feature_importances_})
    df_important=df_important.sort_values(by=["importance"],ascending=False)
    df_important.to_csv("importance_with_user.csv",index=False)
    # print(df_important)
    print("top 30:")
    print(df_important.iloc[:30,:])
    print("down 10:")
    print(df_important.iloc[-10:,:])

    print("Used feature: {}.".format(args.feature))
    print("The process over!")


def parse_arguments():
    parser = argparse.ArgumentParser(description=" SMP Catboost model")
    parser.add_argument("-model", "--model", type=str,
                        choices=["xgboost", "lightgbm", "catboost"],
                        default="catboost",
                        help="which model(default: catboost)")
    parser.add_argument("-f", "--feature", type=str,
                        choices=["Fasttext_tag", "Fasttext_title","Fasttext_ave3", "Bert_tag", "Bert_title", "Bert_ave3", 
                                "Bert_tsvd512_ave3","tfidf_ave3","glove", "glove_ave3","lsa_ave3","uid",
                                "wordchar","category", "other","pathalias", "pathalias_tsvd100", "userdata",
                                "image_resnext_category_ave5", "image_resnext_subcategory_ave5",
                                "TSVD512_ave5_subcategory","TSVD512_ave5_category","TSVD512_subcategory","TSVD512_category",
                                "image_resnext_category_tsvd512_ave5","image_resnext_subcategry_tsvd512_ave5",
                                "image_resnext_pretrain", "image_resnest_pretrain", "image_resnext_category", "image_resnest_category", "image_resnext_subcategory", "image_resnest_subcategory"],
                        nargs="?",
                        const=["Fasttext_tag", "Fasttext_title", "category",
                               "other", "image_resnext_subcategory"],
                        # default=["category","other","userdata"],
                        default=["Fasttext_ave3","Bert_ave3","wordchar","tfidf_ave3","glove_ave3","lsa_ave3","uid","userdata",
                        "pathalias", "category", "other", "image_resnext_subcategory_ave5"],
                        help="which feature will be used")
    # "Fasttext_tag", "Fasttext_title","Bert_tag","Bert_title","category", "other",
    parser.add_argument("-output", "--submission_path", type=str,
                        default=to_submit_path,
                        help="SMP file(.json) will be submit path")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
