# !/usr/bin/env python
# coding: utf-8
'''
@File    :   category.py
@Time    :   2020/04/18 23:54:07
@Author  :   Wang Kai 
@Version :   1.0
@Contact :   wk15@mail.ustc.edu.cn
'''
# This is used to extract Category&Subcategory features by one-hot

import argparse
import pandas as pd


train_category_filepath = "../../../data/data_source/train/train_category.json"
test_category_filepath = "../../../data/data_source/test/train_category.json"
train_feature_filepath = "../../extracted_features/onehot_train_category&subcategory_305613.csv"
test_feature_filepath = "../../extracted_features/onehot_test_category&subcategory_180581.csv"


def main(args):
    train_df = pd.read_json(args.train_dataset_filepath)
    # print(train_df)
    test_df = pd.read_json(args.test_dataset_filepath)
    # print(test_df)
    # concat the dataset train+test
    df = pd.concat([train_df, test_df], axis=0)
    print(df)
    df = pd.get_dummies(df, columns=["Category", "Subcategory"])

    df = df.drop("Concept", axis=1)
    df = df.rename(columns={"Pid": "pid", "Uid": "uid"})

    df_train = df[:305613]
    # print(df_train)
    df_train.to_csv(args.train_feature_filepath, index=False)
    print("train over")

    df_test = df[305613:]
    # print(df_test)
    df_test.to_csv(args.test_feature_filepath, index=False)
    print("test over")
    # print(df)
    print("All Over")
    pass


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="One-Hot Category and Subcategory")
    parser.add_argument("--train_dataset_filepath", type=str,
                        default="/home/wangkai/SMP/data/train/train_category.json",
                        help="train_category.json filepath")
    parser.add_argument("--test_dataset_filepath", type=str,
                        default="/home/wangkai/SMP/data/test/test_category.json",
                        help="test_category.json filepath")
    parser.add_argument("--train_feature_filepath", type=str,
                        default="/home/wangkai/SMP/feature/onehot_train_category&subcategory_305613.csv",
                        help="train category&subcategory feature filepath")
    parser.add_argument("--test_feature_filepath", type=str,
                        default="/home/wangkai/SMP/feature/onehot_test_category&subcategory_180581.csv",
                        help="test category&subcategory feature filepath")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
