# !/usr/bin/env python
# coding: utf-8
'''
@File    :   split_dataset.py
@Time    :   2020/04/18 23:26:56
@Author  :   Wang Kai 
@Version :   1.0
@Contact :   wk15@mail.ustc.edu.cn
'''
# save:[pids,labels]

import argparse
import time

import numpy as np
import pandas as pd

random_seed = 2020

train_temporalspatial_filepath = "../data_source/train/train_temporalspatial.json"
all_train_label_filepath = "../data_source/train/train_label.txt"
splited_train_label_filepath = "../../features/splited_label/train_label.csv"
splited_validate_label_filepath = "../../features/splited_label/validate_label.csv"
is_random_split = False
train_ratio = 0.8


def timestamp_to_time(datetime):
    date_time = []
    for date in datetime.tolist():
        temp = list(time.localtime(date)[:])
        date_time.append(temp)
    return date_time


def split_dataset(args):
    df_temporalspatial = pd.read_json(args.temporalspatial_filepath)
    df_label = pd.read_csv(args.label_filepath, header=None, names=["label"])
    df = pd.concat([df_temporalspatial, df_label], axis=1)
    df.rename(columns={"Pid": "pid"}, inplace=True)
    if args.random_split:
        # random sort
        print("you will split the dataset randomly")
        df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    else:
        # sort by Posttime
        print("you will split the dataset by Postdate")
        df["Posttime"] = timestamp_to_time(df["Postdate"])
        df.sort_values("Posttime", inplace=True, ascending=True)
        df.reset_index(drop=True, inplace=True)

    train_ratio = args.train_ratio

    result = df[["pid", "label"]]
    train = result.iloc[:int(train_ratio*len(result))]
    validate = result.iloc[int(train_ratio*len(result)):]

    if args.is_random_split:
        train.to_csv(args.splited_train_label_filepath, index=False)
        validate.to_csv(
            args.splited_validate_label_filepath, index=False)
    else:
        train.to_csv(args.splited_train_label_filepath, index=False)
        validate.to_csv(
            args.splited_validate_label_filepath, index=False)
    print("split the dataset to train and validate over !")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Split the train and validate dataset")
    parser.add_argument("--temporalspatial_filepath", type=str,
                        default=train_temporalspatial_filepath,
                        help="temporalspatial filepath")
    parser.add_argument("--label_filepath", type=str,
                        default=all_train_label_filepath,
                        help="all label filepath")
    parser.add_argument("--is_random_split", type=bool, default=is_random_split,
                        help="split dataset random ?(default=False)")
    parser.add_argument("--train_ratio", type=float, default=train_ratio,
                        help="train_num/total_dataset (default: 0.95)")
    parser.add_argument("--train_label_filepath", type=str,
                        default=splited_train_label_filepath,
                        help="train label's filepath")
    parser.add_argument("--validate_label_filepath", type=str,
                        default=splited_validate_label_filepath,
                        help="validate label's filepath")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    assert 0 < args.train_ratio < 1, "the input train_ratio must in (0.0, 1.0)"
    split_dataset(args)
