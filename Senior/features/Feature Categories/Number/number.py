# !/usr/bin/env python
# coding: utf-8
'''
@File    :   number.py
@Time    :   2020/04/19 00:05:54
@Author  :   Wang Kai 
@Version :   1.0
@Contact :   wk15@mail.ustc.edu.cn
'''
# This is used to extract other features (except:images,alltags,titles)

import argparse
import csv
import time

import numpy as np
import pandas as pd
from tqdm import tqdm


def process(df):
    data = {}
    data["pid"] = np.array(df["Pid"]).reshape(-1, 1)
    data["uid"] = np.array(df["Uid"]).reshape(-1, 1)

    data["Ispublic"] = np.array(df["Ispublic"] == 1).astype(int).reshape(-1, 1)
    data["Mediatype"] = np.array(
        df["Mediatype"] == "photo").astype(int).reshape(-1, 1)
    data["Mediastatus"] = np.array(
        df["Mediastatus"] == "ready").astype(int).reshape(-1, 1)
    data["ispro"] = np.array(df["ispro"] == 1).astype(int).reshape(-1, 1)
    data["canbuypro"] = np.array(
        df["canbuypro"] == 1).astype(int).reshape(-1, 1)
    data["timezone_offset"] = np.array(
        df["timezone_offset"].astype(int)).reshape(-1, 1)
    data["timezone_id"] = np.array(
        df["timezone_timezone_id"].astype(int)).reshape(-1, 1)
    data["Latitude"] = np.array(df["Latitude"].astype(float)).reshape(-1, 1)
    data["Geoaccuracy"] = np.array(
        df["Geoaccuracy"].astype(int)).reshape(-1, 1)
    data["photo_count"] = np.array(
        df["photo_count"].astype(int)).reshape(-1, 1)

    print("process user_description...")
    user_feature = []  # 399 dim
    for item in tqdm(df["user_description"].tolist()):
        user_feature.append(list(map(float, item.strip().split(","))))
    data["user_description"] = np.array(user_feature).reshape(-1, 399)

    print("process the Postdate")
    Postdate = []  # 9 dim
    for item in df["Postdate"].tolist():
        Postdate.append(list(time.localtime(item)[:]))
    data["Postdate"] = np.array(Postdate).reshape(-1, 9)

    # print("process the photo_firstdatetaken")
    # photo_firstdatetaken=[] # 9 dim
    # for item in df["photo_firstdatetaken"].tolist():
    #     photo_firstdatetaken.append(list(time.localtime(item)[:]))
    # data["photo_firstdatetaken"]=np.array(photo_firstdatetaken)

    # For missing data
    print("process Longitude...")
    Longitude_feature = []  # 1dim
    for item in tqdm(df["Longitude"].tolist()):
        item = item.strip()
        Longitude_feature.append(float(item) if len(item) > 0 else 0.0)
    # only 34913 item in 305613
    data["Longitude"] = np.array(Longitude_feature).reshape(-1, 1)

    print("process location_description...")
    location_feature = []  # 400 dim
    for item in tqdm(df["location_description"].tolist()):
        item = item.strip().split(",")
        location_feature.append(list(map(float, item))
                                if len(item) > 1 else [0. for i in range(400)])
    data["location_description"] = np.array(
        location_feature).reshape(-1, 400)   # 152806 is empty

    # print("process the photo_firstdate")
    # TODO
    pass

    return data


def main(args):

    tag_filepath = args.tag_filepath
    temporalspatial_filepath = args.temporalspatial_filepath
    userdata_filepath = args.userdata_filepath
    additional_filepath = args.additional_filepath

    df_tag = pd.read_json(tag_filepath)
    df_temporalspational = pd.read_json(temporalspatial_filepath)
    df_user = pd.read_json(userdata_filepath)
    df_additional = pd.read_json(additional_filepath)
    df = pd.concat(
        [pd.merge(pd.merge(df_tag, df_temporalspational), df_additional), df_user], axis=1)

    # Process the data
    data = process(df)

    print("start to save date to" + args.feature_filepath)
    with open(file=args.feature_filepath, mode="w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        col = ["year", "mon", "mday", "hour",
               "min", "sec", "wday", "yday", "isdst"]
        columns = ["pid", "uid", "Mediatype", "Latitude", "Longitude", "Geoaccuracy", "Ispublic",
                   "Mediastatus", "ispro", "canbuypro", "timezone_id", "timezone_offset", "photo_count"] + \
            ["Postdate_"+i for i in col] + \
            ["user_description"+str(i+1) for i in range(399)] + \
            ["location_description"+str(i+1) for i in range(400)]
        writer.writerow(columns)

        # (305613,821) dim
        data_values = np.concatenate((data["pid"], data["uid"], data["Mediatype"], data["Latitude"],
                                      data["Longitude"], data["Geoaccuracy"], data["Ispublic"], data["Mediastatus"],
                                      data["ispro"], data["canbuypro"], data["timezone_id"], data["timezone_offset"],
                                      data["photo_count"], data["Postdate"], data["user_description"], data["location_description"]), axis=1)

        print("pid shape : ", data["pid"].shape,
              "all data shape:", data_values.shape)
        writer.writerows(data_values.tolist())
        print("save the data over ! ")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process the number data")
    parser.add_argument("--tag_filepath", type=str,
                        default="/home/wangkai/SMP/data/train/train_tags.json",
                        help="tag filepath")
    parser.add_argument("--temporalspatial_filepath", type=str,
                        default="/home/wangkai/SMP/data/train/train_temporalspatial.json",
                        help="temporalspatial filepath")
    parser.add_argument("--userdata_filepath", type=str,
                        default="/home/wangkai/SMP/data/train/train_userdata.json",
                        help="userdata filepath")
    parser.add_argument("--additional_filepath", type=str,
                        default="/home/wangkai/SMP/data/train/train_additional.json",
                        help="addltional filepath")
    parser.add_argument("--feature_filepath", type=str,
                        default="/home/wangkai/SMP/feature/data_305613.csv",
                        help="time feature filepath")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
