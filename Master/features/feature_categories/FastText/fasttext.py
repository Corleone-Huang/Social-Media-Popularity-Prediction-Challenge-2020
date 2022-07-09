# !/usr/bin/env python
# coding: utf-8
'''
@File    :   fasttext.py
@Time    :   2020/04/19 00:00:51
@Author  :   Wang Kai 
@Version :   1.0
@Contact :   wk15@mail.ustc.edu.cn
'''
# This is used to extract tags and titles feature by FastText model

import argparse
import csv
import io
import re
import time

import numpy as np
import pandas as pd
# import fasttext
from gensim.models import fasttext
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tqdm import tqdm

tags_filepath="../../../data/data_source/train/train_tags.json"
model_filepath="../../pretrained_model/crawl-300d-2M-subword.bin"
title_feature_filepath="../../extracted_features/FastText_title_305613.csv"
tags_feature_filepath="../../extracted_features/FastText_tags_305613.csv"

def clear_str(string):
    stop = set(stopwords.words("english"))
    wnl = WordNetLemmatizer()
    #  clear the word
    # p4=re.compile(r'[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）0-9 , : ; \-\ \[\ \]\ ]')
    p1 = re.compile(r'-\{.*?(zh-hans|zh-cn):([^;]*?)(;.*?)?\}-')
    p2 = re.compile(r'[(][: @ . , ？！\s][)]')
    p3 = re.compile(r'[「『]')
    p4 = re.compile(
        r'[\s+\*.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（） , : ; \-\ \[\ \]\ ]')   # save number 1-9
    line = p1.sub(r' ', string)
    line = p2.sub(r' ', line)
    line = p3.sub(r' ', line)
    line = p4.sub(r' ', line)
    string = re.sub(u" ", " ", line)

    string_data = word_tokenize(string)
    string_data = [wnl.lemmatize(w.lower())
                   for w in string_data if w not in stop]
    # print(string_data)

    return string_data


def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = list(map(int, fin.readline().split()))
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = list(map(float, tokens[1:]))
    return data


def process_titles(model, df, feature_path="./titles_feature.csv"):
    start_time = time.time()

    all_titles = np.array(df["Title"])
    titles_feature = []
    num = 0
    for title in tqdm(all_titles):
        feature = []

        needless = []
        index = -1
        for word in clear_str(title):
            index = index + 1
            try:
                feature.append(model.wv[word])
            except KeyError:
                feature.append([0. for i in range(300)])
                needless.append(index)  # all 0's index
            # continue
        feature = np.delete(np.array(feature), needless, axis=0)
        if 0 == feature.shape[0]:
            # print(np.where(all_titles == title), title)
            # print(title)
            num = num+1
            titles_feature.append([0. for i in range(300)])
        else:
            titles_feature.append(np.mean(feature, axis=0).tolist())

        # titles_feature.append(
        #     np.mean(np.delete(np.array(feature), needless, axis=0), axis=0).tolist())
    print("the number of unporcessed is ", num)
    with open(feature_path, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        colums = ["pid", "uid"]+["Fasttext_title_feature"+str(i+1) for i in range(300)]
        writer.writerow(colums)
        uids = np.array(df["Uid"]).reshape(-1, 1)
        pids = np.array(df["Pid"]).reshape(-1, 1)
        features = np.array(titles_feature)

        print(pids.shape, uids.shape, features.shape)
        writer.writerows(np.concatenate(
            (pids, uids, features), axis=1).tolist())

    time_elapsed = time.time()-start_time
    print('complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))


def process_tags(model, df, feature_path="./tags_feature.csv"):
    start_time = time.time()
    all_tags = np.array(df["Alltags"])
    tags_feature = []
    num = 0
    for tag in tqdm(all_tags):
        feature = []

        needless = []
        index = -1
        for word in clear_str(tag):
            index = index+1
            try:
                feature.append(model.wv[word])
            except KeyError:
                feature.append([0. for i in range(300)])
                needless.append(index)
            # continue
        feature = np.delete(np.array(feature), needless, axis=0)
        if 0 == feature.shape[0]:
            # print(np.where(all_tags == tag), tag)
            # print(tag)
            num = num+1
            tags_feature.append([0. for i in range(300)])
        else:
            tags_feature.append(np.mean(feature, axis=0).tolist())
        # tags_feature.append(
        #     np.mean(np.delete(np.array(feature), needless, axis=0), axis=0).tolist())
    print("the number of unporcessed is ", num)
    with open(feature_path, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        colums = ["pid", "uid"]+["Fasttext_tag_feature"+str(i+1) for i in range(300)]
        writer.writerow(colums)
        uids = np.array(df["Uid"]).reshape(-1, 1)
        pids = np.array(df["Pid"]).reshape(-1, 1)
        features = np.array(tags_feature)

        print(pids.shape, uids.shape, features.shape)
        writer.writerows(np.concatenate(
            (pids, uids, features), axis=1).tolist())

    time_elapsed = time.time()-start_time
    print('complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))


def main(args):
    df = pd.read_json(args.tags_filepath)
    print("Loading the pretrained model...")
    
    model = fasttext.load_facebook_model(args.model_filepath)
    # model = fasttext.load_model(args.model_filepath)
    print("Extracting the title feature by FastText")
    process_titles(model, df[["Uid", "Pid", "Title"]],
                   args.title_feature_filepath)
    print("Extracting the tags feature by FastText")
    process_tags(model, df[["Uid", "Pid", "Alltags"]],
                 args.tags_feature_filepath)
    print("All features extract over !")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Extract tags and titles feature by FastText model")
    parser.add_argument("--tags_filepath", type=str,
                        default=tags_filepath,
                        help="tags_filepath")
    parser.add_argument("--model_filepath", type=str,
                        default=model_filepath,
                        help="FastText model filepath")
    parser.add_argument("--title_feature_filepath", type=str,
                        default=title_feature_filepath,
                        help="titles feature filepath extracted by FastText")
    parser.add_argument("--tags_feature_filepath", type=str,
                        default=tags_feature_filepath,
                        help="tags feature filepath extracted by FastText")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
