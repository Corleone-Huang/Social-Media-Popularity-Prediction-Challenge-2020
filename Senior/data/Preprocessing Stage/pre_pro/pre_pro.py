import os
import sys
import csv
import nltk
import string
import argparse
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# 文本预处理
def textPrecessing(text):
    # 小写化
    text = text.lower()
    # 去除特殊标点
    for c in string.punctuation:
        text = text.replace(c, '')
    # 分词
    wordLst = nltk.word_tokenize(text)
    # 去除停用词
    filtered = [w for w in wordLst if w not in stopwords.words('english')]
    # 词干化
    ps = PorterStemmer()
    filtered = [ps.stem(w) for w in filtered]
    return " ".join(filtered)

def text_preprocess(dframe, args):
    print("preprocessing...")
    docLst = []
    for desc in dframe:
        docLst.append(textPrecessing(desc).encode('utf-8').decode('utf-8', 'strict'))

    with open(args.text_pre_path, 'w') as f:
        for line in docLst:
            f.write(line+'\n')
    return docLst

if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser(description="pre_process")
    parser.add_argument('--do_title_pre', '-D', type=int)
    parser.add_argument('--input_path', '-i', type=str,
                        help='the path of input text corpus，train_tags')
    parser.add_argument('--text_pre_path', '-t', type=str,
                        help="the file path to store text after preprocess")
    args = parser.parse_args()

    docLst = list()
    if args.do_title_pre:
        # 分别提取出title、Alltags
        df_tags = pd.read_json(args.input_path)
        df = df_tags[["Uid", "Pid", "Title", "Alltags"]]
        dframe = df["Title"]
        #dframe = dframe[:200]
        docLst = text_preprocess(dframe, args)
    else:
        df_tags = pd.read_json(args.input_path)
        df = df_tags[["Uid", "Pid", "Title", "Alltags"]]
        dframe = df["Alltags"]
        #dframe = dframe[:200]
        docLst = text_preprocess(dframe, args)
