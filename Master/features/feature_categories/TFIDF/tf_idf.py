# 本代码使用tf-idf提取SMP data中的Alltags, Titles数据的特征
# 结果保存在/home/cx/SMP/data/preprocess文件夹中
# 输出alltags/title_length_305613.csv，保存每一项的数据的tfidf值，维度为20000维，每一行的格式为[uid,pid,feature]
# title/tags特征向量维度为4096
# author： chenxin  2020-4-2
import argparse
import csv
import pandas as pd
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from tqdm import tqdm
import re

np.set_printoptions(threshold=9999999)


data_path = "../../../data/data_source/train/train_tags.json"

title_feature_path = "../../extracted_features/tfidf_title_305613.csv"
tags_feature_path = "../../extracted_features/tfidf_tags_305613.csv"


def parse_args():
    parser = argparse.ArgumentParser(description='------tf-idf任务-------')
    parser.add_argument('--data_path', default=data_path, dest='data_path',
                        type=str, help='文本json数据的路径')
    parser.add_argument('--save_title', default=title_feature_path,
                        dest='save_title',
                        help='存放提取的title特征的路径', type=str)
    parser.add_argument('--save_tags', default=tags_feature_path,
                        dest='save_tags',
                        help='存放提取的tags特征的路径', type=str)
    args = parser.parse_args()
    return args


# 加载数据
args = parse_args()
data_path = args.data_path
save_title = args.save_title
save_tags = args.save_tags
df_data = pd.read_json(data_path)

# map(lambda x:len(str(x).split(" ")),tags)  #split分词，计算每行的词数
# map(lambda x:len(str(x)),tags)  #计算每行的字符数

# 小写，去数字，去符号


def low_findall(input_line):
    result_list = re.findall('[a-zA-Z]+', input_line.lower())
    return result_list


df_data['tags'] = df_data.Alltags.apply(low_findall)
df_data['title'] = df_data.Title.apply(low_findall)
print('数据处理中......')
# 去除停用词
stop = stopwords.words('english')
df_data['tags'] = df_data['tags'].apply(
    lambda sen: " ".join(x for x in sen if x not in stop))
df_data['title'] = df_data['title'].apply(
    lambda sen: " ".join(x for x in sen if x not in stop))
print('请稍侯......')
# 词性还原
wordnet_lemmatizer = WordNetLemmatizer()
df_data['tags'] = df_data['tags'].apply(lambda x: " ".join(
    [wordnet_lemmatizer.lemmatize(x) for x in x.split()]))
df_data['title'] = df_data['title'].apply(lambda x: " ".join(
    [wordnet_lemmatizer.lemmatize(x) for x in x.split()]))
#tags = list(map(lambda element:re.sub(reg,' ',element),tqdm(tags)))

tags = list(df_data['tags'])
title = list(df_data['title'])
pid = list(df_data['Pid'])
uid = list(df_data['Uid'])

# 计算tags和title的tf-idf
tfidf_vectorizer = TfidfVectorizer(max_features=4096, lowercase=True, analyzer='word',
                                   stop_words='english', ngram_range=(1, 1))
tags_vec = tfidf_vectorizer.fit_transform(tags)
title_vec = tfidf_vectorizer.fit_transform(title)

tags_feature = tags_vec.toarray().astype(np.float32)
title_feature = title_vec.toarray().astype(np.float32)

csv_file_tags = open(save_tags, "w", newline='')
csv_file_title = open(save_title, "w", newline='')
file_writer_tags = csv.writer(csv_file_tags)
file_writer_title = csv.writer(csv_file_title)
# file_writer_tags.writerow(["pid", "uid", "tags"])  #写入列名
file_writer_title.writerow(
    ["pid", "uid"]+["title_feature_"+str(i+1) for i in range(4096)])
file_writer_tags.writerow(
    ["pid", "uid"]+["tags_feature_"+str(i+1) for i in range(4096)])

# 保存tags的特征数据
for i, sample in tqdm(enumerate(tags_feature)):
    tags_store = [pid[i], uid[i]] + list(sample)
    file_writer_tags.writerow(tags_store)
csv_file_tags.close()

# 保存title的特征数据
for i, sample in tqdm(enumerate(title_feature)):
    title_store = [pid[i], uid[i]] + list(sample)
    file_writer_title.writerow(title_store)
csv_file_title.close()
