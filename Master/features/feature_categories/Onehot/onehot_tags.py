# 本代码用来提取SMP data中的Alltags数据的特征
# author： chenxin  2020-4-22
import argparse

import pandas as pd
import numpy as np
import re
import csv
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from tqdm import tqdm
from sklearn.decomposition import PCA
import nltk

# nltk.download('stopwords')

tags_filepath = "../../../data/data_source/train/train_tags.json"
save_tags = "../../extracted_features/pca_onehot_tags_10000-256.csv"


np.set_printoptions(threshold=9999999)
def parse_args():
    parser = argparse.ArgumentParser(description='------onehot任务-------')
    parser.add_argument('--data_path', default=tags_filepath,
                        dest='data_path',type=str, help='文本json数据的路径')
    parser.add_argument('--save_tags', default=save_tags,
                        dest='save_tags',help='存放pca之后onehot提取的tags特征的路径', type=str)
    args = parser.parse_args(args=[])
    return args
args =parse_args()
tags_path = args.data_path
df = pd.read_json(tags_path)

def low_findall(input_line):
    result_list = re.findall('[a-zA-Z]+', input_line.lower())
    return result_list

df['tags'] = df.Alltags.apply(low_findall)
print('数据处理中......')
#去除停用词
stop=stopwords.words('english')
df['tags']=df['tags'].apply(lambda sen:" ".join(x for x in sen if x not in stop))

print('请稍侯......')
#稀缺词去除，十个
freq1= pd.Series(' '.join(df['tags']).split()).value_counts()[-10:]
df['tags'] = df['tags'].apply(lambda x: " ".join(x for x in x.split() if x not in freq1))

print('请稍侯......')
#词干提取
stemmer = SnowballStemmer("english")
df['tags'] = df['tags'].apply(lambda x:" ".join([stemmer.stem(x) for x in x.split()]))

print('请稍侯......')
#词性还原
wordnet_lemmatizer = WordNetLemmatizer()
df['tags'] = df['tags'].apply(lambda x:" ".join([wordnet_lemmatizer.lemmatize(x) for x in x.split()]))

print('构建词典中')
# 构建title的字典vocabulary
token_index1 = {}
for sample in tqdm(df['tags']):
    for word in sample.split():
        if word in token_index1:
            token_index1[word] = token_index1[word] + 1
        else:
            token_index1[word] = 1

df1 = pd.DataFrame({'word':list(token_index1.keys()),'number':list(token_index1.values())})
df1.sort_values(by=['number'],ascending=False,inplace=True)
df1.reset_index(drop=True, inplace=True)
df1 = df1.head(10000)

results = np.zeros((305613,10000))
for i,sample in tqdm(enumerate(df['tags'])):
    for word in sample.split():
        if word in df1['word'].values:
            index = df1[word==df1.word].index
            results[i,index[0]]=1

print('pca降维中。。。')
pca=PCA(n_components=256)#降到2048维
pca.fit(results)
results = pca.transform(results)

# print(token_index)
csv_file_path1 = args.save_tags
#存储数据
csv_file_title = open(csv_file_path1, "w",newline='')
file_writer_title = csv.writer(csv_file_title)
file_writer_title.writerow(["pid", "uid"]+["tags_feature_"+str(i+1) for i in range(256)])
for i, sample in tqdm(enumerate(results)):
    title_store = [df['Pid'][i], df['Uid'][i]] + list(sample)
    file_writer_title.writerow(title_store)
csv_file_title.close()
