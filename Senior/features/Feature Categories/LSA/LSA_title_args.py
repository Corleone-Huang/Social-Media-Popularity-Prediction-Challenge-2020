#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import argparse
pd.set_option("display.max_colwidth", 200)

# ###parse args
parser = argparse.ArgumentParser(description="lsa_title")
parser.add_argument('--num_of_topic', '-K', type=int)
parser.add_argument('--num_of_iteration', '-IT', type=int)
parser.add_argument('--input_data_path', '-i', type=str,
                    help='the path of input data，title.csv')
parser.add_argument('--output_doc_topic_path', '-dt', type=str,
                    help='the path of document_topic.csv')
parser.add_argument('--data_path', '-data', type=str,
                    help='the path of data，train_tags.json')
args = parser.parse_args()


# ### Data Preprocessing
#title
csv_file = args.input_data_path
csv_data = pd.read_csv(csv_file, low_memory = False, header=None, skip_blank_lines=False)#防止弹出警告，空行补NaN
csv_data.columns = ['title']
csv_df = pd.DataFrame(csv_data['title'])

# removing everything except alphabets`
csv_df['clean_doc'] = csv_df['title'].str.replace("[^a-zA-Z#]", " ")
csv_df[['clean_doc']] = csv_df[['clean_doc']].astype(str)

# removing short words
csv_df['clean_doc']= csv_df['clean_doc'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))

# make all text lowercase
csv_df['clean_doc'] = csv_df['clean_doc'].apply(lambda x: x.lower())
print(csv_df['clean_doc'])


# It is a good practice to remove the stop-words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

# tokenization
tokenized_doc = csv_df['clean_doc'].apply(lambda x: x.split()) 

# remove stop-words
tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words])

# de-tokenization
detokenized_doc = []
for i in range(len(csv_df)):
    t = ' '.join(tokenized_doc[i])
    detokenized_doc.append(t)
    
csv_df['clean_doc'] = detokenized_doc    


# ### Document-Term Matrix

# This is the first step towards topic modeling.
# We will use sklearn's TfidfVectorizer to create a document-term matrix with 1000 words.
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words='english', 
                             max_features= 1000, # keep top 1000 words
                             max_df = 0.5, 
                             smooth_idf=True)

X = vectorizer.fit_transform(csv_df['clean_doc'])
#X = vectorizer.fit_transform(csv_df)

X.shape # check shape of the document-term matrix

# ## Topic Modeling

# The next step is to represent each and every term and document as a vector.
# We will use the document-term matrix and decompose it into multiple matrices.
# We will use sklearn's TruncatedSVD to perform the task of matrix decomposition.
# let's try to have 20 topics for our text data.
# The number of topics can be specified by using the *n_components* parameter.

from sklearn.decomposition import TruncatedSVD

# SVD represent documents and terms in vectors 
svd_model = TruncatedSVD(n_components=args.num_of_topic, algorithm='randomized', n_iter=args.num_of_iteration, random_state=122)

svd_model.fit(X)
len(svd_model.components_)

X2 = svd_model.fit_transform(X)
X2

# ##print 'title_doc_topic_path.csv'
# The components of svd_model are our topics and we can access them using svd_model.components_.
# Finally let's print a few most important words in each of the 20 topics and see how our model has done.

df_tags = pd.read_json(args.data_path)
df = df_tags[["Uid", "Pid", "Title", "Alltags"]]
uid = np.array(df["Uid"])
pid = np.array(df["Pid"])
print(uid.shape)
uid = uid.reshape(uid.shape[0], 1)
pid = pid.reshape(pid.shape[0], 1)

with open(args.output_doc_topic_path, 'w') as f:
    wr = csv.writer(f)
    wr.writerow(['Uid', 'Pid', 'title'])
    data = np.hstack((uid, pid))
    title_save = np.hstack((data, X2))
    wr.writerows(title_save)

terms = vectorizer.get_feature_names()
for i, comp in enumerate(svd_model.components_):
    terms_comp = zip(terms, comp)
    sorted_terms = sorted(terms_comp, key= lambda x:x[1], reverse=True)[:7]
    print("Topic "+str(i)+": ")
    for t in sorted_terms:
        print(t[0])
        print(" ")





