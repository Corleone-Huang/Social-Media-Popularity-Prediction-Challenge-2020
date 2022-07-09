# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import argparse
from tcn import tcn_full_summary
import os
from tcn import TCN
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense
import json
import pandas as pd
import numpy as np
from tensorflow import keras
import tensorflow as tf


train_label_path = "../../features/splited_label/train_label.csv"
valid_label_path = "../../features/splited_label/validate_label.csv"
train_feature_pklpath = "../../features/extracted_features/train+label_with_user_result_305613.pkl"
test_feature_pklpath = "../../features/extracted_features/test+label_with_user_no_filled_result_180581.pkl"



parser = argparse.ArgumentParser()
parser.add_argument('-f', '--feature_path',
                    type=str, help='The path of concatenated feature in pickle form.',
                    default=train_feature_pklpath)
parser.add_argument('--train_label_path',
                    type=str, help='The path of train set split.',
                    default=train_label_path)
parser.add_argument('--valid_label_path',
                    type=str, help='The path of validate set split.',
                    default=valid_label_path)
args = parser.parse_args()


# %%
# def get_feature(feature_names, path_dict):
#     features = []
#     n_row, n_col = 0, 0

#     for feature_name in feature_names:
#         feature_path = path_dict[feature_name]

#         print('Loading data from: {}...'.format(feature_path))
#         df = pd.read_csv(feature_path, index_col='pid')
#         uids = df.uid
#         df.drop('uid', axis=1, inplace=True)

#         print('Feature shape: {}'.format(df.shape))
#         n_row = df.shape[0]
#         n_col = n_col + df.shape[1]
#         features.append(df)
#     if len(feature_names) > 0:
#         feature = pd.concat(features, axis=1, join='inner', ignore_index=True)
#     else:
#         feature = features[0]
#     feature.insert(0, 'uid', uids)
#     print("Concated feature shape:{}".format(feature.shape))

#     assert n_row == 305613

#     return feature


# # %%
# # 准备数据
# with open(file_path_dict, 'r') as f:
#     path_dict = json.load(f)['train']
# feature_names = ['userdata']
# feature = get_feature(feature_names, path_dict)
# # 按照时间排序

# # 同时还是验证集的label
# df1 = pd.read_csv(train_label_path, index_col='pid')
# df2 = pd.read_csv(valid_label_path, index_col='pid')

# train_set = feature.loc[df1.index]
# valid_set = feature.loc[df2.index]


def get_feature_from_pkl(feature_path, train_label_path, valid_label_path):
    print('Loading data from {}...'.format(feature_path))
    data = pd.read_pickle(feature_path)
    # data = data.set_index('pid')
    train_ref = pd.read_csv(train_label_path)
    valid_ref = pd.read_csv(valid_label_path)
    train = pd.merge(train_ref, data, how='left',on='pid')
    valid = pd.merge(valid_ref, data, how='left',on='pid')
    X_train, Y_train = train.iloc[:, :-1].drop(['label', 'pid'],axis=1), train['label']
    X_valid, Y_valid = valid.iloc[:, :-1].drop(['label', 'pid'],axis=1), valid['label']

    return X_train, Y_train, X_valid, Y_valid


X_train, Y_train, X_valid, Y_valid = get_feature_from_pkl(args.feature_path,
                                                          args.train_label_path,
                                                          args.valid_label_path)

# %%


def get_train_data(feature, label, window_size):
    gb = feature.groupby('uid')
    data, targets = [], []
    for name, grouped in gb:
        grouped = grouped.drop('uid', axis=1)
        data1 = grouped.values
        padding = np.zeros((window_size-1, grouped.shape[1]))
        padded = np.vstack((padding, data1))
        for i in range(len(grouped)):
            tmp = padded[i:i+window_size]
            target = label.loc[grouped.index[i]].values
            # tmp = np.expand_dims(tmp, 0)
            # target = np.expand_dims(target, -1)
            # tmp = K.cast_to_floatx(tmp)
            # target = K.cast_to_floatx(target)
            data.append(tmp)
            targets.append(target)
    data = np.array(data)
    targets = np.array(targets)

    return data, targets


# %%
# 实际测试


gpus = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices([gpus[1], gpus[2]], 'GPU')


timesteps = 8
X, Y = get_train_data(X_train, Y_train, timesteps)
print(X.shape)
print(Y.shape)
x, y = get_train_data(X_valid, Y_valid, timesteps)
print(x.shape)
print(y.shape)


# %%
batchsize, input_dim = args.batch_size, X.shape[2]


i = Input(batch_shape=(None, timesteps, input_dim))
# regression problem here.
o = TCN(return_sequences=False, dilations=[
        1, 2, 4], nb_filters=256, dropout_rate=0.2)(i)
o = Dense(1, activation='relu')(o)
m = Model(inputs=[i], outputs=[o])

m.compile(optimizer='sgd', loss='mse')
# reduce_lr = ReduceLROnPlateau(monitor='val_mse', factor=0.9, patience=20, min_lr=0.0001)

tcn_full_summary(m)
m.fit(X, Y, epochs=100, batch_size=batchsize, shuffle=False)


# %%
m.evaluate(X, Y)
m.evaluate(x, y)

# %%
# 测试get_train_data函数
toy_feat = pd.DataFrame(
    {'uid': [1, 1, 2, 2, 3, 7, 7, 7, 3, 4], 'value': np.arange(10)})
toy_label = pd.DataFrame({'label': np.arange(10)})
x, y = get_train_data(toy_feat, toy_label, 3)
gb = toy_feat.groupby('uid').apply(print)
print(x)
print(y)


# %%
