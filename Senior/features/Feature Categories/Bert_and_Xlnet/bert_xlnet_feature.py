# 本代码用来提取SMP data中的Alltags, Titles数据的特征
# bert-base-uncased预训练模型输出768维特征向量
# bert-large-uncased预训练模型输出1024维特征向量。
# xlnet-base-cased预训练模型输出768维特征向量。
# xlnet-large-cased预训练模型输出1024维特征向量。
# author： Huang Mengqi  2020-3-31

import os
import json
import csv
import numpy as np
import argparse
from tqdm import tqdm, trange

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import *

np.set_printoptions(threshold=9999999)  # 显示完整数组
np.set_printoptions(suppress=True)  # 不使用科学计数法
np.set_printoptions(precision=8)   # 设精度


#          Model          | Tokenizer          | Pretrained weights shortcut
MODELS = [(BertModel,       BertTokenizer,       'bert-base-uncased'),
          (BertModel,       BertTokenizer,       'bert-large-uncased'),
          (XLNetModel,      XLNetTokenizer,      'xlnet-base-cased'),
          (XLNetModel,      XLNetTokenizer,      'xlnet-large-cased'),
          (BertModel,       BertTokenizer,       'bert-base-multilingual-uncased'),
          (BertModel,       BertTokenizer,       'bert-base-multilingual-cased'),
         ]
# 备用的预训练模型
visible_gpu_device = '7'
default_data_dir = '../data/train/train_tags.json'
default_save_title_feature = '../features/'
default_save_alltags_feature = '../features/'
default_gpu_num = 2




os.environ["CUDA_VISIBLE_DEVICES"] = visible_gpu_device


def parse_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--data_dir', default=default_data_dir, type=str,
                        help='存放json数据的路径', dest='data_dir')
    parser.add_argument('--save_title_feature', default=default_save_title_feature, dest='title_feature',
                        help='存放title feature的文件夹路径，/结尾，不是具体csv文件！', type=str)
    parser.add_argument('--save_alltags_feature', default=default_save_alltags_feature, dest='alltags_feature',
                        help='存放alltags feature的文件夹路径，/结尾，不是具体csv文件！', type=str)
    parser.add_argument('--ngpu', default=default_gpu_num, type=int, dest='ngpu', 
                        help='使用的GPU数目')
    parser.add_argument('--bert_base_uncased', default='True', type=str, dest='bert_base_uncased',
                        help='是否提取bert-base-uncased特征')
    parser.add_argument('--bert_large_uncased', default='True', type=str, dest='bert_large_uncased',
                        help='是否提取bert-large-uncased特征')
    parser.add_argument('--xlnet_base_cased', default='True', type=str, dest='xlnet_base_cased',
                        help='是否提取xlnet-base-cased特征')
    parser.add_argument('--xlnet_large_cased', default='True', type=str, dest='xlnet_large_cased',
                        help='是否提取xlnet-large-cased特征')
    parser.add_argument('--bert_base_multilingual_uncased', default='True', type=str, dest='mul_uncased',
                        help='是否提取bert-base-multilingual-uncased特征')
    parser.add_argument('--bert_base_multilingual_cased', default='True', type=str, dest='mul_cased',
                        help='是否提取bert_base_multilingual_cased特征')
    args = parser.parse_args()
    return args


# 建立数据集类 
class TextDataset(Dataset):
    def __init__(self, json_file_path):
        self._file = open(json_file_path)
        self.all_data = json.loads(self._file.readline())
    
    def __len__(self):
        return len(self.all_data)
    
    def __getitem__(self, index):
        Uid = self.all_data[index]['Uid']
        Pid = self.all_data[index]['Pid']
        Title = self.all_data[index]['Title']
        Mediatype = self.all_data[index]['Mediatype']
        Alltags = self.all_data[index]['Alltags']
        return Uid, Pid, Title, Mediatype, Alltags

if __name__ == "__main__":
    
    args = parse_args()
    # 配置路径
    text_data_path = args.data_dir
    title_305613_path = args.title_feature
    alltags_305613_path = args.alltags_feature
    ngpu = args.ngpu

    # 提取哪些预训练模型的特征
    MODELS_LIST = []
    if args.bert_base_uncased == 'True':
        MODELS_LIST.append(MODELS[0])
    if args.bert_large_uncased == 'True':
        MODELS_LIST.append(MODELS[1])
    if args.xlnet_base_cased == 'True':
        MODELS_LIST.append(MODELS[2])
    if args.xlnet_large_cased == 'True':
        MODELS_LIST.append(MODELS[3])
    if args.mul_uncased == 'True':
        MODELS_LIST.append(MODELS[4])
    if args.mul_cased == 'True':
        MODELS_LIST.append(MODELS[5])

    # 数据集实例
    text_dataset = TextDataset(json_file_path = text_data_path)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    for model_class, tokenizer_class, pretrained_weights in MODELS_LIST:
        print('{}{}{}'.format('提取', pretrained_weights, '特征'))

        
        pre_filename = pretrained_weights.replace('-','_')
        title_name = '{}{}{}{}{}'.format(title_305613_path, pre_filename, \
            '_title_', text_dataset.__len__(), '.csv')
        alltags_name = '{}{}{}{}{}'.format(alltags_305613_path, pre_filename, \
            '_alltags_', text_dataset.__len__(), '.csv')
        # ../features/xlnet_large_cased_305613.csv
        
        title_name = '{}{}{}{}{}{}'.format('../test_feature/', 'title_', pretrained_weights.split('-')[0], \
            '_', text_dataset.__len__(), '.csv')
        alltags_name = '{}{}{}{}{}{}'.format('../test_feature/', 'tags_', pretrained_weights.split('-')[0], \
            '_', text_dataset.__len__(), '.csv')

        # 打开准备写入的csv文件
        file_title = open(title_name,"w")
        title_writer = csv.writer(file_title)
        first_title_list = ["pid", "uid"]
        for i in range(768):  # 注意更改！base->768, large->1024
            feature_i = "titles" + str(i+1)
            first_title_list.append(feature_i)
        title_writer.writerow(first_title_list)

        file_alltags = open(alltags_name, "w")
        alltags_writer = csv.writer(file_alltags)
        first_alltags_list = ["pid", "uid"]
        for i in range(768):
            feature_i = "tags" + str(i+1)
            first_alltags_list.append(feature_i)
        alltags_writer.writerow(first_alltags_list)

        # Load pretrained model/tokenizer
        tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        model = model_class.from_pretrained(pretrained_weights)

        if ngpu > 1:
            print("using", device)
            print("Let's use", ngpu, "GPUs!")
            device_id = [i for i in range(ngpu)]
            model = nn.DataParallel(model, device_ids=device_id)
        model.eval()
        model.to(device)

        for index in trange(text_dataset.__len__()):
            uid, pid, title, _, alltags = text_dataset[index]

            # Encode text
            # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens 
            # in the right way for each model.
            token_title_ids = tokenizer.encode(title, add_special_tokens=True)
            # 最多支持512 tokens，超过则截断
            if len(token_title_ids) > 512:
                tmp = token_title_ids
                token_title_ids = token_title_ids[:511]
                token_title_ids.append(tmp[-1])
            title_ids = \
                torch.tensor([token_title_ids])
            title_ids = title_ids.to(device)
            
            token_alltags_ids = tokenizer.encode(alltags, add_special_tokens=True)
            if len(token_alltags_ids) > 512:
                tmp = token_alltags_ids
                token_alltags_ids = token_alltags_ids[:511]
                token_alltags_ids.append(tmp[-1])
            alltags_ids = \
                torch.tensor([token_alltags_ids]).cuda()
            alltags_ids = alltags_ids.to(device)

            with torch.no_grad():
                # last_hidden_states
                # Models outputs are now tuples
                title_output = model(title_ids)[0]  
                alltags_output = model(alltags_ids)[0]  
            
            title_embeddings, alltags_embedding = [], []

            # 一个简单的方法是平均每个token
            title_embeddings = torch.mean(title_output, 1)
            title_embeddings.squeeze(0)
            alltags_embedding = torch.mean(alltags_output, 1)
            alltags_embedding.squeeze(0)

            title_embeddings = np.squeeze(title_embeddings.cpu().numpy())
            alltags_embedding = np.squeeze(alltags_embedding.cpu().numpy())

            # print(title_embeddings)
            # print(alltags_embedding)
            title_embeddings_list = [pid, uid]
            alltags_embedding_list = [pid, uid]

            for index in range(len(title_embeddings)):
                title_embeddings_list.append(title_embeddings[index])
            for index in range(len(alltags_embedding)):
                alltags_embedding_list.append(alltags_embedding[index])

            title_writer.writerow(title_embeddings_list)
            alltags_writer.writerow(alltags_embedding_list)

        # 关闭打开的csv文件
        file_title.close()
        file_alltags.close()







