import torch
import torchtext.vocab as vocab
import numpy as np
import json
from tqdm import tqdm
import csv
import argparse

cache_dir = "../../pretrained_model"
gpu = 0
csv_save_path = '../../extracted_features/glove_test_305613.csv'
title_load_path = '../../../data/data_source/test/test_tags.json'

parser = argparse.ArgumentParser(description='argument parser')

parser.add_argument('--cache_dir', type=str, 
                    default=cache_dir)
parser.add_argument('--gpu', type=str, default=gpu)
parser.add_argument('--csv_save_path', type=str, default=csv_save_path)
parser.add_argument('--title_load_path', type=str, default=title_load_path)

args = parser.parse_args()

cache_dir = args.cache_dir
glove = vocab.pretrained_aliases["glove.6B.300d"](cache=cache_dir)

UNK_embedding = torch.mean(torch.stack([glove.vectors[index] for index in range(len(glove.stoi))]), 0)
print(UNK_embedding)

values = [glove.itos[index] for index in range(len(glove.stoi))]

with open(args.csv_save_path,'w') as f:
  csv_writer = csv.writer(f)
  csv_line = ['pid', 'uid']
  csv_line.extend(['glove{}'.format(i) for i in range(300)])
  csv_writer.writerow(csv_line)
  
with open(args.title_load_path, 'r') as jsonfile:
  json_file = json.load(jsonfile)

title_list = [] 
total_step = len(json_file) 
for index, jsondict in tqdm(enumerate(json_file), total=total_step, ncols=80, unit='title'):
  title_list.append(jsondict["Alltags"].split())


for index, title in tqdm(enumerate(title_list), total=total_step, ncols=80, unit='title' ):
  if len(title) == 0:
    title_list[index] = ['null']
  for i, token in enumerate(title):
    if token not in values:
      title[i] = 'UNK'
print(title_list)

with open(args.csv_save_path,'a+') as f:
  csv_writer = csv.writer(f)
  for index, jsondict in tqdm(enumerate(json_file), total=total_step, ncols=80, unit='title'):
    title = title_list[index] 
    csv_line = [jsondict["Pid"], jsondict["Uid"]]
    embedding_tensor_list = []
    for token in title:
      if token == 'UNK':
        embedding = UNK_embedding
      else:
        embedding = glove.vectors[glove.stoi[token]]
      embedding_tensor_list.append(embedding)
    embedding_avg = torch.mean(torch.stack(embedding_tensor_list), 0)
    embedding_avg_np = np.squeeze(embedding_avg.numpy())
    csv_line.extend(embedding_avg_np)
    csv_writer.writerow(csv_line)

          