import numpy as np
import pandas as pd
import argparse
import json
import gc
import time
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error
import os
import random
from torch import nn
import math

seed = 2020
GPU = "3"
train_label_filepath = "/home/smp/label/train_label_postdate.csv"
validate_label_filepath = "/home/smp/label/validate_label_postdate.csv"
postdate_dir = '/home/zht/MLP/postdate.json'
bert_titles_filepath = '/home/hmq/SMP/features/bert_base_titles_average_3.csv'
bert_alltags_filepath = '/home/hmq/SMP/features/bert_base_alltags_average_3.csv'
uid_info_filepath = '/home/smp/Features/uid_305613_tsvd_256.csv'
seq_len = 5
batch_size = 512
num_workers = 10
input_size = 400
hidden_size = 800
epochs = 100
learning_rate = 0.00001

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

os.environ["CUDA_VISIBLE_DEVICES"] = GPU

parser = argparse.ArgumentParser(description="SMP_LSTM")

parser.add_argument("--train_label_filepath", type=str,
                    default=train_label_filepath)
parser.add_argument("--validate_label_filepath", type=str,
                    default=validate_label_filepath)
parser.add_argument('--postdate_dir',
                    help='The path of postdate data.', type=str, default=postdate_dir)
parser.add_argument("--bert_titles", type=str, 
                    default=bert_titles_filepath)
parser.add_argument("--bert_alltags", type=str, 
                    default=bert_alltags_filepath)
parser.add_argument("--uid_info", type=str, 
                    default=uid_info_filepath)
parser.add_argument('--seq_len',help='The length of input sequence of LSTM.', type=int, default=seq_len)
parser.add_argument("--batch_size", type=int, default=batch_size, help="Batch_size")
parser.add_argument("--num_workers", type=int, default=num_workers, help="num_workers")
parser.add_argument("--input_size", type=int, default=input_size, help="Input size of LSTM")
parser.add_argument("--hidden_size", type=int, default=hidden_size, help="Hidden size of LSTM")
parser.add_argument("--epochs", type=int, default=epochs, help="Training epochs")
parser.add_argument("--learning_rate", type=int, default=learning_rate, help="Learning rate")

args = parser.parse_args()

def add_pad(df):
  dict_toadd = {name:[0] for name, _ in df.iteritems()}
  df_toadd = pd.DataFrame(dict_toadd)
  df_added = df.append(df_toadd, ignore_index=True)
  return df_added


def patches_generator(df_postdate, uid_list):
  df_group_uid = df_postdate.groupby('uid')
  dict_uid=dict(list(df_group_uid))
  pid_patches=[]
  for uid in tqdm(uid_list, ncols=100):
    df_uid = dict_uid[uid]
    dict_topad = {name:[0] for name, _ in df_uid.iteritems()}
    df_topad = pd.DataFrame(dict_topad)
    df_padded = df_uid.append([df_topad]*4, ignore_index=True).sort_values(by = 'postdate',ascending=True)
    pid_list = df_padded['pid'].values.tolist()
    for index in range(len(pid_list)-args.seq_len+1):
      pid_patches.append(pid_list[index:index+args.seq_len])
  pid_patches = np.array(pid_patches)
  return pid_patches


class data(Dataset):
  def __init__(self, pid_patches, df_bert_titles, df_bert_tags, df_uid_info, df_label):
      self.pid_patches = pid_patches
      self.df_bert_titles = df_bert_titles
      self.df_bert_tags = df_bert_tags
      self.df_uid_info = df_uid_info
      self.df_label = df_label

  def __len__(self):
      return len(self.pid_patches)
      
  def patch_gather(self, pid_patch, df):
      df_patch = pd.DataFrame()
      for i in range(args.seq_len):
        df_new = df.loc[df['pid']==pid_patch[i]]
        df_patch = df_patch.append(df_new)
      feature_patch = df_patch.iloc[:, 3:].values
      feature_patch = torch.tensor(feature_patch, dtype=torch.float32)
      return feature_patch
      
  def __getitem__(self, index):
      pid_patch = self.pid_patches[index].tolist()
      bert_titles_patch = self.patch_gather(pid_patch, self.df_bert_titles)
      bert_tags_patch = self.patch_gather(pid_patch, self.df_bert_tags)
      uid_patch = self.patch_gather(pid_patch, self.df_uid_info)
      label = [[0]]*4
      label.append(self.df_label[self.df_label['pid']==pid_patch[-1]]['label'].values)
      label = torch.tensor(label, dtype=torch.float32)
      feature = torch.cat([bert_titles_patch, bert_tags_patch, uid_patch], dim=1)
      return feature, label
      
class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model = 768, dropout = 0.2, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].detach()
        return self.dropout(x)
     
class SMP_transformer(torch.nn.Module):
  def __init__(self, input_size=768, inter_size=800, feature_size=1792):
    super(SMP_transformer, self).__init__()
    self.input_size = 768
    self.output_size = 1
    self.feature_size = feature_size
    self.layer_dropout = 0.15
    self.hidden_size = 768
    self.output_hidden1 = 256
    self.output_hidden2 = 64
    self.position = PositionalEncoding()
    self.dropout_layer = nn.Dropout(self.layer_dropout) 
    
    self.linear_feature_in = nn.Linear(self.feature_size, self.input_size)
    
    encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=8)
    self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
    
    self.conv = torch.nn.Conv1d(5, 1, 1)
    
    self.output_layer1 = nn.Linear(self.hidden_size, self.output_hidden1)
    self.output_layer2 = nn.Linear(self.output_hidden1, self.output_hidden2)
    self.output_layer3 = nn.Linear(self.output_hidden2, self.output_size)
    
  def forward(self, feature):
    feature = feature.permute(1,0,2).contiguous()

    input_feature = self.linear_feature_in(feature)
    embed = self.position(input_feature)
    out = self.transformer_encoder(embed)
    out = out.permute(1,0,2).contiguous()
    
    out = self.conv(out)
    out = out.squeeze(1)
    
    y = self.output_layer1(out)
    y = self.dropout_layer(F.relu(y))
    y = self.output_layer2(y)
    y = self.dropout_layer(F.relu(y))
    y = self.output_layer3(y)

    return y
    
def main():
  print('Loading train label...')
  df_train = pd.read_csv(args.train_label_filepath)
  print('Loading validate label...')
  df_validate = pd.read_csv(args.validate_label_filepath)
  
  print('Loading postdate data and spliting them into train and validate dataset...')
  df_postdate = pd.read_json(args.postdate_dir)
  df_postdate_train = pd.merge(df_train, df_postdate, how='inner') 
  df_postdate_validate = pd.merge(df_validate, df_postdate, how='inner')
  del df_postdate
  
  print('Loading bert_case_titles feature...')
  df_bert_titles = pd.read_csv(args.bert_titles)
  df_bert_titles_train = pd.merge(df_train, df_bert_titles, how='inner') 
  df_bert_titles_validate = pd.merge(df_validate, df_bert_titles, how='inner')
  del df_bert_titles
  
  print('Loading bert_case_alltags feature...')
  df_bert_alltags = pd.read_csv(args.bert_alltags)
  df_bert_alltags_train = pd.merge(df_train, df_bert_alltags, how='inner')
  df_bert_alltags_validate = pd.merge(df_validate, df_bert_alltags, how='inner')
  del df_bert_alltags
  
  print('Loading uid_info feature...')
  df_uid_info = pd.read_csv(args.uid_info)
  df_uid_info_train = pd.merge(df_train, df_uid_info, how='inner')
  df_uid_info_validate = pd.merge(df_validate, df_uid_info, how='inner')
  del df_uid_info
  
  gc.collect()
  
  df_bert_titles_train = add_pad(df_bert_titles_train)
  df_bert_titles_validate = add_pad(df_bert_titles_validate)
  df_bert_alltags_train = add_pad(df_bert_alltags_train)
  df_bert_alltags_validate = add_pad(df_bert_alltags_validate)
  df_uid_info_train = add_pad(df_uid_info_train)
  df_uid_info_validate = add_pad(df_uid_info_validate)

  uid_list_train = df_postdate_train['uid'].values
  uid_list_train = list(set(uid_list_train))
  uid_list_validate = df_postdate_validate['uid'].values
  uid_list_validate = list(set(uid_list_validate))
  
  pid_patches_train = patches_generator(df_postdate_train, uid_list_train)
  pid_patches_validate = patches_generator(df_postdate_validate, uid_list_validate)

  train_dataset = data(pid_patches_train, df_bert_titles_train, df_bert_alltags_train, df_uid_info_train, df_train)
  train_loader = DataLoader(dataset=train_dataset,
                                batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
  
  test_dataset = data(pid_patches_validate, df_bert_titles_validate, df_bert_alltags_validate, df_uid_info_validate, df_validate)
  test_loader = DataLoader(dataset=test_dataset,
                                batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
                              
  model = SMP_transformer()
  if torch.cuda.is_available():
      model = model.cuda()
  
  optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate, weight_decay=1e-5)
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=1, patience=3)
  loss_fn = torch.nn.L1Loss()
  
  avg_losses_f = []
  avg_val_losses_f = []
  
  for epoch in range(args.epochs):
    print('Training epoch{}...'.format(epoch+1))
    model.train()
    start_time = time.time()
    avg_loss = 0.
    for i, (feature_train, label_train) in enumerate(train_loader):
      if torch.cuda.is_available():
        feature_train = feature_train.cuda()
        label_train = label_train.cuda()
      label_train = label_train[:,-1,:]
      y_pred = model(feature_train)
      loss = loss_fn(y_pred, label_train)
      optimizer.zero_grad()   
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)             
      optimizer.step()      
      avg_loss += loss.item() / len(train_loader) 
    
    
    model.eval()
    avg_val_loss = 0. 
    y_preds = []
    y_labels = []
    for i, (feature_validate, label_validate) in enumerate(test_loader):
      if torch.cuda.is_available():
        feature_validate = feature_validate.cuda()
        label_validate = label_validate.cuda()
      label_validate = label_validate[:,-1,:]
      y_pred = model(feature_validate).detach()
      
      avg_val_loss += loss_fn(y_pred, label_validate).item() / len(test_loader)
      y_pred = y_pred.squeeze(1).detach().cpu().numpy().tolist()
      label_validate = label_validate.squeeze(1).detach().cpu().numpy().tolist()
      y_preds.extend(y_pred)
      y_labels.extend(label_validate)
      
    elapsed_time = time.time() - start_time
    MAE = mean_absolute_error(y_labels,y_preds)
    scheduler.step(MAE)
    correction, _ = spearmanr(y_preds, y_labels)
    print('Epoch {}/{} \t loss={:.4f} \t MAE={:.4f} \t SRC:{:.4f} \t time={:.2f}s'.format(epoch + 1, args.epochs, avg_loss, MAE, correction, elapsed_time))
    
    avg_losses_f.append(avg_loss)
    avg_val_losses_f.append(avg_val_loss)
    
if __name__ == "__main__":

  main()
