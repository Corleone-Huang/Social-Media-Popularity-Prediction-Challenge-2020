import numpy as np
import pandas as pd
import argparse
import json
import gc
import h5py
import time
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torch
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error
import os
import random
from torch import nn

seed = 2020
GPU = "7"
train_label_filepath = "/home/smp/label/train_label_postdate.csv"
validate_label_filepath = "/home/smp/label/validate_label_postdate.csv"
postdate_dir = '/home/zht/MLP/postdate.json'
bert_titles_filepath = '/home/hmq/SMP/features/bert_base_titles_average_3.csv'
bert_alltags_filepath = '/home/hmq/SMP/features/bert_base_alltags_average_3.csv'
uid_info_filepath = '/home/smp/Features/uid_305613_tsvd_256.csv'
seq_len = 5
batch_size = 64
num_workers = 10
input_size = 400
hidden_size = 800
epochs = 100
learning_rate = 0.001

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

os.environ["CUDA_VISIBLE_DEVICES"] = GPU
np.set_printoptions(threshold=9999999)  
np.set_printoptions(suppress=True)  
np.set_printoptions(precision=4)   
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

def add_pad(df): # padding 0 at the end of each feature dataframe.
    dict_toadd = {name:[0] for name, _ in df.iteritems()}
    df_toadd = pd.DataFrame(dict_toadd)
    df_added = df.append(df_toadd, ignore_index=True)
    return df_added

def patches_generator(df_postdate, uid_list): 
    '''
    Generates pid_patches list. the pid sequence of each user is padded [0,0,0,0] (4*0) at its first, 
    then generates the pid sequence, such as [[0,0,0,0,1][0,0,0,1,2][0,0,1,2,3]...], for each user(uid).
    '''
    df_group_uid = df_postdate.groupby('uid')
    dict_uid=dict(list(df_group_uid))
    pid_patches=[]
    for uid in tqdm(uid_list, ncols=100):
        df_uid = dict_uid[uid]
        dict_topad = {name:[0] for name, _ in df_uid.iteritems()}
        df_topad = pd.DataFrame(dict_topad)
        df_padded = df_uid.append([df_topad]*4, ignore_index=True).sort_values(by = 'postdate',\
            ascending=True)
        pid_list = df_padded['pid'].values.tolist()
        for index in range(len(pid_list)-args.seq_len+1):
            pid_patches.append(pid_list[index:index+args.seq_len])
    pid_patches = np.array(pid_patches)
    return pid_patches

def feature_list(pid_patches, df):  # generates feature_list (train/test) in the order of pid_patches. 
    feature_list = []
    for patch in tqdm(pid_patches, ncols=100):
        df_patch = pd.DataFrame()
        for i in range(args.seq_len):
            df_new = df.loc[df['pid']==patch[i]]
            df_patch = df_patch.append(df_new)
        feature_patch = df_patch.iloc[:, 3:].values
        feature_list.append(feature_patch)
    return feature_list
  
def label_list(pid_patches, df):  # generates label_list (train/test) in the order of pid_patches. 
    label_list = []
    for patch in tqdm(pid_patches, ncols=100):
        label = [[0]]*(args.seq_len-1)
        #label.append(self.df_label[self.df_label['pid'] == pid_patch[-1]]['label'].values)
        label.append(df[df['pid'] == patch[-1]]['label'].values)
        label_list.append(label)
    return label_list

class data(Dataset):
    def __init__(self, pid_patches, bert_titles, bert_tags, uid_info, label):
        self.pid_patches = pid_patches
        self.bert_titles = bert_titles
        self.bert_tags = bert_tags
        self.uid_info = uid_info
        self.label = label

    def __len__(self):
        return len(self.pid_patches)
      
      
    def __getitem__(self, index):
        bert_titles = torch.tensor(self.bert_titles[index], dtype=torch.float32)
        bert_tags = torch.tensor(self.bert_tags[index], dtype=torch.float32)
        uid_info = torch.tensor(self.uid_info[index], dtype=torch.float32)
        label = torch.tensor(self.label[index], dtype=torch.float32)
        feature = torch.cat((bert_titles, bert_tags, uid_info), dim=1)
        return feature, label
        
class SMP_LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2, inter_size=800, \
            feature_size=1792, bias=True, batch_first=True, dropout=0.08, bidirectional=False):
        super(SMP_LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.bias = bias
        self.inter_size = inter_size
        self.feature_size = feature_size
        self.output_hidden1 = 300
        self.output_hidden2 = 64
        self.batch_first = batch_first
        self.dropout = dropout
        self.layer_dropout = 0.2
        self.bidirectional = bidirectional
        self.num_directions = 1
        if bidirectional:
            self.num_directions = 2
            
        self.dropout_layer = nn.Dropout(self.layer_dropout) 
        
        self.linear_feature_in = nn.Linear(self.feature_size, self.inter_size)
        self.linear_input = nn.Linear(self.inter_size, self.input_size)
        
        self.lstm = torch.nn.LSTM(self.input_size, self.hidden_size, self.num_layers, \
            self.bias, self.batch_first, self.dropout, self.bidirectional)
        
        self.output_layer1 = nn.Linear(self.hidden_size*self.num_directions, self.output_hidden1)
        self.output_layer2 = nn.Linear(self.output_hidden1, self.output_hidden2)
        self.output_layer3 = nn.Linear(self.output_hidden2, self.output_size)
    
    def forward(self, feature):
        batch_size = feature.size(0)
        
        input_inter = self.linear_feature_in(feature)
        input_inter = self.dropout_layer(F.relu(input_inter))
        input_feature = self.linear_input(input_inter)

        h_0 = torch.randn(self.num_layers*self.num_directions, batch_size, self.hidden_size)
        c_0 = torch.randn(self.num_layers*self.num_directions, batch_size, self.hidden_size)
        if torch.cuda.is_available():
            h_0 = h_0.cuda()
            c_0 = c_0.cuda()

        last_hidden, (h_n, c_n) = self.lstm(input_feature, (h_0, c_0))
        
        last_hidden = last_hidden[:,-1,:]
        last_hidden = last_hidden.squeeze(1)
        
        y = self.output_layer1(last_hidden)
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
    
    print("pid patches train/validate generating...")
    pid_patches_train = patches_generator(df_postdate_train, uid_list_train)
    pid_patches_validate = patches_generator(df_postdate_validate, uid_list_validate)

    print("writing dataset.h5...")
    with h5py.File('dataset.h5', 'a') as f: # Writing data into hdf5.
        if 'bert_titles_train' in f.keys():
            pass
        else:
            bert_titles_train = feature_list(pid_patches_train, df_bert_titles_train)
            f.create_dataset('bert_titles_train', dtype='f', data=bert_titles_train)
            del bert_titles_train
            gc.collect()
      
        if 'bert_alltags_train' in f.keys():
            pass
        else:
            bert_alltags_train = feature_list(pid_patches_train, df_bert_alltags_train)
            f.create_dataset('bert_alltags_train', dtype='f', data=bert_alltags_train)
            del bert_alltags_train
            gc.collect()
        
        if 'uid_info_train' in f.keys():
            pass
        else:
            uid_info_train = feature_list(pid_patches_train, df_uid_info_train)
            f.create_dataset('uid_info_train', dtype='f', data=uid_info_train)
            del uid_info_train
            gc.collect()
      
        if 'bert_titles_validate' in f.keys():
            pass
        else:
            bert_titles_validate = feature_list(pid_patches_validate, df_bert_titles_validate)
            f.create_dataset('bert_titles_validate', dtype='f', data=bert_titles_validate)
            del bert_titles_validate
            gc.collect()
        
        if 'bert_alltags_validate' in f.keys():
            pass
        else:
            bert_alltags_validate = feature_list(pid_patches_validate, df_bert_alltags_validate)
            f.create_dataset('bert_alltags_validate', dtype='f', data=bert_alltags_validate)
            del bert_alltags_validate
            gc.collect()
      
        if 'uid_info_validate' in f.keys():
            pass
        else:
            uid_info_validate = feature_list(pid_patches_validate, df_uid_info_validate)
            f.create_dataset('uid_info_validate', dtype='f', data=uid_info_validate)
            del uid_info_validate
            gc.collect()
        
        if 'label_train' in f.keys():
            pass
        else:
            label_train = label_list(pid_patches_train, df_train)
            f.create_dataset('label_train', dtype='f', data=label_train)
            del label_train
            gc.collect()
      
        if 'label_validate' in f.keys():
            pass
        else:
            label_validate = label_list(pid_patches_validate, df_validate)
            f.create_dataset('label_validate', dtype='f', data=label_validate)
            del label_validate
            gc.collect()
      
    with h5py.File('dataset.h5', 'r') as f: # Reading data from hdf5.
        print('Loading data from hdf5...')
        bert_titles_train = f['bert_titles_train'][:]
        bert_alltags_train = f['bert_alltags_train'][:]
        uid_info_train = f['uid_info_train'][:]
        uid_info_validate = f['uid_info_validate'][:]
        bert_titles_validate = f['bert_titles_validate'][:]
        bert_alltags_validate = f['bert_alltags_validate'][:]
        label_train = f['label_train'][:]
        label_validate = f['label_validate'][:]

    train_dataset = data(pid_patches_train, bert_titles_train, bert_alltags_train, \
        uid_info_train, label_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, \
        shuffle=True, num_workers=args.num_workers)
 
    test_dataset = data(pid_patches_validate, bert_titles_validate, bert_alltags_validate, \
        uid_info_validate, label_validate)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True, \
        num_workers=args.num_workers)
                              
    model = SMP_LSTM(args.input_size, args.hidden_size)
    if torch.cuda.is_available():
        model = model.cuda()
  
    optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', \
        factor=0.95, verbose=1, patience=3) # adaptive learning rate.  
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5) # gradient clipping        
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
            with open("bug.txt", 'a') as f:
                f.write(str(y_pred))
                f.write("\n")
                f.write(str(label_validate))
                f.write("\n")
                f.write("-------------------------------------------")

        
        elapsed_time = time.time() - start_time
        MAE = mean_absolute_error(y_labels,y_preds)
        scheduler.step(MAE)  # adaptive learning rate.
        correction, _ = spearmanr(y_preds, y_labels)

        print('Epoch {}/{} \t train_loss={:.4f} \t val_MAE={:.4f} \t val_SRC:{:.4f} \t time={:.2f}s'.\
            format(epoch + 1, args.epochs, avg_loss, MAE, correction, elapsed_time))
    
        avg_losses_f.append(avg_loss)
        avg_val_losses_f.append(avg_val_loss)
    
if __name__ == "__main__":
    main()