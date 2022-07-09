from sklearn.preprocessing import MinMaxScaler,StandardScaler
import torch
import pandas as pd
import numpy as np
import time
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import gc
import argparse
import os
import random
from torch import nn
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error
from torch.utils.data import Dataset
import csv

seed = 2020
GPU = "0,3"
bert_titles_filepath = '/home/hmq/SMP/features/bert_base_titles_average_3.csv'
bert_titles_test_filepath = '/home/smp/SMP_test_feature/title_bert_180581_average3.csv'
bert_alltags_filepath = '/home/hmq/SMP/features/bert_base_alltags_average_3.csv'
bert_alltags_test_filepath = '/home/smp/SMP_test_feature/tags_bert_180581_average3.csv'
glove_filepath = '/home/smp/Features/glove_tags_305613.csv'
resnext_filepath = '/home/smp/Features/Category_1epoch3_ResNext101_image_305613.csv'
resnext_test_filepath = '/home/smp/SMP_test_feature/Subcategory_1epoch3_ResNext101_image.csv'
wordchar_train_filepath = '/home/smp/Features/wordchar_merge.csv'
wordchar_test_filepath = '/home/smp/SMP_test_feature/wordchar_merge_180581.csv'
semantic_filepath = '/home/smp/Features/semantic_img_305613.csv'
postdate_train_filepath = '/home/zjj/SMP2020_test/train_feature/Postdate_305613.csv'
postdate_test_filepath = '/home/zjj/SMP2020_test/test_feature/Postdate_180581.csv'
concept_filepath = '/home/zht/GLOVE/glove_ctrain_305613.csv'
concept_test_filepath = '/home/zht/GLOVE/glove_ctest_180581.csv'
uid_filepath = '/home/smp/Features/uid_305613.csv'
uid_test_filepath = '/home/smp/SMP_test_feature/uid_180581.csv'
train_label_filepath = "/home/zjj/SMP2020_test/label/train_label_postdate.csv"
validate_label_filepath = "/home/zjj/SMP2020_test/label/validate_label_postdate.csv"
label_filepath = "/home/zjj/SMP2020_test/all_label_postdate.csv"
input_size = 100
hidden_size = 300
epochs = 200
batch_size = 128
learning_rate = 0.0006
num_workers = 8
save_path = "/home/zht/MLP/model/"

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.set_printoptions(edgeitems=300)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,3"
parser = argparse.ArgumentParser(description="SMP_MLP")

parser.add_argument("--bert_titles", type=str, 
                    default=bert_titles_filepath)
parser.add_argument("--bert_titles_test", type=str, 
                    default=bert_titles_test_filepath)
parser.add_argument("--bert_alltags", type=str, 
                    default=bert_alltags_filepath)
parser.add_argument("--bert_alltags_test", type=str, 
                    default=bert_alltags_test_filepath)
parser.add_argument("--glove", type=str, 
                    default=glove_filepath)
parser.add_argument("--glove_test", type=str, 
                    default=glove_test_filepath)
parser.add_argument("--resnext", type=str, 
                    default=resnext_filepath)
parser.add_argument("--resnext_test", type=str, 
                    default=resnext_test_filepath)
parser.add_argument("--wordchar_train", type=str, 
                    default=wordchar_train_filepath)
parser.add_argument("--wordchar_test", type=str, 
                    default=wordchar_test_filepath)
parser.add_argument("--semantic", type=str, 
                    default=semantic_filepath)
parser.add_argument("--postdate_train", type=str, 
                    default=postdate_train_filepath)
parser.add_argument("--postdate_test", type=str, 
                    default=postdate_test_filepath)
parser.add_argument("--concept", type=str, 
                    default=concept_filepath)
parser.add_argument("--concept_test", type=str, 
                    default=concept_test_filepath)
parser.add_argument("--uid", type=str, 
                    default=uid_filepath)
parser.add_argument("--uid_test", type=str, 
                    default=uid_test_filepath)
parser.add_argument("--train_label_filepath", type=str,
                    default=train_label_filepath)
parser.add_argument("--validate_label_filepath", type=str,
                    default=validate_label_filepath)
parser.add_argument("--label_filepath", type=str,
                    default=label_filepath")
parser.add_argument("--input_size", type=int, default=input_size)
parser.add_argument("--hidden_size", type=int, default=hidden_size)
parser.add_argument("--epochs", type=int, default=epochs)
parser.add_argument("--batch_size", type=int, default=batch_size)
parser.add_argument("--learning_rate", type=float, default=learning_rate)
parser.add_argument("--save_path", type=str,
                    default=save_path)
parser.add_argument("--num_workers", type=int, default=num_workers)
args = parser.parse_args()
    
class MLP(torch.nn.Module):
  def __init__(self):
    super(MLP, self).__init__()
    self.resnext_size = 2048
    self.glove_size = 300
    self.postdate_size = 9
    self.bert_size = 768
    self.output_size = 1
    self.dropout = torch.nn.Dropout(0.2)
    self.concept_size = 300
    self.uid_size = 384
    
    self.linear_titles_1 = nn.Linear(self.bert_size, 300)
    self.linear_alltags_1 = nn.Linear(self.bert_size, 300)
    self.linear_glove_1 = nn.Linear(self.glove_size, 150)
    self.linear_resnext_1 = nn.Linear(self.resnext_size, 500)
    self.linear_concept_1 = nn.Linear(self.concept_size, 150)
    self.linear_uid_1 = nn.Linear(self.uid_size, 160)
    
    self.linear_titles_2 = nn.Linear(300, 100)
    self.linear_alltags_2 = nn.Linear(300, 100)
    self.linear_glove_2 = nn.Linear(150, 100)
    self.linear_resnext_2 = nn.Linear(500, 150)
    self.linear_concept_2 = nn.Linear(150, 100)
    self.linear_uid_2 = nn.Linear(160, 100)
    
    self.linear_postdate_1 = nn.Linear(9, 32)
    self.linear_postdate_2 = nn.Linear(32, 100)
    
    self.linear_wordchar1 = nn.Linear(4, 32)
    self.linear_wordchar2 = nn.Linear(32, 64)
    
    self.output1 = nn.Linear(814, 360)
    self.output2 = nn.Linear(360, 64)
    self.output3 = nn.Linear(64, 1)


  def forward(self, titles, alltags, glove, resnext, postdate, concept, uid, wordchar):
  
    titles = F.relu(self.linear_titles_1(titles))
    titles = F.relu(self.linear_titles_2(self.dropout(titles)))
    
    alltags = F.relu(self.linear_alltags_1(alltags))
    alltags = F.relu(self.linear_alltags_2(self.dropout(alltags)))

    glove = F.relu(self.linear_glove_1(glove))
    glove = F.relu(self.linear_glove_2(self.dropout(glove)))
  
    resnext = F.relu(self.linear_resnext_1(resnext))
    resnext = F.relu(self.linear_resnext_2(self.dropout(resnext)))
    
    concept = F.relu(self.linear_concept_1(concept))
    concept = F.relu(self.linear_concept_2(self.dropout(concept)))
    
    uid = F.relu(self.linear_uid_1(uid))
    uid = F.relu(self.linear_uid_2(self.dropout(uid)))

    postdate_1 = self.linear_postdate_1(postdate)
    postdate_2 = F.relu(postdate_1)
    postdate_3 = F.relu(self.linear_postdate_2(postdate_2))

    wordchar1 = self.linear_wordchar1(wordchar)
    wordchar2 = F.relu(wordchar1)
    wordchar3 = F.relu(self.linear_wordchar2(self.dropout(wordchar2)))
    
    feature = torch.cat([titles, alltags, glove, resnext, concept, uid, postdate_3, wordchar3], 1)
    
    y = F.relu(self.dropout(self.output1(feature)))
    y = F.relu(self.dropout(self.output2(y)))
    y = self.output3(y)
    
    return y

def train():     
  print('Loading train label...')
  df_train = pd.read_csv(args.train_label_filepath)
  print('Loading validate label...')
  df_validate = pd.read_csv(args.validate_label_filepath)
  
  print('Loading bert_case_titles feature...')
  df_bert_titles = pd.read_csv(args.bert_titles)
  df_bert_titles_train = pd.merge(df_train, df_bert_titles, how='inner') 
  bert_titles_train = df_bert_titles_train.iloc[:, 3:].values
  del df_bert_titles_train
  df_bert_titles_validate = pd.merge(df_validate, df_bert_titles, how='inner')
  bert_titles_validate = df_bert_titles_validate.iloc[:, 3:].values
  del df_bert_titles_validate
  del df_bert_titles
  gc.collect()
  
  print('Loading bert_case_alltags feature...')
  df_bert_alltags = pd.read_csv(args.bert_alltags)
  df_bert_alltags_train = pd.merge(df_train, df_bert_alltags, how='inner')
  bert_alltags_train = df_bert_alltags_train.iloc[:, 3:].values
  del df_bert_alltags_train
  df_bert_alltags_validate = pd.merge(df_validate, df_bert_alltags, how='inner')
  bert_allatgs_validate = df_bert_alltags_validate.iloc[:, 3:].values
  del df_bert_alltags_validate
  del df_bert_alltags
  gc.collect()
  
  print('Loading glove feature...')
  df_glove = pd.read_csv(args.glove)
  df_glove_train = pd.merge(df_train, df_glove, how='inner')
  glove_train = df_glove_train.iloc[:, 3:].values
  del df_glove_train
  df_glove_validate = pd.merge(df_validate, df_glove, how='inner')
  glove_validate = df_glove_validate.iloc[:, 3:].values
  del df_glove_validate
  del df_glove
  gc.collect()
  
  print('Loading image resnext feature...')
  df_resnext = pd.read_csv(args.resnext)
  df_resnext_train = pd.merge(df_train, df_resnext, how='inner')
  resnext_train = df_resnext_train.iloc[:, 3:].values
  del df_resnext_train
  df_resnext_validate = pd.merge(df_validate, df_resnext, how='inner')
  resnext_validate = df_resnext_validate.iloc[:, 3:].values
  del df_resnext_validate
  del df_resnext
  gc.collect()
  
  print('Loading postdate_train numerical feature...')
  df_postdate = pd.read_csv(args.postdate_train)
  df_postdate_train = pd.merge(df_train, df_postdate, how='inner') 
  postdate_train = df_postdate_train.iloc[:, 3:].values
  del df_postdate_train
  df_postdate_validate = pd.merge(df_validate, df_postdate, how='inner')
  postdate_validate = df_postdate_validate.iloc[:, 3:].values
  del df_postdate_validate
  del df_postdate
  gc.collect()
  
  print('Loading postdate_test numerical feature...')
  df_postdate_test = pd.read_csv(args.postdate_test)
  postdate_test = df_postdate_test.iloc[:, 2:].values
  del df_postdate_test
  gc.collect()
  
  postdate = np.concatenate((postdate_train, postdate_validate, postdate_test), axis=0)
  ss = StandardScaler()
  print('Z-score...')
  postdate = ss.fit_transform(postdate)
  postdate_train = postdate[0:len(postdate_train)]
  postdate_validate = postdate[len(postdate_train):len(postdate_train)+len(postdate_validate)]
  print('Z-score... over')
  
  print('Loading concept feature...')
  df_concept = pd.read_csv(args.concept)
  df_concept_train = pd.merge(df_train, df_concept, how='inner')
  concept_train = df_concept_train.iloc[:, 3:].values
  del df_concept_train
  df_concept_validate = pd.merge(df_validate, df_concept, how='inner')
  concept_validate = df_concept_validate.iloc[:, 3:].values
  del df_concept_validate
  del df_concept
  gc.collect()
  
  print('Loading wordchar_train numerical feature...')
  df_wordchar = pd.read_csv(args.wordchar_train)
  df_wordchar_train = pd.merge(df_train, df_wordchar, how='inner') 
  wordchar_train = df_wordchar_train.iloc[:, 3:].values
  del df_wordchar_train
  df_wordchar_validate = pd.merge(df_validate, df_wordchar, how='inner')
  wordchar_validate = df_wordchar_validate.iloc[:, 3:].values
  del df_wordchar_validate
  del df_wordchar
  gc.collect()
  
  print('Loading wordchar_test numerical feature...')
  df_wordchar_test = pd.read_csv(args.wordchar_test)
  wordchar_test = df_wordchar_test.iloc[:, 2:].values
  del df_wordchar_test
  gc.collect()
  
  wordchar = np.concatenate((wordchar_train, wordchar_validate, wordchar_test), axis=0)
  ss = StandardScaler()
  print('Z-score...')
  wordchar = ss.fit_transform(wordchar)
  wordchar_train = wordchar[0:len(wordchar_train)]
  wordchar_validate = wordchar[len(wordchar_train):len(wordchar_train)+len(wordchar_validate)]
  print('Z-score... over')
  
  label_train = df_train["label"].values
  label_validate = df_validate["label"].values
  del df_train
  del df_validate
  gc.collect()
  
  label_train = torch.FloatTensor(label_train).unsqueeze(1)
  label_validate = torch.FloatTensor(label_validate).unsqueeze(1)
  bert_titles_train = torch.FloatTensor(bert_titles_train)
  bert_titles_validate = torch.FloatTensor(bert_titles_validate)
  bert_alltags_train = torch.FloatTensor(bert_alltags_train)
  bert_alltags_validate = torch.FloatTensor(bert_allatgs_validate)
  glove_train = torch.FloatTensor(glove_train)
  glove_validate = torch.FloatTensor(glove_validate)
  resnext_train = torch.FloatTensor(resnext_train)
  resnext_validate = torch.FloatTensor(resnext_validate)
  postdate_train = torch.FloatTensor(postdate_train)
  postdate_validate = torch.FloatTensor(postdate_validate)
  concept_train = torch.FloatTensor(concept_train)
  concept_validate = torch.FloatTensor(concept_validate)
  wordchar_train = torch.FloatTensor(wordchar_train)
  wordchar_validate = torch.FloatTensor(wordchar_validate)  
             
  # Tensordataset
  train = TensorDataset(bert_titles_train, bert_alltags_train, glove_train, resnext_train, label_train, postdate_train, concept_train, wordchar_train)
  validate = TensorDataset(bert_titles_validate, bert_alltags_validate, glove_validate, resnext_validate, label_validate, postdate_validate, concept_validate,wordchar_validate)
  
  # Dataloader
  train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=False)
  valid_loader = DataLoader(validate, batch_size=args.batch_size, shuffle=False)
  
  model = MLP()
  modules = model.named_children()
 # print(modules)
  if torch.cuda.is_available():
      model = model.cuda()
  
  optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate, weight_decay=1e-5)
  #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', \
  #        factor=0.95, verbose=1, patience=5) # adaptive learning rate. 
  loss_fn = torch.nn.L1Loss()
  
  avg_losses_f = []
  avg_val_losses_f = []
  
  for epoch in range(args.epochs):
    print('Training epoch{}...'.format(epoch))
    model.train()
    start_time = time.time()
    model.train()
    avg_loss = 0.
    for i, (bert_titles_train, bert_alltags_train, glove_train, resnext_train, label_train, postdate_train, concept_train, wordchar_train) in enumerate(train_loader):
      if torch.cuda.is_available():
        bert_titles_train = bert_titles_train.cuda()
        bert_alltags_train = bert_alltags_train.cuda()
        wordchar_train = wordchar_train.cuda()
        glove_train = glove_train.cuda()
        resnext_train = resnext_train.cuda()
        concept_train = concept_train.cuda()
        label_train = label_train.cuda()
        postdate_train = postdate_train.cuda()
      y_pred = model(bert_titles_train, bert_alltags_train, glove_train, resnext_train, postdate_train, concept_train, wordchar_train)
  
      loss = loss_fn(y_pred, label_train)
      optimizer.zero_grad()
      loss.backward()           
      optimizer.step()          
      avg_loss += loss.item() / len(train_loader)
      
    model.eval()
    avg_val_loss = 0.
    y_preds = []
    y_labels = []
    for i, (bert_titles_validate, bert_alltags_validate, glove_validate, resnext_validate, label_validate, postdate_validate, concept_validate, wordchar_validate) in enumerate(valid_loader):
      if torch.cuda.is_available():
        bert_titles_validate = bert_titles_validate.cuda()
        bert_alltags_validate = bert_alltags_validate.cuda()
        wordchar_validate = wordchar_validate.cuda()
        glove_validate = glove_validate.cuda()
        resnext_validate = resnext_validate.cuda()
        concept_validate = concept_validate.cuda()
        label_validate = label_validate.cuda()
        postdate_validate = postdate_validate.cuda()
      y_pred = model(bert_titles_validate, bert_alltags_validate, glove_validate, resnext_validate, postdate_validate, concept_validate, wordchar_validate).detach()
      
      avg_val_loss += loss_fn(y_pred, label_validate).item() / len(valid_loader)
      y_pred = y_pred.squeeze(1).detach().cpu().numpy().tolist()
      label_validate = label_validate.squeeze(1).detach().cpu().numpy().tolist()
      y_preds.extend(y_pred)
      y_labels.extend(label_validate)
    elapsed_time = time.time() - start_time
    MAE = mean_absolute_error(y_labels,y_preds)
   # scheduler.step(MAE)
    correction, _ = spearmanr(y_preds, y_labels)
    print('Epoch {}/{} \t loss={:.4f} \t MAE={:.4f} \t SRC:{:.4f} \t time={:.2f}s'.format(epoch + 1, args.epochs, avg_loss, MAE, correction, elapsed_time))
    torch.save(model.state_dict(), os.path.join('/home/zht/MLP/model', 'model{}.pth'.format(str(epoch))))
    avg_losses_f.append(avg_loss)
    avg_val_losses_f.append(avg_val_loss)
    
def train_all():     
  print('Loading train label...')
  df_train = pd.read_csv(args.label_filepath)
  print('Loading validate label...')
  df_validate = pd.read_csv(args.validate_label_filepath)
  
  print('Loading bert_case_titles feature...')
  df_bert_titles = pd.read_csv(args.bert_titles)
  df_bert_titles_train = pd.merge(df_train, df_bert_titles, how='inner') 
  bert_titles_train = df_bert_titles_train.iloc[:, 3:].values
  del df_bert_titles_train
  df_bert_titles_validate = pd.merge(df_validate, df_bert_titles, how='inner')
  bert_titles_validate = df_bert_titles_validate.iloc[:, 3:].values
  del df_bert_titles_validate
  del df_bert_titles
  gc.collect()
  
  print('Loading bert_case_alltags feature...')
  df_bert_alltags = pd.read_csv(args.bert_alltags)
  df_bert_alltags_train = pd.merge(df_train, df_bert_alltags, how='inner')
  bert_alltags_train = df_bert_alltags_train.iloc[:, 3:].values
  del df_bert_alltags_train
  df_bert_alltags_validate = pd.merge(df_validate, df_bert_alltags, how='inner')
  bert_allatgs_validate = df_bert_alltags_validate.iloc[:, 3:].values
  del df_bert_alltags_validate
  del df_bert_alltags
  gc.collect()
  
  print('Loading glove feature...')
  df_glove = pd.read_csv(args.glove)
  df_glove_train = pd.merge(df_train, df_glove, how='inner')
  glove_train = df_glove_train.iloc[:, 3:].values
  del df_glove_train
  df_glove_validate = pd.merge(df_validate, df_glove, how='inner')
  glove_validate = df_glove_validate.iloc[:, 3:].values
  del df_glove_validate
  del df_glove
  gc.collect()
  
  print('Loading image resnext feature...')
  df_resnext = pd.read_csv(args.resnext)
  df_resnext_train = pd.merge(df_train, df_resnext, how='inner')
  resnext_train = df_resnext_train.iloc[:, 3:].values
  del df_resnext_train
  df_resnext_validate = pd.merge(df_validate, df_resnext, how='inner')
  resnext_validate = df_resnext_validate.iloc[:, 3:].values
  del df_resnext_validate
  del df_resnext
  gc.collect()
  
  print('Loading postdate_train numerical feature...')
  df_postdate = pd.read_csv(args.postdate_train)
  df_postdate_train = pd.merge(df_train, df_postdate, how='inner') 
  postdate_train = df_postdate_train.iloc[:, 3:].values
  del df_postdate_train
  df_postdate_validate = pd.merge(df_validate, df_postdate, how='inner')
  postdate_validate = df_postdate_validate.iloc[:, 3:].values
  del df_postdate_validate
  del df_postdate
  gc.collect()
  
  print('Loading postdate_test numerical feature...')
  df_postdate_test = pd.read_csv(args.postdate_test)
  postdate_test = df_postdate_test.iloc[:, 2:].values
  del df_postdate_test
  gc.collect()
  
  postdate = np.concatenate((postdate_train, postdate_validate, postdate_test), axis=0)
  ss = StandardScaler()
  print('Z-score...')
  postdate = ss.fit_transform(postdate)
  postdate_train = postdate[0:len(postdate_train)]
  postdate_validate = postdate[len(postdate_train):len(postdate_train)+len(postdate_validate)]
  print('Z-score... over')
  
  print('Loading concept feature...')
  df_concept = pd.read_csv(args.concept)
  df_concept_train = pd.merge(df_train, df_concept, how='inner')
  concept_train = df_concept_train.iloc[:, 3:].values
  del df_concept_train
  df_concept_validate = pd.merge(df_validate, df_concept, how='inner')
  concept_validate = df_concept_validate.iloc[:, 3:].values
  del df_concept_validate
  del df_concept
  gc.collect()
  
  print('Loading uid feature...')
  df_uid = pd.read_csv(args.uid)
  df_uid_train = pd.merge(df_train, df_uid, how='inner')
  uid_train = df_uid_train.iloc[:, 3:].values
  del df_uid_train
  df_uid_validate = pd.merge(df_validate, df_uid, how='inner')
  uid_validate = df_uid_validate.iloc[:, 3:].values
  del df_uid_validate
  del df_uid
  gc.collect()
  
  print('Loading wordchar_train numerical feature...')
  df_wordchar = pd.read_csv(args.wordchar_train)
  df_wordchar_train = pd.merge(df_train, df_wordchar, how='inner') 
  wordchar_train = df_wordchar_train.iloc[:, 3:].values
  del df_wordchar_train
  df_wordchar_validate = pd.merge(df_validate, df_wordchar, how='inner')
  wordchar_validate = df_wordchar_validate.iloc[:, 3:].values
  del df_wordchar_validate
  del df_wordchar
  gc.collect()
  
  print('Loading wordchar_test numerical feature...')
  df_wordchar_test = pd.read_csv(args.wordchar_test)
  wordchar_test = df_wordchar_test.iloc[:, 2:].values
  del df_wordchar_test
  gc.collect()
  
  wordchar = np.concatenate((wordchar_train, wordchar_validate, wordchar_test), axis=0)
  ss = StandardScaler()
  print('Z-score...')
  wordchar = ss.fit_transform(wordchar)
  wordchar_train = wordchar[0:len(wordchar_train)]
  wordchar_validate = wordchar[len(wordchar_train):len(wordchar_train)+len(wordchar_validate)]
  print('Z-score... over')
  
  label_train = df_train["label"].values
  label_validate = df_validate["label"].values
  del df_train
  del df_validate
  gc.collect()
  
  label_train = torch.FloatTensor(label_train).unsqueeze(1)
  label_validate = torch.FloatTensor(label_validate).unsqueeze(1)
  bert_titles_train = torch.FloatTensor(bert_titles_train)
  bert_titles_validate = torch.FloatTensor(bert_titles_validate)
  bert_alltags_train = torch.FloatTensor(bert_alltags_train)
  bert_alltags_validate = torch.FloatTensor(bert_allatgs_validate)
  glove_train = torch.FloatTensor(glove_train)
  glove_validate = torch.FloatTensor(glove_validate)
  resnext_train = torch.FloatTensor(resnext_train)
  resnext_validate = torch.FloatTensor(resnext_validate)
  postdate_train = torch.FloatTensor(postdate_train)
  postdate_validate = torch.FloatTensor(postdate_validate)
  concept_train = torch.FloatTensor(concept_train)
  concept_validate = torch.FloatTensor(concept_validate)
  uid_train = torch.FloatTensor(uid_train)
  uid_validate = torch.FloatTensor(uid_validate)
  wordchar_train = torch.FloatTensor(wordchar_train)
  wordchar_validate = torch.FloatTensor(wordchar_validate)  
             
  # Tensordataset
  train = TensorDataset(bert_titles_train, bert_alltags_train, glove_train, resnext_train, label_train, postdate_train, concept_train, uid_train, wordchar_train)
  validate = TensorDataset(bert_titles_validate, bert_alltags_validate, glove_validate, resnext_validate, label_validate, postdate_validate, concept_validate, uid_validate, wordchar_validate)
  
  # Dataloader
  train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=False)
  valid_loader = DataLoader(validate, batch_size=args.batch_size, shuffle=False)
  
  model = MLP()
  modules = model.named_children()
  print(modules)
  if torch.cuda.is_available():
      model = model.cuda()
  
  optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate, weight_decay=1e-5)
  #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', \
  #        factor=0.95, verbose=1, patience=5) # adaptive learning rate. 
  loss_fn = torch.nn.L1Loss()
  
  avg_losses_f = []
  avg_val_losses_f = []
  
  for epoch in range(args.epochs):
    print('Training epoch{}...'.format(epoch))
    model.train()
    start_time = time.time()
    model.train()
    avg_loss = 0.
    for i, (bert_titles_train, bert_alltags_train, glove_train, resnext_train, label_train, postdate_train, concept_train, uid_train, wordchar_train) in enumerate(train_loader):
      if torch.cuda.is_available():
        bert_titles_train = bert_titles_train.cuda()
        bert_alltags_train = bert_alltags_train.cuda()
        wordchar_train = wordchar_train.cuda()
        glove_train = glove_train.cuda()
        resnext_train = resnext_train.cuda()
        concept_train = concept_train.cuda()
        uid_train = uid_train.cuda()
        label_train = label_train.cuda()
        postdate_train = postdate_train.cuda()
      y_pred = model(bert_titles_train, bert_alltags_train, glove_train, resnext_train, postdate_train, concept_train, uid_train, wordchar_train)
  
      loss = loss_fn(y_pred, label_train)
      optimizer.zero_grad()
      loss.backward()           
      optimizer.step()          
      avg_loss += loss.item() / len(train_loader)
      
    model.eval()
    avg_val_loss = 0.
    y_preds = []
    y_labels = []
    for i, (bert_titles_validate, bert_alltags_validate, glove_validate, resnext_validate, label_validate, postdate_validate, concept_validate, uid_validate, wordchar_validate) in enumerate(valid_loader):
      if torch.cuda.is_available():
        bert_titles_validate = bert_titles_validate.cuda()
        bert_alltags_validate = bert_alltags_validate.cuda()
        wordchar_validate = wordchar_validate.cuda()
        glove_validate = glove_validate.cuda()
        resnext_validate = resnext_validate.cuda()
        concept_validate = concept_validate.cuda()
        uid_validate = uid_validate.cuda()
        label_validate = label_validate.cuda()
        postdate_validate = postdate_validate.cuda()
      y_pred = model(bert_titles_validate, bert_alltags_validate, glove_validate, resnext_validate, postdate_validate, concept_validate, uid_validate, wordchar_validate).detach()
      
      avg_val_loss += loss_fn(y_pred, label_validate).item() / len(valid_loader)
      y_pred = y_pred.squeeze(1).detach().cpu().numpy().tolist()
      label_validate = label_validate.squeeze(1).detach().cpu().numpy().tolist()
      y_preds.extend(y_pred)
      y_labels.extend(label_validate)
    elapsed_time = time.time() - start_time
    MAE = mean_absolute_error(y_labels,y_preds)
   # scheduler.step(MAE)
    correction, _ = spearmanr(y_preds, y_labels)
    print('Epoch {}/{} \t loss={:.4f} \t MAE={:.4f} \t SRC:{:.4f} \t time={:.2f}s'.format(epoch + 1, args.epochs, avg_loss, MAE, correction, elapsed_time))
    torch.save(model.state_dict(), os.path.join('/home/zht/MLP/model_uid', 'model{}.pth'.format(str(epoch))))
    avg_losses_f.append(avg_loss)
    avg_val_losses_f.append(avg_val_loss)


def extractor_train():
  print('Loading train pid and uid...')
  df_train = pd.read_csv(args.postdate_train)
  uid_train = df_train.iloc[:, 1].values
  pid_train = df_train.iloc[:, 0].values
    
  print('Loading test pid and uid...')
  df_test = pd.read_csv(args.postdate_test)
  pid_test = df_test.iloc[:, 0].values
  uid_test = df_test.iloc[:, 1].values
  
  print('Loading bert_case_titles feature...')
  df_bert_titles = pd.read_csv(args.bert_titles)
  df_bert_titles = pd.merge(df_train, df_bert_titles, how='inner')
  bert_titles_train = df_bert_titles.iloc[:, 11:].values
  del df_bert_titles
  gc.collect()

  print('Loading bert_case_alltags feature...')
  df_bert_alltags = pd.read_csv(args.bert_alltags)
  df_bert_alltags = pd.merge(df_train, df_bert_alltags, how='inner')
  bert_alltags_train = df_bert_alltags.iloc[:, 11:].values
  del df_bert_alltags
  gc.collect()
  
  print('Loading glove feature...')
  df_glove = pd.read_csv(args.glove)
  df_glove = pd.merge(df_train, df_glove, how='inner')
  glove_train = df_glove.iloc[:, 11:].values
  del df_glove
  gc.collect()
  
  print('Loading image resnext feature...')
  df_resnext = pd.read_csv(args.resnext)
  df_resnext = pd.merge(df_train, df_resnext, how='inner')
  resnext_train = df_resnext.iloc[:, 11:].values
  del df_resnext
  gc.collect()
  
  print('Loading postdate_train numerical feature...')
  df_postdate = pd.read_csv(args.postdate_train)
  postdate_train = df_postdate.iloc[:, 2:].values
  del df_postdate
  gc.collect()
  
  print('Loading postdate_test numerical feature...')
  df_postdate_test = pd.read_csv(args.postdate_test)
  postdate_test = df_postdate_test.iloc[:, 2:].values
  del df_postdate_test
  gc.collect()
  
  postdate = np.concatenate((postdate_train, postdate_test), axis=0)
  ss = StandardScaler()
  print('Z-score...')
  postdate = ss.fit_transform(postdate)
  postdate_train = postdate[0:len(postdate_train)]
  postdate_test = postdate[len(postdate_train):]
  print('Z-score... over')
  
  print('Loading concept feature...')
  df_concept = pd.read_csv(args.concept)
  df_concept = pd.merge(df_train, df_concept, how='inner')
  concept_train = df_concept.iloc[:, 11:].values
  del df_concept
  gc.collect()
  
  print('Loading uid feature...')
  df_uid = pd.read_csv(args.uid)
  df_uid = pd.merge(df_train, df_uid, how='inner')
  uid_feature_train = df_uid.iloc[:, 11:].values
  del df_uid
  gc.collect()
  
  print('Loading wordchar_train numerical feature...')
  df_wordchar = pd.read_csv(args.wordchar_train)
  df_wordchar = pd.merge(df_train, df_wordchar, how='inner')
  wordchar_train = df_wordchar.iloc[:, 11:].values
  del df_wordchar
  gc.collect()
  
  print('Loading wordchar_test numerical feature...')
  df_wordchar_test = pd.read_csv(args.wordchar_test)
  df_wordchar_test = pd.merge(df_test, df_wordchar_test, how='inner')
  wordchar_test = df_wordchar_test.iloc[:, 11:].values
  del df_wordchar_test
  gc.collect()
  
  wordchar = np.concatenate((wordchar_train, wordchar_test), axis=0)
  ss = StandardScaler()
  print('Z-score...')
  wordchar = ss.fit_transform(wordchar)
  wordchar_train = wordchar[0:len(wordchar_train)]
  wordchar_test = wordchar[len(wordchar_train): ]
  print('Z-score... over')
  
  del df_train
  del df_test
  gc.collect()
  
  class data(Dataset):
    def __init__(self, pid, uid, titles, alltags, glove, resnext, postdate, concept, uid_feature, wordchar):
        self.pid = pid
        self.uid = uid
        self.titles = titles
        self.alltags = alltags
        self.glove = glove
        self.resnext = resnext
        self.postdate = postdate
        self.concept = concept
        self.wordchar = wordchar
        self.uid_feature = uid_feature
        
    def __len__(self):
        return len(self.pid)

    def __getitem__(self, index):
        return self.pid[index], self.uid[index], self.titles[index], self.alltags[index], self.glove[index],\
                  self.resnext[index], self.postdate[index], self.concept[index], self.uid_feature[index], self.wordchar[index]
  
  
  
  dataset = data(pid_train, uid_train, bert_titles_train, bert_alltags_train, 
                    glove_train, resnext_train, postdate_train, concept_train, uid_feature_train, wordchar_train)     
  dataloader = DataLoader(dataset=dataset,
                              batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
  
  cache = []
  def hook(module, input, output):
    cache.append(output.clone().detach())
    
  with open('./mlpv4_train.csv', 'w') as f:
    csv_writer = csv.writer(f)
    csv_line = ['pid','uid']
    csv_line.extend(['mlp{}'.format(i) for i in range(814)])
    csv_writer.writerow(csv_line)
  
  model = MLP()
  model.load_state_dict(torch.load('/home/zht/MLP/model_uid/model24.pth'))
  model.linear_titles_2.register_forward_hook(hook)
  model.linear_alltags_2.register_forward_hook(hook)
  model.linear_glove_2.register_forward_hook(hook)
  model.linear_resnext_2.register_forward_hook(hook)
  model.linear_concept_2.register_forward_hook(hook)
  model.linear_uid_2.register_forward_hook(hook)
  model.linear_postdate_2.register_forward_hook(hook)
  model.linear_wordchar2.register_forward_hook(hook)
  model.eval()
  model.cuda()
  
  with open('./mlpv4_train.csv', 'a+') as f:
    csv_writer = csv.writer(f)
    with torch.no_grad():
      total_step = len(dataloader)
      for i, (pid, uid, bert_titles, bert_alltags, glove, \
                            resnext, postdate, concept, uid_feature, wordchar) in tqdm(enumerate(dataloader), total=total_step, ncols=100):
        bert_titles = bert_titles.float().cuda()
        bert_alltags = bert_alltags.float().cuda()
        glove = glove.float().cuda()
        resnext = resnext.float().cuda()
        postdate = postdate.float().cuda()
        concept = concept.float().cuda()
        uid_feature = uid_feature.float().cuda()
        wordchar = wordchar.float().cuda()
        y = model(bert_titles, bert_alltags, glove, resnext, postdate, concept, uid_feature, wordchar)
        feature = torch.cat(cache, dim=1)
        del cache[:]
        feature = feature.detach().cpu().numpy()
        pid = pid.numpy()
        uid = np.array(uid)
        pid = pid[:, np.newaxis]
        uid = uid[:, np.newaxis]
        csv_lines = np.hstack([pid, uid, feature])
        csv_writer.writerows(csv_lines)


def extractor_test():
  
  print('Loading train pid and uid...')
  df_train = pd.read_csv(args.postdate_train)
  uid_train = df_train.iloc[:, 1].values
  pid_train = df_train.iloc[:, 0].values
  
  print('Loading test pid and uid...')
  df_test = pd.read_csv(args.postdate_test)
  pid_test = df_test.iloc[:, 0].values
  uid_test = df_test.iloc[:, 1].values
  
  print('Loading bert titles feature...')
  df_bert_titles = pd.read_csv(args.bert_titles_test)
  df_bert_titles = pd.merge(df_test, df_bert_titles, how='inner')
  bert_titles_test = df_bert_titles.iloc[:, 11:].values
  del df_bert_titles
  gc.collect()
  
  print('Loading bert alltags feature...')
  df_bert_alltags = pd.read_csv(args.bert_alltags_test)
  df_bert_alltags = pd.merge(df_test, df_bert_alltags, how='inner')
  bert_alltags_test = df_bert_alltags.iloc[:, 11:].values
  del df_bert_alltags
  gc.collect()
  
  print('Loading glove feature...')
  df_glove = pd.read_csv(args.glove_test)
  df_glove = pd.merge(df_test, df_glove, how='inner')
  glove_test = df_glove.iloc[:, 11:].values
  del df_glove
  gc.collect()
  
  print('Loading image resnext feature...')
  df_resnext = pd.read_csv(args.resnext_test)
  df_resnext = pd.merge(df_test, df_resnext, how='inner')
  resnext_test = df_resnext.iloc[:, 11:].values
  del df_resnext
  gc.collect()
  
  print('Loading postdate_train numerical feature...')
  df_postdate = pd.read_csv(args.postdate_train) 
  postdate_train = df_postdate.iloc[:, 2:].values
  del df_postdate
  gc.collect()
  
  print('Loading postdate_test numerical feature...')
  df_postdate_test = pd.read_csv(args.postdate_test)
  postdate_test = df_postdate_test.iloc[:, 2:].values
  del df_postdate_test
  gc.collect()
  
  postdate = np.concatenate((postdate_train, postdate_test), axis=0)
  ss = StandardScaler()
  print('Z-score...')
  postdate = ss.fit_transform(postdate)
  postdate_train = postdate[0:len(postdate_train)]
  postdate_test = postdate[len(postdate_train):]
  print('Z-score... over')
  
  print('Loading concept feature...')
  df_concept = pd.read_csv(args.concept_test)
  df_concept = pd.merge(df_test, df_concept, how='inner')
  concept_test = df_concept.iloc[:, 11:].values
  del df_concept
  gc.collect()
  
  print('Loading concept feature...')
  df_uid = pd.read_csv(args.uid_test)
  df_uid = pd.merge(df_test, df_uid, how='inner')
  uid_feature_test = df_uid.iloc[:, 11:].values
  del df_uid
  gc.collect()
  
  print('Loading wordchar_train numerical feature...')
  df_wordchar = pd.read_csv(args.wordchar_train)
  df_wordchar = pd.merge(df_train, df_wordchar, how='inner')
  wordchar_train = df_wordchar.iloc[:, 11:].values
  del df_wordchar
  gc.collect()
  
  print('Loading wordchar_test numerical feature...')
  df_wordchar_test = pd.read_csv(args.wordchar_test)
  df_wordchar_test = pd.merge(df_test, df_wordchar_test, how='inner')
  wordchar_test = df_wordchar_test.iloc[:, 11:].values
  del df_wordchar_test
  gc.collect()
  
  wordchar = np.concatenate((wordchar_train, wordchar_test), axis=0)
  ss = StandardScaler()
  print('Z-score...')
  wordchar = ss.fit_transform(wordchar)
  wordchar_train = wordchar[0:len(wordchar_train)]
  wordchar_test = wordchar[len(wordchar_train): ]
  print('Z-score... over')
  
  del df_test
  del df_train
  gc.collect()
  
  class data(Dataset):
    def __init__(self, pid, uid, titles, alltags, glove, resnext, postdate, concept, uid_feature, wordchar):
        self.pid = pid
        self.uid = uid
        self.titles = titles
        self.alltags = alltags
        self.glove = glove
        self.resnext = resnext
        self.postdate = postdate
        self.concept = concept
        self.wordchar = wordchar
        self.uid_feature = uid_feature

    def __len__(self):
        return len(self.pid)

    def __getitem__(self, index):
        return self.pid[index], self.uid[index], self.titles[index], self.alltags[index], self.glove[index],\
                          self.resnext[index], self.postdate[index], self.concept[index], self.uid_feature[index], self.wordchar[index]
  
  
  
  dataset = data(pid_test, uid_test, bert_titles_test, bert_alltags_test, \
                    glove_test, resnext_test, postdate_test, concept_test, uid_feature_test, wordchar_test)     
  dataloader = DataLoader(dataset=dataset,
                              batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
  
  cache = []
  def hook(module, input, output):
    cache.append(output.clone().detach())
    
  with open('./mlpv4_test.csv', 'w') as f:
    csv_writer = csv.writer(f)
    csv_line = ['pid','uid']
    csv_line.extend(['mlp{}'.format(i) for i in range(814)])
    csv_writer.writerow(csv_line)
  
  model = MLP()
  model.load_state_dict(torch.load('/home/zht/MLP/model_uid/model24.pth'))
  model.linear_titles_2.register_forward_hook(hook)
  model.linear_alltags_2.register_forward_hook(hook)
  model.linear_glove_2.register_forward_hook(hook)
  model.linear_resnext_2.register_forward_hook(hook)
  model.linear_concept_2.register_forward_hook(hook)
  model.linear_uid_2.register_forward_hook(hook)
  model.linear_postdate_2.register_forward_hook(hook)
  model.linear_wordchar2.register_forward_hook(hook)
  model.eval()
  model.cuda()
  
  with open('./mlpv4_test.csv', 'a+') as f:
    csv_writer = csv.writer(f)
    with torch.no_grad():
      total_step = len(dataloader)
      for i, (pid, uid, bert_titles, bert_alltags, glove, \
                            resnext, postdate, concept, uid_feature, wordchar) in tqdm(enumerate(dataloader), total=total_step, ncols=100):
        bert_titles = bert_titles.float().cuda()
        bert_alltags = bert_alltags.float().cuda()
        glove = glove.float().cuda()
        resnext = resnext.float().cuda()
        postdate = postdate.float().cuda()
        concept = concept.float().cuda()
        uid_feature = uid_feature.float().cuda()
        wordchar = wordchar.float().cuda()
        y = model(bert_titles, bert_alltags, glove, resnext, postdate, concept, uid_feature, wordchar)
        feature = torch.cat(cache, dim=1)
        del cache[:]
        feature = feature.detach().cpu().numpy()
        pid = pid.numpy()
        uid = np.array(uid)
        pid = pid[:, np.newaxis]
        uid = uid[:, np.newaxis]
        csv_lines = np.hstack([pid, uid, feature])
        csv_writer.writerows(csv_lines)
        
def test():
  
  print('Loading train pid and uid...')
  df_train = pd.read_csv(args.postdate_train)
  uid_train = df_train.iloc[:, 1].values
  pid_train = df_train.iloc[:, 0].values
  
  print('Loading test pid and uid...')
  df_test = pd.read_csv(args.postdate_test)
  pid_test = df_test.iloc[:, 0].values
  uid_test = df_test.iloc[:, 1].values
  
  print('Loading bert titles feature...')
  df_bert_titles = pd.read_csv(args.bert_titles_test)
  df_bert_titles = pd.merge(df_test, df_bert_titles, how='inner')
  bert_titles_test = df_bert_titles.iloc[:, 11:].values
  del df_bert_titles
  gc.collect()
  
  print('Loading bert alltags feature...')
  df_bert_alltags = pd.read_csv(args.bert_alltags_test)
  df_bert_alltags = pd.merge(df_test, df_bert_alltags, how='inner')
  bert_alltags_test = df_bert_alltags.iloc[:, 11:].values
  del df_bert_alltags
  gc.collect()
  
  print('Loading glove feature...')
  df_glove = pd.read_csv(args.glove_test)
  df_glove = pd.merge(df_test, df_glove, how='inner')
  glove_test = df_glove.iloc[:, 11:].values
  del df_glove
  gc.collect()
  
  print('Loading image resnext feature...')
  df_resnext = pd.read_csv(args.resnext_test)
  df_resnext = pd.merge(df_test, df_resnext, how='inner')
  resnext_test = df_resnext.iloc[:, 11:].values
  del df_resnext
  gc.collect()
  
  print('Loading postdate_train numerical feature...')
  df_postdate = pd.read_csv(args.postdate_train) 
  postdate_train = df_postdate.iloc[:, 2:].values
  del df_postdate
  gc.collect()
  
  print('Loading postdate_test numerical feature...')
  df_postdate_test = pd.read_csv(args.postdate_test)
  postdate_test = df_postdate_test.iloc[:, 2:].values
  del df_postdate_test
  gc.collect()
  
  postdate = np.concatenate((postdate_train, postdate_test), axis=0)
  ss = StandardScaler()
  print('Z-score...')
  postdate = ss.fit_transform(postdate)
  postdate_train = postdate[0:len(postdate_train)]
  postdate_test = postdate[len(postdate_train):]
  print('Z-score... over')
  
  print('Loading concept feature...')
  df_concept = pd.read_csv(args.concept_test)
  df_concept = pd.merge(df_test, df_concept, how='inner')
  concept_test = df_concept.iloc[:, 11:].values
  del df_concept
  gc.collect()
  
  print('Loading uid feature...')
  df_uid = pd.read_csv(args.uid_test)
  df_uid = pd.merge(df_test, df_uid, how='inner')
  uid_feature_test = df_uid.iloc[:, 11:].values
  del df_uid
  gc.collect()
  
  print('Loading wordchar_train numerical feature...')
  df_wordchar = pd.read_csv(args.wordchar_train)
  df_wordchar = pd.merge(df_train, df_wordchar, how='inner')
  wordchar_train = df_wordchar.iloc[:, 11:].values
  del df_wordchar
  gc.collect()
  
  print('Loading wordchar_test numerical feature...')
  df_wordchar_test = pd.read_csv(args.wordchar_test)
  df_wordchar_test = pd.merge(df_test, df_wordchar_test, how='inner')
  wordchar_test = df_wordchar_test.iloc[:, 11:].values
  del df_wordchar_test
  gc.collect()
  
  wordchar = np.concatenate((wordchar_train, wordchar_test), axis=0)
  ss = StandardScaler()
  print('Z-score...')
  wordchar = ss.fit_transform(wordchar)
  wordchar_train = wordchar[0:len(wordchar_train)]
  wordchar_test = wordchar[len(wordchar_train): ]
  print('Z-score... over')
  
  del df_test
  del df_train
  gc.collect()
  
  class data(Dataset):
    def __init__(self, pid, uid, titles, alltags, glove, resnext, postdate, concept, uid_feature, wordchar):
        self.pid = pid
        self.uid = uid
        self.titles = titles
        self.alltags = alltags
        self.glove = glove
        self.resnext = resnext
        self.postdate = postdate
        self.concept = concept
        self.wordchar = wordchar
        self.uid_feature = uid_feature

    def __len__(self):
        return len(self.pid)

    def __getitem__(self, index):
        return self.pid[index], self.uid[index], self.titles[index], self.alltags[index], self.glove[index],\
                          self.resnext[index], self.postdate[index], self.concept[index], self.uid_feature[index], self.wordchar[index]
  
  
  
  dataset = data(pid_test, uid_test, bert_titles_test, bert_alltags_test, \
                    glove_test, resnext_test, postdate_test, concept_test, uid_feature_test, wordchar_test)     
  dataloader = DataLoader(dataset=dataset,
                              batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
                              
  with open('./mlpuid20_result.csv', 'w') as f:
    csv_writer = csv.writer(f)
    csv_line = ['pid','label']
    csv_writer.writerow(csv_line) 
    print(1)
                           
  model = MLP()
  model.load_state_dict(torch.load('/home/zht/MLP/model_uid/model19.pth'))
  model.eval()
  model.cuda()
  
  with open('./mlpuid20_result.csv', 'a+') as f:
    csv_writer = csv.writer(f)
    with torch.no_grad():
      total_step = len(dataloader)
      for i, (pid, uid, bert_titles, bert_alltags, glove, \
                            resnext, postdate, concept, uid_feature, wordchar) in tqdm(enumerate(dataloader), total=total_step, ncols=100):
        bert_titles = bert_titles.float().cuda()
        bert_alltags = bert_alltags.float().cuda()
        glove = glove.float().cuda()
        resnext = resnext.float().cuda()
        postdate = postdate.float().cuda()
        concept = concept.float().cuda()
        uid_feature = uid_feature.float().cuda()
        wordchar = wordchar.float().cuda()
        y = model(bert_titles, bert_alltags, glove, resnext, postdate, concept, uid_feature, wordchar)
        y = y.detach().cpu().numpy()
        pid = pid.numpy().astype(int)
        pid = pid[:, np.newaxis]
        csv_lines = np.hstack([pid, y])
        csv_writer.writerows(csv_lines)
        
if __name__ == "__main__":
  
# train()
#  train_all()
#  extractor_train()
  extractor_test()
 # test()