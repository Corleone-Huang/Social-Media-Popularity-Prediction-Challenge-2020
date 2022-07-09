from sklearn.preprocessing import MinMaxScaler, StandardScaler
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
import matplotlib.pyplot as plt

seed = 2020
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
parser = argparse.ArgumentParser()

train_feature_path = "../../features/extracted_features/train+label_no_user_305613.pkl"
test_feature_path = "../../features/extracted_features/test+label_no_user_180581.pkl"
model_path = "../../features/extracted_features/mlp.pth"

parser.add_argument("--train_feature", type=str,
                    default=train_feature_path)

parser.add_argument("--test_feature", type=str,
                    default=test_feature_path)
parser.add_argument("--model_path", type=str,
                    default=model_path)
parser.add_argument("--batch_size", type=int, default=256)
args = parser.parse_args()

t0 = time.time()
print('Loading train feature...')
train_data = pd.read_pickle(args.train_feature)
train_pid = train_data.pid.values
train_uid = train_data.uid.values
train_data.drop(['label', 'uid', 'pid'], axis=1, inplace=True)
train_data = train_data.values.astype(float)
train_data = torch.FloatTensor(train_data)
t1 = time.time()
print('Load train data in {}sec'.format(t1-t0))

print('Loading test feature...')
test_data = pd.read_pickle(args.test_feature)
test_pid = test_data.pid.values
test_uid = test_data.uid.values
test_data.drop(['uid', 'pid'], axis=1, inplace=True)
test_data = test_data.values.astype(float)
test_data = torch.FloatTensor(test_data)
t2 = time.time()
print('Load test feature in {}sec'.format(t2-t1))

# ###test
# train_data = np.random.randn(args.batch_size*2, 5604)
# train_data = torch.FloatTensor(train_data)
# train_pid = np.arange(args.batch_size*2)
# train_uid = train_pid

# test_data = np.random.randn(args.batch_size*2, 5604)
# test_data = torch.FloatTensor(test_data)
# test_pid = np.arange(args.batch_size*2)
# test_uid = test_pid


train_loader = DataLoader(
    train_data, batch_size=args.batch_size, shuffle=False, num_workers=8)
test_loader = DataLoader(
    test_data, batch_size=args.batch_size, shuffle=False, num_workers=8)


class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.input_size = 5604
        self.output_size = 1
        self.dropout = torch.nn.Dropout(0.2)

        self.linear = nn.Linear(5604, 1024)

        self.output1 = nn.Linear(1024, 384)
        self.output2 = nn.Linear(384, 1)

    def forward(self, feature, output_feature=False):

        feature = F.relu(self.linear(feature))
        y0 = self.output1(feature)
        y1 = F.relu(self.dropout(y0))

        y2 = self.output2(y1)

        if output_feature:
            return y0
        else:
            return y2


model = MLP()
model.load_state_dict(torch.load(args.model_path))
model.eval()


print(torch.cuda.is_available())
if torch.cuda.is_available():

    model = model.cuda()

train_feats = []
test_feats = []
for ind, train_batch in tqdm(enumerate(train_loader)):
    mlp_feat = model(train_batch.cuda(), output_feature=True).cpu().detach().numpy()
    train_feats.append(mlp_feat)

for ind, test_batch in tqdm(enumerate(test_loader)):
    mlp_feat = model(test_batch.cuda(), output_feature=True).cpu().detach().numpy()
    # assert mlp_feat.shape == (args.batch_size, 384)
    test_feats.append(mlp_feat)

train_feats = np.vstack(train_feats)
test_feats = np.vstack(test_feats)

print('Write train feature...')
train_df = pd.DataFrame(train_feats)
train_df.insert(0, 'pid', train_pid)
train_df.insert(1,'uid',train_uid)
train_df.to_csv('train_mlp_305613.csv', index=0)

print('Writing test feature...')
train_df = pd.DataFrame(test_feats)
train_df.insert(0, 'pid', test_pid)
train_df.insert(1,'uid', test_uid)
train_df.to_csv('test_mlp_180581.csv', index=0)
