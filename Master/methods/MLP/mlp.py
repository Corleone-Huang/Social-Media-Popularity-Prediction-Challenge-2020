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

train_label_path = "../../features/splited_label/train_label.csv"
valid_label_path = "../../features/splited_label/validate_label.csv"
train_feature_pklpath = "../../features/extracted_features/train+label_with_user_result_305613.pkl"

parser.add_argument("--all_feature", type=str,
                    default=all_feature_path)


parser.add_argument("--train_label_filepath", type=str,
                    default=train_label_path)
parser.add_argument("--validate_label_filepath", type=str,
                    default=validate_label_path)
parser.add_argument("--output_feature", action='store_true')
parser.add_argument("--input_size", type=int, default=100)
parser.add_argument("--hidden_size", type=int, default=300)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--learning_rate", type=float, default=0.0007)
args = parser.parse_args()

t_start = time.time()
print('Loading train label...')
df_train = pd.read_csv(args.train_label_filepath, index_col='pid')
print('Loading validate label...')
df_validate = pd.read_csv(args.validate_label_filepath, index_col='pid')

# print('Loading feature...')
# df_all = pd.read_pickle(args.all_feature)
# df_all.set_index(['pid'], inplace=True)
# df_all.drop(['uid', 'label'], axis=1, inplace=True)
# time_elapsed = time.time() - t_start
# print('Load feature in {}sec'.format(time_elapsed))

# # cols = df_all.select_dtypes('category').columns
# # df_all[cols] = df_all[cols].astype(int)
# t4 = time.time()
# feature_train = df_all.loc[df_train.index]
# feature_train.to_pickle('feature_train.pkl')
# feature_train = feature_train.values.astype(float)

# feature_validate = df_all.loc[df_validate.index]
# feature_validate.to_pickle('feature_validate.pkl')
# feature_validate = feature_validate.values.astype(float)
# print('Locate index in {}sec'.format(time.time()-t4))
# print('data ok.')
print('loading train and validata set...')
feature_train = pd.read_pickle('feature_train.pkl')
feature_validate = pd.read_pickle('feature_validate.pkl')
feature_train = feature_train.values.astype(float)
feature_validate = feature_validate.values.astype(float)
print('Load data in {}sec'.format(time.time()-t_start))


# del df_all

t2 = time.time()
label_train = df_train["label"].values
label_validate = df_validate["label"].values
print('re-index data in {}sec'.format(time.time()-t2))

del df_train
del df_validate
gc.collect()
t1 = time.time()
label_train = torch.FloatTensor(label_train).unsqueeze(1)
label_validate = torch.FloatTensor(label_validate).unsqueeze(1)
feature_train = torch.FloatTensor(feature_train)
feature_validate = torch.FloatTensor(feature_validate)
print('Transfer data into tensor in {}sec'.format(time.time()-t1))


train = TensorDataset(feature_train, label_train)
validate = TensorDataset(feature_validate, label_validate)

# Dataloader
print('load data')
t0 = time.time()
train_loader = DataLoader(
    train, batch_size=args.batch_size, shuffle=False, num_workers=8)
valid_loader = DataLoader(
    validate, batch_size=args.batch_size, shuffle=False, num_workers=8)
print('load data in {}sec'.format(time.time()-t0))


class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.input_size = 5604
        self.output_size = 1
        self.dropout1 = torch.nn.Dropout(0.4)
        self.dropout2 = torch.nn.Dropout(0.2)

        self.linear = nn.Linear(5604, 768)

        self.output1 = nn.Linear(768, 384)
        self.output2 = nn.Linear(384, 1)

    def forward(self, feature, output_feature=False):
        feature = self.dropout1(self.linear(feature))

        feature = F.relu(feature)
        y0 = self.output1(feature)
        y1 = F.relu(self.dropout2(y0))

        y2 = self.output2(y1)

        if output_feature:
            return y0
        else:
            return y2


model = MLP()
print(torch.cuda.is_available())
if torch.cuda.is_available():

    model = model.cuda()

optimizer = torch.optim.Adam(
    model.parameters(), lr=args.learning_rate, weight_decay=1e-5)


loss_fn1 = torch.nn.MSELoss()


def corr(x, y):
    vx = x-torch.mean(x)
    vy = y-torch.mean(y)
    corr = vx*vy*torch.rsqrt(torch.sum(vx**2))*torch.rsqrt(torch.sum(vy**2))
    return corr


avg_losses_f = []
avg_val_losses_f = []

for epoch in range(args.epochs):
    print('Training epoch{}...'.format(epoch))
    model.train()
    start_time = time.time()
    model.train()
    avg_loss = 0.
    for i, (feature, label) in enumerate(train_loader):
        if torch.cuda.is_available():
            feature = feature.cuda()
            label = label.cuda()
            # postdate_train = postdate_train.cuda()
        y_pred = model(feature)

        loss = loss_fn1(y_pred, label)
        optimizer.zero_grad()       # clear gradients for next train
        loss.backward()             # -> accumulates the gradient (by addition) for each parameter
        optimizer.step()            # -> update weights and biases
        avg_loss += loss.item() / len(train_loader)
        # avg_auc += round(roc_auc_score(y_batch.cpu(),y_pred.detach().cpu()),4) / len(train_loader)

    model.eval()
    avg_val_loss = 0.
    # avg_val_auc = 0.
    y_preds = []
    y_labels = []
    for i, (feature, label) in enumerate(valid_loader):
        if torch.cuda.is_available():
            feature = feature.cuda()
            label = label.cuda()
        y_pred = model(feature).detach()

        # avg_val_auc += round(roc_auc_score(y_batch.cpu(),sigmoid(y_pred.cpu().numpy())[:, 0]),4) / len(valid_loader)
        avg_val_loss += loss_fn1(y_pred, label).item() / len(valid_loader)
        y_pred = y_pred.squeeze(1).detach().cpu().numpy().tolist()
        label = label.squeeze(1).detach().cpu().numpy().tolist()
        y_preds.extend(y_pred)
        y_labels.extend(label)
    elapsed_time = time.time() - start_time
    MAE = mean_absolute_error(y_labels, y_preds)
    correction, _ = spearmanr(y_preds, y_labels)
    print('Epoch {}/{} \t loss={:.4f} \t MAE={:.4f} \t SRC:{:.4f} \t time={:.2f}s'.format(
        epoch + 1, args.epochs, avg_loss, MAE, correction, elapsed_time))

    avg_losses_f.append(avg_loss)
    avg_val_losses_f.append(avg_val_loss)

plt.figure()
plt.plot(avg_losses_f, label='Trainning loss')
plt.plot(avg_val_losses_f, label='Validate loss')
plt.legend()
plt.savefig('mlp_trainning_curve.png')
# save model
torch.save(model.state_dict(), 'mlp_model.pth')
