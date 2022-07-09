# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from __future__ import unicode_literals, print_function, division
import time
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import json
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset
import pandas as pd
import numpy as np

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


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
parser.add_argument('-t','--timesteps',
                    type=int, help='The window size of spliting data.',
                    default=5)
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


# %%
def get_train_data(feature, label, window_size):
    gb = feature.groupby('uid')
    all_data, targets = [], []
    for name, grouped in gb:
        grouped = grouped.drop('uid', axis=1)
        n_pad = window_size-len(grouped) % window_size
        padding = np.zeros((n_pad, grouped.shape[1]))
        data = np.vstack((padding, grouped.values))
        # all_data.extend(data.tolist())
        all_data.append(data)

        target = label.loc[grouped.index].values
        label_padding = np.zeros((n_pad, 1))
        target = np.vstack((label_padding, target))
        targets.append(target)

    all_data = np.vstack(all_data)
    targets = np.vstack(targets)

    all_data = all_data.reshape(-1, window_size, grouped.shape[1])
    targets = targets.reshape(-1, window_size, 1)

    all_data = torch.FloatTensor(all_data)
    targets = torch.FloatTensor(targets)

    return all_data, targets


# %%
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


timesteps = args.timesteps
MAX_LENGTH = timesteps
X_train, Y_train = get_train_data(X_train, Y_train, timesteps)
X_valid, Y_valid = get_train_data(X_valid, Y_valid, timesteps)


# %%

train_set = torch.utils.data.TensorDataset(X_train, Y_train)
valid_set = torch.utils.data.TensorDataset(X_valid, Y_valid)
train_loader = DataLoader(train_set, batch_size=32,
                          shuffle=False, num_workers=4, drop_last=True)
valid_loader = DataLoader(valid_set, batch_size=32,
                          shuffle=False, num_workers=4, drop_last=True)


# %%
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # self.embedding = nn.Embedding(input_size, hidden_size)

        self.gru = nn.GRU(input_size, hidden_size)

    def forward(self, input, hidden):

        # input = torch.unsqueeze(1)
        output, hidden = self.gru(input, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 32, self.hidden_size, device=device)


# %%
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        # embedded = self.embedding(input)
        # embedded = self.dropout(embedded)
        # 不用embedding。
        embedded = input

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded, hidden), 2)), dim=2)
        # print(attn_weights.unsqueeze(0).shape)
        # print(encoder_outputs.unsqueeze(0).shape)
        attn_applied = torch.bmm(attn_weights.permute(1, 0, 2),
                                 encoder_outputs.permute(1, 0, 2))

        output = torch.cat((embedded, attn_applied.permute(1, 0, 2)), 2)
        output = self.attn_combine(output)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = self.out(F.relu(output[0]))
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


# %%
X_train.shape


# %%
# Training
teacher_forcing_ratio = 0.5


def train(input_tensor, target, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=timesteps):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target.size(0)

    encoder_outputs = torch.zeros(max_length, 32, 512, device=device)

    loss = 0

    input_tensor = input_tensor.permute(1, 0, 2)
    target = target.permute(1, 0, 2)
    input_tensor = input_tensor.to(device)
    target = target.to(device)
    encoder_hidden = encoder_hidden.to(device)
    
    for ei in range(input_length):
        # for ei in range(MAX_LENGTH):
        # print(input_tensor.shape)
        # print(encoder_hidden.shape)
        # single step.
    
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei].unsqueeze(0), encoder_hidden)
        # print(encoder_output.shape)
        encoder_outputs[ei] = encoder_output[0]

    # decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_input = torch.zeros_like(1, 32, 512)

    decoder_hidden = encoder_hidden
    print(decoder_hidden.shape)


    # use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    use_teacher_forcing = False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target[di])
            decoder_input = target[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            # topv, topi = decoder_output.topk(1)
            # decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target[di])
            decoder_input = decoder_output[di]
            # if decoder_input.item() == EOS_token:
            #     break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


# %%
def trainIters(train_loader, encoder, decoder, n_iters=100, print_every=10, plot_every=5, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    # n_iters = seq.shape[0]

# TODO:
    criterion = nn.MSELoss(reduction='sum')

    for ei, (input_batch, target_batch) in tqdm(enumerate(train_loader)):
        # input_tensor = input_tensor.to(device)

        # target = target.to(device)

        loss = train(input_batch, target_batch, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if (iter+1) % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if (iter+1) % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)


# %%
plt.switch_backend('agg')


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.savefig('./plot.png')


# %%
def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


# %%
def evaluate(encoder, decoder, input_tensor, max_length=timesteps):
    with torch.no_grad():
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(
            max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        # decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
        decoder_input = torch.zeros_like(1, 32)

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            # topv, topi = decoder_output.data.topk(1)
            # if topi.item() == EOS_token:
            #     decoded_words.append('<EOS>')
            #     break
            # else:
            decoded_words.append(decoder_output)

            decoder_input = decoder_output

        return decoded_words, decoder_attentions[:di + 1]


# %%


hidden_size = 512
input_size = X_train.shape[2]
output_size = 1
encoder1 = EncoderRNN(input_size, hidden_size).to(device)
# encoder1 = nn.DataParallel(encoder1)
attn_decoder1 = AttnDecoderRNN(
    hidden_size, output_size, dropout_p=0.1).to(device)
# attn_decoder1 = nn.DataParallel(attn_decoder1)

trainIters(train_loader, encoder1, attn_decoder1)


# %%
