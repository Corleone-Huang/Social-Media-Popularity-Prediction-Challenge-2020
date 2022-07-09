import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument(
    'data_dir', help='The path of the csv file.', type=str)
parser.add_argument(
    'output_dir', help='The path to save results', type=str)
parser.add_argument(
    '--bins', help='Number of bins of histgram', type=int, default=None)
args = parser.parse_args()

print('Loading...')
df = pd.read_csv(args.data_dir) 
os.mkdir(args.output_dir)


for col in tqdm(df.columns):
    if col in ['Pid', 'Uid', 'pid', 'uid', 'PID', 'UID']:
        continue
    plt.figure()

    plt.hist(df[col], bins=args.bins)
    plt.savefig('{}/{}.png'.format(args.output_dir, col), dpi=64)
    plt.close()
