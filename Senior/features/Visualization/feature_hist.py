import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import os
import argparse
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('source', help='The path of the csv file.', type=str)
parser.add_argument('target', help='The path to save results', type=str)
args = parser.parse_args()

print('Loading...')
df = pd.read_csv(args.source)
os.mkdir(args.target)


for col in tqdm(df.columns):
    if col in ['Pid','Uid','pid','uid','PID', 'UID']:
        continue
    plt.figure()
    
    plt.hist(df[col])
    plt.savefig('{}/{}.png'.format(args.target,col), dpi=64)
    plt.close()

