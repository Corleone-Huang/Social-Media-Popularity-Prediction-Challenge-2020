import pandas as pd
import argparse
import json
import time

input_feature = "../../extracted_features/Glove_tags_486194.csv"
output_feature = "../../extracted_features/Glove_tags_486194_average_5.csv"
n_windows = 5 

t1 = time.time()
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_dir',
                    help='The path of input data.',
                    default=input_feature, type=str)
parser.add_argument('-o', '--output_dir',
                    help='The path of output data.',
                    default=output_feature,
                    type=str)
parser.add_argument('-n', '--n_window',default=n_windows,
                    help='The number of feature to conduct moving average.', type=int, choices=[3, 5])
args = parser.parse_args(args=[])

with open('../../../data/data_source/train/train_temporalspatial.json') as f:
    timespetio1 = pd.read_json(f)
with open("../../../data/data_source/test/test_temporalspatial.json") as fi:
    timespetio2 = pd.read_json(fi)


timespetio=pd.concat([timespetio1,timespetio2],keys='Pid')
print(timespetio)
#postdate = [int(item['Postdate']) for item in timespetio]
#pids = [int(item['Pid']) for item in timespetio]
postdate_df = pd.DataFrame({'pid':timespetio['Pid'],'uid': timespetio['Uid'], 'postdate': timespetio['Postdate']})
print(postdate_df)
#postdate_df = pd.DataFrame({'pid': pids, 'postdate': postdate})
postdate_df.sort_values(by='postdate', inplace=True)
index_pid = postdate_df['pid'].values
# 按照时序对特征进行排�?
df = pd.read_csv(args.data_dir, index_col='pid')
print(df)
df = df.loc[index_pid]
uids = df['uid']


def moving_average(y):

    if len(y) <= 2:
        return y
    else:
        return (y+y.shift(fill_value=0)+y.shift(2, fill_value=0))/3


def moving_average2(y):

    if len(y) <= 4:
        return y
    else:
        return (y+y.shift(fill_value=0)+y.shift(2, fill_value=0)+y.shift(3, fill_value=0)+y.shift(4, fill_value=0))/5


gb = df.groupby('uid')
if args.n_window == 3:
    df = gb.apply(moving_average)
elif args.n_window == 5:
    df = gb.apply(moving_average2)

df=df.reset_index()
df1=pd.DataFrame({'pid':timespetio['Pid'],'uid': timespetio['Uid'], 'postdate': timespetio['Postdate']})
df=pd.merge(df1[["pid","uid"]],df)
# df.insert(0, 'uid', uids)
df
print('存成文件')
print(df)
df.to_csv(args.output_dir,index=False)
df.iloc[:305613,:].to_csv("../../extracted_features/Glove_tags_305613_average_5.csv",index=False)
df.iloc[305613:,:].to_csv("../../extracted_features/Glove_tags_180581_average_5.csv",index=False)
print('Done!')
t2 = time.time()
print('Running time: {}s'.format(t2-t1))
