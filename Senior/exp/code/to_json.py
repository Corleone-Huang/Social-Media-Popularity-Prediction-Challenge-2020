import csv
import json
import argparse
import pandas as pd
import numpy as np

count1 = 0
count2 = 0
default_add_csv = "results/lgbm_add.csv"
default_noadd_csv = "results/noadd.csv"
default_number = 0.95
default_pid116338_path = "results/test_pid_116338.csv"
default_pid64243_path = "results/test_pid_64243.csv"
default_df1_path = "results/v14_alltrain20000_add.csv"
default_df2_path = "results/cat_knn_fill_40600e_cpu.csv"
default_csv_path = "results/submit_results.csv"



def parse_args():
    parser = argparse.ArgumentParser(description='------LightGBM-------')
    parser.add_argument('--add_csv', default=default_add_csv, dest='add_csv',
                        type=str, help='the path of add resluts csv' )
    parser.add_argument('--noadd_csv', default=default_noadd_csv, dest='noadd_csv',
                        type=str, help='the path of noadd resluts csv')
    parser.add_argument('--number', default=default_number, dest='number',
                        type=float, help='the process number')
    args = parser.parse_args(args=[])
    return args
args = parse_args()
pid116338 = pd.read_csv(default_pid116338_path)
pid64243 = pd.read_csv(default_pid64243_path)
df1 = pd.read_csv(default_df1_path)
del df1["Unnamed: 0"]
df1 = pd.merge(df1, pid64243)
del df1['uid']



df2 = pd.read_csv(default_df2_path)
df2 = pd.merge(df2, pid116338)
label_raw = df2['label'].tolist()
label = [i-args.number for i in label_raw]
del df2['uid']

# df1 = pd.read_csv(args.add_csv)
# df2 = pd.read_csv(args.noadd_csv)








with open(default_csv_path, "w") as writer:
    reader = csv.reader(writer)
    result = []
    for item in reader:
        if reader.line_num == 1:
            continue
        result.append({'post_id': 'post'+item[0], 'popularity_score': round(float(item[1]), 4)})
        count1 = count1+1
output = {
    "version": "VERSION 1.0",
    "result": result,
    "external_data": {
        "used": "true",
        "details": "glove, bert, 7-d user, count"
    }
}
with open(default_csv_path, 'w', encoding='utf-8') as writer:
    json.dump(output, writer, ensure_ascii=False, indent=4)
print("success")
print(count1, count2, count1+count2)
