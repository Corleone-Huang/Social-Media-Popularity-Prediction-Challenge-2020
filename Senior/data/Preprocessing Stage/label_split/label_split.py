# ------------------------------------------------------------------------------
# -*- coding: utf-8 -*-
# @Date    : 2020-06-19
# @Time    : 21:04
# @Author  : Zhang Jingjing
# @FileName: label_split.py
# @Software: PyCharm
# ------------------------------------------------------------------------------
import pandas as pd
import json

default_label1_path = "/home/zjj/SMP2020_test/label/train_label_postdate.csv"
default_label2_path = "/home/zjj/SMP2020_test/label/validate_label_postdate.csv"
default_json = '/home/zjj/challenges/SMP/data/train_temporalspatial.json'
default_savepath1 = "/home/zjj/SMP2020_test/label/train_label_postdate_95.csv"
default_savepath2 = "/home/zjj/SMP2020_test/label/validate_label_postdate_05.csv"


if __name__ == "__main__":
    label1 = pd.read_csv(default_label1_path)
    label2 = pd.read_csv(default_label1_path)
    label = pd.concat([label1, label2], axis=0)
    with open(default_json) as f:
        timespetio = json.load(f)
    postdate = [int(item['Postdate']) for item in timespetio]
    pids = [int(item['Pid']) for item in timespetio]
    postdate_df = pd.DataFrame({'pid': pids, 'postdate': postdate})
    # 按postdate升序排序
    label_postdate_df = pd.merge(postdate_df, label, on='pid')
    label_postdate_df.sort_values(by='postdate', inplace=True)
    pid = label_postdate_df['pid'].values.tolist()
    label = label_postdate_df['label'].values.tolist()
    train_pid = pid[:290332]
    val_pid = pid[290332:]
    train_label = label[:290332]
    val_label = label[290332:]
    train_label = pd.DataFrame({"pid":train_pid, "label":train_label})
    val_label = pd.DataFrame({"pid":val_pid, "label":val_label})
    train_label.to_csv(default_savepath1, index=None)
    val_label.to_csv(default_savepath2, index=None)
