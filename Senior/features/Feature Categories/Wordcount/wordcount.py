# 本代码使用wordcount提取SMP data中的Alltags, Titles数据的特征
# 结果保存在../results/wordcount文件夹中
# wordcount：输出alltags/title_length_305613.csv，保存每一项的词数(包括特殊字符项和数字项在内！)每一行都是[uid,pid,feature]
# Character Count：输出alltags/title_Charcount_305613.csv，保存每一项的字符数, 每一行都是list[uid,pid,feature]
# average count：输出alltags/title_avercount_1.csv，保存平均词数,存储类型是['average']
# author： Zhang Jingjing  2020-3-31
import json
import re
import os
import csv
import argparse

train_data_path = '/home/zjj/smpdata/train_tags.json'
save_feature_dir = '../results'
task_item = 'wordcount'
tag_category = 'Alltags'
def parse_args():
    parser = argparse.ArgumentParser(description='------wordcount系列任务-------')
    parser.add_argument('--data_dir', default= train_data_path, dest='data_dir',
                        type=str, help='json数据的路径')
    parser.add_argument('--save_dir', default= save_feature_dir, dest='save_dir',
                        help='存放wordcount特征的文件夹', type=str)
    parser.add_argument('--task', default= task_item, type=str, dest='task',
                        help='wordcount、averwordcount or charcount')
    parser.add_argument('--category', default= tag_category, type=str, dest='category',
                        help='Title or Alltags')
    args = parser.parse_args()
    return args


# 统计某条title/alltags含有的单词数，包括特殊字符和纯数字
def wocount(str):
    # 以空格分开字符串
    strlist = str.lower().split(' ')
    # # 跳过特殊字符统计str的单词个数
    # count = 0
    # for str in strl_ist:
    #     test_str = re.search(r"\W", str)
    # if test_str != None or str.isdigit():
    #     continue
    # count += 1
    return len(strlist)


# 统计某条title/alltags里面所有的字符数，包括特殊字符/空格等
def charcount(str):
    return len(str)


if __name__ == "__main__":
    args = parse_args()

    # 加载和存储路径
    data_dir = args.data_dir
    save_dir = args.save_dir
    task = args.task
    category = args.category
    with open(data_dir, 'r') as load_f:
        # a list of #305613
        load_tags = json.load(load_f)
        number = len(load_tags)
    load_f.close()
    save_wordcount = os.path.join(save_dir, task + '_'+category.lower() + '_'+ str(number) + '.csv')

    # 特征提取部分
    # 统计单词数
    if task == 'wordcount':
        # 打开准备写入的csv文件
        with open(save_wordcount, "w") as csvfile:
            task_writer = csv.writer(csvfile)
            # 先写入columns_name
            task_writer.writerow(["uid", "pid", category.lower() + '_' + task])
        for lineitem in load_tags:
            item_count = []
            # lineitem is a dict of 5: Alltags,Pid,Uid,Title,Meidatype
            uid = lineitem['Uid']
            pid = lineitem['Pid']
            item = lineitem[category]
            # 统计字符长度
            item_length = wocount(item)
            # print(wordcount_title)
            item_count.append([uid, pid, item_length])
            with open(save_wordcount, "a+") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(item_count)
    # 统计平均单词数
    elif task == 'averwordcount':
        averwc = []
        allcount = 0
        for lineitem in load_tags:
            item = lineitem[category]
            allcount += wocount(item)
        averwc.append(str(allcount / number))
        # 打开准备写入的csv文件
        with open(save_wordcount, "w") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(averwc)
    # # 统计字符数
    else:
        with open(save_wordcount, "w") as csvfile:
            task_writer = csv.writer(csvfile)
            # 先写入columns_name
            task_writer.writerow(["uid", "pid", category.lower() + '_' + task])
        char_length = 0
        for lineitem in load_tags:
            char_count = []
            # lineitem is a dict of 5: Alltags,Pid,Uid,Title,Meidatype
            uid = lineitem['Uid']
            pid = lineitem['Pid']
            item = lineitem[category]
            # 统计字符长度
            char_length = charcount(item)
            # print(wordcount_title)
            char_count.append([uid, pid, char_length])
            with open(save_wordcount, "a+") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(char_count)
