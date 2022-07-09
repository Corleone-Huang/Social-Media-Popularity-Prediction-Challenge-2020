# -*- coding: utf-8 -*-
# @Date    : 2020-04-16
# @Time    : 15:47
# @Author  : Zhang Jingjing
# @FileName: Hu.py
# @Software: PyCharm
import cv2
import os
import pandas as pd
import csv
import glob
from scipy import misc
import matplotlib.pyplot as plt
import argparse
import json
from tqdm import tqdm
save_feature_dir= '../../extracted_features'
img_path_json= '/home/zjj/smp/data/train_img_path.json'
def parse_args():
    parser = argparse.ArgumentParser(description='------Hu矩特征提取-------')
    parser.add_argument('--save_dir', default= save_feature_dir, dest='save_dir',
                        help='存放特征结果的文件夹,不是.csv的路径', type=str)
    parser.add_argument('--imgpth_dir',default= img_path_json, dest='imgpth_dir',
                        help='存放图片路径.json文件，与train_tags中uid/pid顺序是一致的', type=str)
    args = parser.parse_args()
    return args

# 对每张图片二值处理
def img_pre(img_path):
    imgs = {}
    print("image processing.......")
    for img_item in tqdm(img_path):
        uid = img_item.split("/")[-2]
        pid = img_item.split("/")[-1].split('.')[-2]
        # 读取灰度图像
        gray = cv2.imread(img_item, cv2.IMREAD_GRAYSCALE)
        # 二值化处理
        thresh = 128
        _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
        imgs[str(uid + '_' + pid)] = binary
    nums = len(imgs)
    return imgs, nums


# 计算Hu矩并存入.csv文件
def cal_hu(imgs):
    print("calculate Hu Moments.......")
    for key, value in tqdm(imgs.items()):
        moment = cv2.moments(value)
        huMonents = cv2.HuMoments(moment)
        huMonents = huMonents.reshape(-1).tolist()
        uid = key.split('_')[0]
        pid = key.split('_')[1]
        imgs_Hu = [uid, pid]
        for index in range(len(huMonents)):
            imgs_Hu.append(huMonents[index])
        writer.writerow(imgs_Hu)


if __name__ == "__main__":
    args = parse_args()
    # 加载和存储路径
    save_dir = args.save_dir
    imgpth_dir = args.imgpth_dir
    # 获取所有img的路径
    pathdile = open(imgpth_dir, 'r', encoding='utf8')
    img_path = json.load(pathdile)
    imgs, nums = img_pre(img_path)
    save_hu = os.path.join(save_dir, 'hu_img_'+str(nums) + '.csv')
    with open(save_hu, "w+") as csvfile:
        writer = csv.writer(csvfile)
        # 先写入columns_name
        header = ["uid", "pid"]
        for i in range(7):
            header.append("img_hu_feature" + str(i + 1))
        writer.writerow(header)
        # 写入多行用writerows
        cal_hu(imgs)
