import pickle
import numpy as np
from skimage.feature import hog
from tqdm import tqdm
import pandas as pd
import argparse
# 所有图像加到列表里面，列表是提前创建的


parser = argparse.ArgumentParser()
parser.add_argument(
    'data_dir', help='Data dir of train_img_list.pkl file', type=str)
parser.add_argument('output_dir', help='Dir to save hog features', type=str)
args = parser.parse_args()

print('Loading...')
with open(args.data_dir, 'rb') as f:
    imgs = pickle.load(f)

batch_size = 2000
all_size = len(imgs)
epoch = all_size//batch_size
res = all_size % batch_size
#
for step in range(epoch):
    imgs_batch = imgs[step*batch_size:(step+1)*batch_size]
    # 主要是这一步利用hog函数提取特征
    hog_features = [hog(img, pixels_per_cell=(14, 14),
                        cells_per_block=(2, 2)) for img in tqdm(imgs_batch)]
# 因为数据量比较大， 所以每隔2000个数据往文件里面写一次
    df = pd.DataFrame(np.array(hog_features).astype(np.float32))
    df.to_csv(args.output_dir,
              index=0, header=None, mode='a')

imgs_batch = imgs[(step+1)*batch_size:all_size]
hog_features = [hog(img, pixels_per_cell=(14, 14), cells_per_block=(2, 2))
                for img in tqdm(imgs_batch)]
df = pd.DataFrame(np.array(hog_features).astype(np.float32))
df.to_csv(args.output_dir, index=0, header=None, mode='a')

print('done!')
