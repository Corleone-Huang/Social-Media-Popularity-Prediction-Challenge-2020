import cv2
import numpy as np
import json
from tqdm import tqdm
import pandas as pd
import argparse



parser = argparse.ArgumentParser()
parser.add_argument(
    'data_dir', help='The path of the image path json file.', type=str)
parser.add_argument(
    'output_dir', help='The path to save results', type=str)
parser.add_argument(
    '--keypoints', help='Number of keypoints of surf', type=int)
args = parser.parse_args()


with open(args.data_dir, 'r') as f:
    img_paths = json.load(f)

surf = cv2.xfeatures2d.SURF_create(1000)
keys_num = args.keypoints
surf_features = []

for idx, img_path in tqdm(enumerate(img_paths)):
    img = cv2.imread(img_path, 0)
    # 利用这个函数提取特征，kps是返回的关键点，dst是返回的特征向量，维度是kps*64
    kps, dst = surf.detectAndCompute(img, mask=None)
    # 因为关键点的个数每张图检测的数量不一样，所以要控制数量。一张图如果特征点的数量多余100，那就把后面的舍去；少于100，补零
    if dst is None:
        surf_features.append(np.zeros(keys_num*64))
    elif len(kps) >= keys_num:
        surf_features.append(dst[:keys_num, :].flatten())
    else:
        padding = np.zeros((keys_num-len(kps), dst.shape[1]), dtype=np.float)
        feature_tmp = np.vstack([dst, padding])
        surf_features.append(feature_tmp.flatten())
        # 这里也是分别对文件写入，每次写入2000个
    if (idx+1) % 2000 == 0:
        df = pd.DataFrame(np.array(surf_features).astype(np.float32))
        df.to_csv(args.output_dir,
                  mode='a', header=None, index=0)
        # print('Add 5000*{} data!'.format((idx+1)//5000))
        surf_features = []

df = pd.DataFrame(np.array(surf_features).astype(np.float32))
df.to_csv(args.output_dir, mode='a', header=None, index=0)
print('done')
