# Introduction

This project completed the extraction of Hu, DenseSIFTh series.

## results

**feature extraction results exist under ``./extracted_features``**

``Hu.py`` outputs 7-dimensional feature vectors for each image, a total of 305,613 images, and generates feature CSV file column names as follows:

```python
uid | pid | hu_img_1 | ... | hu_img_7
```

## usage

```sh run_hu.sh```

```python
  -h, --help  show this help message and exit
  --save_dir SAVE_DIR  存放特征结果的文件夹,不是.csv的路径
  --imgpth_dir IMGPTH_DIR  存放图片路径.json文件，与train_tags中uid/pid顺序是一致的
```
