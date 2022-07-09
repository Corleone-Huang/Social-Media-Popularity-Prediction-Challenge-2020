# Introduction
This project completed the extraction of three features, including Hu, DenseSIFTh and Wordcount series.

# results
**All feature extraction results exist under./results directory**
1. Hu.py outputs 7-dimensional feature vectors for each image, a total of 305,613 images, and generates feature CSV file column names as follows:

```python
uid | pid | hu_img_1 | ... | hu_img_7
```
2. DenseSIFT.py

```python
uid | pid | img_dsift_feature_1 | ... | img_dsift_feature_6272
```
3. wordcount.py includes wordCount, Charcount and AverwordCount, respectively count the number of words, characters and average words of title/ Alltags.
The worcount task generates feature CSV file column names as follows:

```python
uid | pid | title_worcount
uid | pid | alltags_worcount
```

# usage
1. ```sh run_hu.sh```

```python
  -h, --help  show this help message and exit
  --save_dir SAVE_DIR  存放特征结果的文件夹,不是.csv的路径
  --imgpth_dir IMGPTH_DIR  存放图片路径.json文件，与train_tags中uid/pid顺序是一致的
```
2. ```sh run_dsift.sh```

```python
  -h, --help  show this help message and exit
  --save_dir SAVE_DIR  存放特征结果的文件夹,不是.csv的路径
  --imgpth_dir IMGPTH_DIR  存放图片路径.json文件，与train_tags中uid/pid顺序是一致的
  --size SIZE 图片统一的尺寸 
```
3. ```sh run_wc.sh```

```python
  -h, --help           show this help message and exit
  --data_dir DATA_DIR  json数据的路径
  --save_dir SAVE_DIR  存放wordcount特征的文件夹
  --task TASK          wordcount、averwordcount or charcount
  --category CATEGORY  Title or Alltags
```

# reference
[dsift.](https://github.com/Yangqing/dsift-python)