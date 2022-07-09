# Introduction

This project completed the extraction of DenseSIFT series.

## results

**feature extraction results exist under ``./extracted_features``**

DenseSIFT.py

``` python
uid | pid | img_dsift_feature_1 | ... | img_dsift_feature_6272
```

## usage

```sh run_dsift.sh```

```python
  -h, --help  show this help message and exit
  --save_dir SAVE_DIR  存放特征结果的文件夹,不是.csv的路径
  --imgpth_dir IMGPTH_DIR  存放图片路径.json文件，与train_tags中uid/pid顺序是一致的
  --size SIZE 图片统一的尺寸 
```

## reference

[dsift.](https://github.com/Yangqing/dsift-python)