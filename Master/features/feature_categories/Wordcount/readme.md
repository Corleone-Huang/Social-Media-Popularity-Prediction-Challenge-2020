# Introduction

This project completed the extraction Wordcount series.

## results

**feature extraction results exist under ``./extracted_features``**

``wordcount.py`` includes wordCount, Charcount and AverwordCount, respectively count the number of words, characters and average words of title/ Alltags.
The worcount task generates feature CSV file column names as follows:

```python
uid | pid | title_worcount
uid | pid | alltags_worcount
```

## usage

```sh run_wc.sh```

```python
  -h, --help           show this help message and exit
  --data_dir DATA_DIR  json数据的路径
  --save_dir SAVE_DIR  存放wordcount特征的文件夹
  --task TASK          wordcount、averwordcount or charcount
  --category CATEGORY  Title or Alltags
```
