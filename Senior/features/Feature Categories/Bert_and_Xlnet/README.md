# Introduction
This project is used to extract the characteristics of Title and Alltags data in ACM MM SMP Challenges Data.This project takes the mean value of each token feature generated in the end as the feature of alltags and title corresponding to each PID.

# Result
This project will generate feature files in CSV format for Title and Alltags in the addresses pointed to by the '--save_title_feature' and '--save_alltags_feature' parameters.In this project, the eigenvectors of each token in the last layer are averaged to obtain the eigenvectors of the final sentence.

# Usage
Simply run 

```python
conda env create -f environment.yml
conda activate env
sh run_train_data.sh
```

Specific parameters for reference:
```python
usage: bert_and_xlnet_feature.py [-h] [--data_dir DATA_DIR]
                                 [--save_title_feature TITLE_FEATURE]
                                 [--save_alltags_feature ALLTAGS_FEATURE]
                                 [--ngpu NGPU]
                                 [--bert_base_uncased BERT_BASE_UNCASED]
                                 [--bert_large_uncased BERT_LARGE_UNCASED]
                                 [--xlnet_base_cased XLNET_BASE_CASED]
                                 [--xlnet_large_cased XLNET_LARGE_CASED]

optional arguments:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR   存放json数据的路径
  --save_title_feature TITLE_FEATURE   存放title feature的文件夹路径，/结尾，不是具体csv文件！
  --save_alltags_feature ALLTAGS_FEATURE   存放alltags feature的文件夹路径，/结尾，不是具体csv文件！
  --ngpu NGPU           使用的GPU数目(取1可能会出错，建议>1)
  --bert_base_uncased BERT_BASE_UNCASED   是否提取bert-base-uncased特征
  --bert_large_uncased BERT_LARGE_UNCASED   是否提取bert-large-uncased特征
  --xlnet_base_cased XLNET_BASE_CASED   是否提取xlnet-base-cased特征
  --xlnet_large_cased XLNET_LARGE_CASED   是否提取xlnet-large-cased特征
```


# Reference
[Transformers:State-of-the-art Natural Language Processing for TensorFlow 2.0 and PyTorch.](https://github.com/huggingface/transformers#Quick-tour-TF-20-training-and-PyTorch-interoperability)