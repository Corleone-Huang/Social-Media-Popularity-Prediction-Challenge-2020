# Introduction

Extract tags and titles features by one-hot.

## Usage

### for ``onehot_category.py``

```shell
usage: onehot_category.py [-h]
                          [--train_dataset_filepath TRAIN_DATASET_FILEPATH]
                          [--test_dataset_filepath TEST_DATASET_FILEPATH]
                          [--train_feature_filepath TRAIN_FEATURE_FILEPATH]
                          [--test_feature_filepath TEST_FEATURE_FILEPATH]

One-Hot Category and Subcategory

optional arguments:
  -h, --help            show this help message and exit
  --train_dataset_filepath TRAIN_DATASET_FILEPATH
                        train_category.json filepath
  --test_dataset_filepath TEST_DATASET_FILEPATH
                        test_category.json filepath
  --train_feature_filepath TRAIN_FEATURE_FILEPATH
                        train category&subcategory feature filepath
  --test_feature_filepath TEST_FEATURE_FILEPATH
                        test category&subcategory feature filepath
```

### for ``onehot_tags.py``

```shell
optional arguments:
  -h, --help     show this help message and exit
  --data_path    The path of data.  
  --save_tags    The path to save title feature.
  --save_title   The path to save tags feature.
```

### simply run

```shell
bash onehot.sh
```

## Saved feature

The features are in this format ```pid,uid,feature1,feature2,...```
