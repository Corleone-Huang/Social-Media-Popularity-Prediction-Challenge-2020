# Introduction

Extract tags and titles features by pre-trained FastText model.

## Usage

```shell
usage: fasttext.py [-h] [--tags_filepath TAGS_FILEPATH]
                   [--model_filepath MODEL_FILEPATH]
                   [--title_feature_filepath TITLE_FEATURE_FILEPATH]
                   [--tags_feature_filepath TAGS_FEATURE_FILEPATH]

Extract tags and titles feature by FastText model

optional arguments:
  -h, --help            show this help message and exit
  --tags_filepath TAGS_FILEPATH
                        tags_filepath
  --model_filepath MODEL_FILEPATH
                        FastText model filepath
  --title_feature_filepath TITLE_FEATURE_FILEPATH
                        titles feature filepath extracted by FastText
  --tags_feature_filepath TAGS_FEATURE_FILEPATH
                        tags feature filepath extracted by FastText
```

simply run

```shell
bash Fasttext.sh
```

## Saved feature

The features are in this format ```pid,uid,feature1,feature2,...,feature300```

## Reference

[Pre-trained FastText model](https://fasttext.cc/docs/en/english-vectors.html)
