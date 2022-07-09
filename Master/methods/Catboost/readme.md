# Introduction

Use Catboost to regress popularity.

## Usage

example:

```shell
usage: resuly_with_user.py [-h] [--model {xgboost,lightgbm,catboost}]
                         [--feature [{Fasttext_tag,Fasttext_title,Fasttext_ave3,Bert_tag,Bert_title,Bert_ave3,Bert_tsvd512_ave3,tfidf_ave3,glove,glove_ave3,lsa_ave3,uid,wordchar,category,other,pathalias,pathalias_tsvd100,userdata,image_resnext_category_ave5,image_resnext_subcategory_ave5,TSVD512_ave5_subcategory,TSVD512_ave5_category,TSVD512_subcategory,TSVD512_category,image_resnext_category_tsvd512_ave5,image_resnext_subcategry_tsvd512_ave5,image_resnext_pretrain,image_resnest_pretrain,image_resnext_category,image_resnest_category,image_resnext_subcategory,image_resnest_subcategory}]]
                         [-output SUBMISSION_PATH]

SMP Catboost model

optional arguments:
  -h, --help            show this help message and exit
  --model {xgboost,lightgbm,catboost}
                        which model(default: catboost)
  --feature [{Fasttext_tag,Fasttext_title,Fasttext_ave3,Bert_tag,Bert_title,Bert_ave3,Bert_tsvd512_ave3,tfidf_ave3,glove,glove_ave3,lsa_ave3,uid,wordchar,category,other,pathalias,pathalias_tsvd100,userdata,image_resnext_category_ave5,image_resnext_subcategory_ave5,TSVD512_ave5_subcategory,TSVD512_ave5_category,TSVD512_subcategory,TSVD512_category,image_resnext_category_tsvd512_ave5,image_resnext_subcategry_tsvd512_ave5,image_resnext_pretrain,image_resnest_pretrain,image_resnext_category,image_resnest_category,image_resnext_subcategory,image_resnest_subcategory}]
                        which feature will be used
  -output SUBMISSION_PATH, --submission_path SUBMISSION_PATH
                        SMP file(.json) will be submit path

 ```

simply run

```shell
bash catboost.sh
```

## Reference

- [CatBoost](https://catboost.ai/docs/concepts/python-reference_parameters-list.html)