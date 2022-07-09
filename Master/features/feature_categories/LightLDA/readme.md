# Introduction

This project is used to extract the characteristics of Title and Alltags data in ACM MM SMP Challenges Data.
Document_topic probability distribution, i.e. feature, is obtained through topic modeling of title and Alltags.The dimension depends on the number of topics and can be set freely.
A topic_word distribution is also generated to reflect the meaning of each topic.

## Result

Document_topic probability distribution and TOPic_Word distribution feature files in CSV format are generated in the addresses pointed to by '--output_topic_word_path' and '--output_doc_topic_path' parameters.
The characteristic dimension is determined by '-- num_OF_topic '.
'-- num_OF_iteration 'refers to the number of model training iterations

## Usage

Simply run

For title:

```shell
bash run_title.sh
```

For alltags：

```shell
bash run_tag.sh
```

Specific parameters for reference:

``` shell
  --num_of_topic        主题数量
  --num_of_iteration    迭代次数
  --input_processed_data_path 存放已预处理数据的路径
  --output_topic_word_path    存放topic_word分布csv文件的路径
  --output_doc_topic_path     存放doc_topic分布csv文件的路径
  --input_data_path           存放原始数据的路径
```
