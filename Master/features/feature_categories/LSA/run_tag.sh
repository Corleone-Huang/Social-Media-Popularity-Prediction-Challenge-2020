#!/bin/bash
python LSA_tag_args.py  --num_of_topic 11 \
  --num_of_iteration 100 \
  --input_data_path /home/wangting/smp/data/pre_pro/pre_tag.txt \
  --output_doc_topic_path /home/wangting/smp/features/LSA/doc_topic_tag_args.csv \
  --data_path ../../../data/data_source/train/train_tags.json
