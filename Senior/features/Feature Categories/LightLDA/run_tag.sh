#!/bin/bash
python lightlda_tag_args.py  --num_of_topic 11 \
  --num_of_iteration 100 \
  --input_processed_data_path /home/wangting/smp/data/pre_pro/pre_tag.txt \
  --output_topic_word_path /home/wangting/smp/features/LightLDA/topic_word_tag_args.csv \
  --output_doc_topic_path /home/wangting/smp/features/LightLDA/doc_topic_tag_args.csv \
  --data_path /home/wangting/smp/data/train/train_tags.json
