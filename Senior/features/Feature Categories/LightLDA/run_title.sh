#!/bin/bash
python lightlda_title_args.py  --num_of_topic 11 \
  --num_of_iteration 100 \
  --input_processed_data_path /home/wangting/smp/data/pre_pro/pre_title.txt \
  --output_topic_word_path /home/wangting/smp/features/LightLDA/topic_word_title_args.csv \
  --output_doc_topic_path /home/wangting/smp/features/LightLDA/doc_topic_title_args.csv \
  --data_path /home/wangting/smp/data/train/train_tags.json
