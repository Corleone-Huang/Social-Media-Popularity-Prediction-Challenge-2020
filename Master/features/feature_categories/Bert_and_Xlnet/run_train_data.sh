python bert_xlnet_feature.py \
--data_dir ../../../data/data_source/train/train_tags.json \
--save_title_feature ../../extracted_features/ \
--save_alltags_feature ../../extracted_features/ \
--ngpu 1 \
--bert_base_uncased False \
--bert_large_uncased False \
--xlnet_base_cased False \
--xlnet_large_cased False \
--bert_base_multilingual_uncased True \
--bert_base_multilingual_cased False