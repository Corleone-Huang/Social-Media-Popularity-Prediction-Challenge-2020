python onehot_title.py \
--data_path ../../../data/data_source/train/train_tags.json \
--save_title ../../extracted_features/pca_onehot_title_10000-256.csv \
wait

python onehot_tags.py \
--data_path ../../../data/data_source/train/train_tags.json \
--save_tags ../../extracted_features/pca_onehot_tags_10000-256.csv \
wait

python onehot_category.py \
--train_dataset_filepath ../../../data/data_source/train/train_category.json \
--test_dataset_filepath ../../../data/data_source/train/test_category.json \
--train_feature_filepath ../../extracted_features/onehot_train_category&subcategory_305613.csv \
--test_feature_filepath ../../extracted_features/onehot_test_category&subcategory_180581.csv \
wait

echo "over"