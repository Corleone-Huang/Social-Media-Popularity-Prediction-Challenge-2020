python onehot_tags.py \
--data_path /home/cx/SMP/data/train/train_tags.json \
--save_title /home/cx/SMP/data/处理/onehot_title_305613.csv 
wait

python onehot_tags.py \
--data_path /home/cx/SMP/data/train/train_tags.json \
--save_tags /home/cx/SMP/data/处理/onehot_tags_305613.csv \
wait

python onehot_category.py \
--train_dataset_filepath /home/wangkai/SMP/data/train/train_category.json \
--test_dataset_filepath /home/wangkai/SMP/data/test/test_category.json \
--train_feature_filepath /home/wangkai/SMP/feature/onehot_train_category&subcategory_305613.csv \
--test_feature_filepath /home/wangkai/SMP/feature/onehot_test_category&subcategory_180581.csv
wait

echo "over"