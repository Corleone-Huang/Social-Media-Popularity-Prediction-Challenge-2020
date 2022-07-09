python finetune_image.py \
--use_gpu True \
--model resnext101 \
--finetune_category_feature_filepath ../../extracted_features/Category_ResNext101_image_486194.csv \
--finetune_subcategory_feature_filepath ../../extracted_features/Subcategory_ResNext101_image_486194.csv \
--use_pretrained True \
--finetune_by_category False \
--batch_size 256 \
--learning_rate 0.0001 \
--num_epochs 1 \
--num_workers 16

