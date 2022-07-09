# Introduction

Use ResNeXt model to extract image feature finetuned by Category or Subcategory.

## Usage

```shell
usage: finetune_test.py [-h] [--use_gpu USE_GPU]
                        [--model {resnext101,resnext50,resnet152,resnest269}]
                        [--finetune_category_feature_filepath FINETUNE_CATEGORY_FEATURE_FILEPATH]
                        [--finetune_subcategory_feature_filepath FINETUNE_SUBCATEGORY_FEATURE_FILEPATH]
                        [--use_pretrained USE_PRETRAINED]
                        [--finetune_by_category FINETUNE_BY_CATEGORY]
                        [--batch_size BATCH_SIZE]
                        [--learning_rate LEARNING_RATE]
                        [--num_epochs NUM_EPOCHS] [--num_workers NUM_WORKERS]

optional arguments:
  -h, --help            show this help message and exit
  --use_gpu USE_GPU     will use gpu(default: True)
  --model {resnext101,resnext50,resnet152,resnest269}
                        model name
  --finetune_category_feature_filepath FINETUNE_CATEGORY_FEATURE_FILEPATH
                        image feature filepath extract by finetune category
                        model
  --finetune_subcategory_feature_filepath FINETUNE_SUBCATEGORY_FEATURE_FILEPATH
                        image feature filepath extract by finetune subcategory
                        model
  --use_pretrained USE_PRETRAINED
                        use pretrain weight in ImageNet(default: True)
  --finetune_by_category FINETUNE_BY_CATEGORY
                        is finetuned by Category(default: False)
  --batch_size BATCH_SIZE
                        batch size(default: 256)
  --learning_rate LEARNING_RATE
                        learning rate (default: 0.001)
  --num_epochs NUM_EPOCHS
                        number epochs(default: 1)
  --num_workers NUM_WORKERS
                        number workers(default: 8)
```

simply run

```shell
bash image.sh
```

## Saved feature

The features are in this format ```pid,uid,image_feature1,...,image_feature2048```

## Reference

- [pytorch tutorials](https://pytorch.org/tutorials/)
- [ResNeXt](https://github.com/facebookresearch/ResNeXt)
