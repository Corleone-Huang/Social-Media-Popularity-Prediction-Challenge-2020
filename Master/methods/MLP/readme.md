# Introduction

Using MLP(Multilayer Perceptron) to predict popularity.

## Usage

example:

```shell
  python mlp.py \
   --all_feature train_feature.pkl \
   --train_label_path train_label.csv \
   --validate_label_path valid_label.csv \
   --batch_size 128 \
   --learning_rate 0.0007
 ```

 parameters:

 ```shell
  usage: mlp.py [-h][--all_feature][--train_label_path]
                [--validate_label_path][--batch_size][--learning_rate]
  
  optional arguments:
    --all_feature          The path of input features(pickle).
    --train_label_path     The path of train label(csv).
    --validate_label_path  The path of validate label.
    --batch_size           Training batch size.
    --learning_rate        Initial learning rate.
 ```
  
## Extract mlp features using pre-trained mlp model

 example:

 ```shell
 python mlp_feature.py train_feature.pkl test_feature.pkl mlp_model.pth
 ```

 parameters:

 ```shell
 usage: mlp_feature.py [-h][--train_feature][--test_feature][--model_path]
 optional arguments:
   --train_feature   The path of train feature.
   --test_feature    The path of test feature.
   --model_path      The path of pre-trained model path.
 ```
