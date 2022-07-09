# Introduction

Use tcn to predict popularity. After input features and feeding them into model, tcn will deal with them as a time series. [TCN](https://arxiv.org/pdf/1803.01271.pdf) is a model based on CNN and has a better performance then RNN in many cases.

## Usage

By default, run ``tcn.py``.

``` shell
python tcn.py
```

parameters

```shell
 tcn [-h][-f][--train_label_path][--valid_label_path][--timesteps]
   -f, --feature_path    The path of input feature.
   --train_label_path    The path of train data split.
   --valid_label_path    The path of valid data split.
   --timesteps           The size of window to slice data.
 ```

## References

- [keras-tcn](https://github.com/philipperemy/keras-tcn)
- [TCN for Pytorch](https://github.com/locuslab/TCN/)
- [An Empirical Evaluation of Generic Convolutional and Recurrent Networks
for Sequence Modeling](https://arxiv.org/pdf/1803.01271.pdf)
- [WAVENET: A GENERATIVE MODEL FOR RAW AUDIO](https://arxiv.org/pdf/1609.03499.pdf)
