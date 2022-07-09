# Introduction

Using Seq2Seq to predict popularity. [Seq2Seq](https://arxiv.org/abs/1409.3215) is a famous framework in neural machine translation and other related cases. And it mainly based on RNN and Attention mechanism. Due to SMDP's temporal character, we employ Seq2Seq to the mission.

## Usage

By default, run ``seq2seq.py``, remember to change your root dir.
```python seq2seq.py```

parameters

```shell
seq2seq.py [-h][-f][--train_label_path][--valid_label_path]
  -f, --feature_path       The path of input feature.
  --train_label_path       The path of train data split.
  --valid_label_path       The path of valid data split.
```

## References

- [seq2seq_translation_tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)
- [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)
- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
