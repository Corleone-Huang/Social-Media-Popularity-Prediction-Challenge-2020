# Introduction

Feature moving average, group by uid.

# Usage

```shell
usage: feature_moving_average.py [-h] -d DATA_DIR -o OUTPUT_DIR [-n {3,5}]

optional arguments:
  -h, --help            show this help message and exit
  -d DATA_DIR, --data_dir DATA_DIR
                        The path of input data.
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        The path of output data.
  -n {3,5}, --n_window {3,5}
                        The number of feature to conduct moving average.
```

simlpe run

```shell
bash feature_moving_average.sh
```