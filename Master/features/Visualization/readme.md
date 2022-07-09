# Introduction

```feature_hist.py``` is used to draw the histogram distribution of each dimension according to the extracted features to verify the validity of the features.

## Usage

example:

```shell
python feature_hist.py csv_file hist --bins 50
```

parameters:

```shell
usage: feature_hist.py [-h] [--bins BINS] data_dir output_dir

positional arguments:
  data_dir     The path of the csv file.
  output_dir   The path to save results

optional arguments:
  -h, --help   show this help message and exit
  --bins BINS  Number of bins of histgram
```
