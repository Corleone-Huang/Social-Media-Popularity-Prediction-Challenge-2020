# Introduction

Extract SURF(Speeded Up Robust Features) feature of images.

# Usage
example:
 ```shell
 python surf.py  image_path_list.json surf_img.csv
 ```
 
 Parameters:
 ```shell
 usage: surf.py [-h] [--keypoints] --data_dir --output_dir
 
 positional arguments:
  data_dir     The path of the pickle file warpping the list of imgs.
  output_dir   The path to save results
 
 optional arguments:
  --keypoints  Number of keypoints of surf.
 ```
