# Introduction

Extract other number features, like "Ispublic", "photo_count" and so on.

## Usage

```shell
usage: number.py [-h] [--tag_filepath TAG_FILEPATH]
                 [--temporalspatial_filepath TEMPORALSPATIAL_FILEPATH]
                 [--userdata_filepath USERDATA_FILEPATH]
                 [--additional_filepath ADDITIONAL_FILEPATH]
                 [--feature_filepath FEATURE_FILEPATH]

Process the number data

optional arguments:
  -h, --help            show this help message and exit
  --tag_filepath TAG_FILEPATH
                        tag filepath
  --temporalspatial_filepath TEMPORALSPATIAL_FILEPATH
                        temporalspatial filepath
  --userdata_filepath USERDATA_FILEPATH
                        userdata filepath
  --additional_filepath ADDITIONAL_FILEPATH
                        addltional filepath
  --feature_filepath FEATURE_FILEPATH
                        time feature filepath
```

simply run

```shell
bash number.sh
```

## Saved feature

The features are in this format ```pid,uid,feature1,feature2,...```
