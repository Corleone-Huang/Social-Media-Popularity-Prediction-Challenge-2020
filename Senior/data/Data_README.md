# Data
This subfolder includes the information of social media prediction dataset and the code of the data preprocessing part.

## Overview of Social Media Prediction Dataset

The SMPD (Social Media Prediction Dataset) contains 486K social multimedia posts from 70K users and various social media information including anonymized photo-sharing records, user profile, web image, text, time, location, category, etc. SMPD is a multi-faced, large-scale, temporal web data collection, which collected from Flickr (one of the largest photo-sharing platforms). For the time-series forecasting task, we split training/testing data into chronological sets (commonly, by date and time). The tables below show the statistics of the dataset. Both train and test dataset can be download [here](http://smp-challenge.com/download).

|Dataset|Post|User|Categories|Temporal Range (Months)|Avg. Title Length|Customize Tags|
| ------ | ------ | ------ |------|----|---|---|
|SMPD2019|	486k|70k|756|16|29|250k|

## Data Preprocessing
### Split Train Dataset
We further divide the training set into new training set and verification set according to random and time sequence.The ratio of the new training set to the validation set is 0.95:0.05.

```python
cd label_split
sh label_split.sh
```

### Preprocessing of Tag and Title
This part is used for the pretreatment process of "lowercase", "remove special punctuation", "segmentation" and "remove stop words" of tag and title.

```python
cd pre_pro
sh pre_pro_tag.sh
sh pre_pro_title.sh
```



