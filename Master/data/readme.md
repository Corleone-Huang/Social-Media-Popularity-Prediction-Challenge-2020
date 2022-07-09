# Data

This subfolder includes the original social media popularity prediction dataset and the prepocessing code.

## Dataset Overview 

The `SMPD (Social Media Prediction Dataset)` contains 486K social multimedia posts from 70K users and various social media information including anonymized photo-sharing records, user profile, web image, text, time, location, category, etc. SMPD is a multi-faced, large-scale, temporal web data collection, which collected from Flickr (one of the largest photo-sharing platforms). For the time-series forecasting task, we split training/testing data into chronological sets (commonly, by date and time). The tables below show the statistics of the dataset.

|Dataset|Post|User|Categories|Temporal Range (Months)|Avg. Title Length|Customize Tags|
| ------ | ------ | ------ |------|----|---|---|
|SMPD2020|486k|70k|756|16|29|250k|

Both train and test dataset can be download [here](http://smp-challenge.com/download). Please download the dataset to `data_source` subdirectory

## Data Preprocessing

### Split Training set

We select 80% of training set for training and 20% of the data for
validating the performance of our model in chronological order.

``` shell
cd preprocess_stage
python split_dataset.py
```

Besides, in `All Train Data Report.html`, you can see some statistics analysis about this dataset.

## Data details

### Histogram of Labels

![histogram of labels](../figure/histogram%20of%20labels.png)

### Label

[train](https://drive.google.com/file/d/1FW14l4jkTVV6hBymdrFeszq5B993axh_/view?usp=sharing)

Each row contains the popularity score(log-views) of the corresponding post.

``` shell
Popularityscore
...
3.2
2.3
...
```

### Image

[train](https://drive.google.com/file/d/196ssXYxA6YUbrFriVUP2m9rDN8_guZLU/view?usp=sharing), [test](https://drive.google.com/file/d/1QyhI6kloakDpG6d54G47Cb2lXkEGXiDR/view?usp=sharing)

Each row contains the URL of the cprrsponding photo or video

``` shell
...
"https://www.flickr.com/photos/58708830@N00/385070026"
"https://www.flickr.com/photos/97042891@N00/943750056"
...
```

![image](../figure/image%20sample.png)

### Category

[train](https://drive.google.com/file/d/1pQRC6fdFRPx48_iOlhuqt2K5TM50P5iB/view?usp=sharing), [test](https://drive.google.com/file/d/1gHfFiRU04SVs6zRxNOWqtHxBqw3JEWS4/view?usp=sharing)

Each row in the file corresponds a category set of the post.

``` shell
Uid  Pid  Category  Subcategory  Concept
...
"70478@N10" "564687" "Whether&Season" "Raining" "umbrella"
"37810@N60" "565202" "Fashion" "Girls,Fashion" "skirt"
"25893@N22" "565381" "Whether&Season" "Raining" "puddle"
"3175@N73" "16603"  "Entertainment" "Music" "rnb
...
```

**Uid**: the user this post belongs to.

**Pid**: the photo along with the post. One Pid can locate a particular post.

**Category**: the first category of the post.(11 classes)

**Subcategory**: there are 77 classes in 2nd level category.

**Concept**: there are 668 different description.

## Text

[train](https://drive.google.com/file/d/1OyrxlbCqE0qDwb9Ks6WrEYdpCacH--qA/view?usp=sharing), [test](https://drive.google.com/file/d/1gw3c6aI3hWsyZSGsvjYd0l1T37mvn726/view?usp=sharing)

Each row represents a text information of the post.

``` shell
Uid Pid Title Mediatype Alltags
...
"70478@N10" "564687" "Sarah Moon 3" "photo" "black hat fashion yellow umbrella"
"37810@N60" "565202" "2016-03-06 22.19.08" "photo" "orange sexy philadelphia hockey nhl sweater bra skirt blonde flowing cheerleader cleavage plaid flyers philadelphiaflyers icegirls"
"25893@N22" "565381" "Tristesse at the Federal Chancellery" "photo" "blackandwhite bw white black reflection berlin wet water rain canon germany puddle deutschland eos blackwhite wasser symmetry sw schwarzweiss puddles reflexion weiss federal schwarz regen tristesse trist reflektion kanzleramt pfuetze 6d nass bundeskanzleramt 2016 symmetrie pftze weis pftzen chancellery schwarzweis federalchancellery pfuetzen canoneos6d hoonose68 againstautotagging sgrossien grossien"
"3175@N73" "16594" "Amari DJ Mona-Lisa" "photo" "newyork celebrity brooklyn radio flickr photos itunes images singer singers celebrities hiphop reggae rb songwriter recordingartists broadcaster rnb songwriters amari hiphopartists cdbaby newreleases femaleartists reverbnation soundcloud famouscelebrities reggaeartists femaleperformers mtvartists amaridjmonalisa amazonmusic newyorkperformers reverbnationartists spotifyartists soundcloudartists jamgo cdbabyartists googleplayartists"
...
```

**Title**: the tile of the post defined by the user.

**Mediatype**: the type of the attached media file, including 'photo' and 'video'.

**Alltags**: the customized tags from users.

### Temporal-Spatial Information

[train](https://drive.google.com/file/d/1LN2-4rin05TdlnKt2KN4L8irrjGJ6SpX/view?usp=sharing), [test](https://drive.google.com/file/d/1tc5okCzHCAHn5SGVdJlf4Z7Mx8PaPXBH/view?usp=sharing)

Each row offers the date and geographic information of the post.

``` shell
Uid Pid Postdate Latitude Longitude Geoaccuracy
...
"70478@N10" "564687" "1457068974" "0" "0" "0"
"37810@N60" "565202" "1457273948" "0" "0" "0"
"25893@N22" "565381" "1457239452" "52.520213" "13.373097" "16"
"3263@N23" "17776" "1445400000" "39.051935" "-94.48068 14"
...
```

**Postdate**: the publish timestamp of the post. It can be converted to Datetime by following python code:

``` python
import time
timestamp = 1457068974
timeArray = time.localtime(timestamp)
datetime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
```

**Latitude**: the latitude whose valid range is -90 to 90. Anything more than 6 decimal places will be truncated.

**Longitude**: the longitude whose valid range is -180 to 180. Anything more than 6 decimal places will be truncated.

**Geoaccuracy**: recorded accuracy level of the location information. World level is 1, Country is ~3, Region ~6, City ~11, Street ~16. The current range is 1-16. Defaults to 16 if not specified.

### User Profile

[train](https://drive.google.com/file/d/1eQjmXp2x9L5EhyvEpBVpiMuiHOijuzxV/view?usp=sharing), [test](https://drive.google.com/file/d/1x9UOhi6dw527IjjtodIjYBjSEyS49eIU/view?usp=sharing)

Each row contains the user data of the post.

``` shell
photo_firstdate photo_count ispro canbuypro timezone_offset photo_firstdatetaken timezone_id user_description location_description
...
"1213743830" "6828" "1" "0" "1" "1904010100" "9" "0.0866962,-0.0752717,..." "0,0,..."
...
```

**Photo_firstdate**: the date of the first photo uploaded by the user.

**Photo_count**: the number of posted photo by the user.

**Ispro**: is the user belong to pro member.

**Photo_firstdatetaken**: the date of the first photo taken by the user.

**Timezone_offset**: the time zone of the user.

**User description**: the feature used to describe the user data.

**Location description**: the feature used to describe the user location.

### Additional Information

[train](https://drive.google.com/file/d/1c8TeShlNFE-_mdN5zYKutcVqTP_iZ3J8/view?usp=sharing), [test](https://drive.google.com/file/d/13kfrkZd5YK1vyVDMIokDZeVOJtARdJCk/view?usp=sharing)

Each row offers the supplimental information of the post.

``` shell
Uid Pid Pathalias Ispublic Mediastatus
...
"70478@N10" "564687" "None" "1" "ready"
"37810@N60" "565202" "None" "1" "ready"
"25893@N22" "565381" "hoo_nose_68" "1" "ready"
"3652@N11" "19388" "angelo_nairod" "1" "ready"
...
```

**Pathalias**: the path alias provided by the user.

**Ispublic**: indicates that the post is authenticated with 'read' permissions.

**Mediastatus**: indicates that the attached media is ready to access by others.

### Image Dataset

[train]() ~ 22G 

[test](https://drive.google.com/u/0/open?id=1W99A_q8JGGHwFIrzQTzRgRAwIhYsgQE1) ~ 14G
