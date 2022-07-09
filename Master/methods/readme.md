# Introduction

Our best model for SMP Challenage is Combined CatBoost in ``CatBoost``  subdirectory. There are some ideas for this popularity prediction task.

## Combined CatBoost idea

- Train two catboost models: one model trained with all features and crawler data, the other one trained with all features but without crawler data
- For predicting popularity of the same post, the result will to be submitted is the weighted average of the two models

## Usage

### Feature Extraction

- Extract all features, visual features, textual features, numerical feature.
- Use sliding window average to process image and text features to get short-term dependencies on the same user.
- Convert all features csv files to a big file in pkl format.(None is ok)

### Example features

Download the SMP features and model here :

- In China, by Baidu Netdisk
  - link: [Baidu Netdisk](https://pan.baidu.com/s/1wRMKmb3OIol_Yd_ltYyAwg)
  - verification code: ``539j``

- Others, by Google Drive
  - link: [Google Drive](https://drive.google.com/drive/folders/1y7KuegsM_vtm9shiAfQCN9yuT4xjI0kk?usp=sharing)

### Model Training

Train two catboost models

simply run

```shell
bash ./methods/CatBoost/catboost.sh
```

### Prediction

Average the two models prediction, using `submission.py`
