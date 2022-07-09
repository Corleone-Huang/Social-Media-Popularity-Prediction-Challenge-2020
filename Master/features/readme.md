# Features

## Directory

- extracted_features/ (store the extracted features)
- pretrained_model/ (store the pretrained models)
- splited_label/ (store the splited label)
- visualization/ (draw frequency histograms to roughly verify the correctness of features)
- feature_categories/ (methods to extract features)

## Methods to extract features

The ``feature_categories``subfolder contains the following feature extract codes:

### Visual

- ResNeXt
- DenseSIFT
- SURF
- HoG
- Hu

### Textual

- Bert and Xlnet feature
- FastText
- Glove
- LightLDA
- LSA
- TFIDF
- Wordcount
- Onehot

### Numeral

- Number

### Short-term dependence

- Sliding_window_average

Each subdirectory contains the readme file and usage of the corresponding codes.

## Feature Performance

![feature](../figure/performance%20on%20combines%20of%20features%20in%20validation%20set.png)

## Sliding Window Performance

![sliding](../figure/sliding%20performance%20on%20validation%20set.png)