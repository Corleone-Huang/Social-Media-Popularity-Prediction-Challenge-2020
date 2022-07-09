# Methods
This folder contains the following methods code:

- Lightgbm
- MLP
- RNN
- Transformer

Each subfile contains the running environment and usage of the corresponding code.If the subfolder does not contain the environment, environment_default.yml is used by default.


## Usage for each methods

### Lightgbm
#### Introduction
We apply the liaghtgbm to implement the regression task based on the extracted featres.

#### Run
```python
conda env create -f environment_default.yml
sh sh lightgbm.sh
```

#### Options
```python
  -h, --help           show this help message and exit
  ----feature_path FAETURE_PATH  the path of the feature.csv
  ----train_path TRAIN_PATH  the path of the divided training dataset
  ----val_path VAL_PATH        the path of the divided val dataset
  ----save_DIR  SAVE_DIR the folder path to save the results
```

### MLP
#### Introduction
Embedding features related to post, including image/text/user information, are mapped into embedding features with similar dimensions, and embedding splicing features of various embedding technologies are inputted into THE MLP network to eventually obtain the corresponding popularity value of POST.

#### Struture
The code consists of the data read part, train(), train_all(), extractor_train(), extractor_test(), and test () parts. Among them:

- Trian () and TRAIN_all () are both training codes for MLP popularity prediction task. The difference between them is: Train () USES 8 (training set) /2 (test set) to divide data set, while train_all() USES all data for training, making full use of available data and giving up the referability of experimental results on verification set.

- The extractor_train() part extracts the output of a particular layer of the MLP training set as a pre-training (MLP) feature of the POST.

- The extractor_test() part extracts the output of the same layer of the test set as the PRE-training (MLP) feature of the POST.

- The test() section directly USES the trained MLP network to predict the post popularity of the test set.

#### Run
```python
python mlp.py
```

### RNN
#### Introduction
In the user's POST set, the fixed-length PID list set used for training RNN is generated in the order of Postdate, such as the PID list of a certain UID:

>[1,2,3,4,5,6]

Can produce:

> [0,0,0,0,1], [0,0,0,1,2], [0,0,1,2,3], [0,1,2,3,4], [1,2,3,4,5], [2,3,4,5,6]

A set of PID lists.

The corresponding POST features are then obtained according to the PID list and entered into the LSTM in postdate chronological order.

For each PID list, select hidden State of the last time step of RNN and feed it into the regression MLP to obtain the predicted value of the popularity of PID corresponding to the final time step of RNN.

#### Struture
- Add_pad () sets the padding pid to 0 (filled PID) vector to the full zero vector of the specific dimension.
- Patches_generator () generates a set of PID lists according to the postDate sequence.
- Feature_list () gets the corresponding (same order) feature list in pid order from the PID list set
- Label_list () generates the label list corresponding to each PID list.

#### Run
```python
conda env create -f environment.yml
python rnn_hdf5_v2.py
```

### Transformer
In the user's POST set, the fixed-length PID list set used for training RNN is generated in the order of Postdate, such as the PID list of a certain UID:

>[1,2,3,4,5,6]

Can produce:

> [0,0,0,0,1], [0,0,0,1,2], [0,0,1,2,3], [0,1,2,3,4], [1,2,3,4,5], [2,3,4,5,6]

A set of PID lists.

The corresponding POST characteristics are then obtained from the PID list and entered into the Transformer in postdate order.

For each PID list, the last layer hidden State of all time steps is selected. After stitching them together and performing one-dimensional convolution, input them into the regression MLP to obtain the predicted value of the popularity of pid corresponding to the last time step.

#### Struture
* Add_pad () sets the padding pid to 0 (filled PID) vector to the full zero vector of the specific dimension.
* Patches_generator () generates a set of PID lists according to the postDate sequence.
* Feature_list () gets the corresponding (same order) feature list in pid order from the PID list set.
* Label_list () generates the label list corresponding to each PID list.
* PositionalEncoding() class produces class embedding vectors for Transformer input.
* this code realization of transformer, based on nn. TransformerEncoderLayer ().

#### Run
```python
sh transformer.sh
```
