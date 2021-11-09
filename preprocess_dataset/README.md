# Preprocess the dataset
In order to prepare the dataset for our (Tensorflow) model, we used the *preprocess_datasets_IDS2017.py*. You can run this code by simplify using:
```
    python preprocess_datasets_IDS2017.py
```

There we apply a series of preprocessing steps so as to adapt the original IDS2017 dataset into a suitable dataset for graph-based learning.

To simplify this setting, however, we provide the *preprocessed_IDS2017* dataset, which you can simply use for the Tensorflow code, after unziping using:
```
    unzip preprocessed_IDS2017.zip
```