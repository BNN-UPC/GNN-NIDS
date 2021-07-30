# GNN-NIDS native TensorFlow implementation

In this directory you may find a functional implementation of our GNN-based NIDS. The concrete GNN architecture in encoded in the *GNN.py* file, meanwhile the whole model can be trained by simply executing the *main.py* file, as shown below.

```
    python main.py
```

Notice that we do not provide the original dataset, which consists on [CIC-IDS 2017](https://www.unb.ca/cic/datasets/ids-2017.html). Thus, to use it, simply download it from the provided website. Once done that, provide in the *config.ini* file the corresponding files to the train and validation dataset (as shown below).

```
# PATH of the training data
train: ./data/TRAIN

#PATH of the validation data
validation: ./data/VAL
```


One may notice that some preprocessing may be required. The provided dataset is not sorted, which is a critical requirement given that in our paper we construct the graphs by grouping together consecutive flows. To this end, simply sort the dataset according to the datetime attribute that each of the flows has.
