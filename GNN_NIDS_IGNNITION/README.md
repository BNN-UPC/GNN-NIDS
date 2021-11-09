# GNN-NIDS in IGNNITION

In this directory you may find a functional implementation of our GNN-based NIDS implemented using IGNNITION. 

## Generate the dataset
Notice that the first requirement for IGNNITION is to migrate your original dataset into the supported format. To ease this, we provide already the migrated dataset in [data](./data).
Observe that before using it, you need to uncompress it, using:

```
    unzip data.zip
```

We, however, also provide the *migrate.py* file that we used, which allows to easily make adaptations to our provided settings. For this, run:

```
    python migrate.py
```

This will create the dataset in the *data* directory as well. Observe that you need to indicate the corresponding paths in the *migrate.ini* file.

## Training from scratch
Once the dataset is create, simply train the model using:
```
    python main.py
```

## Pre-trained model
In order to improve the usability of this model, we also provide a pre-trained model in [model](./CheckPoint/experiment_2021_11_08_17_21_38). To use it, simply indicate the path in the *train_options.yaml* file, in the field *load_model_path*.
This will allow IGNNITION to start a new training using that checkpoint or to directly do predictions using it.


## More information...
The concrete model that we used is implemented in *model_description.yaml*. For more information on how to train/predict/evaluate the models, we refer users to [ignnition](https://ignnition.net/doc/quick_tutorial/).
