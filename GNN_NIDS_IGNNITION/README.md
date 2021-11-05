# GNN-NIDS in IGNNITION

In this directory you may find a functional implementation of our GNN-based NIDS implemented using IGNNITION. 

Notice that the first requirement for IGNNITION is to migrate your original dataset into the supported format. To easen this, we provide already the migrated dataset in [data](./data).
We, however, also provide the *migrate.py* file that we used, which allows to easily make adaptations to our provided settings. For this, run:

```
    python migrate.py
```

This will create the dataset in the *data* directory as well. Observe that you need to indicate the corresponding paths in the *migrate.ini* file.

Once the dataset is create, simply train the model using:
```
    python main.py
```

The rest of the files are provided, and concretely, the model is implemented in *model_description.yaml*. For more information, we refer users to [ignnition](https://ignnition.net/doc/quick_tutorial/).
