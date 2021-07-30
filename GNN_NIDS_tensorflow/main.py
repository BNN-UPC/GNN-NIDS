import configparser
import tensorflow as tf
from GNN_NIDS_tensorflow.utils import make_model
from GNN_NIDS_tensorflow.generator import input_fn
import configparser

params = configparser.ConfigParser()
params._interpolation = configparser.ExtendedInterpolation()
params.read('./config.ini')

model = make_model( params=params)

# callbacks to save the model
callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath= params["DIRECTORIES"]["logs"] + "/ckpt/weights.{epoch:02d}-{loss:.2f}.hdf5", save_freq='epoch', monitor='loss', save_best_only=True), tf.keras.callbacks.TensorBoard(log_dir=params["DIRECTORIES"]["logs"] + "/logs", update_freq=1000)]

train_dataset = input_fn(data_path=params["DIRECTORIES"]["train"])  # load the training dataset
val_dataset = input_fn(data_path=params["DIRECTORIES"]["validation"])   # load the validation dataset

# DO THE TRAINING HERE
model.fit(train_dataset,
          validation_data= val_dataset,
          validation_steps = 600,
          steps_per_epoch=10000,
          batch_size=32,
          epochs=100,
          verbose=True,
          callbacks=callbacks,
          use_multiprocessing=True)
