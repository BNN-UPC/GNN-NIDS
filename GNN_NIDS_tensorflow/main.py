import configparser
import time
import numpy as np
import os
import tensorflow as tf
from utils import make_or_restore_model
from generator import input_fn
import configparser

params = configparser.ConfigParser()
params._interpolation = configparser.ExtendedInterpolation()
params.read('./config.ini')

model = make_or_restore_model(params=params)

# callbacks to save the model
path_logs = os.path.abspath(params['DIRECTORIES']['logs'])
callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath=  path_logs + "/ckpt/weights.{epoch:02d}-{loss:.2f}.hdf5", save_freq='epoch', monitor='loss', save_best_only=False), tf.keras.callbacks.TensorBoard(log_dir=path_logs + "/logs", update_freq=1000)]

train_dataset = input_fn(data_path=os.path.abspath(params["DIRECTORIES"]["train"]), validation=False)
val_dataset = input_fn(data_path=os.path.abspath(params["DIRECTORIES"]["validation"]), validation=True)

# Training the model
model.fit(train_dataset,
          validation_data= val_dataset,
          validation_steps = 600,
          steps_per_epoch=1600,
          batch_size=16,
          epochs=2000,
          callbacks=callbacks,
          use_multiprocessing=True)
