"""
   Copyright 2020 Universitat Politècnica de Catalunya
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
       http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

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
