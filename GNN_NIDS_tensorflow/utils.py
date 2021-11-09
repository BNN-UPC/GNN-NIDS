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

import tensorflow as tf
import tensorflow_addons as tfa
from GNN import GNN
import os


def _get_compiled_model(params):
    model = GNN(params)
    decayed_lr = tf.keras.optimizers.schedules.ExponentialDecay(float(params['HYPERPARAMETERS']['learning_rate']),
                                                                int(params['HYPERPARAMETERS']['decay_steps']),
                                                                float(params['HYPERPARAMETERS']['decay_rate']),
                                                                staircase=False)

    optimizer = tf.keras.optimizers.Adam(learning_rate=decayed_lr)
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    metrics = [tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.SpecificityAtSensitivity(0.1),
               tf.keras.metrics.Recall(top_k=1,class_id=0, name='rec_0'), tf.keras.metrics.Precision(top_k=1, class_id=0, name='pre_0'),
               tf.keras.metrics.Recall(top_k=1,class_id=1, name='rec_1'), tf.keras.metrics.Precision(top_k=1,class_id=1, name='pre_1'),
               tf.keras.metrics.Recall(top_k=1,class_id=2, name='rec_2'), tf.keras.metrics.Precision(top_k=1,class_id=2, name='pre_2'),
               tf.keras.metrics.Recall(top_k=1,class_id=3, name='rec_3'), tf.keras.metrics.Precision(top_k=1,class_id=3, name='pre_3'),
               tf.keras.metrics.Recall(top_k=1,class_id=4, name='rec_4'), tf.keras.metrics.Precision(top_k=1,class_id=4, name='pre_4'),
               tf.keras.metrics.Recall(top_k=1,class_id=5, name='rec_5'), tf.keras.metrics.Precision(top_k=1,class_id=5, name='pre_5'),
               tf.keras.metrics.Recall(top_k=1,class_id=6, name='rec_6'), tf.keras.metrics.Precision(top_k=1,class_id=6, name='pre_6'),
               tf.keras.metrics.Recall(top_k=1,class_id=7, name='rec_7'), tf.keras.metrics.Precision(top_k=1,class_id=7, name='pre_7'),
               tf.keras.metrics.Recall(top_k=1,class_id=8, name='rec_8'), tf.keras.metrics.Precision(top_k=1,class_id=8, name='pre_8'),
               tf.keras.metrics.Recall(top_k=1,class_id=9, name='rec_9'), tf.keras.metrics.Precision(top_k=1,class_id=9, name='pre_9'),
               tf.keras.metrics.Recall(top_k=1,class_id=10, name='rec_10'), tf.keras.metrics.Precision(top_k=1, class_id=10, name='pre_10'),
               tf.keras.metrics.Recall(top_k=1,class_id=11, name="rec_11"), tf.keras.metrics.Precision(top_k=1, class_id=11, name="pre_11"),
               tf.keras.metrics.Recall(top_k=1,class_id=12, name="rec_12"), tf.keras.metrics.Precision(top_k=1, class_id=12, name='prec_12'),
               tf.keras.metrics.Recall(top_k=1,class_id=13, name='rec_13'), tf.keras.metrics.Precision(top_k=1, class_id=13, name='prec_13'),
               tf.keras.metrics.Recall(top_k=1,class_id=14, name='rec_14'), tf.keras.metrics.Precision(top_k=1, class_id=14, name='prec_14'),
               tfa.metrics.F1Score(15,average='macro',name='macro_F1'),tfa.metrics.F1Score(15,average='weighted',name='weighted_F1')]#, tfma.metrics.MultiClassConfusionMatrixPlot(name='multi_class_confusion_matrix_plot'),],

    model.compile(loss=loss_object,
                  optimizer=optimizer,
                  metrics= metrics,
                  run_eagerly=False)
    return model


import glob
def make_or_restore_model(params):
    # Either restore the latest model, or create a fresh one

    checkpoint_dir = os.path.abspath(params['DIRECTORIES']['logs'] + '/ckpt')
    # if there is no checkpoint available.
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    checkpoints = glob.glob(checkpoint_dir + "/weights*")
    
    gnn = _get_compiled_model(params)
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print("Restoring from", latest_checkpoint)
        return gnn.load_weights(latest_checkpoint)
    print("Creating a new model")
    return gnn
