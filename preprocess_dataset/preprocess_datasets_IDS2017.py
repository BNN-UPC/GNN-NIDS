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

import csv
import os
import numpy as np
import glob
import configparser
import generator
from random import random, shuffle
import operator

params = configparser.ConfigParser()
params._interpolation = configparser.ExtendedInterpolation()
params.read('./config.ini')

def preprocess(path):
    files = glob.glob(path + '/*.csv')
    
    #files = [f for f in files if 'Monday' not in f]     #we donnot include the monday file (as it contains no attacks)
    all_traces = []

    training_graphs = []
    evaluation_graphs = []

    print("Starting to preprocess the dataset...")
    
    # Iterate over all the data files
    # Each of the files contains attacks of one type
    for file in files:
        print("Processing file: ", file)
        with open(file, encoding="utf8", errors='ignore') as csvfile:
            data = csv.reader(csvfile, delimiter=',', quotechar='|')

            # sort the traces by timestamp
            data = sorted(data, key=operator.itemgetter(6))

            current_time_traces = []
            counter = 0
            n = len(data)
            total_counter = 0
            for row in data:
                if len(row) > 1:
                    # remains to fix this criterion
                    if counter >= 200 or (total_counter+1) >= n:
                        if current_time_traces != []:
                            G = generator.traces_to_graph(current_time_traces)
                            features, label = generator.graph_to_dict(G)
                            benign_labels = np.array([l == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0] for l in label])
                            if benign_labels.all(): # do random undersampling
                                if random() < 0.1:
                                    all_traces.append(current_time_traces)
                            else:
                                all_traces.append(current_time_traces)
                            #all_traces.append(current_time_traces)

                        current_time_traces = []
                        counter = 0
                    else:
                        current_time_traces.append(row)

                    counter += 1
                    total_counter += 1

        shuffle(all_traces)
        n = len(all_traces)
        training_graphs += all_traces[0:int(n*0.8)]
        evaluation_graphs += all_traces[-int(n*0.2):]

        all_traces = []

    shuffle(training_graphs)
    shuffle(evaluation_graphs)

    output_path = os.path.abspath(params['DIRECTORIES']['output_dir'])
    with open(output_path + '/TRAIN/train_dataset.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for g in training_graphs:
            for row in g:
                writer.writerow(row)

    with open(output_path + '/EVAL/eval_dataset.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for g in evaluation_graphs:
            for row in g:
                writer.writerow(row)

if __name__ == '__main__':
    preprocess(os.path.abspath(params['DIRECTORIES']['original_dataset']))