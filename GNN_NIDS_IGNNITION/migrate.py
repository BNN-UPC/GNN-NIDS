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
import tarfile
import networkx as nx
from random import random
import json
from networkx.readwrite import json_graph
import os
import csv
import sys

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import tensorflow as tf
import glob
import configparser

params_norm = configparser.ConfigParser()
params_norm._interpolation = configparser.ExtendedInterpolation()
params_norm.read('./normalization_parameters.ini')

params = configparser.ConfigParser()
params._interpolation = configparser.ExtendedInterpolation()
params.read('./migrate.ini')


# --------------------------------------
# IDS 2017

# MAP THAT TELLS US, GIVEN A FEATURE, ITS POSITION (IDS 2017)
features = ['Flow ID','Source IP','Source Port','Destination IP','Destination Port','Protocol','Timestamp','Flow Duration','Total Fwd Packets','Total Backward Packets','Total Length of Fwd Packets','Total Length of Bwd Packets','Fwd Packet Length Max','Fwd Packet Length Min','Fwd Packet Length Mean','Fwd Packet Length Std','Bwd Packet Length Max','Bwd Packet Length Min','Bwd Packet Length Mean','Bwd Packet Length Std','Flow Bytes/s','Flow Packets/s','Flow IAT Mean','Flow IAT Std','Flow IAT Max','Flow IAT Min','Fwd IAT Total','Fwd IAT Mean','Fwd IAT Std','Fwd IAT Max','Fwd IAT Min','Bwd IAT Total','Bwd IAT Mean','Bwd IAT Std','Bwd IAT Max','Bwd IAT Min','Fwd PSH Flags','Bwd PSH Flags','Fwd URG Flags','Bwd URG Flags','Fwd Header Length','Bwd Header Length','Fwd Packets/s','Bwd Packets/s','Min Packet Length','Max Packet Length','Packet Length Mean','Packet Length Std','Packet Length Variance','FIN Flag Count','SYN Flag Count','RST Flag Count','PSH Flag Count','ACK Flag Count','URG Flag Count','CWE Flag Count','ECE Flag Count','Down/Up Ratio','Average Packet Size','Avg Fwd Segment Size','Avg Bwd Segment Size','Fwd Avg Bytes/Bulk','Fwd Avg Packets/Bulk','Fwd Avg Bulk Rate','Bwd Avg Bytes/Bulk','Bwd Avg Packets/Bulk','Bwd Avg Bulk Rate','Subflow Fwd Packets','Subflow Fwd Bytes','Subflow Bwd Packets','Subflow Bwd Bytes','Init_Win_bytes_forward','Init_Win_bytes_backward','act_data_pkt_fwd','min_seg_size_forward','Active Mean','Active Std','Active Max','Active Min','Idle Mean','Idle Std','Idle Max','Idle Min','Label']
indices = range(len(features))
zip_iterator = zip(features,indices)
features_dict = dict(zip_iterator)


# ATTACKS IDS 2017
attack_names = ['SSH-Patator', 'DoS GoldenEye', 'PortScan', 'DoS Slowhttptest', 'Web Attack  Brute Force', 'Bot', 'Web Attack  Sql Injection', 'Web Attack  XSS', 'Infiltration', 'DDoS', 'DoS slowloris', 'Heartbleed', 'FTP-Patator', 'DoS Hulk','BENIGN']
indices = range(len(attack_names))
zip_iterator = zip(attack_names,indices)
attacks_dict = dict(zip_iterator)


chosen_connection_features = ['Source Port', 'Destination Port', 'Bwd Packet Length Min', 'Subflow Fwd Packets',
                   'Total Length of Fwd Packets', 'Fwd Packet Length Mean', 'Total Length of Fwd Packets',
                   'Fwd Packet Length Std', 'Fwd IAT Min', 'Flow IAT Min', 'Flow IAT Mean', 'Bwd Packet Length Std',
                   'Subflow Fwd Bytes', 'Flow Duration', 'Flow IAT Std', 'Active Min','Active Mean', 'Bwd IAT Mean',
                   'Subflow Bwd Bytes', 'Init_Win_bytes_forward', 'ACK Flag Count','Fwd PSH Flags','SYN Flag Count',
                   'Flow Packets/s', 'PSH Flag Count', 'Average Packet Size']

indices = range(len(chosen_connection_features))
zip_iterator = zip(chosen_connection_features, indices)
chosen_features_dict = dict(zip_iterator)


possible_protocols = {'6':[0.0,0.0,1.0],'17':[0.0,1.0,0.0], '0':[1.0,0.0,0.0],'':[0.0,0.0,0.0]}

# --------------------------------------

def normalization_function(feature, name):
    if name in chosen_connection_features and (name+'_mean') in params_norm['PARAMS'] and float(params_norm['PARAMS'][name + '_mean']) != 0:
        feature = (feature - float(params_norm['PARAMS'][name + '_mean'])) / float(params_norm['PARAMS'][name + '_std'])
    return feature


def transform_ips(ip):
    # transform it into a 12 bit string
    ip = ip.split('.')
    for i in range(len(ip)):
        ip[i] = '0'*(3 - len(ip[i])) + ip[i]

    ip = ''.join(ip)
    try:
        result = [float(v) for v in ip if v != '.']
    except:
        result = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]

    return result


def get_feature(trace, feature_name,  parse=True):
    if parse:
        if feature_name == 'Label':
            attack = trace[-1]
            return attacks_dict.get(attack)
        else:
            idx = features_dict[feature_name]
            feature = trace[idx]

            if 'ID' in feature_name:
                return feature
            elif 'IP' in feature_name:
                return transform_ips(feature)
            elif feature_name == 'Protocol':
                # Transform to a one-hot encoding
                return possible_protocols.get(feature)
            else:
                try:
                    value = float(feature)
                    if value != float('+inf') and value != float('nan'):
                        return value
                    else:
                        return 0
                except:
                    return 0
    else:
        idx = features_dict[feature_name]
        return trace[idx]

# constructs a dictionary with all the chosen features of the ids 2017
def get_connection_features(trace, final_feature):
    connection_features = {}
    aux = []
    for f in chosen_connection_features:
        feat = get_feature(trace, f)
        norm_feats = normalization_function(feat, f)
        aux.append(norm_feats)
    connection_features['Label'] = final_feature
    connection_features['conect_feats'] = aux
    return connection_features


def traces_to_graph(traces):
    G = nx.DiGraph()
    # G = nx.MultiDiGraph()
    # G = nx.MultiGraph()

    n = len(traces)
    for i in range(n):
        trace = traces[i]

        dst_name = 'Destination IP'
        src_name = 'Source IP'

        # For now we create the IP features as a list of 128
        if get_feature(trace, dst_name, parse=False) not in G.nodes():
            G.add_node(get_feature(trace, dst_name, parse=False), entity='ip', ip_feats = list(np.ones(128)))

        if get_feature(trace, src_name, parse=False) not in G.nodes():
            G.add_node(get_feature(trace, src_name, parse=False), entity='ip', ip_feats = list(np.ones(128)))

        label_num = get_feature(trace, 'Label')
        final_label = np.zeros(15)
        if label_num != -1: # if it is an attack
            final_label[label_num] = 1
        final_label = final_label.tolist()

        connection_features = get_connection_features(trace, final_label)
        connection_features['entity'] = 'connection'
        G.add_node('con_' + str(i), **connection_features)

        # these edges connect the ports with the IP node (connecting all the servers together)
        G.add_edge('con_' + str(i), get_feature(trace, dst_name, parse=False))
        G.add_edge('con_' + str(i), get_feature(trace, src_name, parse=False))
        G.add_edge(get_feature(trace, dst_name, parse=False), 'con_' + str(i))
        G.add_edge(get_feature(trace, src_name, parse=False), 'con_' + str(i))

    return G

# This function must return the corresponding graphs
def generator(path):
    files = glob.glob(path + '/*.csv')
    
    for file in files:
        print("Processing file:", file)
        with open(file, encoding="utf8", errors='ignore') as csvfile:
            data = csv.reader(csvfile, delimiter=',', quotechar='|')

            current_time_traces = []
            counter = 0
            for row in data:
                if len(row) > 1:
                    # remains to fix this criterion (for now we set the windows to be 200 connections big)
                    if counter >= 200:
                        if current_time_traces != []:
                            G = traces_to_graph(current_time_traces)
                            yield G

                        counter = 0
                        current_time_traces = []
                    else:
                        current_time_traces.append(row)
                    
                    counter += 1

def migrate_dataset(input_path, output_path, max_per_file=100):
    print("Starting to do the migration...")

    gen = generator(input_path)
    data = []
    file_ctr = 0
    counter = 0

    while True:
       try:
            G = next(gen)
            parser_graph = json_graph.node_link_data(G)
            data.append(parser_graph)
            if max_per_file is not None and counter == max_per_file:
                with open(output_path + '/data_' + str(file_ctr) + '.json', 'w') as json_file:
                        json.dump(data, json_file)

                data = []
                counter = 0
                file_ctr += 1
            else:
                counter +=1

        #when finished, save all the remaining ones
       except:
            with open(output_path + '/data_' + str(file_ctr) + '.json', 'w') as json_file:
                json.dump(data, json_file)
            return
            
if __name__ == "__main__":
    input_path = os.path.abspath(params['DIRECTORIES']['original_dataset_path'])
    output_path = os.path.abspath(params['DIRECTORIES']['output_path'])
    
    # Create the output directories if necessary
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    migrate_dataset(input_path, output_path)
