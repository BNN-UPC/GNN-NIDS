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
import sys

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import tensorflow as tf
import glob
import configparser


# --------------------------------------
# MAP THAT TELLS US, GIVEN A FEATURE, ITS POSITION (IDS 2017)
features = ['Flow ID','Source IP','Source Port','Destination IP','Destination Port','Protocol','Timestamp','Flow Duration','Total Fwd Packets','Total Backward Packets','Total Length of Fwd Packets','Total Length of Bwd Packets','Fwd Packet Length Max','Fwd Packet Length Min','Fwd Packet Length Mean','Fwd Packet Length Std','Bwd Packet Length Max','Bwd Packet Length Min','Bwd Packet Length Mean','Bwd Packet Length Std','Flow Bytes/s','Flow Packets/s','Flow IAT Mean','Flow IAT Std','Flow IAT Max','Flow IAT Min','Fwd IAT Total','Fwd IAT Mean','Fwd IAT Std','Fwd IAT Max','Fwd IAT Min','Bwd IAT Total','Bwd IAT Mean','Bwd IAT Std','Bwd IAT Max','Bwd IAT Min','Fwd PSH Flags','Bwd PSH Flags','Fwd URG Flags','Bwd URG Flags','Fwd Header Length','Bwd Header Length,Fwd Packets/s','Bwd Packets/s','Min Packet Length','Max Packet Length','Packet Length Mean','Packet Length Std','Packet Length Variance','FIN Flag Count','SYN Flag Count','RST Flag Count','PSH Flag Count','ACK Flag Count','URG Flag Count','CWE Flag Count','ECE Flag Count','Down/Up Ratio','Average Packet Size','Avg Fwd Segment Size','Avg Bwd Segment Size','Fwd Header Length,Fwd Avg Bytes/Bulk','Fwd Avg Packets/Bulk','Fwd Avg Bulk Rate','Bwd Avg Bytes/Bulk','Bwd Avg Packets/Bulk','Bwd Avg Bulk Rate','Subflow Fwd Packets','Subflow Fwd Bytes','Subflow Bwd Packets','Subflow Bwd Bytes','Init_Win_bytes_forward','Init_Win_bytes_backward','act_data_pkt_fwd','min_seg_size_forward','Active Mean','Active Std','Active Max','Active Min','Idle Mean','Idle Std','Idle Max','Idle Min','Label']
indices = range(len(features))
zip_iterator = zip(features,indices)
features_dict = dict(zip_iterator)

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

possible_protocols = {'6':[0.0,0.0,1.0],'17':[0.0,1.0,0.0], '0':[1.0,0.0,0.0],'':[0.0,0.0,0.0]}

# --------------------------------------

def normalization_function(feature, labels):
    n = len(features)
    for i in range(n):
        name = features[i]
        if name in features_dict and name in params['PARAMS']:
            feature['ip_to_ip_params'][name] = feature['ip_to_ip_params'][name] - float(params['PARAMS'][name + '_mean']) / float(params['PARAMS'][name + '_std'])
    return feature, labels


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


def get_feature(trace, feature_name, parse=True):
    if parse:
        if feature_name == 'Label':
            attack = trace[-1]
            attack_encoding = attacks_dict.get(attack, -1)

            return attack_encoding
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
    for f in chosen_connection_features:
        connection_features[f] = get_feature(trace, f)

    connection_features['Label'] = final_feature
    return connection_features


def traces_to_graph(traces):
    G = nx.MultiDiGraph()
    #G = nx.MultiGraph()

    n = len(traces)
    endpoints = {}
    for i in range(n):
        trace = traces[i]

        label_num = get_feature(trace, 'Label')
        final_label = np.zeros(15)
        final_label[label_num] = 1

        connection_features = get_connection_features(trace, final_label)
        G.add_node('con_' + str(i), **connection_features)

        src_ip = get_feature(trace,'Source IP', parse=False)
        dst_ip = get_feature(trace, 'Destination IP', parse=False)
        if(src_ip not in endpoints):
            endpoints[src_ip] = []

        if(dst_ip not in endpoints):
            endpoints[dst_ip] = []

        endpoints[src_ip].append('con_' + str(i))
        endpoints[dst_ip].append('con_' + str(i))


    # Now add all the connections between the nodes
    # Iterate over all the endpoints
    for _,adj_connections in endpoints.items():
        # Iterate over all the pairs of connections that are adjacent with this given endpoint
        n_cons = len(adj_connections)
        for i in range(n_cons):
            for j in range(i,n_cons):
                n1 = adj_connections[i]
                n2 = adj_connections[j]
                G.add_edge(n1,n2)
                G.add_edge(n2,n1)

    return G

def assign_indices(G):
    indices_connection = {}
    counter_connection = 0

    for v in G.nodes():
        indices_connection[v] = counter_connection
        counter_connection += 1

    return indices_connection, counter_connection

def process_adjacencies(G):
    indices_connection, counter_connection = assign_indices(G)
    src_connection_to_connection, dst_connection_to_connection  = [], []

    for e in G.edges(): # each edge is a pair of the source, dst node
        connection_node1 = e[0]  # connection
        connection_node2 = e[1]  # connection

        # connection to ip and ip to connection
        src_connection_to_connection.append(indices_connection[connection_node1])
        dst_connection_to_connection.append(indices_connection[connection_node2])

    return src_connection_to_connection, dst_connection_to_connection,counter_connection


def graph_to_dict(G):
    #edge features
    connection_features = np.array([])
    first = True
    for f in chosen_connection_features:
        aux = np.array(list(nx.get_node_attributes(G, f).values()))

        if first:
            connection_features = np.expand_dims(aux, axis=-1)
            first = False
        else:
            if len(aux.shape) == 1:
                aux = np.expand_dims(aux, -1)

            connection_features = np.concatenate([connection_features, aux], axis=1)

    # obtain the labels of the nodes (indicator r.v indicating whether it has been infected or not)
    label = np.array(list(nx.get_node_attributes(G, 'Label').values())).astype('float32')

    # obtain the adjacencies
    src_connection_to_connection, dst_connection_to_connection, n_c = process_adjacencies(G)

    features = {
        'feature_connection': connection_features,
        'n_c': n_c,
        'src_connection_to_connection': src_connection_to_connection,
        'dst_connection_to_connection': dst_connection_to_connection
    }
    return (features, label)
