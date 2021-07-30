
import csv
import sys

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import tensorflow as tf
import glob
import configparser

params = configparser.ConfigParser()
params._interpolation = configparser.ExtendedInterpolation()
params.read('./normalization_parameters.ini')

# --------------------------------------
# CIC-IDS 2017
# Produce a map of the index of each of the possible features
features = ['Flow ID','Source IP','Source Port','Destination IP','Destination Port','Protocol','Timestamp','Flow Duration','Total Fwd Packets','Total Backward Packets','Total Length of Fwd Packets','Total Length of Bwd Packets','Fwd Packet Length Max','Fwd Packet Length Min','Fwd Packet Length Mean','Fwd Packet Length Std','Bwd Packet Length Max','Bwd Packet Length Min','Bwd Packet Length Mean','Bwd Packet Length Std','Flow Bytes/s','Flow Packets/s','Flow IAT Mean','Flow IAT Std','Flow IAT Max','Flow IAT Min','Fwd IAT Total','Fwd IAT Mean','Fwd IAT Std','Fwd IAT Max','Fwd IAT Min','Bwd IAT Total','Bwd IAT Mean','Bwd IAT Std','Bwd IAT Max','Bwd IAT Min','Fwd PSH Flags','Bwd PSH Flags','Fwd URG Flags','Bwd URG Flags','Fwd Header Length','Bwd Header Length','Fwd Packets/s','Bwd Packets/s','Min Packet Length','Max Packet Length','Packet Length Mean','Packet Length Std','Packet Length Variance','FIN Flag Count','SYN Flag Count','RST Flag Count','PSH Flag Count','ACK Flag Count','URG Flag Count','CWE Flag Count','ECE Flag Count','Down/Up Ratio','Average Packet Size','Avg Fwd Segment Size','Avg Bwd Segment Size','Fwd Avg Bytes/Bulk','Fwd Avg Packets/Bulk','Fwd Avg Bulk Rate','Bwd Avg Bytes/Bulk','Bwd Avg Packets/Bulk','Bwd Avg Bulk Rate','Subflow Fwd Packets','Subflow Fwd Bytes','Subflow Bwd Packets','Subflow Bwd Bytes','Init_Win_bytes_forward','Init_Win_bytes_backward','act_data_pkt_fwd','min_seg_size_forward','Active Mean','Active Std','Active Max','Active Min','Idle Mean','Idle Std','Idle Max','Idle Min','Label']
indices = range(len(features))
zip_iterator = zip(features,indices)
features_dict = dict(zip_iterator)

# Produce a map of the index of each of the considered attacks
attack_names = ['SSH-Patator', 'DoS GoldenEye', 'PortScan', 'DoS Slowhttptest', 'Web Attack  Brute Force', 'Bot', 'Web Attack  Sql Injection', 'Web Attack  XSS', 'Infiltration', 'DDoS', 'DoS slowloris', 'Heartbleed', 'FTP-Patator', 'DoS Hulk','BENIGN']
indices = range(len(attack_names))
zip_iterator = zip(attack_names,indices)
attacks_dict = dict(zip_iterator)

# List of the chosen features
chosen_connection_features = ['Source Port', 'Destination Port', 'Bwd Packet Length Min', 'Subflow Fwd Packets',
                   'Total Length of Fwd Packets', 'Fwd Packet Length Mean', 'Total Length of Fwd Packets',
                   'Fwd Packet Length Std', 'Fwd IAT Min', 'Flow IAT Min', 'Flow IAT Mean', 'Bwd Packet Length Std',
                   'Subflow Fwd Bytes', 'Flow Duration', 'Flow IAT Std', 'Active Min','Active Mean', 'Bwd IAT Mean',
                   'Subflow Bwd Bytes', 'Init_Win_bytes_forward', 'ACK Flag Count','Fwd PSH Flags','SYN Flag Count',
                   'Flow Packets/s', 'PSH Flag Count', 'Average Packet Size']
indices = range(len(chosen_connection_features))
zip_iterator = zip(chosen_connection_features, indices)
chosen_features_dict = dict(zip_iterator)

# --------------------------------------
# Apply a z-score normalization to all the features that have a non-zero normalization parameter (precomputed)
def normalization_function(feature, labels):
    for name in features:
        if name in chosen_connection_features and (name+'_mean') in params['PARAMS'] and float(params['PARAMS'][name + '_mean']) != 0:
            idx = chosen_features_dict[name]
            feature['feature_connection'][idx] = feature['feature_connection'][idx] - float(params['PARAMS'][name + '_mean']) / float(params['PARAMS'][name + '_std'])

    return feature, labels

# Given a trace and a feature name, returns the specific feature value of such trace
def get_feature(trace, feature_name, parse=True):
    if parse:
        if feature_name == 'Label':
            attack = trace[-1]
            attack_encoding = attacks_dict.get(attack)

            return attack_encoding
        else:
            idx = features_dict[feature_name]

            feature = trace[idx]
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

# Constructs a dictionary with all the chosen features of a connection node
# Additionally, we add each of their labels, and a type (indicating the type of node)
def get_connection_features(trace, final_feature, type):
    connection_features = {}
    for f in chosen_connection_features:
        connection_features[f] = get_feature(trace, f)

    connection_features['Label'] = final_feature
    connection_features['type'] = type
    return connection_features

# Given a flow trace, it produces the corresponding Networkx graph (incorporating all the nodes described in the paper, along with their corresponding features, if any)
def traces_to_graph(traces):
    G = nx.MultiGraph()

    # iterate over all captured flows
    n = len(traces)
    for i in range(n):
        trace = traces[i]
        dst_name = 'Destination IP'
        src_name = 'Source IP'

        # create the source node (if not created already)
        if get_feature(trace, src_name) not in G.nodes():
            G.add_node(get_feature(trace, src_name, parse=False), type=1)

        # create the destination node (if not created already)
        if get_feature(trace, dst_name, parse=False) not in G.nodes():
            G.add_node(get_feature(trace, dst_name, parse=False), type=1)

        # get the one hot encoding of the labels
        label_num = get_feature(trace, 'Label')
        final_label = np.zeros(15)
        if label_num != -1: # if it is an attack
            final_label[label_num] = 1

        # create the connection nodes with their corresponding features
        connection_features = get_connection_features(trace, final_label, 2)
        G.add_node('con_' + str(i), **connection_features)

        # add the edges that connect the connection nodes with the corresponding source and destination IP node
        G.add_edge('con_' + str(i), get_feature(trace, dst_name, parse=False))
        G.add_edge('con_' + str(i), get_feature(trace, src_name, parse=False))
        G.add_edge(get_feature(trace, dst_name, parse=False), 'con_' + str(i))
        G.add_edge(get_feature(trace, src_name, parse=False), 'con_' + str(i))
    return G

# Assigns an index to each of the nodes in the graph (depending on the type of node that it represents)
def assign_indices(G):
    indices_ip = {}
    indices_connection = {}
    counter_ip = 0
    counter_connection = 0

    for v in G.nodes():
        if G.nodes()[v]['type'] == 1:
            if v not in indices_ip:
                indices_ip[v] = counter_ip
                counter_ip += 1
        else:
            if v not in indices_connection:
                indices_connection[v] = counter_connection
                counter_connection += 1
    return indices_ip, indices_connection, counter_ip, counter_connection

# Return arrays that indicate, for each of the edges of the graph, the corresponding index of the src and dst node (endpoints).
# These arrays also differentiate between the type of nodes in the endpoints, effectively creating ip_to_connection and connection_to_ip arrays.
def process_adjacencies(G):
    indices_ip, indices_connection, counter_ip, counter_connection = assign_indices(G)
    src_ip_to_connection, dst_ip_to_connection, src_connection_to_ip, dst_connection_to_ip  = [], [], [], []

    # iterate over all the edges in the graph
    for e in G.edges():
        if 'con' not in e[0]:   # if it is an edge ip->connection
            ip_node = e[0] # capture the source node (ip node)
            connection_node = e[1]  # capture the destination node (connection node)

            src_ip_to_connection.append(indices_ip[ip_node])    # indices of the source nodes of these edges
            dst_ip_to_connection.append(indices_connection[connection_node]) # indices of the dest nodes of these edges
        
        else:   # if it is an edge connection->ip
            ip_node = e[1]  # capture the destination node (ip node)
            connection_node = e[0] # capture the source node (connection node)

            src_connection_to_ip.append(indices_connection[connection_node])  # indices of the source nodes of these edges
            dst_connection_to_ip.append(indices_ip[ip_node])  # indices of the dest nodes of these edges

    return src_ip_to_connection, dst_ip_to_connection, src_connection_to_ip, dst_connection_to_ip, counter_ip, counter_connection


# This function returns the final dictionary that includes all the relevant information of a given networkx graph (which is then fed to the model)
def graph_to_dict(G):
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

    # obtain the adjacencies of the graph (separated by endpoint types)
    src_ip_to_connection, dst_ip_to_connection, src_connection_to_ip, dst_connection_to_ip, n_i, n_c = process_adjacencies(G)

    features = {
        'feature_connection': connection_features,
        'n_i': n_i,
        'n_c': n_c,
        'src_ip_to_connection': src_ip_to_connection,
        'dst_ip_to_connection': dst_ip_to_connection,
        'src_connection_to_ip': src_connection_to_ip,
        'dst_connection_to_ip': dst_connection_to_ip
    }
    features, label =  normalization_function(features, label)  # apply the normalization function to the features
    return (features, label)



from random import random
def generator(path):
    path = path.decode('utf-8')
    files = glob.glob(path + '*.csv')

    for file in files:
        with open(file, encoding="utf8", errors='ignore') as csvfile:
            data = csv.reader(csvfile, delimiter=',', quotechar='|')

            current_time_traces = []
            counter = 0
            for row in data:
                if len(row) > 1:
                    # this separates the given flows with fixed windows of 200 flows. Notice that this is a somewhat arbitrary criterion, as the number could be different (or depend on the flow's time, so as to consider time-based windows)
                    if counter >= 200:
                        if current_time_traces != []:
                            #try:
                                G = traces_to_graph(current_time_traces)    # produce the corresponding graph
                                features, label = graph_to_dict(G)  # obtain its corresponding features and labels, which can be fed to the model

                                # check if the graph contains only benign samples
                                benign_only = np.array([l == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0] for l in label])

                                # do random undersampling (discard 90% of graphs containing only benign flows)
                                if benign_only.all():
                                    if random() < 0.1:
                                        yield (features, label)
                                else:
                                    yield (features, label)
                            #except:
                            #    print("A sample was discarted")

                        counter = 0
                        current_time_traces = []
                    else:
                        # include this flow to the current window
                        current_time_traces.append(row)
                    
                    counter += 1

# This function generates the actual generator that is fed to the model
def input_fn(data_path, val=False):
    ds = tf.data.Dataset.from_generator(generator,
                                        args=[data_path],
                                        output_types=(
                                            {'feature_connection':tf.float32,
                                             'n_i': tf.int64,
                                             'n_c': tf.int64,
                                             'src_ip_to_connection': tf.int64,
                                             'dst_ip_to_connection': tf.int64,
                                             'src_connection_to_ip': tf.int64,
                                             'dst_connection_to_ip': tf.int64}, tf.float32),
                                        output_shapes=(
                                            {
                                            'feature_connection':tf.TensorShape(None),
                                            'n_i': tf.TensorShape([]),
                                            'n_c': tf.TensorShape([]),
                                            'src_ip_to_connection': tf.TensorShape(None),
                                            'dst_ip_to_connection': tf.TensorShape(None),
                                            'src_connection_to_ip': tf.TensorShape(None),
                                            'dst_connection_to_ip': tf.TensorShape(None)}, tf.TensorShape(None))
                                        )

    #ds = ds.map(lambda x, y: normalization_function(x, y,gen), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    if not val:
        ds = ds.repeat()
    
    return ds
