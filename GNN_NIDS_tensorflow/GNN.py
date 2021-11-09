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

class GNN(tf.keras.Model):

    def __init__(self, config):
        super(GNN, self).__init__()

        # Configuration dictionary. It contains the needed Hyperparameters for the model.
        # All the Hyperparameters can be found in the config.ini file
        self.config = config

        # GRU Cells used in the Message Passing step
        self.ip_update = tf.keras.layers.GRUCell(int(self.config['HYPERPARAMETERS']['node_state_dim']), name='update_ip')
        self.connection_update = tf.keras.layers.GRUCell(int(self.config['HYPERPARAMETERS']['node_state_dim']), name='update_connection')

        self.message_func1 = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=int(self.config['HYPERPARAMETERS']['node_state_dim'])*2 ),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(int(self.config['HYPERPARAMETERS']['node_state_dim']),
                                      activation=tf.nn.relu)
            ]
        )
        self.message_func2 = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=int(self.config['HYPERPARAMETERS']['node_state_dim'])*2 ),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(int(self.config['HYPERPARAMETERS']['node_state_dim']),
                                      activation=tf.nn.relu)
            ]
        )


        # Readout Neural Network. It expects as input the path states and outputs the per-path delay
        self.readout = tf.keras.Sequential([
            tf.keras.layers.Input(shape=int(self.config['HYPERPARAMETERS']['node_state_dim'])),
            tf.keras.layers.Dense(128,
                                  activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(64,
                                  activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(15, activation = tf.nn.softmax)
        ])

    @tf.function
    def call(self, inputs):
        # connection features
        feature_connection = tf.squeeze(inputs['feature_connection'])

        # number of ip
        n_ips = inputs['n_i']

        # number of connections
        n_connections = inputs['n_c']

        # adjacencies
        src_ip_to_connection = tf.squeeze(inputs['src_ip_to_connection'])
        dst_ip_to_connection = tf.squeeze(inputs['dst_ip_to_connection'])
        src_connection_to_ip = tf.squeeze(inputs['src_connection_to_ip'])
        dst_connection_to_ip = tf.squeeze(inputs['dst_connection_to_ip'])

        # CREATE THE IP NODES
        #Encode only ones in the IP states
        ip_state = tf.ones((n_ips, int(self.config['HYPERPARAMETERS']['node_state_dim'])))


        # CREATE THE CONNECTION NODES
        # Compute the shape for the  all-zero tensor for link_state
        shape = tf.stack([
            n_connections,
            int(self.config['HYPERPARAMETERS']['node_state_dim']) - 26
        ], axis=0)

        # Initialize the initial hidden state for id nodes
        connection_state = tf.concat([
            feature_connection,
            tf.zeros(shape)
        ], axis=1)


        # MESSAGE PASSING: ALL with ALL simoultaniously
        # We simply use sum aggregation for all, RNN for the update. The messages are formed with the source and edge parameters (NN)
        # Iterate t times doing the message passing
        for _ in range(int(self.config['HYPERPARAMETERS']['t'])):
            # IP to CONNECTION
            # compute the hidden-states
            ip_node_gather = tf.gather(ip_state, src_ip_to_connection)
            connection_gather = tf.gather(connection_state, dst_ip_to_connection)
            connection_gather = tf.squeeze(connection_gather)
            ip_gather = tf.squeeze(ip_node_gather)
            
            # apply the message function on the ip nodes
            nn_input = tf.concat([ip_gather, connection_gather], axis=1) #([port1, ... , portl, param1, ..., paramk])
            nn_input = tf.ensure_shape(nn_input,[None, int(self.config['HYPERPARAMETERS']['node_state_dim'])*2])
            ip_message = self.message_func1(nn_input)

            # apply the aggregation function on the ip nodes
            ip_mean = tf.math.unsorted_segment_mean(ip_message, dst_ip_to_connection, n_connections)

            # CONNECTION TO IP
            # compute the hidden-states
            connection_node_gather = tf.gather(connection_state, src_connection_to_ip)
            ip_gather = tf.gather(ip_state, dst_connection_to_ip)
            ip_gather = tf.squeeze(ip_gather)
            connection_gather = tf.squeeze(connection_node_gather)

            # apply the message function on the connection nodes
            nn_input = tf.concat([connection_gather, ip_gather], axis=1)
            nn_input = tf.ensure_shape(nn_input, [None, int(self.config['HYPERPARAMETERS']['node_state_dim']) * 2])
            connection_messages = self.message_func2(nn_input)

            # apply the aggregation function on the connection nodes
            connection_mean = tf.math.unsorted_segment_mean(connection_messages, dst_connection_to_ip, n_ips)


            # UPDATE (both IP and connection simoultaniously)
            #update of ip nodes
            connection_mean = tf.ensure_shape(connection_mean, [None, int(self.config['HYPERPARAMETERS']['node_state_dim'])])
            ip_state, _ = self.ip_update(connection_mean, [ip_state])
            
            #update of connection nodes
            ip_mean = tf.ensure_shape(ip_mean, [None, int(self.config['HYPERPARAMETERS']['node_state_dim'])])
            connection_state, _ = self.connection_update(ip_mean, [connection_state])

        # apply the feed-forward nn
        nn_output = self.readout(connection_state)
        return nn_output

