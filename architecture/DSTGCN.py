import random
import numpy as np
import tensorflow as tf
import keras
from architecture.spatial_layers import stackedSpatialGCNs, GCN
from architecture.temporal_layers import StackedSTBlocks, STBlock

@keras.saving.register_keras_serializable(package="DSTGCN")
class DSTGCN(keras.Model):

    def __init__(self, feature_sizes, **kwargs):
        super().__init__(**kwargs)
        '''
        out_features should be num_nodes
        '''

        spatial_features, st_features, external_features, weather_features, out_features = feature_sizes

        # spatial embedding layer to embed spatial features from nodes
        self.spatial_embedding = keras.Sequential([
            keras.layers.Dense(20, activation="relu"),
            keras.layers.Dense(15, activation="relu"),
        ])


        # stacked spatial GCN blocks
        self.spatial_gcn = stackedSpatialGCNs(GCN([15, 15, 15], 15),
                                           GCN([15, 15, 15], 15),
                                           GCN([14, 13, 12, 11], 10))
        
        # embedding temporal features
        self.temporal_blocks = StackedSTBlocks(STBlock(st_features, 4), STBlock(5, 5), STBlock(10, 10))
        # average pooling of temporal 
        self.temporal_agg = keras.layers.AveragePooling1D(pool_size=24)

        embedding_sizes = [(external_features * (4 - i) + 10 * i) // 4 for i in (1, 4)]
        external_embedding_layers = [
            keras.layers.Dense(embedding_sizes[0]),
            keras.layers.Dense(embedding_sizes[1]),
            keras.layers.Dense(10),
        ]
        self.external_embedding = keras.Sequential(external_embedding_layers)

        # weather time series encoder using GRU with dropout
        self.weather_gru = keras.layers.GRU(
            units=weather_features,
            return_sequences=False,
            dropout=0.2,
            recurrent_dropout=0.2,
        )

        # classifier head
        head = [
            keras.layers.ReLU(),
            keras.layers.Dense(out_features),
        ]
        self.classifier = keras.Sequential(head)

    
    def call(self, inputs, training=False):
        '''
        inputs = tuple of five feature tensors
        spatial_features   shape=[num_nodes, spatial_dim] = node static spatial features
        temporal_features  shape=[num_nodes, temporal_dim, T] = node time series for the given time window
        external_features  shape=[1, F3] = per-graph external features
        weather_features   shape=[1,T,weather_dim] = weather series for the given time window
        A                  shape=[N, N] = adjacency of the B graphs
        '''
        spatial_features, temporal_features, external_features, weather_features, A = inputs
        N = tf.shape(A)[0]

        # spatial branch: [N, spatial_dim] --> [N, dim_GCN_output]
        embedded_spatial_features = self.spatial_embedding(spatial_features)  # [N,15]
        spatial_out = self.spatial_gcn((embedded_spatial_features, A), training=training) # [N,10]

        # temporal branch: [N, temporal_dim, T] --> [N, dim_stgcn_output, T]
        embedded_temporal_features = self.temporal_blocks((temporal_features, A), training=training) # [N,10,T]
        etf = tf.transpose(embedded_temporal_features, [0, 2, 1]) # [N,T,10]
        pooled_etf = self.temporal_agg(etf) # [N,1,10]
        etf_out = tf.squeeze(pooled_etf, axis=1) # [N,10]     

        # external static features
        external_static_embedding = self.external_embedding(external_features, training=training) # [1, external_dim]
        ese_full = tf.tile(external_static_embedding, [N, 1]) # [N, external_dim]

        # GRUUUUUU weather encoding
        weather_embedding = self.weather_gru(weather_features, training=training) # [1, weather_dim]
        weather_full = tf.tile(weather_embedding, [N, 1]) # [N, weather_dim]

        # concatenate node features
        full_features = tf.concat([spatial_out, etf_out, ese_full, weather_full], axis=-1) # [N, spatial_dim+temporal_dim+external_dim+weather_dim]

        # node level prediction
        return self.classifier(full_features, training=training) # [N, 1]
