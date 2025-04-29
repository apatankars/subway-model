import tensorflow as tf
import keras
from architecture.spatial_layers import GCN

class STBlock(keras.Model):
    def __init__(self, input_features: int, output_features: int, **kwargs):
        '''
        spatio-temporal conv block
        input_features: num of input feats
        output_features: num of output feats
        '''
        super().__init__(**kwargs)

        self.hidden_sizes =  [(input_features * (4 - i) + output_features * i) // 4 for i in (1, 4)]

        self.spatial_embedding = GCN(self.hidden_sizes, output_features)
        self.temporal_embedding = tf.keras.layers.Conv1D(
            filters=output_features,
            kernel_size=3,
            padding="same",
            data_format="channels_last"
            # data_format="channels_first"  # uncomment for (B, C, T), as opposed to default (B, T, C)
            )

    def call(self, inputs, training=False):
        '''
        inputs: tuple(temporal_features, adjacency)
                adjacency: graph connectivity (pass intoo to GCN)
                temporal_features: tensor shaped [n_nodes, input_features, time_steps]
        :return: tensor shaped [n_nodes, output_features, time_steps]
        '''
        temporal_features,adjacency,  = inputs
        # swap shape to be [n_nodes, time_steps, input_features]
        x = tf.transpose(temporal_features, perm=[0, 2, 1])
        # aply GCN per time slice
        time_steps = tf.shape(x)[1]
        spatial_outputs = []
        for t in range(time_steps):
            feat_t = x[:, t, :] # [n_nodes, input_features]
            out_t = self.spatial_embedding((feat_t, adjacency), training=training)
            spatial_outputs.append(out_t)  # each out_t is [n_nodes, output_features]
        # stack into [n_nodes, time_steps, output_features]
        x_spatial = tf.stack(spatial_outputs, axis=1)
        # temp conv, with input [batch=n_nodes, length=time_steps, channels=output_features]
        x_temporal = self.temporal_embedding(x_spatial)

        return tf.transpose(x_temporal, perm=[0, 2, 1])
    
class StackedSTBlocks(keras.layers.Layer):
    '''
    seq of STBlocks with feature concatenation (i.e. residual stacking).
    '''
    def __init__(self, *blocks, **kwargs):
        super().__init__(**kwargs)
        self.blocks = list(blocks) # will be STBlocks

    def call(self, inputs, training=False):
        '''
        inputs: tuple(features, adjacency)
                features: [n_nodes, input_features, time_steps]
        
        return: tensor shaped [n_nodes, output_features * num_blocks + input_features, time_steps]
        '''
        f, adjacency = inputs
        # seq run each block on input and concatenate the output for next iter
        for block in self.blocks:
            out = block((f, adjacency), training=training)  # [n_nodes, f_out, time_steps]
            f = tf.concat([f, out], axis=1)  # concat along features
        return f