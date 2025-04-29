from spektral.layers.convolutional import GCNConv
import tensorflow as tf
import keras
from keras.layers import BatchNormalization, ReLU

class GCN(keras.Model):
    def __init__(self, hidden_sizes, out_feats):
        '''
        gcns with hidden layers

        params:
            hidden_sizes: array of sizes for hidden layers
            out_feats: number of output features
        '''
        super().__init__()
        self.layers_list = []
        # stack hidden GCN layers
        for h in hidden_sizes:
            self.layers_list += [
                GCNConv(h, activation=None, mask=None),
                BatchNormalization(),
                ReLU(),
            ]
        # final output gcn 
        self.layers_list += [
            GCNConv(out_feats, activation=None, mask=None),
            BatchNormalization(),
            ReLU(),
        ]
        # TODO DOES TF TRACK LAYERS_LIST WEIGHTS

    def call(self, inputs: tuple):
        '''
        inputs: tuple(x, a) where x=node features, a=adjacency
        '''
        x, a = inputs    
        for layer in self.layers_list:
            # GCNConv input order is [X, A]
            if isinstance(layer, GCNConv):
                x = layer([x, a], mask=None)
            else:
                x = layer(x)
        return x
    
class stackedSpatialGCNs(keras.layers.Layer):
    def __init__(self, *blocks, **kwargs):
        '''
        blocks = sequence of layers that each take (x,a) and return x+block(x,a)
        add residual connections between each block
        '''
        super().__init__(**kwargs)
        self.blocks = list(blocks)

    def call(self, inputs, training=False):
        '''
        inputs: tuple(x,a)
                x is node_features
                a is adjacency
        '''
        x, a = inputs
        # apply all but last with residual
        for block in self.blocks[:-1]:
            x = x + block((x, a), training=training)
        # apply the last one without residual
        x = self.blocks[-1]((x, a), training=training)
        return x



