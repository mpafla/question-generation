import numpy as np
import tensorflow as tf
from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.point_wise_feed_forward_network import PointWiseFeedForwardNetwork

class Transformer():
    def __init__(self, config, embeddings_matrix):
        pass


    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    #https://datascience.stackexchange.com/questions/51065/what-is-positional-encoding-in-transformer-model
    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                np.arange(d_model)[np.newaxis, :],
                                d_model)

        # apply sin to even indices in the array; 2i
        #0::2 means to start from index 0 and then always jump 2
        sines = np.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        # 1::2 means to start from index 1 and then always jump 2
        cosines = np.cos(angle_rads[:, 1::2])

        #this concatonates rather than interleaving
        pos_encoding = np.concatenate([sines, cosines], axis=-1)

        pos_encoding = pos_encoding[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)