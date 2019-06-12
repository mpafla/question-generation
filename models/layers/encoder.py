from tensorflow.python.keras.layers import Layer, Embedding, Dropout
from utils.positional_encoding import positional_encoding
from models.layers.encoder_layer import EncoderLayer

import tensorflow as tf


class Encoder(Layer):
    def __init__(self, num_layers, d_model, number_of_features ,num_heads, dff, input_vocab_size,
                 rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers


        #number of features have to be substracted
        self.embedding = Embedding(input_vocab_size, d_model - number_of_features)
        self.pos_encoding = positional_encoding(input_vocab_size, self.d_model - number_of_features)


        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dropout = Dropout(rate)


    def call(self, x, features, training, mask):

        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        features = tf.cast(features, tf.float32)

        x = tf.concat([x, features], axis=2)

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)