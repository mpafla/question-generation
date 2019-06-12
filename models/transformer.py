import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense
from models.layers.encoder import Encoder
from models.layers.decoder import Decoder


class Transformer(Model):
    def __init__(self, config, embeddings_layer, vocab):
        super(Transformer, self).__init__()
        self.number_of_features = config["model"]["number_of_features"]
        self.num_layers = config["model"]["num_layers"]
        self.d_model = config["model"]["d_model"]
        self.num_heads = config["model"]["num_heads"]
        self.dff = config["model"]["dff"]
        self.input_vocab_size = vocab.get_vocab_size()
        self.target_vocab_size = vocab.get_vocab_size()
        self.dropout_rate = config["model"]["dropout_rate"]
        self.embeddings_layer = embeddings_layer

        self.encoder = Encoder(self.num_layers, self.d_model, self.number_of_features, self.num_heads, self.dff,
                               self.input_vocab_size, self.dropout_rate)

        self.decoder = Decoder(self.num_layers, self.d_model, self.num_heads, self.dff,
                               self.target_vocab_size, self.dropout_rate)

        self.final_layer = Dense(self.target_vocab_size)

    def call(self, inp, features, tar, training, enc_padding_mask,
             look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(inp, features, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights