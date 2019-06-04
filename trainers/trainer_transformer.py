import numpy as np
import tensorflow as tf
from tqdm import tqdm
from trainers.trainer import Trainer

class TrainerTransformer(Trainer):
    def __init__(self, config, vocab):
        super(TrainerTransformer, self).__init__(config, vocab)

    def train(self, model, dataset_train, dataset_test):
        pass


    def create_padding_mask(self, seq):
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

        # add extra dimensions so that we can add the padding
        # to the attention logits.
        return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

    #masks all the future tokens because they are irrelevant for current token - creates a triangular matrix because for the first word all other words are masked, for the second only the first word is not masked, and so on
    def create_look_ahead_mask(size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask  # (seq_len, seq_len)