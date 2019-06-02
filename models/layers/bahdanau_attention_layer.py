import tensorflow as tf
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.layers import Dense

class BahdanauAttentionLayer(Layer):

    def __init__(self, units, **kwargs):
        self.output_dim = 1
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)
        super(BahdanauAttentionLayer, self).__init__(**kwargs)

    def call(self, inputs):
        query, values = inputs
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, hidden_size)
        score = self.V(tf.nn.tanh(
            self.W1(values) + self.W2(hidden_with_time_axis)))

        # attention_weights shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights