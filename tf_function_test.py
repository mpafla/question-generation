import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Input, Embedding
from tensorflow.python.keras.models import Model

class EmbeddingsModel(Model):

    def __init__(self, embeddings_matrix):
        super(EmbeddingsModel, self).__init__()
        self.embeddings_matrix = embeddings_matrix



    @tf.function
    def call(self, input_data):
        tf.print(self.embeddings_matrix.shape)
        self.embeddings_layer = Embedding(3,
                                          2,
                                          mask_zero=False,
                                          weights=[embeddings_matrix],
                                          trainable=False)
        return self.embeddings_layer(input_data)

'''

embeddings_matrix = tf.Variable(embeddings_matrix)
input_layer = Input(shape=(None,))
embeddings_layer = Embedding(3,
                            2,
                            mask_zero=False,
                            weights=[embeddings_matrix],
                            trainable=False)(input_layer)

model = Model(input_layer, embeddings_layer)

'''
embeddings_matrix = np.array([[0,1], [2,3], [4,5]])

model = EmbeddingsModel(embeddings_matrix)

data = [1,2,0,1,2,1]
dataset = tf.data.Dataset.from_tensor_slices(data)

#@tf.function
def train(data, model):
    output = model(data)
    tf.print(output)

for data in dataset:
    train(data, model)

