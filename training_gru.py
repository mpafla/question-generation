from __future__ import absolute_import, division, print_function, unicode_literals

import pickle
from tensorflow.python.keras.utils import *
from utils.preprocessing import *
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.layers import Embedding, Input, Add, Flatten, Bidirectional, CuDNNGRU, Concatenate, Dense
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.losses import CategoricalCrossentropy
from tensorflow.python.keras.metrics import Mean, CategoricalAccuracy
from tensorflow.python.keras.optimizer_v2.rmsprop import RMSprop

import tensorflow

import numpy as np
from collections import Counter
import tensorflow as tf

from gensim.models import Word2Vec
from data_generator import DataGenerator


VALIDATION_SAMPLE_SIZE = 1000
BATCH_SIZE = 32
SEQUENCE_LIMIT = 50
GRU_HIDDEN_STATES = 512
EPOCHS = 150
LEARNING_RATE = 0.001

with open("data/questions_indexed.pickle", 'rb') as handle:
    questions_indexed = pickle.load(handle)

with open("data/answers_indexed.pickle", 'rb') as handle:
    answers_indexed = pickle.load(handle)

with open("data/sources_indexed.pickle", 'rb') as handle:
    sources_indexed = pickle.load(handle)

with open("data/questions_input_indexed.pickle", 'rb') as handle:
    questions_input_indexed = pickle.load(handle)

with open("data/embedding_matrix.pickle", 'rb') as handle:
    embedding_matrix = pickle.load(handle)

vocabulary_size = embedding_matrix.shape[0]
print("Vocabulary size: {}".format(vocabulary_size))

w2v_dim = embedding_matrix.shape[1]
print("Embeddings dimension: {}".format(w2v_dim))


def element_length_fn(x, y):
    return tf.shape(x["input_3"])[0]

bucket_boundaries = list(range(1, SEQUENCE_LIMIT + 1))
bucket_batch_sizes = [BATCH_SIZE] * (len(bucket_boundaries) + 1)

def createDataset():
    def generator():
        for question, answer, source, questions_input in zip(questions_indexed, answers_indexed, sources_indexed, questions_input_indexed):
            yield {"input_1": answer, "input_2": source, "input_3": questions_input}, to_categorical(question, num_classes=vocabulary_size)

    dataset = tf.data.Dataset.from_generator(generator,
                                             output_shapes=({'input_1': [None], 'input_2': [None], 'input_3': [None]}, [None, None]),
                                             output_types=({"input_1": tf.int32, "input_2": tf.int32, "input_3": tf.int32}, tf.int32))

    padding_value = special_token2index(vocabulary_size-4, PAD)
    dataset = dataset.apply(tf.data.experimental.bucket_by_sequence_length(element_length_func=element_length_fn,
                                                                           bucket_batch_sizes=bucket_batch_sizes,
                                                                          bucket_boundaries=bucket_boundaries,
                                                                           padding_values=({"input_1": padding_value, "input_2": padding_value, "input_3": padding_value}, padding_value)))

    dataset = dataset.prefetch(BATCH_SIZE)
    return dataset

dataset = createDataset()



test_dataset = dataset.take(VALIDATION_SAMPLE_SIZE)
train_dataset = dataset.skip(VALIDATION_SAMPLE_SIZE)
#train_dataset = train_dataset.take(2)

max_sequence_length = len(questions_indexed[0])

def createEmbeddingLayer():
    embedding_layer = Embedding(vocabulary_size,
                      w2v_dim,
                      mask_zero=False,
                      weights=[embedding_matrix],
                      trainable=False)
    return embedding_layer


def createModel():
    encoder_input_answers = Input(shape=(None,))
    encoder_input_sources = Input(shape=(None,))
    encoder_embedding_answers = createEmbeddingLayer()(encoder_input_answers)
    encoder_embedding_source = createEmbeddingLayer()(encoder_input_sources)
    encoder_input = Add()([encoder_embedding_answers, encoder_embedding_source])
    encoder_gru = CuDNNGRU(GRU_HIDDEN_STATES, return_state=True)
    _, state_h_encoder = encoder_gru(encoder_input)
    #encoder_states = [state_h, state_c]



    decoder_input_questions = Input(shape=(None,))
    decoder_embedding_questions = createEmbeddingLayer()(decoder_input_questions)
    decoder_gru = CuDNNGRU(GRU_HIDDEN_STATES, return_sequences=True, return_state=True)
    decoder_outputs, _ = decoder_gru(decoder_embedding_questions,
                                         initial_state=state_h_encoder)
    #decoder_outputs_flattend = Flatten()(decoder_outputs)
    decoder_dense = Dense(vocabulary_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_input_answers, encoder_input_sources, decoder_input_questions], decoder_outputs)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    model.summary()
    return model

model = createModel()

#model.fit(train_dataset,
#          epochs=EPOCHS)

#model.save('s2s.h5')



loss_function = CategoricalCrossentropy()
optimizer = RMSprop(lr=LEARNING_RATE)


train_loss = Mean(name='train_loss')
train_accuracy = CategoricalAccuracy(name='train_accuracy')

test_loss = Mean(name='test_loss')
test_accuracy = CategoricalAccuracy(name='test_accuracy')

#@tf.function
def train_step(input, target):
    with tf.GradientTape() as tape:
        predictions = model(input)
        loss = loss_function(target, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(target, predictions)

#@tf.function
def test_step(input, target):
    predictions = model(input)
    t_loss = loss_function(target, predictions)

    test_loss(t_loss)
    test_accuracy(target, predictions)

def diagnostics(epoch, step):
    template = 'Epoch: {}/{}, Step: {}/{}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    diagnostics = template.format(epoch + 1,
                                  EPOCHS,
                                  step,
                                  int((len(questions_indexed) - (VALIDATION_SAMPLE_SIZE * BATCH_SIZE)) / BATCH_SIZE),
                                  train_loss.result(),
                                  train_accuracy.result() * 100,
                                  test_loss.result(),
                                  test_accuracy.result() * 100)
    return diagnostics


print("Start training")
for epoch in range(EPOCHS):
    step = 0

    for input, target in train_dataset:
        train_step(input, target)

        print(diagnostics(epoch, step))
        step = step + 1


    for test_input, test_target in test_dataset:
        test_step(test_input, test_target)

    with open("models/diagnostics.txt", "a") as text_file:
        print(diagnostics(epoch, step), file=text_file)

    model_path = "models/s2s_gru{}_epoch{}.h5".format(GRU_HIDDEN_STATES, epoch)
    model.save(model_path)




