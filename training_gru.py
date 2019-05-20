from __future__ import absolute_import, division, print_function, unicode_literals

import pickle
import os
import time
from tensorflow.python.keras.utils import *
from utils.preprocessing import *
from bahdanau_attention import BahdanauAttention
from bahdanau_attention_layer import BahdanauAttentionLayer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.layers import Embedding, Input, Add, Flatten, Bidirectional, CuDNNGRU, Concatenate, Dense
from tensorflow.python.keras.models import Model, save_model, model_from_yaml
from tensorflow.python.saved_model.save import save
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
BATCH_SIZE = 4
SEQUENCE_LIMIT = 50
GRU_HIDDEN_STATES = 512
EPOCHS = 150
LEARNING_RATE = 0.001
ATTENTION_LAYER_UNITS = 10

with open("data/questions_indexed.pickle", 'rb') as handle:
    questions_indexed = pickle.load(handle)

with open("data/answers_indexed.pickle", 'rb') as handle:
    answers_indexed = pickle.load(handle)

with open("data/sources_indexed.pickle", 'rb') as handle:
    sources_indexed = pickle.load(handle)

#with open("data/questions_input_indexed.pickle", 'rb') as handle:
#    questions_input_indexed = pickle.load(handle)

with open("data/embedding_matrix.pickle", 'rb') as handle:
    embedding_matrix = pickle.load(handle)

vocabulary_size = embedding_matrix.shape[0]
print("Vocabulary size: {}".format(vocabulary_size))

w2v_dim = embedding_matrix.shape[1]
print("Embeddings dimension: {}".format(w2v_dim))


def element_length_fn(x, y):
    return tf.shape(y)[0]

bucket_boundaries = list(range(1, SEQUENCE_LIMIT + 1))
bucket_batch_sizes = [BATCH_SIZE] * (len(bucket_boundaries) + 1)

def createDataset():
    def generator():
        for question, answer, source in zip(questions_indexed, answers_indexed, sources_indexed):
            yield {"input_1": answer, "input_2": source, "input_3": question}, to_categorical(question,
                                                                                                     num_classes=vocabulary_size)

    dataset = tf.data.Dataset.from_generator(generator,
                                             output_shapes=(
                                             {'input_1': [None], 'input_2': [None], 'input_3': [None]}, [None, None]),
                                             output_types=(
                                             {"input_1": tf.int32, "input_2": tf.int32, "input_3": tf.int32}, tf.int32))

    padding_value = special_token2index(vocabulary_size - 4, PAD)
    dataset = dataset.apply(tf.data.experimental.bucket_by_sequence_length(element_length_func=element_length_fn,
                                                                           bucket_batch_sizes=bucket_batch_sizes,
                                                                           bucket_boundaries=bucket_boundaries,
                                                                           padding_values=({"input_1": padding_value,
                                                                                            "input_2": padding_value,
                                                                                            "input_3": padding_value},
                                                                                           padding_value)))

    dataset = dataset.prefetch(BATCH_SIZE)
    return dataset

dataset = createDataset()



test_dataset = dataset.take(VALIDATION_SAMPLE_SIZE)
train_dataset = dataset.skip(VALIDATION_SAMPLE_SIZE)
#train_dataset = train_dataset.take(2)
#test_dataset = test_dataset.take(2)

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
    encoder_gru = CuDNNGRU(GRU_HIDDEN_STATES,return_sequences=True, return_state=True)
    encoder_output, state_h_encoder = encoder_gru(encoder_input)


    #encoder_states = [state_h, state_c]



    decoder_input_questions = Input(shape=(None,))
    decoder_embedding_questions = createEmbeddingLayer()(decoder_input_questions)

    decoder_hidden_states_input = Input(shape=(GRU_HIDDEN_STATES,))
    #print(decoder_hidden_states_input.shape)
    decoder_encoder_output_input = Input(shape=(None, encoder_output.shape[2]))
    #print(decoder_encoder_output_input.shape)
    attention_layer = BahdanauAttentionLayer(ATTENTION_LAYER_UNITS)
    context_vector, attention_weights = attention_layer((decoder_hidden_states_input, decoder_encoder_output_input))

    #context_vector = tf.reshape(context_vector,
    #                           [tf.shape(decoder_embedding_questions)[0], tf.shape(decoder_embedding_questions)[1], context_vector.shape[1]])
    #print(context_vector.shape )
    #print(decoder_embedding_questions.shape)
    decoder_inputs = tf.concat([tf.expand_dims(context_vector, 1), decoder_embedding_questions], axis=-1)
    #print(decoder_inputs.shape)
    #decoder_input_attention = tf.concat([context_vector, decoder_embedding_questions],axis=-1)


    decoder_gru = CuDNNGRU(GRU_HIDDEN_STATES, return_sequences=True, return_state=True)
    decoder__gru_outputs, decoder_gru_hidden = decoder_gru(decoder_inputs)
    decoder__gru_outputs = tf.reshape(decoder__gru_outputs, (-1, decoder__gru_outputs.shape[2]))
    #decoder_outputs_flattend = Flatten()(decoder_outputs)
    decoder_dense = Dense(vocabulary_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder__gru_outputs)

    #model = Model([encoder_input_answers, encoder_input_sources, decoder_input_questions], decoder_outputs)
    #model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    #model.summary()

    encoder = Model([encoder_input_answers, encoder_input_sources], [encoder_output, state_h_encoder])
    encoder.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    #encoder.summary()

    decoder = Model([decoder_input_questions, decoder_hidden_states_input, decoder_encoder_output_input], [decoder_outputs, decoder_gru_hidden])
    decoder.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    #decoder.summary()


    return encoder, decoder

encoder, decoder = createModel()

#encoder_yaml = encoder.to_yaml()
#decoder_yaml = decoder.to_yaml()

#with open("models/encoder.yaml", "w") as yaml_file:
#    yaml_file.write(encoder_yaml)

#with open("models/decoder.yaml", "w") as yaml_file:
#    yaml_file.write(decoder_yaml)

loss_function = CategoricalCrossentropy()
optimizer = RMSprop(lr=LEARNING_RATE)


train_loss = Mean(name='train_loss')
#Calculates the accuracy by dividing the correct classification by the all instances (independent from sequence length)
train_accuracy = CategoricalAccuracy(name='train_accuracy')

test_loss = Mean(name='test_loss')
test_accuracy = CategoricalAccuracy(name='test_accuracy')

checkpoint_dir = 'models'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

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

def run_through_step(input, target_one_hot, training=True):
    loss = 0

    target = input["input_3"]

    with tf.GradientTape() as tape:


        enc_output, enc_hidden = encoder.call([input["input_1"], input["input_2"]])

        dec_hidden = enc_hidden

        dec_input = tf.expand_dims([special_token2index(vocabulary_size - 4, SOS)] * target.shape[0], 1)
        #dec_input = to_categorical([special_token2index(vocabulary_size - 4, SOS)] * BATCH_SIZE)

        target_predictions = np.zeros(target_one_hot.shape)

        # Teacher forcing - feeding the target as the next input
        for t in range(target.shape[1]):
            # passing enc_output to the decoder

            predictions, dec_hidden = decoder.call([dec_input, dec_hidden, enc_output])

            #print(predictions.shape)
            #print(target_predictions[:, t].shape)
            target_predictions[:, t] = predictions

            loss += loss_function(target_one_hot[:, t], predictions)

            #if training:
            #    train_accuracy(target_one_hot[:, t], predictions)
            #else:
            #    test_accuracy(target_one_hot[:, t], predictions)

            # using teacher forcing
            dec_input = tf.expand_dims(target[:, t], 1)


    batch_loss = (loss / int(target.shape[1]))

    if training:
        try:
            variables = encoder.trainable_variables + decoder.trainable_variables
            gradients = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(gradients, variables))
            train_accuracy(target_one_hot, target_predictions)
            train_loss(batch_loss)
        except:
            print("Optimazation could not be performed")
    else:
        test_loss(batch_loss)
        test_accuracy(target_one_hot, target_predictions)

    return (target_predictions)


print("Start training")
for epoch in range(EPOCHS):
    step = 0
    start = time.time()

    print("Training the model")
    for input, target in train_dataset:
        run_through_step(input, target, training=True)
        step = step + 1


        #if (step % 10 == 0):
        #    print(diagnostics(epoch, step))
        print(diagnostics(epoch, step))

    print("Getting accuracy from test dataset")
    for test_input, test_target in test_dataset:
        run_through_step(test_input, test_target, training=False)

    print(diagnostics(epoch, step))
    print('Time taken for training this epoch is {} sec'.format(time.time() - start))

    with open("models/diagnostics.txt", "a") as text_file:
        print(diagnostics(epoch, step), file=text_file)

    #model_path = lambda model: "models/s2s_{}_gru{}_epoch{}.h5".format(model, GRU_HIDDEN_STATES, epoch)
    #encoder_path = model_path("encoder")
    #decoder_path = model_path("decoder")

    #encoder.save_weights(encoder_path)
    #decoder.save_weights(encoder_path)

    #new_encoder, new_decoder = createModel()

    #encoder.load_weights(encoder_path)
    #new_decoder.load_weights(decoder_path)

    #save(decoder, decoder_path)

    #save_model(decoder, model_path, )
    #print(decoder.to_yaml())
    if (epoch % 10 == 0):
        checkpoint.save(file_prefix=checkpoint_prefix)



    #new_encoder, new_decoder = createModel()
    #new_optimizer = RMSprop(lr=LEARNING_RATE)

    #test_checkpoint = tf.train.Checkpoint(optimizer=new_optimizer,
    #                                      encoder=new_encoder,
    #                                      decoder=new_decoder)


    #test_checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    #for input, target in test_dataset:
    #    pred1 = run_through_step(input, target, training=False)
    #    encoder = new_encoder
    #    decoder = new_decoder
    #    pred2 = run_through_step(input, target, training=False)
    #    print(tf.debugging.assert_equal(pred1, pred2))


