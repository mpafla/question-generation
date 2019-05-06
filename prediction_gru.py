

from __future__ import absolute_import, division, print_function, unicode_literals

import pickle
from tensorflow.python.keras.models import load_model
from utils.preprocessing import *
import csv
import numpy as np
from tensorflow.python.keras.utils import *
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.layers import Embedding, Input, Add, Flatten, Bidirectional,Concatenate, Dense
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

#special_tokens = [UNK, EOS, PAD, SOS]

max_sentence_len = 144
GRU_HIDDEN_STATES = 512


with open("data/answers.pickle", 'rb') as handle:
    answers = pickle.load(handle)

with open("data/sources.pickle", 'rb') as handle:
    sources = pickle.load(handle)

#Embeddings
w2v_path = "models/word2vec.model"

try:
    print("Loading word2vec model")
    w2v_model = Word2Vec.load(w2v_path)
    print("Loaded word2vec model")
except:
    print("Loading of word2vec failed")

vocab = w2v_model.wv.vocab
vocabulary_size = len(vocab)

model = load_model('models/s2s_gru512_epoch2.h5')



encoder_inputs = [model.get_layer("input_1").output, model.get_layer("input_2").output]
encoder_outputs, state_h = model.get_layer("cu_dnngru").output
encoder_states = state_h
encoder_model = Model(encoder_inputs, encoder_states)

#print("ENCODER")
#for layer in encoder_model.layers:
#    print(layer.name)

decoder_inputs = model.get_layer("input_3").output
decoder_embedding = model.get_layer("embedding_2")(decoder_inputs)
#decoder_inputs = model.get_layer("embedding_2").output
decoder_state_input_h = Input(shape=(GRU_HIDDEN_STATES,))
decoder_gru = model.get_layer("cu_dnngru_1")
decoder_outputs, state_h= decoder_gru(
    decoder_embedding, initial_state=decoder_state_input_h)
decoder_states = state_h
decoder_dense = model.get_layer("dense")
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs, decoder_state_input_h],
    [decoder_outputs, decoder_states])


#decoder_model.summary()

#print("DECODER")
#for layer in decoder_model.layers:
#    print(layer.name)






def decode_sequence(source, answer):
    preprocess_functions = [word_tokenize, removePunctuation, removeEmptyTokens, lowerCase, removeStopWords]
    source = reduce(lambda x, y: y(x), preprocess_functions, source)
    answer = reduce(lambda x, y: y(x), preprocess_functions, answer)

    source_indexed = getIndexesForSentence(source, w2v_model)
    answer_indexed = getIndexesForSentence(answer, w2v_model)

    answer_indexed = addEOS(answer_indexed, vocabulary_size)
    source_indexed = addEOS(source_indexed, vocabulary_size)

    if (len(answer_indexed) < len(source_indexed)):
        padSequence(answer_indexed, len(source_indexed), vocabulary_size)
    elif (len(answer_indexed) > len(source_indexed)):
        padSequence(source_indexed, len(answer_indexed), vocabulary_size)

    # Encode the input as state vectors.
    states_value = encoder_model.predict([source_indexed, answer_indexed])
    target_question = [[special_token2index(vocabulary_size, SOS)]]

    stop_condition = False
    decoded_question = []
    while not stop_condition:
        output_tokens, h = decoder_model.predict(
            [target_question] + [states_value])

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = index2token(w2v_model, sampled_token_index)
        decoded_question = decoded_question + [sampled_word]

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_word == EOS or
                len(decoded_question) > max_sentence_len):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_question[0].append(sampled_token_index)

        # Update states
        states_value = h
    return decoded_question

source_example = "Canada is a parliamentary democracy and a constitutional monarchy in the Westminster tradition, with Elizabeth II as its queen and a prime minister who serves as the chair of the federal cabinet and head of government."
answer_example = "a parliamentary democracy"

decoded = decode_sequence(source_example, answer_example)
print(source_example)
print(decoded)
print(answer_example)

source_example = "Marvin is a human."
answer_example = "a human"

decoded = decode_sequence(source_example, answer_example)
print(source_example)
print(decoded)
print(answer_example)

with open('data/data_cleaned.csv', encoding="utf8") as csvfile:
    reader = csv.reader(csvfile, delimiter=',')

    # Get rid of rowname
    next(reader)

    for i, row in enumerate(reader):
        question_example = row[0]
        answer_example = row[1]
        source_example = row[2]

        decoded = decode_sequence(source_example, answer_example)
        print(source_example)
        print(answer_example)
        print(question_example)
        print(decoded)

        if i > 100:
            break

