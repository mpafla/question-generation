import string
import numpy as np
import spacy
import tensorflow as tf
from nltk.tokenize import word_tokenize
from functools import reduce
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
nlp = spacy.load("en_core_web_sm")


from utils.constants import Constants

PAD_index = Constants.special_token2index[Constants.PAD]
PAD = Constants.PAD

def get_trimmed_sequence(sequence, final_index, length):
    return(sequence[final_index - length:final_index])

def get_padded_sequence(sequence, final_index, length):
    sequence_length = len(sequence)
    pads_to_add_before = final_index - sequence_length
    pads_to_add_after = length - final_index

    #Check if all instances in the sequence are str and change PAD symbol accordingly
    pad_symbol = PAD if all(isinstance(elem, str) for elem in sequence) else PAD_index

    sequence = np.concatenate([[pad_symbol] * pads_to_add_before, sequence, [pad_symbol] * pads_to_add_after])
    return(sequence)

def get_length_adjusted_sequence(sequence, desired_length, padding_pos = "back", trimming_pos = "front"):
    values_to_adjust = desired_length - len(sequence)

    # padding
    if values_to_adjust > 0:
        if padding_pos == "front":
            return (get_padded_sequence(sequence, desired_length, desired_length))
        elif padding_pos == "back":
            return (get_padded_sequence(sequence, len(sequence), desired_length))
    # trimming
    elif values_to_adjust < 0:
        if trimming_pos == "front":
            return (get_trimmed_sequence(sequence, len(sequence), desired_length))
        elif trimming_pos == "back":
            return (get_trimmed_sequence(sequence, desired_length, desired_length))
    # do nothing
    else:
        return (sequence)


def get_answer_processed(answer, context):
    #answer_lemma = [token.lemma_ for token in answer]
    #context_lemma = [token.lemma_ for token in context]

    answer_start = None
    answer_end = None
    for i in range(len(context) - len(answer)):
        if answer == context[i:i+len(answer)]:
            answer_start = i
            break

    #answer_processed = np.zeros(len(context))
    answer_processed = np.array([PAD_index] * len(context))


    if answer_start is not None:
        answer_end = answer_start + len(answer)
        for i in range(answer_start, answer_end):
            if i < len(answer_processed):
                answer_processed[i] = 1

    #answer_processed = self.pad_or_trim_sequence(list(answer_processed), pad="back", trim="front", seq_length=self.sequence_length_input)
    return answer_processed, answer_start, answer_end

def pad_one_sequence(sequence, length, dtype='int32', padding='post', truncating='pre', value=0.0):
    sequence = tf.expand_dims(sequence, axis=0)
    sequence_padded = pad_sequences(sequence, length, dtype, padding, truncating, value)
    sequence_padded = tf.squeeze(sequence_padded, axis=0)
    return(sequence_padded)

#foo = list(range(10))
#bar = "hello i am very hungry what about you".split()
#print(get_length_adjusted_sequence(bar, 12, padding_pos = "back", trimming_pos = "front"))

'''

foo = list(range(10))
print(foo)
print(get_trimmed_sequence(foo,len(foo), 3))
print(get_padded_sequence(foo, len(foo), 100))
print(len(get_padded_sequence(foo, 100, 100)))

example = "I am hungry and I want to eat a burger, please."
example = nlp(example)
answer = "eat a burger"
answer = nlp(answer)
answer_processed, answer_start, answer_end = get_answer_processed(answer, example)
print([example.vocab.strings[token.lower] for token in example])
print(answer_processed)
print(answer_start)
print(answer_end)



if (answer_end is not None):
    answer_trimmed = get_trimmed_sequence(answer_processed, answer_end, 5)
    print(answer_trimmed)

SOS = "<SOS>"
EOS = "<EOS>"
UNK = "<UNK>"
PAD = "<PAD>"

offset2special_token = {1 : SOS, 2 : EOS, 3 : UNK, 4 : PAD}
special_token2offset = {SOS : 1, EOS : 2, UNK : 3, PAD :4}
len_special_tokens = len(special_token2offset.keys())

def special_token2index(vocabulary_size, special_token):
    return(vocabulary_size + special_token2offset[special_token])

def index2special_token(vocabulary_size, index):
    return(offset2special_token[index - vocabulary_size])

def token2index(w2v, token):
    vocab = w2v.wv.vocab
    if vocab.get(token):
        return(vocab.get(token).index)
    else:
        return (special_token2index(len(vocab), token))

def index2token(w2v, index):
    vocabulary_size = len(w2v.wv.vocab)
    if (index < vocabulary_size):
        return (w2v.wv.index2word[index])
    else:
        return (index2special_token(vocabulary_size, index))


#Preprocess
removePunctuation = lambda sentence: [''.join([c for c in word if not c in string.punctuation]) for word in sentence]
removeEmptyTokens = lambda sentence: [x for x in sentence if not x == ""]
lowerCase = lambda sentence: [x.lower() for x in sentence]
removeStopWords = lambda sentence: [x for x in sentence if not x in "for a of the and to in".split()]

def addEOS(sentence, vocabulary_size):
    sentence.append(special_token2index(vocabulary_size, EOS))
    return sentence

def addSOS(sentence, vocabulary_size):
    sentence.insert(0, special_token2index(vocabulary_size, SOS))
    return sentence

def padSequence(sentence, max_sentence_len, vocabulary_size):
    #max sentence length + SOS + EOS
    while len(sentence) < max_sentence_len:
        sentence.insert(0, special_token2index(vocabulary_size, PAD))

def getIndexesForSentence(sentence, w2v):
    vocab = w2v.wv.vocab
    unk = special_token2index(len(vocab), UNK)
    return([vocab.get(word).index  if vocab.get(word) is not None else unk for word in sentence])

def getIndexesForCorpus(corpus, w2v):
    return ([getIndexesForSentence(sentence, w2v) for sentence in corpus])



def createEmbeddingsMatrix(vocabulary_size, W2V_DIM, w2v_model):
    embedding_matrix = np.zeros((vocabulary_size + len_special_tokens, W2V_DIM + len_special_tokens))
    for i in range(vocabulary_size):
        embedding_vector = w2v_model.wv[w2v_model.wv.index2word[i]]
        if embedding_vector is not None:
            embedding_matrix[i] = np.append(embedding_vector, np.zeros(len_special_tokens))
    for i in range(vocabulary_size, vocabulary_size + len_special_tokens):
        offset = i - vocabulary_size
        appendix = to_categorical(offset, len_special_tokens)
        embedding_matrix[i] = np.append(np.zeros(W2V_DIM), appendix)
    return (embedding_matrix)
    
'''






