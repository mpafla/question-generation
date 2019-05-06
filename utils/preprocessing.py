import string
import numpy as np
from nltk.tokenize import word_tokenize
from functools import reduce
from tensorflow.python.keras.utils import to_categorical

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
