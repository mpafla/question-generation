import pickle
import string
from collections import Counter
from utils.constants import Constants

class Vocabulary:
    def __init__(self):
        self.vocabulary_size = 20000
        self.vocab_path = "data/vocab.pickle"
        self.vocab = None
        self.vocab_limited = None
        self.index2token = {}
        self.token2index = {}
        self.stop_words = "for a of the and to in".split()
        self.punctuation = [p for p in string.punctuation]
        self.filter_words = self.stop_words + self.punctuation + ["-PRON-"]

        try:
            self.load_vocab()
            self.create_limited_vocab()

        except:
            self.vocab = Counter()

    #tokenized sentence
    def add_sentence_to_vocab(self, sentence):
        self.vocab.update(sentence)

    def create_limited_vocab(self):
        self.vocab_limited = self.vocab
        for filter_word in self.filter_words:
            del self.vocab_limited[filter_word]
        self.vocab_limited = self.vocab_limited.most_common(self.vocabulary_size)
        self.vocab_limited = {k: v for (k,v) in self.vocab_limited}
        self.initialize_index_vocabularies()

    def initialize_index_vocabularies(self):
        self.index2token = Constants.index2special_token
        self.token2index = Constants.special_token2index
        number_of_special_tokens = len(self.index2token.keys())
        for i, token in enumerate(self.vocab_limited.keys(), number_of_special_tokens):
            self.token2index[token] = i
            self.index2token[i] = token

    def get_index_for_token(self, token):
        if self.is_token_in_vocabulary(token):
            return(self.token2index[token])
        else:
            return(self.token2index[Constants.UNK])

    def get_token_for_index(self, index):
        return(self.index2token[index])

    def is_token_in_vocabulary(self, token):
        return(token in self.token2index.keys())

    def load_vocab(self):
        with open(self.vocab_path, 'rb') as handle:
            self.vocab = pickle.load(handle)

    def save_vocab(self):
        with open(self.vocab_path, 'wb') as handle:
            pickle.dump(self.vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)


