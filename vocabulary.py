import pickle
from collections import Counter

class Vocabulary:
    def __init__(self):
        self.vocabulary_size = 20000
        self.vocab_path = "data/vocab.pickle"
        self.vocab = None
        self.vocab_limited = None
        self.index2token = {}
        self.token2index = {}

        try:
            self.load_vocab()
            self.create_limited_vocab()
        except:
            self.vocab = Counter()

    #tokenized sentence
    def add_sentence_to_vocab(self, sentence):
        self.vocab.update(sentence)

    def create_limited_vocab(self):
        self.vocab_limited = self.vocab.most_common(self.vocabulary_size)
        self.initialize_index_vocabularies()

    def initialize_index_vocabularies(self):
        for i, token in enumerate(self.vocab_limited):
            self.token2index[token] = i
            self.index2token[i] = token

    def get_index_for_token(self, token):
        return(self.token2index[token])

    def get_token_for_index(self, index):
        return(self.index2token[index])

    def is_token_in_vocabulary(self, token):
        return(token in self.vocab_limited)

    def load_vocab(self):
        with open(self.vocab_path, 'rb') as handle:
            self.vocab = pickle.load(handle)

    def save_vocab(self):
        with open(self.vocab_path, 'wb') as handle:
            pickle.dump(self.vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)

