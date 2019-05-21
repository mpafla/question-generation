import time
import numpy as np
import pickle
from gensim import models
from gensim.models import KeyedVectors
from tensorflow.python.keras.utils import to_categorical
from utils.constants import Constants

class Embedder():
    def __init__(self, vocab, config):
        self.wv_path = config["embedder"]["wv_path"]
        self.google_w2v_path = config["embedder"]["google_w2v_path"]
        self.embeddings_matrix_path = config["embedder"]["embeddings_matrix_path"]
        self.vocab = vocab

    def load_embeddings_model(self):
        try:
            print("Loading {}".format(self.wv_path))
            start_time_loading = time.time()
            self.wv = KeyedVectors.load(self.wv_path, mmap='r')
            print("Loaded in {} sec".format(time.time() - start_time_loading))
        except:
            print("Loading failed")
            print("Loading {} into memory". format(self.google_w2v_path))
            start_time_loading = time.time()
            w2v = models.KeyedVectors.load_word2vec_format(self.google_w2v_path, binary=True)
            print("Loaded in {} sec".format(time.time() - start_time_loading))

            self.wv = w2v.wv

            start_time_loading = time.time()
            print("Saving {}".format(self.wv_path))
            w2v.wv.save(self.wv_path)
            print("Saved in {} sec".format(time.time() - start_time_loading))

        self.embedding_dim = self.wv[self.wv.index2word[0]].shape[0]
        self.index_len = len(self.wv.index2word)

        self.special_tokens_list = list(Constants.special_token2index.keys())
        self.number_of_special_tokens = len(self.special_tokens_list)
        self.special_tokens = []
        for i, special_token in enumerate(self.special_tokens_list):
            self.special_tokens.append({
                "token": special_token,
                "index": self.index_len + i,
                "wv": np.append(np.zeros(self.embedding_dim), to_categorical(i, len(self.special_tokens_list)))
            })

    def _get_data_for_special_token(self, data, data_type):
        for special_token in self.special_tokens:
            if data == special_token[data_type]:
                return (special_token)
        return(None)

    def get_special_token(self, token, encoding):
        special_token = self._get_data_for_special_token(token, "token")
        if special_token is not None:
            return (special_token[encoding])
        else:
            return (self._get_data_for_special_token(Constants.UNK, "token")[encoding])

    def from_token(self, token, encoding):
        #token_in_vocab = self.wv.vocab.get(token)
        #if token_in_vocab is not None:
        if self.vocab.is_token_in_vocabulary(token):
            token_in_vocab = self.wv.vocab.get(token)
            if token_in_vocab is not None:
                if encoding == "index":
                    return(token_in_vocab.index)
                elif encoding == "wv":
                    return(np.append(self.wv[token], np.zeros(self.number_of_special_tokens)))
            else:
                return (self.get_special_token(token, encoding))
        else:
            #raise Exception("Token {} is not part of vocabulary.".format(token))
            return (self.get_special_token(token, encoding))


    def from_index(self, index, encoding):
        if index < self.vocab.get_vocab_size():
            token = self.vocab.get_token_for_index(index)
            if encoding == "token":
                return(token)
            elif encoding == "wv":
                return(self.from_token(token, "wv"))
        else:
            raise Exception("Index {} is not part of vocabulary.".format(index))

    def create_embeddings_matrix(self):
        vocab_size = self.vocab.get_vocab_size()
        self.embeddings_matrix = np.zeros((vocab_size, self.embedding_dim + self.number_of_special_tokens))
        for i in range(vocab_size):
            self.embeddings_matrix[i] = self.from_index(i, "wv")

    def get_embeddings_matrix(self):
        try:
            self.load_embeddings_matrix()
        except:
            self.load_embeddings_model()
            self.create_embeddings_matrix()
            print("Embeddings matrix created")
            self.save_embeddings_matrix()
            print("Embeddings matrix saved")
        return(self.embeddings_matrix)

    def save_embeddings_matrix(self):
        with open(self.embeddings_matrix_path, 'wb') as handle:
            pickle.dump(self.embeddings_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_embeddings_matrix(self):
        with open(self.embeddings_matrix_path, 'rb') as handle:
            self.embeddings_matrix = pickle.load(handle)

        #if index < self.index_len:
        #    if(encoding == "token"):
        #        return(self.wv.index2word[index])
        #    elif encoding == "wv":
        #        return(np.append(self.wv[self.wv.index2word[index]], np.zeros(self.number_of_special_tokens)))
        #else:
        #    special_token = self._get_data_for_special_token(index, "index")
        #    if special_token is not None:
        #        return(special_token[encoding])
        #    else:
        #        raise Exception("Index {} is not part of vocabulary.".format(index))

#embedder = Embedder()

#print(embedder.from_token("foo", "wv"))
#print(embedder.from_token("foo", "index"))

#print(embedder.from_token("<UNK>", "wv"))
#print(embedder.from_token("<UNK>", "index"))

#print(embedder.from_token("skv[okd[vds", "wv"))
#print(embedder.from_token("skv[okd[vds", "index"))