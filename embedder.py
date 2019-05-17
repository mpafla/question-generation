import time
import numpy as np
from gensim import models
from gensim.models import KeyedVectors
from tensorflow.python.keras.utils import to_categorical

class Embedder():
    def __init__(self):
        self.wv_path = "models/google_wv"

        try:
            print("Loading {}".format(self.wv_path))
            start_time_loading = time.time()
            self.wv = KeyedVectors.load(self.wv_path, mmap='r')
            print("Loaded in {} sec".format(time.time() - start_time_loading))
        except:
            print("Loading failed")
            w2v_path = 'models/GoogleNews-vectors-negative300.bin'
            print("Loading {} into memory". format(w2v_path))
            start_time_loading = time.time()
            w2v = models.KeyedVectors.load_word2vec_format(w2v_path, binary=True)
            print("Loaded in {} sec".format(time.time() - start_time_loading))

            self.wv = w2v.wv

            start_time_loading = time.time()
            print("Saving {}".format(self.wv_path))
            w2v.wv.save(self.wv_path)
            print("Saved in {} sec".format(time.time() - start_time_loading))

        self.embedding_dim = self.wv[self.wv.index2word[0]].shape[0]
        self.index_len = len(self.wv.index2word)
        self.SOS = "<SOS>"
        self.EOS = "<EOS>"
        self.UNK = "<UNK>"
        self.PAD = "<PAD>"
        self.special_tokens_list = [self.SOS, self.EOS, self.UNK, self.PAD]
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
            return (self._get_data_for_special_token(self.UNK, "token")[encoding])

    def from_token(self, token, encoding):
        token_in_vocab = self.wv.vocab.get(token)
        if token_in_vocab is not None:
            if encoding == "index":
                return(token_in_vocab.index)
            elif encoding == "wv":
                return(np.append(self.wv[token], np.zeros(self.number_of_special_tokens)))
        else:
            return(self.get_special_token(token, encoding))

    def from_index(self, index, encoding):
        if index < self.index_len:
            if(encoding == "token"):
                return(self.wv.index2word[index])
            elif encoding == "wv":
                return(np.append(self.wv[self.wv.index2word[index]], np.zeros(self.number_of_special_tokens)))
        else:
            special_token = self._get_data_for_special_token(index, "index")
            if special_token is not None:
                return(special_token[encoding])
            else:
                raise Exception("Index {} is not part of vocabulary.".format(index))

#embedder = Embedder()

#print(embedder.from_token("foo", "wv"))
#print(embedder.from_token("foo", "index"))

#print(embedder.from_token("<UNK>", "wv"))
#print(embedder.from_token("<UNK>", "index"))

#print(embedder.from_token("skv[okd[vds", "wv"))
#print(embedder.from_token("skv[okd[vds", "index"))