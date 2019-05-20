import pickle
import numpy as np
import time
import os
import tensorflow as tf
from vocabulary import Vocabulary
from embedder import Embedder
from utils.constants import Constants
from utils.preprocessing import *

class DatasetCreator():
    #def __init__(self, vocab, embedder):
    def __init__(self, vocab, config):
        self.vocab = vocab
        #self.embedder = embedder
        self.folder_prefix_train = config["dataset_creator"]["folder_prefix_train"]
        self.folder_prefix_dev =  config["dataset_creator"]["folder_prefix_dev"]
        self.files_folder_train = os.listdir(self.folder_prefix_train)
        self.files_folder_dev = os.listdir(self.folder_prefix_dev)
        self.padding_value = self.vocab.get_index_for_token(Constants.PAD)
        self.sequence_length_input = config["dataset_creator"]["sequence_length_input"]
        self.sequence_length_target = config["dataset_creator"]["sequence_length_target"]

    def preprocess(self, chunk):
        input = []
        target = []

        for paragraph in chunk["paragraphs"]:
            context = paragraph["context"]
            context_tokenized = [context.vocab.strings[token.lower] for token in context] + [Constants.EOS]
            context_processed = [self.vocab.get_index_for_token(token) for token in context_tokenized]

            for qa in paragraph["qas"]:
                question = qa["question"]
                question_tokenized = [question.vocab.strings[token.lower] for token in question] + [Constants.EOS]
                question_processed = [self.vocab.get_index_for_token(token) for token in question_tokenized]

                answer = qa["answers"][0]["text"]
                answer_tokenized = [answer.vocab.strings[token.lower] for token in answer]
                answer_processed, answer_start, answer_end = get_answer_processed(answer_tokenized, context_tokenized)

                answer_processed = get_length_adjusted_sequence(answer_processed, desired_length=self.sequence_length_input,  padding_pos="back", trimming_pos="front")
                context_processed = get_length_adjusted_sequence(context_processed, desired_length=self.sequence_length_input,  padding_pos="back", trimming_pos="front")
                question_processed = get_length_adjusted_sequence(question_processed, desired_length=self.sequence_length_target, padding_pos="back", trimming_pos="back")
                question_processed = to_categorical(question_processed, self.vocab.get_vocab_size())

                #Check if answer was not trimmed and encoded correctly
                if (max(answer_processed) > 0):
                    input.append((answer_processed, context_processed))
                    target.append(question_processed)

        return input, target

    def load_and_preprocess(self, chunk_path, files_folder):

        #this is a bit painful - transformation of EagerTensor into string
        chunk_path = tf.compat.as_str_any(chunk_path.numpy())
        files_folder = tf.compat.as_str_any(files_folder.numpy())

        with open(files_folder + chunk_path, 'rb') as handle:
            chunk = pickle.load(handle)

        return self.preprocess(chunk)

    def create_sub_dataset(self, chunk_path, files_folder):
        input, target = tf.py_function(self.load_and_preprocess, [chunk_path, files_folder],  [tf.float32, tf.float32])

        input_dataset = tf.data.Dataset.from_tensor_slices(input)
        target_dataset = tf.data.Dataset.from_tensor_slices(target)

        data_set = tf.data.Dataset.zip((input_dataset, target_dataset))
        return data_set

    def create_datasets(self):
        dataset_train = tf.data.Dataset.from_tensor_slices(self.files_folder_train)
        dataset_train = dataset_train.flat_map(lambda chunk_path: self.create_sub_dataset(chunk_path, self.folder_prefix_train))

        dataset_dev = tf.data.Dataset.from_tensor_slices(self.files_folder_dev)
        dataset_dev = dataset_dev.flat_map(lambda chunk_path: self.create_sub_dataset(chunk_path, self.folder_prefix_dev))
        return(dataset_train, dataset_dev)


#vocab = Vocabulary()
#embedder = Embedder()


#dataset_creator = DatasetCreator(vocab)
#dataset_train, dataset_dev = dataset_creator.create_datasets()



#start = time.time()

#for i, data in enumerate(dataset_train):
#    print(data)
#    if i > 1:
#        break


#print("{} time pased".format(time.time() - start))

'''
    def preprocess_tokens(self, doc, max_seq_len):
        wv = []
        for token in doc:
            token_lemma = token.lemma_
            if self.vocab.is_token_in_vocabulary(token_lemma):
                wv.append(self.embedder.from_token(token_lemma, "wv"))
            else:
                wv.append(self.embedder.get_special_token(token_lemma, "wv"))

        #Add end of line token
        wv.append(self.embedder.get_special_token(self.embedder.EOS, "wv"))
        #return wv
        return np.ones(5)
'''

