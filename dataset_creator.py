import pickle
import numpy as np
import time
import os
import spacy
import tensorflow as tf
from vocabulary import Vocabulary
from embedder import Embedder
from constants import Constants

class DatasetCreator():
    def __init__(self, vocab, embedder):
        self.vocab = vocab
        self.embedder = embedder
        self.folder_prefix_train = "data/train/"
        self.folder_prefix_dev = "data/dev/"
        self.files_folder_train = os.listdir(self.folder_prefix_train)
        self.files_folder_dev = os.listdir(self.folder_prefix_dev)
        self.padding_value = self.vocab.get_index_for_token(Constants.PAD)
        self.sequence_length_input = 300
        self.sequence_length_target = 30


    def trim_sequence(self, sequence, trim_count, trim):
        if trim == "front":
            return(sequence[trim_count:])
        elif trim == "back":
            return(sequence[:len(sequence) - trim_count])

    def pad_or_trim_sequence(self, sequence, pad, trim, seq_length):
        pads_to_add = seq_length - len(sequence)
        #padding
        if pads_to_add > 0:
            if pad == "front":
                return([self.padding_value] * pads_to_add + sequence)
            elif pad == "back":
                return (sequence + [self.padding_value] * pads_to_add)
        #trimming
        elif pads_to_add < 0:
            trim_count = abs(pads_to_add)
            return(self.trim_sequence(sequence, trim_count, trim))
        #do nothing
        else:
            return(sequence)

    def preprocess_tokens(self, doc, pad="back", trim="back", index_only=False, seq_length=100):
        indices = []
        for token in doc:
            token_lemma = token.lemma_
            indices.append(self.vocab.get_index_for_token(token_lemma))

        #Add end of line token
        indices.append(self.vocab.get_index_for_token(Constants.EOS))

        if not index_only:
            indices = self.pad_or_trim_sequence(indices, pad, trim, seq_length)

        return indices

    def get_answer_processed(self, answer, context):
        answer_lemma = []
        context_lemma = []
        for token in answer:
            answer_lemma.append(token.lemma_)
        for token in context:
            context_lemma.append(token.lemma_)
        answer_start = None
        for i in range(len(context) - len(answer_lemma)):
            if answer_lemma == context_lemma[i:i+len(answer_lemma)]:
                answer_start = i
                break

        answer_processed = np.zeros(len(context))

        if answer_start is not None:
            for i in range(answer_start, answer_start + len(answer)):
                if i < len(answer_processed):
                    answer_processed[i] = 1

        answer_processed = self.pad_or_trim_sequence(list(answer_processed), pad="back", trim="front", seq_length=self.sequence_length_input)
        return answer_processed


    def preprocess(self, chunk):
        input = []
        target = []
        #data_points = []

        for paragraph in chunk["paragraphs"]:
            context = paragraph["context"]

            for qa in paragraph["qas"]:
                question = qa["question"]
                answer = qa["answers"][0]["text"]
                answer_start = qa["answers"][0]["answer_start"]

                #because context might be trimmed
                answer_offset = max(0, len(context) - self.sequence_length_input)

                #max_seq_len = max([len(answer),len(context), len(question)])



                answer_processed = self.get_answer_processed(answer, context)
                original_answer_processed = self.preprocess_tokens(answer, pad="back", trim="front", seq_length=self.sequence_length_input)
                context_processed = self.preprocess_tokens(context, pad="back", trim="front", seq_length=self.sequence_length_input)
                question_processed = self.preprocess_tokens(question, seq_length=self.sequence_length_target)

                #data_point = (answer_processed, context_processed, question_processed)
                #data_point = (np.ones(3), np.ones(3), np.ones(3))
                #data_points.append(data_point)

                input.append((answer_processed, original_answer_processed, context_processed))
                target.append(question_processed)

        #return data_points
        #return np.ones((2,3,2)), np.zeros((2,2))
        return input, target

    def load_and_preprocess(self, chunk_path, files_folder):

        #this is painful - transformation of EagerTensor into string
        chunk_path = tf.compat.as_str_any(chunk_path.numpy())
        files_folder = tf.compat.as_str_any(files_folder.numpy())

        with open(files_folder + chunk_path, 'rb') as handle:
            chunk = pickle.load(handle)

        return self.preprocess(chunk)

    def create_sub_dataset(self, chunk_path, files_folder):
        #data_points = tf.py_function(self.load_and_preprocess, [chunk_path, files_folder],  [tf.float32, tf.float32, tf.float32])
        #data_set = tf.data.Dataset.from_tensor_slices(data_points)

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


vocab = Vocabulary()
embedder = Embedder()


dataset_creator = DatasetCreator(vocab, embedder)
dataset_train, dataset_dev = dataset_creator.create_datasets()



start = time.time()

for i, data in enumerate(dataset_train):
    print(data)
    if i > 1:
        break


print("{} time pased".format(time.time() - start))

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

