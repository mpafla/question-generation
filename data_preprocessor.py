import os
import spacy
import json
from tqdm import tqdm
import pickle
import tensorflow as tf
from utils.constants import Constants
from utils.preprocessing import *

class DataPreprocessor():
    def __init__(self, vocab, config):
        self.vocab = vocab
        self.preprocess = config["data_preprocessor"]["preprocess"]
        self.folder_prefix_train = config["data_preprocessor"]["folder_prefix_train"]
        self.folder_prefix_dev = config["data_preprocessor"]["folder_prefix_dev"]
        self.chunks_included_train = config["data_preprocessor"]["chunks_included_train"]
        self.chunks_included_dev = config["data_preprocessor"]["chunks_included_dev"]

        self.cycle_length = config["data_preprocessor"]["cycle_length"]

        self.sequence_length_input = config["data_preprocessor"]["sequence_length_input"]
        self.sequence_length_target = config["data_preprocessor"]["sequence_length_target"]


        if self.preprocess:
            self.squad_train_path = config["data_preprocessor"]["squad_train_path"]
            self.squad_dev_path = config["data_preprocessor"]["squad_dev_path"]
            self.spacy_model = config["data_preprocessor"]["spacy_model"]
            self.nlp = spacy.load(self.spacy_model)

            print("Loading raw data")
            self.load_raw_data()

            print("Preprocessing raw data")
            self.preprocess_raw_data()

    def load_raw_data(self):
        with open(self.squad_train_path, "r", encoding='utf-8') as reader:
            self.json_train = json.load(reader)["data"]

        with open(self.squad_dev_path, "r", encoding='utf-8') as reader:
            self.json_dev = json.load(reader)["data"]


    def create_spacy_and_vocab(self, text):
        doc = self.nlp(text)

        #Fill vocab
        text_lower = [doc.vocab.strings[token.lower] for token in doc]
        self.vocab.add_sentence_to_vocab(text_lower)

        return(doc)

    def preprocess_json(self, input_data, folder_path):
        data_id = 0
        for data in tqdm(input_data):
            for paragraph in data["paragraphs"]:
                paragraph["context"] = self.create_spacy_and_vocab(paragraph["context"])
                for qa in paragraph["qas"]:
                    qa["question"] = self.create_spacy_and_vocab(qa["question"])
                    for a in qa["answers"]:
                        a["text"] = self.create_spacy_and_vocab(a["text"])
            with open(folder_path + str(data_id) + ".pickle", 'wb') as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            data_id = data_id + 1

    def preprocess_raw_data(self):

        self.preprocess_json(self.json_train, self.folder_prefix_train)
        self.preprocess_json(self.json_dev, self.folder_prefix_dev)

        self.vocab.save_vocab()
        self.vocab.create_limited_vocab()

    def preprocess_chunk(self, chunk):
        input = []
        target = []
        target_one_hot = []

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

                answer_processed = get_length_adjusted_sequence(answer_processed, desired_length=self.sequence_length_input,
                                                                padding_pos="back", trimming_pos="front")
                context_processed = get_length_adjusted_sequence(context_processed,
                                                                 desired_length=self.sequence_length_input,
                                                                 padding_pos="back", trimming_pos="front")
                question_processed = get_length_adjusted_sequence(question_processed,
                                                                  desired_length=self.sequence_length_target,
                                                                  padding_pos="back", trimming_pos="back")
                question_processed_one_hot = to_categorical(question_processed, self.vocab.get_vocab_size())

                # Check if answer was not trimmed and encoded correctly
                if (max(answer_processed) > 0):
                    input.append((answer_processed, context_processed))
                    target.append(question_processed)
                    #target_one_hot.append(question_processed)
                    target_one_hot.append(question_processed_one_hot)
                    # target.append((question_processed, question_processed_one_hot))

        return input, target, target_one_hot


    def load_and_preprocess(self, chunk_path, files_folder):
        # this is a bit painful - transformation of EagerTensor into string
        chunk_path = tf.compat.as_str_any(chunk_path.numpy())
        files_folder = tf.compat.as_str_any(files_folder.numpy())

        with open(files_folder + chunk_path, 'rb') as handle:
            chunk = pickle.load(handle)

        return self.preprocess_chunk(chunk)


    def create_sub_dataset(self, chunk_path, files_folder):
        input, target, target_one_hot = tf.py_function(self.load_and_preprocess, [chunk_path, files_folder],
                                                       [tf.float32, tf.float32, tf.float32])

        input_dataset = tf.data.Dataset.from_tensor_slices(input)
        target_dataset = tf.data.Dataset.from_tensor_slices(target)
        target_one_hot_dataset = tf.data.Dataset.from_tensor_slices(target_one_hot)

        data_set = tf.data.Dataset.zip((input_dataset, target_dataset, target_one_hot_dataset))
        return data_set


    def create_datasets(self):
        files_folder_train = os.listdir(self.folder_prefix_train)
        files_folder_train = files_folder_train[:self.chunks_included_train]

        files_folder_dev = os.listdir(self.folder_prefix_dev)
        files_folder_dev = files_folder_dev[:self.chunks_included_dev]

        dataset_train = tf.data.Dataset.from_tensor_slices(files_folder_train)
        dataset_train = dataset_train.interleave(
            lambda chunk_path: self.create_sub_dataset(chunk_path, self.folder_prefix_train), cycle_length=self.cycle_length, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        dataset_dev = tf.data.Dataset.from_tensor_slices(files_folder_dev)
        dataset_dev = dataset_dev.interleave(lambda chunk_path: self.create_sub_dataset(chunk_path, self.folder_prefix_dev), cycle_length=self.cycle_length, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return (dataset_train, dataset_dev)