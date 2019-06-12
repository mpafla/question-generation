import os
import spacy
import json
from tqdm import tqdm
import pickle
import tensorflow as tf
from utils.constants import Constants
from utils.preprocessing import *
from tensorflow.python.keras.utils import to_categorical

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

        self.spacy_model = config["data_preprocessor"]["spacy_model"]
        self.nlp = spacy.load(self.spacy_model)

        self.pos_list = list(nlp.tokenizer.vocab.morphology.tag_map.keys())
        #reserve 0 index for padding
        #self.pos_list.insert(0, "PAD")
        self.number_of_pos = len(self.pos_list)


        if self.preprocess:
            self.squad_train_path = config["data_preprocessor"]["squad_train_path"]
            self.squad_dev_path = config["data_preprocessor"]["squad_dev_path"]

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
        feature = []
        target = []

        for paragraph in chunk["paragraphs"]:
            context = paragraph["context"]
            context_tokenized = [context.vocab.strings[token.lower] for token in context] + [Constants.EOS]

            #check if context is within limit
            if len(context_tokenized) < self.sequence_length_input:
                context_indexed = [self.vocab.get_index_for_token(token) for token in context_tokenized]

                for qa in paragraph["qas"]:
                    question = qa["question"]
                    question_tokenized = [Constants.SOS] + [question.vocab.strings[token.lower] for token in question] + [Constants.EOS]

                    # check if question is witin limit
                    if len(question_tokenized) < self.sequence_length_target:
                        question_indexed = [self.vocab.get_index_for_token(token) for token in question_tokenized]

                        answer = qa["answers"][0]["text"]
                        answer_tokenized = [answer.vocab.strings[token.lower] for token in answer]
                        answer_processed, answer_start, answer_end = get_answer_processed(answer_tokenized, context_tokenized)
                        answer_expanded = tf.expand_dims(answer_processed, axis=1)

                        # check if there is at least one answer word
                        if (max(answer_processed) > 0):

                            #plus one to have index 0 reserved for padding
                            pos_tags_context = [to_categorical(self.pos_list.index(token.tag_), num_classes=self.number_of_pos, dtype="int32") for token in context] + [[0] * self.number_of_pos]
                            pos_tags_context = tf.convert_to_tensor(pos_tags_context)

                            features = tf.concat([answer_expanded, pos_tags_context], axis=1)

                            context_padded = pad_one_sequence(context_indexed, self.sequence_length_input)
                            features_padded = pad_one_sequence(features, self.sequence_length_input)
                            question_padded = pad_one_sequence(question_indexed, self.sequence_length_target)

                            input.append(context_padded)
                            feature.append(features_padded)
                            target.append(question_padded)
                            #target_one_hot.append(question_processed_one_hot)
                            # target.append((question_processed, question_processed_one_hot))

        #return input, target_one_hot
        return input, feature, target


    def load_and_preprocess(self, chunk_path, files_folder):
        # this is a bit painful - transformation of EagerTensor into string
        chunk_path = tf.compat.as_str_any(chunk_path.numpy())
        files_folder = tf.compat.as_str_any(files_folder.numpy())

        with open(files_folder + chunk_path, 'rb') as handle:
            chunk = pickle.load(handle)

        return self.preprocess_chunk(chunk)


    def create_sub_dataset(self, chunk_path, files_folder):
        input, feature, target_one_hot = tf.py_function(self.load_and_preprocess, [chunk_path, files_folder],
                                                       [tf.float32, tf.int32, tf.float32])

        input_dataset = tf.data.Dataset.from_tensor_slices(input)
        feature_dataset = tf.data.Dataset.from_tensor_slices(feature)
        target_one_hot_dataset = tf.data.Dataset.from_tensor_slices(target_one_hot)

        #data_set = tf.data.Dataset.zip((input_dataset, target_one_hot_dataset))
        data_set = tf.data.Dataset.zip((input_dataset, feature_dataset, target_one_hot_dataset))
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