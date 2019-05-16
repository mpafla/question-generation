import pickle
import time
import os
import spacy
import tensorflow as tf
from vocabulary import Vocabulary

class DatasetCreator():
    def __init__(self, vocab):
        self.vocab = vocab
        self.folder_prefix_train = "data/train/"
        self.folder_prefix_dev = "data/dev/"
        self.files_folder_train = os.listdir(self.folder_prefix_train)
        self.files_folder_dev = os.listdir(self.folder_prefix_dev)

    def preprocess(self, chunk):
        answers = []
        contexts = []
        questions = []
        for paragraph in chunk["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                question = qa["question"]
                answer = qa["answers"][0]["text"]
                answers.append(answer)
                contexts.append(context)
                questions.append(question)


        return [1,2], [3,4], [5,6]

    def load_and_preprocess(self, chunk_path, files_folder):

        #this is painful - transformation of EagerTensor into string
        chunk_path = tf.compat.as_str_any(chunk_path.numpy())
        files_folder = tf.compat.as_str_any(files_folder.numpy())



        with open(files_folder + chunk_path, 'rb') as handle:
            chunk = pickle.load(handle)

        return self.preprocess(chunk)

    def create_sub_dataset(self, chunk_path, files_folder):
        answers, contexts, questions = tf.py_function(self.load_and_preprocess, [chunk_path, files_folder],  [tf.int32, tf.int32, tf.int32])
        return tf.data.Dataset.from_tensor_slices((answers, contexts, questions))

    def create_datasets(self):
        dataset_train = tf.data.Dataset.from_tensor_slices(self.files_folder_train)
        dataset_train = dataset_train.flat_map(lambda chunk_path: self.create_sub_dataset(chunk_path, self.folder_prefix_train))

        dataset_dev = tf.data.Dataset.from_tensor_slices(self.files_folder_dev)
        dataset_dev = dataset_dev.flat_map(lambda chunk_path: self.create_sub_dataset(chunk_path, self.folder_prefix_dev))
        return(dataset_train, dataset_dev)


vocab = Vocabulary()


dataset_creator = DatasetCreator(vocab)
dataset_train, dataset_dev = dataset_creator.create_datasets()

for i, data in enumerate(dataset_train):
    print(data)
    if i > 1:
        break

