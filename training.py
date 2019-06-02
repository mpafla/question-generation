import sys
import os
import yaml
import time
import numpy as np

from vocabulary import Vocabulary
from dataset_creator import DatasetCreator
from embedder import Embedder
from trainer import Trainer
import tensorflow as tf
from models.layers.embeddings_layer import get_embeddings_layer
from utils.constants import Constants
from tensorflow.python.keras.metrics import CategoricalAccuracy

#@tf.function
def main():
    configs_path = sys.argv[1:][0]
    with open(configs_path, 'r') as file:
        config = yaml.safe_load(file)

    model_file_name = "models." + config["model"]["file_name"]
    model_class_name = config["model"]["class_name"]

    vocab = Vocabulary(config)

    dataset_creator = DatasetCreator(vocab, config)
    dataset_train, dataset_dev = dataset_creator.create_datasets()

    embedder = Embedder(vocab, config)
    embeddings_matrix = embedder.get_embeddings_matrix()

    module = __import__(model_file_name, fromlist=[model_class_name])
    model = getattr(module, model_class_name)(config, embeddings_matrix)

    trainer = Trainer(config, vocab)

    #print(tf.autograph.to_code(trainer.train.python_function))

    trainer.train(model, dataset_train, dataset_dev)

    #for data in dataset_train:
    #    print(data)
    #    break


if __name__ == "__main__":
    main()