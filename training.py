import sys
import os
import yaml
import time
import numpy as np

from vocabulary import Vocabulary
from embedder import Embedder
from trainer import Trainer
from data_preprocessor import DataPreprocessor
import tensorflow as tf
from models.layers.embeddings_layer import get_embeddings_layer
from utils.constants import Constants
from tensorflow.python.keras.metrics import CategoricalAccuracy


def main():
    configs_path = sys.argv[1:][0]
    with open(configs_path, 'r') as file:
        config = yaml.safe_load(file)

    vocab = Vocabulary(config)

    data_preprocessor = DataPreprocessor(vocab, config)
    dataset_train, dataset_dev = data_preprocessor.create_datasets()

    model_file_name = "models." + config["model"]["file_name"]
    model_class_name = config["model"]["class_name"]

    embedder = Embedder(vocab, config)
    embeddings_matrix = embedder.get_embeddings_matrix()

    module = __import__(model_file_name, fromlist=[model_class_name])
    model = getattr(module, model_class_name)(config, embeddings_matrix)

    trainer = Trainer(config, vocab)

    trainer.train(model, dataset_train, dataset_dev)





if __name__ == "__main__":
    main()